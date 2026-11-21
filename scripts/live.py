#!/usr/bin/env python3
"""Clip a live Twitch stream while it is still going.

    python live.py <channel> <workdir> [--minutes 60] [--model sonnet] [--min-score 7]

What it does, all at once:
  1. pulls the live video in 2-second pieces and keeps the last ~20 minutes on disk
  2. reads the public chat anonymously (Twitch IRC, read-only "justinfan" login)
  3. watches two signals: chat suddenly much faster than usual, streamer suddenly much louder
  4. when both point at a moment: cuts 40 s around it, transcribes it, asks Claude (headless
     `claude -p`) to score it 0-10 and write a title
  5. 7+ moments are rendered as vertical clips into <workdir>/clips/

Events go to <workdir>/events.log, one line each, for a monitor to follow.
Only for streamers who allow clipping (official campaign or written permission).
"""
import argparse, json, os, re, socket, subprocess, sys, threading, time, random, urllib.request, glob, shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UA = {"User-Agent": "Mozilla/5.0"}
HYPE = re.compile(r"(LMAO|LMFAO|KEKW|OMEGALUL|LUL|💀|😭|😂|\?\?\?|NAHH+|W+\b|CLIP|BRO+|OMG|WTF|HOLY|NO WAY|LOL)", re.I)


def log(wd, msg):
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(wd, "events.log"), "a") as f:
        f.write(line + "\n")


def get(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def pdt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Video(threading.Thread):
    """Follows the live playlist, saves live (non-ad) pieces, measures loudness."""

    def __init__(self, channel, wd, fmt):
        super().__init__(daemon=True)
        self.channel, self.wd, self.fmt = channel, wd, fmt
        self.segdir = os.path.join(wd, "seg")
        os.makedirs(self.segdir, exist_ok=True)
        self.segs = deque()          # (t_broadcast, dur, path)
        self.loud = deque()          # (t_broadcast, dB) every 0.5 s
        self.seen = set()
        self.lock = threading.Lock()
        self.latency = 8.0           # our delay behind the broadcast, measured
        self.url = None
        self.ex = ThreadPoolExecutor(4)
        self.alive = True
        self.inits = {}              # fMP4 streams: init piece per EXT-X-MAP url
        self.init_len = {}           # saved fMP4 piece -> size of the init in front of it

    def refresh(self):
        r = subprocess.run(["yt-dlp", "-g", "--no-warnings", "-f", self.fmt, f"https://www.twitch.tv/{self.channel}"],
                           capture_output=True, text=True)
        u = [l for l in r.stdout.split() if l.startswith("http")]
        self.url = u[0] if u else None

    def measure(self, path, t0):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                           capture_output=True)
        a = np.frombuffer(r.stdout, np.int16).astype(np.float32)
        n = 4000
        out = []
        for i in range(0, len(a) - n + 1, n):
            rms = np.sqrt(np.mean(a[i:i + n] ** 2)) + 1e-6
            out.append((t0 + i / 8000, 20 * np.log10(rms)))
        return out

    def fetch(self, t, dur, url, init=None):
        # old Twitch streams: MPEG-TS pieces that play on their own; newer ones: fMP4 pieces that need
        # the init piece (EXT-X-MAP) in front, so every saved piece is a standalone file either way
        p = os.path.join(self.segdir, f"{int(t * 1000)}.{'mp4' if init else 'ts'}")
        try:
            data = get(url, 20)
            if init:
                if init not in self.inits:
                    self.inits[init] = get(init, 20)
                data = self.inits[init] + data
                self.init_len[p] = len(self.inits[init])
            open(p, "wb").write(data)
        except Exception:
            return
        lv = self.measure(p, t)
        with self.lock:
            self.segs.append((t, dur, p))
            self.segs = deque(sorted(self.segs))
            self.loud.extend(lv)
            self.loud = deque(sorted(self.loud))
            while self.segs and self.segs[0][0] < t - 1200:      # keep 20 minutes
                old = self.segs.popleft()
                try: os.remove(old[2])
                except OSError: pass
            while self.loud and self.loud[0][0] < t - 1200:
                self.loud.popleft()

    def run(self):
        self.refresh()
        fails = 0
        while self.alive:
            try:
                text = get(self.url).decode()
                fails = 0
            except Exception:
                fails += 1
                if fails >= 3:
                    self.refresh(); fails = 0
                    if not self.url:
                        log(self.wd, "STREAM offline or unreachable"); time.sleep(20)
                time.sleep(2); continue
            cur_t, cur_d, title, cur_map = None, None, "", None
            for line in text.splitlines():
                if line.startswith("#EXT-X-MAP:"):
                    cur_map = line.split('URI="', 1)[1].split('"', 1)[0]
                elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                    cur_t = pdt(line.split(":", 1)[1])
                elif line.startswith("#EXTINF:"):
                    d, _, title = line[8:].partition(",")
                    cur_d = float(d)
                elif line.startswith("http") and cur_t is not None:
                    if title.startswith("live") and cur_t not in self.seen:
                        self.seen.add(cur_t)
                        self.ex.submit(self.fetch, cur_t, cur_d, line, cur_map)
                        self.latency = max(2.0, time.time() - (cur_t + cur_d))
                    cur_t = None
            time.sleep(1.5)

    def covered_until(self):
        with self.lock:
            return self.segs[-1][0] + self.segs[-1][1] if self.segs else 0

    def cut(self, t0, t1, out):
        with self.lock:
            sel = [s for s in self.segs if s[0] + s[1] > t0 and s[0] < t1]
        if not sel:
            return False
        # join by bytes: TS pieces simply concatenate; fMP4 = one init + all fragments
        lst = out + ".txt"
        joined = out + (".join.ts" if sel[0][2].endswith(".ts") else ".join.mp4")
        with open(joined, "wb") as fo:
            for k, (_, _, p) in enumerate(sel):
                b = open(p, "rb").read()
                fo.write(b if k == 0 else b[self.init_len.get(p, 0):])
        open(lst, "w").close()
        off = max(0.0, t0 - sel[0][0])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-ss", f"{off:.2f}", "-t", f"{t1 - t0:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-r", "30", "-c:a", "aac", "-b:a", "192k", out],
                       check=True)
        os.remove(lst); os.remove(joined)
        return True


class Chat(threading.Thread):
    """Anonymous read-only Twitch chat."""

    def __init__(self, channel, wd):
        super().__init__(daemon=True)
        self.channel, self.wd = channel, wd
        self.msgs = deque()          # (wall_time, user, text)
        self.lock = threading.Lock()
        self.f = open(os.path.join(wd, "chat.jsonl"), "a")

    def run(self):
        while True:
            try:
                s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=300)
                nick = f"justinfan{random.randint(10000, 99999)}"
                s.sendall(f"NICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
                buf = b""
                while True:
                    data = s.recv(65536)
                    if not data:
                        break
                    buf += data
                    *lines, buf = buf.split(b"\r\n")
                    for raw in lines:
                        l = raw.decode("utf8", "ignore")
                        if l.startswith("PING"):
                            s.sendall(b"PONG :tmi.twitch.tv\r\n"); continue
                        m = re.match(r":(\w+)!\S+ PRIVMSG #\w+ :(.*)", l)
                        if m:
                            t = time.time()
                            with self.lock:
                                self.msgs.append((t, m[1], m[2]))
                                while self.msgs and self.msgs[0][0] < t - 1200:
                                    self.msgs.popleft()
                            self.f.write(json.dumps({"t": round(t, 2), "u": m[1], "m": m[2]}) + "\n"); self.f.flush()
            except Exception as e:
                log(self.wd, f"CHAT reconnect ({e})")
                time.sleep(5)

    def window(self, a, b):
        with self.lock:
            return [x for x in self.msgs if a <= x[0] < b]


def chat_signal(chat, now):
    """Messages in the last 10 s vs the usual 10 s over the last 10 minutes."""
    with chat.lock:
        ts = np.array([m[0] for m in chat.msgs if m[0] > now - 600])
    if len(ts) < 30 or now - ts.min() < 120:
        return 0.0, 0, 0
    recent = chat.window(now - 10, now)
    bins = np.histogram(ts, bins=np.arange(now - 600, now + 0.01, 10))[0]
    base = max(np.median(bins[:-1][bins[:-1] > 0]) if (bins[:-1] > 0).any() else 1, 1)
    hype = sum(1 for _, _, m in recent if HYPE.search(m))
    clipit = sum(1 for _, _, m in recent if re.search(r"\bclip", m, re.I))
    return len(recent) / base, hype, clipit


def loud_signal(video, a, b):
    with video.lock:
        L = np.array(video.loud)
    if len(L) < 120:
        return 0.0, None
    base = np.median(L[L[:, 0] > b - 600][:, 1])
    w = L[(L[:, 0] >= a) & (L[:, 0] <= b)]
    if not len(w):
        return 0.0, None
    i = int(np.argmax(w[:, 1]))
    return float(w[i, 1] - base), float(w[i, 0])


JUDGE = """You are judging one moment from {channel}'s live Twitch stream for a short-form clip (TikTok / YouTube Shorts / Reels).
The window runs from 0 to {dur:.0f} s. The suspected peak is at about {peak:.1f} s.

Transcript (seconds, word):
{words}

Chat around it (seconds, user: message) — chat reacts a few seconds after the moment:
{chat}

Frames from the window (you MUST open each one with the Read tool before scoring; the score depends on what is on screen): {frames}

Score 0-10 the way a stranger scrolling would:
9-10 instantly gets it and stops (clear funny / shocking event + strong reaction); 7-8 strong reaction with clear cause;
4-6 loud but needs context (yelling, donations, inside jokes, reading chat); 0-3 ads, menus, nothing happening, music.
Then pick the clip like a top clipping channel would: start exactly on the line or action that sets it up (the
first second must already hook: a question, a dare, "watch this"), peak = the reaction on screen, end right after
the punchline or payoff — no trailing chatter. 8-25 s long, inside 0..{dur:.0f}. Dead air gets jump-cut automatically.
beats: up to 2 short action captions for moments where he is silent but doing something the viewer must follow,
one or two words in asterisks, e.g. {{"t": 12.0, "text": "*CHECKING*"}} or "*CRASHES*". Empty list if not needed.
Title: streamer name first, what happens, plain words, max 55 characters, no emojis, no clickbait lies.

Answer with ONLY one JSON object, nothing else:
{{"score": 0, "start": 0.0, "peak": 0.0, "end": 0.0, "beats": [], "title": "...", "reason": "one short line"}}"""


def judge(channel, cdir, dur, peak, words, chat, model):
    frames = []
    for k, t in enumerate([peak - 6, peak - 1, peak + 2, peak + 6]):
        p = os.path.abspath(os.path.join(cdir, f"f{k}.jpg"))   # absolute: claude runs with cwd=cdir
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, min(dur - 0.5, t)):.2f}", "-i",
                        os.path.join(cdir, "raw.mp4"), "-frames:v", "1", "-vf", "scale=640:-2", p])
        if os.path.exists(p):
            frames.append(p)
    prompt = JUDGE.format(channel=channel, dur=dur, peak=peak,
                          words=" ".join(f"[{t:.1f}] {w}" for w, t, _ in words) or "(no speech)",
                          chat="\n".join(f"[{t:.1f}] {u}: {m}" for t, u, m in chat[-60:]) or "(no chat)",
                          frames=", ".join(frames))
    r = subprocess.run(["claude", "-p", prompt, "--model", model, "--allowedTools", "Read", "--output-format", "text"],
                       capture_output=True, text=True, timeout=300, cwd=cdir)
    open(os.path.join(cdir, "judge_raw.txt"), "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    dec = json.JSONDecoder()
    for i, ch in enumerate(r.stdout):            # first {...} that parses and has a score
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(r.stdout[i:])
                if isinstance(obj, dict) and "score" in obj:
                    return obj
            except ValueError:
                pass
    return {"score": 0, "reason": "judge gave no JSON: " + r.stdout[-150:].replace("\n", " ")}


def auto_layout(raw):
    """Facecam box from a frame: small face -> box around it; big face -> camera stream, crop around it."""
    # The webcam face sits still in every frame; faces in the game move around. Look at 8 frames and
    # keep the face that stays in the same place in most of them.
    import cv2
    from facelib_live import faces_boxes
    dets = []
    for k, t in enumerate([2, 7, 12, 17, 22, 27, 32, 37]):
        fr = os.path.join(os.path.dirname(raw), f"layout{k}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", raw, "-frames:v", "1", "-vf", "scale=1920:1080", fr])
        img = cv2.imread(fr)
        if img is not None:
            dets.append(faces_boxes(img))
    if not dets:
        return {"facecam": None}
    best, best_n = None, 0
    for frame in dets:
        for (x, y, w, h) in frame:
            cx, cy = x + w / 2, y + h / 2
            same = [b for f in dets for b in f if abs(b[0] + b[2] / 2 - cx) < 40 and abs(b[1] + b[3] / 2 - cy) < 40]
            if len(same) > best_n:
                best_n, best = len(same), (np.median([b[0] for b in same]), np.median([b[1] for b in same]),
                                           np.median([b[2] for b in same]), np.median([b[3] for b in same]))
    if best is None or best_n < len(dets) * 0.5:
        return {"facecam": None}
    x, y, w, h = best
    if h > 1080 * 0.35:                          # camera fills the frame (IRL / just chatting)
        return {"facecam": None, "crop_x": int(x + w / 2)}
    cx, cy = x + w / 2, y + h / 2
    bw = min(1920, w * 5.5); bh = min(1080, bw * 9 / 16)
    bx = int(max(0, min(1920 - bw, cx - bw / 2))); by = int(max(0, min(1080 - bh, cy - bh * 0.5)))
    return {"facecam": [bx, by, int(bw), int(bh)], "face": [int(x), int(y), int(w), int(h)]}


def process(args, video, chat, cid, peak_t, why):
    from render import words_of, render
    wd = args.workdir
    cdir = os.path.join(wd, "cands", cid)
    os.makedirs(cdir, exist_ok=True)
    t0, t1 = peak_t - 25, peak_t + 22
    while video.covered_until() < t1 + 2:
        time.sleep(2)
    raw = os.path.join(cdir, "raw.mp4")
    if not video.cut(t0, t1, raw):
        log(wd, f"{cid} SKIP could not cut"); return
    wav = os.path.join(cdir, "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "16000", wav], check=True)
    words = words_of(wav, os.path.join(cdir, "w"))
    lat = video.latency + 3
    ch = [(m[0] - lat - t0, m[1], m[2]) for m in chat.window(t0 + lat, t1 + lat + 8)]
    v = judge(args.channel, cdir, t1 - t0, peak_t - t0, words, ch, args.model)
    v.update({"id": cid, "why": why, "wall": datetime.now().strftime("%H:%M:%S")})
    json.dump(v, open(os.path.join(cdir, "verdict.json"), "w"), indent=1)
    log(wd, f"{cid} SCORE {v.get('score')} | {v.get('title', '')} | {v.get('reason', '')}")
    if (v.get("score") or 0) < args.min_score:
        return
    s, e = float(v.get("start", 0)), float(v.get("end", t1 - t0))
    pk = float(v.get("peak", peak_t - t0))
    s, e = max(0, s), min(t1 - t0, e)
    if e - s < 8:
        s, e = max(0, pk - 15), min(t1 - t0, pk + 8)
    lay = auto_layout(raw)
    tmp = os.path.join(wd, "tmp", cid)
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(raw, os.path.join(tmp, "raw_full.mp4"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", raw, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "17", "-c:a", "aac", "-b:a", "192k", os.path.join(tmp, "raw.mp4")], check=True)
    p = {"id": cid, "start": 0.0, "peak": max(0.5, pk - s), "end": e - s, "title": v.get("title", args.channel),
         "beats": [{"t": float(b["t"]) - s, "text": b["text"]} for b in v.get("beats") or []
                   if isinstance(b, dict) and "t" in b and "text" in b and s <= float(b["t"]) <= e]}
    if "crop_x" in lay:
        p["crop_x"] = lay["crop_x"]
    try:
        r = render(p, lay, os.path.join(tmp, "raw.mp4"), wd, "best")
        log(wd, f"{cid} CLIP {r['file']} ({r['duration_s']} s) score {v.get('score')} | {p['title']}")
    except SystemExit as ex:
        log(wd, f"{cid} RENDER failed {str(ex)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("workdir")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--min-score", type=float, default=7)
    ap.add_argument("--format", default="1080p60/1080p/720p60/best")
    args = ap.parse_args()
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    video, chat = Video(args.channel, wd, args.format), Chat(args.channel, wd)
    video.start(); chat.start()
    log(wd, f"START {args.channel} for {args.minutes:.0f} min")
    workers = ThreadPoolExecutor(2)
    last, n, end = 0.0, 0, time.time() + args.minutes * 60
    stat_t = 0
    while time.time() < end:
        time.sleep(2)
        now = time.time()
        ratio, hype, clipit = chat_signal(chat, now)
        lat = video.latency
        a, b = now - lat - 16, now - lat - 2       # where the moment happened in broadcast time
        db, peak = loud_signal(video, a, b)
        if now - stat_t > 60:
            log(wd, f"STAT chat x{ratio:.1f} hype {hype} loud +{db:.1f} dB latency {lat:.0f}s segs {len(video.segs)} chat_msgs {len(chat.msgs)}")
            stat_t = now
        fire = (ratio >= 2.0 and hype >= 15) or (ratio >= 1.3 and db >= 8) or clipit >= 5 or (db >= 16 and ratio >= 1.0)
        if fire and peak and peak - last > 40:
            n += 1
            cid = f"m{n:02d}"
            why = f"chat x{ratio:.1f}, hype {hype}, 'clip' x{clipit}, loud +{db:.1f} dB"
            log(wd, f"{cid} MOMENT {why}")
            last = peak
            workers.submit(process, args, video, chat, cid, peak, why)
    log(wd, "STOP time is up, finishing open moments")
    video.alive = False
    workers.shutdown(wait=True)
    log(wd, "DONE")


if __name__ == "__main__":
    main()
