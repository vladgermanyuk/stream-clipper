#!/usr/bin/env python3
"""Turn picked moments into vertical 1080x1920 clips.

Default style "split": webcam big on top, gameplay below, one word at a time on the seam with exact word
timing (clean voice -> whisper large-v3 -> wav2vec2 alignment), jump cuts through dead air, a short red
flash on the payoff. "classic": title card on the seam, the peak shown first, then the whole moment.

    python render.py <url_or_video> <workdir> [--only ID] [--format FORMAT]

Needs in <workdir>:
  picks.json   [{"id": "c1", "start": 3410.0, "peak": 3431.5, "end": 3446.0, "title": "...", "beats": []}]
  layout.json  {"style": "split", "facecam": [x, y, w, h] or null, "face": [x, y, w, h]}  (1920x1080 pixels;
               scripts/find_layout.py writes it)
Output: <workdir>/clips/<id>.mp4
"""
import argparse, json, os, re, shutil, subprocess, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")
# Captions need exact word timing. large-v3-turbo with DTW alignment hears speech over game audio far better
# than small/base and does not collapse words onto one timestamp. Falls back to small/base if not downloaded.
MODEL_CHOICES = [("ggml-large-v3-q5_0.bin", "large.v3"), ("ggml-large-v3.bin", "large.v3"),
                 ("ggml-large-v3-turbo-q5_0.bin", "large.v3.turbo"), ("ggml-large-v3-turbo.bin", "large.v3.turbo"),
                 ("ggml-small.en.bin", "small.en"), ("ggml-base.en.bin", "base.en")]
MODEL, DTW = next(((os.path.join(MODELS_DIR, f), d) for f, d in MODEL_CHOICES if os.path.exists(os.path.join(MODELS_DIR, f))),
                  (os.path.join(MODELS_DIR, "ggml-base.en.bin"), None))
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
COND_CANDIDATES = [  # narrow bold caption font of the "split" style (as on top clipping channels)
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
]
COND = next((f for f in COND_CANDIDATES if os.path.exists(f)), FONT)


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd[:6])}...\n{r.stderr[-1500:]}")
    return r.stdout


def fetch_section(src, start, end, out, fmt):
    if os.path.exists(out):
        return
    sh([sys.executable, os.path.join(HERE, "fetch.py"), "video", src, str(start), str(end), out, "--format", fmt])


def text_png(path, text, size, wrap=None, box=None, fill="white", stroke=8, maxw=W - 120):
    font = ImageFont.truetype(FONT, size)
    if wrap and len(text) > wrap:
        ws = text.split()
        best = min(range(1, len(ws)), key=lambda i: abs(len(" ".join(ws[:i])) - len(" ".join(ws[i:]))))
        lines = [" ".join(ws[:best]), " ".join(ws[best:])]
    else:
        lines = [text]
    d0 = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    widths = [d0.textbbox((0, 0), l, font=font, stroke_width=stroke)[2] for l in lines]
    while max(widths) > maxw and size > 30:
        size -= 4
        font = ImageFont.truetype(FONT, size)
        widths = [d0.textbbox((0, 0), l, font=font, stroke_width=stroke)[2] for l in lines]
    lh = int(size * 1.18)
    pad = 28 if box else 0
    img = Image.new("RGBA", (max(widths) + pad * 2, lh * len(lines) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if box:
        d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=26, fill=box)
    for i, l in enumerate(lines):
        x = (img.width - widths[i]) // 2
        d.text((x, pad + i * lh), l, font=font, fill=fill, stroke_width=stroke,
               stroke_fill="black" if stroke else None)
    img.save(path)
    return img.size


TAG = re.compile(r"[\(\[\*♪]|^-+$")


ALIGN_PY = os.path.join(HERE, "..", ".align-venv", "bin", "python")


def exact_align(audio_wav, words, base):
    """Pin every word to the audio with wav2vec2 forced alignment (scripts/align_words.py, own venv).
    Returns the words unchanged if the aligner is not installed or fails."""
    if not words or not os.path.exists(ALIGN_PY):
        return words
    src, dst = base + ".words.json", base + ".aligned.json"
    json.dump([list(w) for w in words], open(src, "w"))
    r = subprocess.run([ALIGN_PY, os.path.join(HERE, "align_words.py"), audio_wav, src, dst], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        return words
    out, last = [], 0.0
    for w, a, b in json.load(open(dst)):
        a = max(a, last); b = max(b, a + 0.08)
        out.append((w, a, b)); last = a + 0.02
    return out


def voice_only(audio_wav, base):
    """The streamer's voice without game sound and music (Demucs htdemucs, in .align-venv). Speech over a
    game is what makes whisper mishear ("five" -> "well"); on the clean voice it hears it. Falls back to
    the original audio if Demucs is not installed."""
    if not os.path.exists(ALIGN_PY):
        return audio_wav
    out_dir = base + "_sep"
    r = subprocess.run([ALIGN_PY, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", out_dir, audio_wav],
                       capture_output=True, text=True)
    stem = os.path.join(out_dir, "htdemucs", os.path.splitext(os.path.basename(audio_wav))[0], "vocals.wav")
    if r.returncode != 0 or not os.path.exists(stem):
        return audio_wav
    voc = base + "_voice16.wav"
    sh(["ffmpeg", "-v", "error", "-y", "-i", stem, "-ac", "1", "-ar", "16000", voc])
    return voc


def words_of(audio_wav, base):
    """Exact words: clean voice -> whisper large-v3 (+DTW) -> wav2vec2 forced alignment on the clean voice."""
    voc = voice_only(audio_wav, base)
    return exact_align(voc, _whisper_words(voc, base), base)


def _whisper_words(audio_wav, base):
    """Word list [(word, start, end)] in seconds. With large-v3-turbo: DTW token times (accurate to ~0.1 s);
    otherwise whisper.cpp per-word segments, repaired."""
    if DTW and DTW.startswith("large"):
        sh(["whisper-cli", "-m", MODEL, "-f", audio_wav, "-l", "en", "-oj", "-ojf", "-of", base, "-np", "-t", "8",
            "-dtw", DTW, "-nfa"])                    # DTW needs flash attention off
        toks = [t for seg in json.load(open(base + ".json"))["transcription"] for t in seg.get("tokens", [])
                if not t["text"].startswith("[_") and t.get("t_dtw", -1) >= 0]
        words = []
        for t in toks:
            txt = t["text"]
            if txt.startswith(" ") or not words:
                words.append([txt.strip(), t["t_dtw"] / 100 - 0.06])
            else:
                words[-1][0] += txt
        words = [w for w in words if w[0]]
        if words:
            out, depth = [], 0
            for k, (w, a) in enumerate(words):
                opened = w.count("(") + w.count("["); closed = w.count(")") + w.count("]")
                inside = depth > 0 or opened > 0 or "*" in w or "♪" in w
                depth = max(0, depth + opened - closed)
                nxt = words[k + 1][1] if k + 1 < len(words) else a + 0.4
                w = w.strip(" -\"")
                if w and not inside and not TAG.search(w):
                    out.append([w, max(0.0, a), max(a + 0.1, min(nxt, a + 0.6))])
            return repair_timing(out)
    cmd = ["whisper-cli", "-m", MODEL, "-f", audio_wav, "-l", "en", "-ml", "1", "-sow", "-oj", "-of", base, "-np", "-t", "8"]
    sh(cmd)
    raw, depth = [], 0
    for seg in json.load(open(base + ".json"))["transcription"]:
        w = seg["text"].strip()
        # sound tags like "(OVERLAPPING CHATTERING)", "[MUSIC]", "*laughs*", "♪" can span several words: drop them
        opened = w.count("(") + w.count("[")
        closed = w.count(")") + w.count("]")
        inside = depth > 0 or opened > 0 or "*" in w or "♪" in w
        depth = max(0, depth + opened - closed)
        w = w.strip(" -\"")
        if w and not inside and not TAG.search(w):
            raw.append([w, seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000])
    return repair_timing(raw)


# Captions are burned into the video: slurs never appear, swears are masked like clipping channels do.
SLURS = re.compile(r"^(nigg\w*|niga\w*|nigs?|faggot\w*|fag|retard\w*|chink\w*|spic\w*|tranny\w*)$", re.I)
SWEARS = [(re.compile(p, re.I), r) for p, r in (
    (r"^motherfuck", "MOTHERF*CK"), (r"^fuck", "F*CK"), (r"^shit", "SH*T"), (r"^bitch", "B*TCH"),
    (r"^dick", "D*CK"), (r"^pussy", "P*SSY"), (r"^ass(hole)?$", "A**"), (r"^damn", "D*MN"), (r"^cock", "C*CK"))]


def clean_word(w):
    """None for a slur, masked text for a swear, the word otherwise."""
    core = re.sub(r"[^\w']", "", w)
    if SLURS.match(core):
        return None
    for rx, rep in SWEARS:
        m = rx.match(core)
        if m:
            return (rep + core[m.end():]).upper()
    return w


def repair_timing(words, min_dur=0.14):
    """whisper.cpp sometimes stacks several words on one timestamp or gives them zero length. Spread each
    stacked run across the time up to the next real word, in proportion to word length; keep order strict."""
    out, i = [], 0
    while i < len(words):
        j = i + 1
        while j < len(words) and (words[j][1] - words[i][1] < 0.05 or words[j - 1][2] - words[j - 1][1] < 0.03):
            j += 1
        run = words[i:j]
        t0 = run[0][1]
        nxt = words[j][1] if j < len(words) else max(w[2] for w in run) + 0.25 * len(run)
        t1 = max(max(w[2] for w in run), min(nxt, t0 + 0.45 * len(run)))
        if len(run) == 1:
            out.append((run[0][0], t0, max(run[0][2], t0 + min_dur)))
        else:
            tot = sum(len(w[0]) + 2 for w in run); acc = t0
            for w in run:
                d = (t1 - t0) * (len(w[0]) + 2) / tot
                out.append((w[0], acc, acc + d)); acc += d
        i = j
    fixed, last = [], 0.0
    for w, a, b in out:                       # strictly increasing, no overlaps
        a = max(a, last); b = max(b, a + 0.08)
        fixed.append((w, a, b)); last = a + 0.05
    return fixed


def caption_groups(words, max_words=3, max_gap=0.25):
    groups, cur = [], []
    for w in words:
        new_sentence = cur and w[0][:1].isupper() and w[0] not in ("I", "I'm", "I'll", "I've", "I'd")
        if cur and (len(cur) >= max_words or w[1] - cur[-1][2] > max_gap or new_sentence
                    or cur[-1][0][-1:] in ".?!,"):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    res = []
    for i, g in enumerate(groups):
        t0 = g[0][1]
        t1 = groups[i + 1][0][1] if i + 1 < len(groups) else g[-1][2] + 0.3
        t1 = min(t1, g[-1][2] + 0.6)
        txt = " ".join(x[0] for x in g).upper().strip(" .,")
        if txt:
            res.append((txt, t0, max(t1, t0 + 0.25)))
    return res


def cap_png(path, text, size=128, stroke=9, shadow=8):
    """White condensed caps, thick black outline, hard black drop shadow (the reference look)."""
    font = ImageFont.truetype(COND, size)
    d0 = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    while d0.textbbox((0, 0), text, font=font, stroke_width=stroke)[2] > W - 80 and size > 50:
        size -= 6; font = ImageFont.truetype(COND, size)
    l, t, r, b = d0.textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (r - l + shadow + 4, b - t + shadow + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((-l + shadow, -t + shadow), text, font=font, fill="black", stroke_width=stroke, stroke_fill="black")
    d.text((-l, -t), text, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
    img.save(path)
    return img.size


def keep_segments(words, dur, peak, max_gap=1.2, keep_edge=0.35, protect=(1.5, 3.0)):
    """Jump cuts: long silent stretches (no words, away from the payoff) shrink to two short pieces."""
    keep, cur = [], 0.0
    lo, hi = peak - protect[0], peak + protect[1]          # the payoff always plays in full
    times = [(0.0, 0.0)] + [(w[1], w[2]) for w in words] + [(dur, dur)]
    for (a0, a1), (b0, b1) in zip(times, times[1:]):
        gap0, gap1 = a1, b0
        # silent stretches, minus the protected payoff window, each shrink to two short pieces
        for g0, g1 in ((gap0, min(gap1, lo)), (max(gap0, hi), gap1)) if gap0 < hi and gap1 > lo else ((gap0, gap1),):
            if g1 - g0 > max_gap:
                keep.append((cur, g0 + keep_edge))
                cur = g1 - keep_edge
    keep.append((cur, dur))
    return [(max(0, a), min(dur, b)) for a, b in keep if b - a > 0.2]


def remap(t, keep):
    acc = 0.0
    for a, b in keep:
        if a <= t <= b:
            return acc + t - a
        acc += b - a
    return None


def render(p, layout, src, wd, fmt):
    if p.get("style", layout.get("style", "split")) == "split":
        return render_split(p, layout, src, wd, fmt)
    return render_classic(p, layout, src, wd, fmt)


def render_split(p, layout, src, wd, fmt):
    """Reference style: webcam big on top (41%), game below, one word at a time on the seam,
    *ACTION* captions over silent action, jump cuts through dead air, a short red flash on the payoff,
    no title card and no repeated hook (the clip starts on its own hook line)."""
    cid = p["id"]
    tmp = os.path.join(wd, "tmp", cid)
    os.makedirs(tmp, exist_ok=True)
    raw = os.path.join(tmp, "raw.mp4")
    fetch_section(src, p["start"], p["end"], raw, fmt)
    starts = {s_["codec_type"]: float(s_.get("start_time", 0)) for s_ in json.loads(sh(["ffprobe", "-v", "error",
              "-show_entries", "stream=codec_type,start_time", "-of", "json", raw]))["streams"]}
    if abs(starts.get("video", 0) - starts.get("audio", 0)) > 0.03:     # never build on a shifted section
        fixed = os.path.join(tmp, "raw_sync.mp4")
        sh(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ss", f"{max(starts.values()):.3f}", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", fixed])
        raw = fixed
    probe = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "json", raw]))["streams"][0]
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw]))
    sx, sy = probe["width"] / 1920, probe["height"] / 1080
    fc = p["facecam"] if "facecam" in p else layout.get("facecam")
    face = p.get("face", layout.get("face"))
    crop_x = p.get("crop_x", layout.get("crop_x"))
    top_h = 796; bot_h = H - top_h; seam = top_h
    if fc or face:
        if face:   # tight crop around the streamer's face, face a bit above centre
            fx, fy, fw, fh = face
            cw = min(1920, fw * float(p.get("face_zoom", layout.get("face_zoom", 3.4)))); ch = cw * top_h / W
            if fc:  # stay inside the webcam box when it is known
                cw = min(cw, fc[2]); ch = cw * top_h / W
            cx = fx + fw / 2; cy = fy + fh / 2 + fh * 0.15
            x = max(0, min(1920 - cw, cx - cw / 2)); y = max(0, min(1080 - ch, cy - ch / 2))
        else:
            x, y, cw, ch = fc
        x, y, cw, ch = [int(v) // 2 * 2 for v in (x * sx, y * sy, cw * sx, ch * sy)]
        gz = float(p.get("game_zoom", layout.get("game_zoom", 1.2)))
        gh = int(probe["height"] / gz) // 2 * 2; gw = int(gh * W / bot_h) // 2 * 2
        gx = (probe["width"] - gw) // 2; gy = (probe["height"] - gh) // 2
        # game loading screens (black after "wasted"/"busted") would leave half the Short black while the
        # streamer still talks: freeze the last game frame over them instead
        bl = subprocess.run(["ffmpeg", "-v", "info", "-i", raw, "-vf", f"crop={gw}:{gh}:{gx}:{gy},blackdetect=d=0.4:pix_th=0.10",
                             "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
        blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", bl)]
        parts, cur = [], 0.0
        for b0, b1 in blacks:
            if b0 < 0.3:           # clip opens on black: nothing to freeze from
                continue
            if b0 > cur:
                parts.append(("play", cur, b0))
            parts.append(("hold", b0, min(b1, dur)))
            cur = min(b1, dur)
        if cur < dur - 0.02:
            parts.append(("play", cur, dur))
        if len(parts) <= 1:
            bot = f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{bot_h}[bot]"
        else:
            bot = f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{bot_h},split={len(parts)}" + "".join(f"[g{i}]" for i in range(len(parts))) + ";"
            for i, (kind, a0, a1) in enumerate(parts):
                if kind == "play":
                    bot += f"[g{i}]trim={a0:.3f}:{a1:.3f},setpts=PTS-STARTPTS[h{i}];"
                else:
                    bot += (f"[g{i}]trim={max(0, a0 - 0.1):.3f}:{a0:.3f},setpts=PTS-STARTPTS,"
                            f"tpad=stop_mode=clone:stop_duration={max(0.0, a1 - a0 - 0.1):.3f},trim=0:{a1 - a0:.3f},setpts=PTS-STARTPTS[h{i}];")
            bot += "".join(f"[h{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[bot]"
        vf = (f"split=2[a][b];[a]crop={cw}:{ch}:{x}:{y},scale={W}:{top_h}:force_original_aspect_ratio=increase,"
              f"crop={W}:{top_h}[top];{bot};[top][bot]vstack")
    elif crop_x is not None:
        gh = probe["height"]; gw = int(gh * 9 / 16) // 2 * 2
        gx = int(max(0, min(probe["width"] - gw, crop_x * sx - gw / 2)))
        vf = f"crop={gw}:{gh}:{gx}:0,scale={W}:{H}"; seam = int(H * 0.62)
    else:
        vf = (f"split=2[a][b];[a]scale=-2:{H},crop={W}:{H},boxblur=30:2,eq=brightness=-0.15[bg];"
              f"[b]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2")
        seam = (H - int(W * 9 / 16)) // 2

    wav = os.path.join(tmp, "a.wav")
    sh(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "16000", wav])
    words = words_of(wav, os.path.join(tmp, "w"))
    off = float(p.get("start", 0)) if float(p.get("peak", 0)) >= float(p.get("start", 0)) > 0 else 0.0
    peak = float(p.get("peak", off + dur / 2)) - off      # picks.json uses stream time; live uses clip time
    p = dict(p, beats=[dict(b, t=float(b["t"]) - off) for b in p.get("beats", [])])
    # the judge's peak can be a few seconds off: snap it to the loudest half-second within +-5 s
    import numpy as np
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-f", "s16le", "-"], capture_output=True).stdout,
                      np.int16).astype(np.float32)
    if len(a) > 16000:
        win = 8000
        rms = [np.sqrt(np.mean(a[i:i + win] ** 2)) for i in range(0, len(a) - win, win)]
        lo, hi = max(0, int((peak - 5) * 2)), min(len(rms), int((peak + 5) * 2) + 1)
        if hi > lo:
            peak = (lo + int(np.argmax(rms[lo:hi]))) / 2
    keep = keep_segments(words, dur, peak) if p.get("tighten", True) else [(0.0, dur)]
    n = len(keep)
    parts = "".join(f"[v{i}]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[vk{i}];[x{i}]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[ak{i}];"
                    for i, (a, b) in enumerate(keep))
    pk = remap(peak, keep) or 0
    tl = (f"[0:v]{vf},split={n}" + "".join(f"[v{i}]" for i in range(n)) + ";"
          f"[0:a]asplit={n}" + "".join(f"[x{i}]" for i in range(n)) + ";" + parts +
          "".join(f"[vk{i}][ak{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[cv0][ca];"
          f"[cv0]drawbox=x=0:y=0:w=iw:h=ih:color=red@0.32:t=fill:enable='between(t,{pk:.2f},{pk + 0.13:.2f})'[cv];"
          f"[ca]loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    base = os.path.join(tmp, "base.mp4")
    sh(["ffmpeg", "-v", "error", "-y", "-i", raw, "-filter_complex", tl, "-map", "[cv]", "-map", "[aout]",
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "aac", "-b:a", "160k", base])

    # captions: one word at a time, held until the next word, never two on screen at once
    words = [(clean_word(w), t0, t1) for w, t0, t1 in words]
    words = [x for x in words if x[0]]
    tw = [(w.upper().strip(" .,!?"), remap(t0, keep), remap(t1, keep)) for w, t0, t1 in words]
    tw = [(w, t0, t1 if t1 is not None else t0 + 0.4) for w, t0, t1 in tw if t0 is not None and w]
    tw.sort(key=lambda x: x[1])
    caps = []
    for i, (w, t0, t1) in enumerate(tw):
        nxt = tw[i + 1][1] if i + 1 < len(tw) else t1 + 0.6
        caps.append((w, t0, max(t0 + 0.05, min(nxt, t1 + 0.4))))
    # *ACTION* beats go into silent gaps only (>= 0.7 s with no word), the nearest one within 2 s of the judge's time
    total = sum(b - a for a, b in keep)
    gaps, prev = [], 0.0
    for _, a, b in caps + [("", total, total)]:
        if a - prev >= 0.7:
            gaps.append((prev, a))
        prev = max(prev, b)
    for bt in p.get("beats", []):
        t = remap(float(bt["t"]), keep)
        if t is None or not gaps:
            continue
        g = min(gaps, key=lambda g: 0 if g[0] <= t <= g[1] else min(abs(t - g[0]), abs(t - g[1])))
        if (g[0] <= t <= g[1]) or min(abs(t - g[0]), abs(t - g[1])) <= 2.0:
            a = min(max(t, g[0]), g[1] - 0.6); b = min(g[1], a + 2.0)
            caps.append((bt["text"].upper(), a, b)); gaps.remove(g)
    caps.sort(key=lambda c: c[1])
    caps = [(t, a, min(b, caps[i + 1][1]) if i + 1 < len(caps) else b) for i, (t, a, b) in enumerate(caps)]
    caps = [c for c in caps if c[2] - c[1] >= 0.04]     # never two captions on screen at once
    inputs, chain, last = [], [], "0:v"
    for i, (txt, t0, t1) in enumerate(caps):
        png = os.path.join(tmp, f"c{i}.png")
        cw_, ch_ = cap_png(png, txt)
        inputs += ["-i", png]
        chain.append(f"[{last}][{i + 1}:v]overlay=(W-w)/2:{seam - ch_ // 2}:enable='between(t,{t0:.2f},{t1:.2f})'[t{i}]")
        last = f"t{i}"
    os.makedirs(os.path.join(wd, "clips"), exist_ok=True)
    out = os.path.join(wd, "clips", f"{cid}.mp4")
    if chain:
        script = os.path.join(tmp, "overlay.txt"); open(script, "w").write(";".join(chain))
        sh(["ffmpeg", "-v", "error", "-y", "-i", base, *inputs, "-filter_complex_script", script, "-map", f"[{last}]",
            "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "copy", "-movflags", "+faststart", out])
    else:
        shutil.copy(base, out)
    d = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out]))
    return {"id": cid, "file": out, "duration_s": round(d, 1), "captions": len(caps), "cuts": n - 1}


def render_classic(p, layout, src, wd, fmt):
    cid = p["id"]
    tmp = os.path.join(wd, "tmp", cid)
    os.makedirs(tmp, exist_ok=True)
    hook0 = max(p["start"], p["peak"] - 0.4)
    hook1 = min(p["end"], p["peak"] + 2.4)
    raw = os.path.join(tmp, "raw.mp4")
    fetch_section(src, p["start"], p["end"], raw, fmt)

    # 1. picture: facecam on top, game below (or blurred fill if no facecam)
    fc = p["facecam"] if "facecam" in p else layout.get("facecam")
    probe = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "json", raw]))["streams"][0]
    sx, sy = probe["width"] / 1920, probe["height"] / 1080
    crop_x = p.get("crop_x", layout.get("crop_x"))
    if crop_x is not None and not fc:
        # camera fills the frame (IRL / just chatting): one 9:16 window around the streamer
        gh = probe["height"]; gw = int(gh * 9 / 16) // 2 * 2
        gx = int(max(0, min(probe["width"] - gw, crop_x * sx - gw / 2)))
        vf = f"[0:v]crop={gw}:{gh}:{gx}:0,scale={W}:{H}[v]"
        seam = 330
    elif fc:
        x, y, w, h = [int(v) for v in (fc[0] * sx, fc[1] * sy, fc[2] * sx, fc[3] * sy)]
        top_h = min(760, int(W * h / w) // 2 * 2)
        bot_h = H - top_h
        gh = probe["height"]
        gw = int(gh * W / bot_h) // 2 * 2
        gx = (probe["width"] - gw) // 2
        vf = (f"[0:v]split=2[a][b];[a]crop={w}:{h}:{x}:{y},scale={W}:{top_h}:force_original_aspect_ratio=increase,"
              f"crop={W}:{top_h}[top];[b]crop={gw}:{gh}:{gx}:0,scale={W}:{bot_h}[bot];[top][bot]vstack[v]")
        seam = top_h
    else:
        vf = (f"[0:v]split=2[a][b];[a]scale=-2:{H},crop={W}:{H},boxblur=30:2,eq=brightness=-0.15[bg];"
              f"[b]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2[v]")
        seam = (H - int(W * 9 / 16)) // 2
    # 2. timeline: peak first, then the whole moment from the setup
    a0, a1 = hook0 - p["start"], hook1 - p["start"]
    tl = (f"{vf};[v]split=2[v1][v2];[v1]trim={a0}:{a1},setpts=PTS-STARTPTS[h];"
          f"[v2]setpts=PTS-STARTPTS[m];[0:a]asplit=2[x1][x2];[x1]atrim={a0}:{a1},asetpts=PTS-STARTPTS[ha];"
          f"[x2]asetpts=PTS-STARTPTS[ma];[h][ha][m][ma]concat=n=2:v=1:a=1[cv][ca];"
          f"[ca]loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    base = os.path.join(tmp, "base.mp4")
    sh(["ffmpeg", "-v", "error", "-y", "-i", raw, "-filter_complex", tl, "-map", "[cv]", "-map", "[aout]",
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", base])

    # 3. captions: transcribe the source section once, then lay the words onto both parts of the clip
    wav = os.path.join(tmp, "a.wav")
    sh(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "16000", wav])
    src_words = words_of(wav, os.path.join(tmp, "w"))
    hook_len = a1 - a0
    timeline = [(w, t0 - a0, t1 - a0) for w, t0, t1 in src_words if t0 >= a0 - 0.05 and t1 <= a1 + 0.05]
    timeline += [(w, t0 + hook_len, t1 + hook_len) for w, t0, t1 in src_words]
    groups = caption_groups(timeline)
    inputs, chain, last = [], [], "0:v"
    tw, th = text_png(os.path.join(tmp, "title.png"), p["title"], 54, wrap=26, box=(255, 255, 255, 240),
                      fill="black", stroke=0)
    inputs += ["-i", os.path.join(tmp, "title.png")]
    chain.append(f"[{last}][1:v]overlay=(W-w)/2:{max(20, seam - th // 2)}[t0]")
    last = "t0"
    cap_y = seam + (H - seam) * 55 // 100
    for i, (txt, t0, t1) in enumerate(groups):
        png = os.path.join(tmp, f"c{i}.png")
        cw, ch = text_png(png, txt, 92, stroke=9)
        inputs += ["-i", png]
        chain.append(f"[{last}][{i + 2}:v]overlay=(W-w)/2:{cap_y - ch // 2}:enable='between(t,{t0:.2f},{t1:.2f})'[t{i + 1}]")
        last = f"t{i + 1}"
    os.makedirs(os.path.join(wd, "clips"), exist_ok=True)
    out = os.path.join(wd, "clips", f"{cid}.mp4")
    script = os.path.join(tmp, "overlay.txt")
    open(script, "w").write(";".join(chain))
    sh(["ffmpeg", "-v", "error", "-y", "-i", base, *inputs, "-filter_complex_script", script, "-map", f"[{last}]",
        "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "copy",
        "-movflags", "+faststart", out])
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out]))
    return {"id": cid, "file": out, "duration_s": round(dur, 1), "captions": len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="stream URL or local video file")
    ap.add_argument("workdir")
    ap.add_argument("--only")
    ap.add_argument("--format", default="1080p60/1080p/720p60/best[height<=1080]")
    a = ap.parse_args()
    if not FONT:
        sys.exit("no bold font found; install Arial Black or DejaVu Sans")
    picks = json.load(open(os.path.join(a.workdir, "picks.json")))
    layout = json.load(open(os.path.join(a.workdir, "layout.json")))
    done = []
    for p in picks:
        if a.only and p["id"] != a.only:
            continue
        r = render(p, layout, a.src, a.workdir, a.format)
        print(f"{r['id']}: {r['duration_s']} s, {r['captions']} caption groups -> {r['file']}")
        done.append(r)
    json.dump(done, open(os.path.join(a.workdir, "rendered.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
