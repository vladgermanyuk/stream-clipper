#!/usr/bin/env python3
"""Find loud moments in a long stream.

Reads the audio with ffmpeg (8 kHz mono), measures loudness every 0.5 s, compares each moment with
the stream's own level over the surrounding 10 minutes, and writes the biggest spikes.

    python signals.py <audio> <workdir> [--top 60] [--chat chat.json]

Output: <workdir>/signals.json  (list of candidates with time, excess dB, chat rate if any)
"""
import argparse, json, subprocess, sys
import numpy as np

SR = 8000
HOP = 0.5  # seconds per loudness step


def loudness_curve(audio):
    cmd = ["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    step = int(SR * HOP)
    out = []
    buf = b""
    while True:
        chunk = p.stdout.read(step * 2 * 2000)
        if not chunk:
            break
        buf += chunk
        n = len(buf) // (step * 2)
        if n == 0:
            continue
        a = np.frombuffer(buf[: n * step * 2], dtype=np.int16).astype(np.float32).reshape(n, step)
        buf = buf[n * step * 2 :]
        rms = np.sqrt((a ** 2).mean(axis=1)) + 1e-3
        out.append(20 * np.log10(rms / 32768))
    p.wait()
    return np.concatenate(out) if out else np.array([])


def rolling_median(x, win):
    # median over a sliding window, computed on a coarse grid for speed
    stride = max(1, win // 20)
    centers = np.arange(0, len(x), stride)
    med = np.empty(len(centers))
    for i, c in enumerate(centers):
        lo, hi = max(0, c - win // 2), min(len(x), c + win // 2)
        med[i] = np.median(x[lo:hi])
    return np.interp(np.arange(len(x)), centers, med)


def chat_rate(chat_path, n_steps):
    """messages per 10 s around each step, divided by the stream's median rate"""
    msgs = json.load(open(chat_path))
    t = np.array([m.get("time_in_seconds", m.get("t", 0)) for m in msgs], dtype=float)
    t = t[t >= 0]
    bins = np.bincount((t / HOP).astype(int), minlength=n_steps)[:n_steps].astype(float)
    k = int(10 / HOP)
    rate = np.convolve(bins, np.ones(k), mode="same")
    base = np.median(rate[rate > 0]) if (rate > 0).any() else 1.0
    return rate / max(base, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("workdir")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--chat")
    ap.add_argument("--merge", type=float, default=30.0, help="seconds: spikes closer than this are one moment")
    ap.add_argument("--all-hours", action="store_true",
                    help="also count spikes in quiet stretches (by default sleep/AFK hours are ignored)")
    a = ap.parse_args()

    db = loudness_curve(a.audio)
    if len(db) == 0:
        sys.exit("no audio")
    # smooth over ~1.5 s so a single clap does not count
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    base = rolling_median(sm, int(600 / HOP))  # stream's own level over 10 minutes
    excess = sm - base
    # ignore near-silence (menus, breaks)
    excess[base < -55] = 0
    if not a.all_hours:
        # marathons: while the streamer sleeps any noise is "much louder than usual". A moment must also be
        # loud in absolute terms (above the stream's 70th percentile of active audio) and the surrounding
        # 10 minutes must not be a quiet stretch (sleep, AFK).
        active = sm[sm > np.percentile(sm, 20)]
        p70, gmed = np.percentile(active, 70), np.median(active)
        excess[(sm < p70) | (base < gmed - 6)] = 0

    chat = chat_rate(a.chat, len(db)) if a.chat else None
    score = excess.copy()
    if chat is not None:
        score = excess + 4 * np.clip(np.log2(np.maximum(chat, 1e-3)), 0, 3)

    order = np.argsort(-score)
    picked = []
    for i in order:
        t = i * HOP
        if score[i] <= 0 or len(picked) >= a.top:
            break
        if any(abs(t - p["t"]) < a.merge for p in picked):
            continue
        picked.append({
            "t": round(t, 1),
            "excess_db": round(float(excess[i]), 1),
            "level_db": round(float(sm[i]), 1),
            "chat_x": round(float(chat[i]), 2) if chat is not None else None,
            "score": round(float(score[i]), 2),
        })
    picked.sort(key=lambda p: p["t"])
    out = {
        "audio": a.audio,
        "duration_s": round(len(db) * HOP, 1),
        "hop_s": HOP,
        "chat": bool(a.chat),
        "median_level_db": round(float(np.median(db)), 1),
        "candidates": picked,
    }
    json.dump(out, open(f"{a.workdir}/signals.json", "w"), indent=1)
    np.save(f"{a.workdir}/loudness.npy", np.stack([sm, base]).astype(np.float32))
    print(f"{len(picked)} candidates from {out['duration_s']/3600:.1f} h of audio")


if __name__ == "__main__":
    main()
