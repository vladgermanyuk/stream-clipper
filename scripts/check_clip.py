#!/usr/bin/env python3
"""Check finished clips before anyone posts them.

    python check_clip.py <workdir> [id ...]

For every clip in <workdir>/clips (or the ids given) it reports:
  - audio and video start together and have the same length
  - no two captions on screen at once
  - every caption word starts on real voice (clean voice track from Demucs, not on game noise)
  - the words, with times, to read through
and writes <workdir>/check/<id>.jpg: the caption band at 5 frames per second, to look at.
Exit code 1 if any clip fails a hard check.
"""
import json, os, re, subprocess, sys
import numpy as np


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def voice_db(wav):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = 320
    e = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, max(0, len(a) - hop), hop)])
    return 20 * np.log10(e + 1e-6)


def check(wd, cid):
    clip = os.path.join(wd, "clips", f"{cid}.mp4"); tmp = os.path.join(wd, "tmp", cid)
    problems, notes = [], []
    st = json.loads(sh(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time,duration", "-of", "json", clip]))["streams"]
    s = {x["codec_type"]: (float(x.get("start_time", 0)), float(x.get("duration", 0))) for x in st}
    if abs(s["video"][0] - s["audio"][0]) > 0.03:
        problems.append(f"audio/video start differ by {s['video'][0] - s['audio'][0]:+.2f} s")
    if abs(s["video"][1] - s["audio"][1]) > 0.15:
        problems.append(f"audio/video length differ by {s['video'][1] - s['audio'][1]:+.2f} s")
    ov = os.path.join(tmp, "overlay.txt")
    caps = [(float(a), float(b)) for a, b in re.findall(r"between\(t,([\d.]+),([\d.]+)\)", open(ov).read())] if os.path.exists(ov) else []
    overl = sum(1 for i in range(len(caps) - 1) if caps[i + 1][0] < caps[i][1] - 0.001)
    if overl:
        problems.append(f"{overl} overlapping captions")
    al = os.path.join(tmp, "w.aligned.json")
    words = json.load(open(al)) if os.path.exists(al) else []
    voc = next((os.path.join(tmp, f) for f in ("w_voice16.wav", "a.wav") if os.path.exists(os.path.join(tmp, f))), None)
    off_voice = []
    if words and voc:
        db = voice_db(voc); thr = np.percentile(db, 90) - 25
        for w, a, b in words:
            i = int(a / 0.02); seg = db[i:i + 12]
            if len(seg) and seg.mean() < thr:
                off_voice.append(f"{w}@{a:.2f}")
        if off_voice:
            notes.append("words not on clear voice (look/listen): " + ", ".join(off_voice))
    notes.append("words: " + " ".join(f"{w}@{a:.2f}" for w, a, b in words))
    os.makedirs(os.path.join(wd, "check"), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", "fps=5,crop=1080:330:0:630,scale=216:66,tile=5x16",
                    "-frames:v", "1", os.path.join(wd, "check", f"{cid}.jpg")])
    return problems, notes, len(caps), s["video"][1]


def main():
    wd = sys.argv[1]
    ids = sys.argv[2:] or sorted(f[:-4] for f in os.listdir(os.path.join(wd, "clips")) if f.endswith(".mp4"))
    bad = 0
    for cid in ids:
        problems, notes, n, dur = check(wd, cid)
        print(f"{'FAIL' if problems else 'OK  '} {cid}: {dur:.1f} s, {n} captions" + (" | " + "; ".join(problems) if problems else ""))
        for x in notes:
            print("     " + x)
        bad += bool(problems)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
