#!/usr/bin/env python3
"""Render a live candidate by hand (for moments the judge scored just under the bar).

    python render_candidate.py <live_workdir> <id> [<id> ...]
Uses cands/<id>/raw.mp4 and cands/<id>/verdict.json written by live.py. Output: <workdir>/clips/<id>.mp4
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live
from render import render

rejudge = "--rejudge" in sys.argv
args = [a for a in sys.argv[1:] if a != "--rejudge"]
wd = args[0]
for cid in args[1:]:
    cdir = os.path.join(wd, "cands", cid)
    v = json.load(open(os.path.join(cdir, "verdict.json")))
    if rejudge:   # ask Claude again with the current prompt (chat is not stored per candidate)
        words = json.load(open(os.path.join(cdir, "w.json")))["transcription"]
        ws = [(x["text"].strip(), x["offsets"]["from"] / 1000, 0) for x in words if x["text"].strip()]
        rawd = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     os.path.join(cdir, "raw.mp4")], capture_output=True, text=True).stdout)
        nv = live.judge(os.path.basename(wd.rstrip("/")).split("-")[1], cdir, rawd, float(v.get("peak", 25)), ws, [], "sonnet")
        if nv.get("score"):
            v.update(nv)
        print(cid, "rejudged:", json.dumps(nv)[:300])
    raw = os.path.join(cdir, "raw.mp4")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw],
                               capture_output=True, text=True).stdout)
    s, e, pk = max(0, float(v["start"])), min(dur, float(v["end"])), float(v["peak"])
    if e - s < 8:
        s, e = max(0, pk - 15), min(dur, pk + 8)
    lay = live.auto_layout(raw)
    tmp = os.path.join(wd, "tmp", cid); os.makedirs(tmp, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", raw, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "17", "-c:a", "aac", "-b:a", "192k", os.path.join(tmp, "raw.mp4")], check=True)
    p = {"id": cid, "start": 0.0, "peak": max(0.5, pk - s), "end": e - s, "title": v["title"],
         "beats": [{"t": float(b["t"]) - s, "text": b["text"]} for b in v.get("beats") or [] if s <= float(b["t"]) <= e]}
    if "crop_x" in lay:
        p["crop_x"] = lay["crop_x"]
    r = render(p, lay, os.path.join(tmp, "raw.mp4"), wd, "best")
    print(cid, r["file"], r["duration_s"], "s |", v["score"], "|", v["title"])
