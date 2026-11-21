#!/usr/bin/env python3
"""Find the streamer's webcam and face automatically and write layout.json.

    python find_layout.py <stream_url_or_video> <workdir> [--at SECONDS]

Takes 40 s of video (around --at, default 600 s into the stream), looks at 8 frames, keeps the face that
stays in the same place (the webcam; faces in the game move), and writes <workdir>/layout.json:
  {"style": "split", "facecam": [x, y, w, h], "face": [x, y, w, h], "face_zoom": 3.4, "game_zoom": 1.2}
A face taller than 35% of the frame means a camera stream: {"crop_x": ...} (9:16 crop around the streamer).
No steady face: {"facecam": null} (game with a blurred fill). Look at the result before rendering.
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live import auto_layout

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("workdir"); ap.add_argument("--at", type=float, default=600)
a = ap.parse_args()
os.makedirs(a.workdir, exist_ok=True)
probe = os.path.join(a.workdir, "layout_probe.mp4")
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "fetch.py"), "video", a.src, str(a.at), str(a.at + 40), probe], check=True)
lay = {"style": "split", "face_zoom": 3.4, "game_zoom": 1.2, **auto_layout(probe)}
json.dump(lay, open(os.path.join(a.workdir, "layout.json"), "w"))
print(json.dumps(lay))
