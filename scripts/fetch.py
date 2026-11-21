#!/usr/bin/env python3
"""Fast downloads for long streams.

Twitch serves VODs as thousands of 10-second pieces; yt-dlp pulls them slowly. This asks yt-dlp only
for the playlist address, then downloads the pieces in parallel.

    python fetch.py audio <url> <out.mp4>
    python fetch.py video <url> <start_s> <end_s> <out.mp4> [--format 1080p60/720p60/best]

Non-HLS sources (YouTube, local files) fall back to yt-dlp / ffmpeg.
"""
import argparse, os, subprocess, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0"}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise


def playlist_url(src, fmt):
    r = subprocess.run(["yt-dlp", "-g", "-f", fmt, src], capture_output=True, text=True)
    urls = [l for l in r.stdout.split() if l.startswith("http")]
    return urls[0] if urls else None


def parse_m3u8(url):
    text = get(url).decode()
    base = url.rsplit("/", 1)[0]
    init, frags, t = None, [], 0.0
    dur = None
    for line in text.splitlines():
        if line.startswith("#EXT-X-MAP:"):
            init = line.split('URI="', 1)[1].split('"', 1)[0]
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#"):
            frags.append((t, dur or 0.0, line))
            t += dur or 0.0
            dur = None
    absu = lambda u: u if u.startswith("http") else f"{base}/{u}"
    return (absu(init) if init else None), [(s, d, absu(u)) for s, d, u in frags]


def download_pieces(init, frags, out, jobs=24):
    with ThreadPoolExecutor(jobs) as ex:
        data = list(ex.map(lambda f: get(f[2]), frags))
    with open(out, "wb") as fh:
        if init:
            fh.write(get(init))
        for d in data:
            fh.write(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["audio", "video"])
    ap.add_argument("src")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--format", default=None)
    a = ap.parse_args()

    if a.kind == "audio":
        out = a.rest[0]
        fmt = a.format or "Audio_Only/bestaudio"
        if os.path.exists(a.src):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-vn", "-c:a", "aac", out], check=True)
            return
        pl = playlist_url(a.src, fmt)
        if pl and ".m3u8" in pl:
            init, frags = parse_m3u8(pl)
            download_pieces(init, frags, out)
        else:
            subprocess.run(["yt-dlp", "-q", "-N", "8", "-f", fmt, "-o", out, a.src], check=True)
        print(f"audio -> {out}")
        return

    start, end, out = float(a.rest[0]), float(a.rest[1]), a.rest[2]
    fmt = a.format or "1080p60/1080p/720p60/best[height<=1080]"
    if os.path.exists(a.src):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", a.src,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", out], check=True)
        return
    pl = playlist_url(a.src, fmt)
    if not pl or ".m3u8" not in pl:
        subprocess.run(["yt-dlp", "-q", "-f", fmt, "--download-sections", f"*{start}-{end}",
                        "--force-keyframes-at-cuts", "-o", out, a.src], check=True)
        return
    init, frags = parse_m3u8(pl)
    sel = [f for f in frags if f[0] + f[1] > start and f[0] < end]
    if not sel:
        sys.exit("requested time is outside the stream")
    with tempfile.TemporaryDirectory() as td:
        joined = os.path.join(td, "joined.mp4")
        download_pieces(init, sel, joined, jobs=8)
        off = start - sel[0][0]
        # seek AFTER opening (output seeking): Twitch pieces start audio and video at different times, and
        # input seeking keeps that gap (video landed 1.5 s off the audio in testing). This cuts both at the same instant.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-ss", f"{off:.3f}", "-t", f"{end - start:.3f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", out],
                       check=True)
    print(f"video {start}-{end} -> {out}")


if __name__ == "__main__":
    main()
