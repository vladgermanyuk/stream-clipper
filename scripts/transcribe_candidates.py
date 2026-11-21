#!/usr/bin/env python3
"""Transcribe only the loud moments, not the whole stream.

For each candidate in signals.json: cut [t-25 s, t+15 s] of audio, run whisper.cpp with one word per
segment, count reaction words, and write a readable sheet for Claude to score.

    python transcribe_candidates.py <audio> <workdir> [--model path] [--before 25] [--after 15]

Output: <workdir>/candidates.json and <workdir>/candidates.md
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

REACT = re.compile(
    r"\b(no way|oh my god|omg|bro+|what+|wait|yo+|let'?s go+|holy|bruh|nah+|dude|"
    r"oh no|help|run|stop|ha(ha)+|lol|insane|crazy|what the|are you serious|how)\b",
    re.I,
)

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "ggml-base.en.bin")


def fmt(t):
    t = int(t)
    return f"{t//3600}:{t%3600//60:02d}:{t%60:02d}"


def timeline(audio, wd):
    """Whole stream as raw 16 kHz mono PCM, decoded front to back. Seeking (-ss) inside a stream that was
    stitched from thousands of HLS pieces drifts or fails after ~20 h; byte offsets in this file do not."""
    raw = os.path.join(wd, "audio16k.s16")
    if not os.path.exists(raw):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio, "-ac", "1", "-ar", "16000", "-f", "s16le", raw], check=True)
    return raw


def run_one(c, audio, wd, model, before, after):
    s = max(0.0, c["t"] - before)
    d = before + after
    name = f"cand_{int(c['t'])}"
    wav = os.path.join(wd, "cand", name + ".wav")
    with open(timeline(audio, wd), "rb") as f:
        f.seek(int(s * 16000) * 2)
        pcm = f.read(int(d * 16000) * 2)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-", wav],
                   input=pcm, check=True)
    base = os.path.join(wd, "cand", name)
    subprocess.run(["whisper-cli", "-m", model, "-f", wav, "-ml", "1", "-sow", "-oj", "-of", base,
                    "-np", "-t", "4"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    words = []
    for seg in json.load(open(base + ".json"))["transcription"]:
        w = seg["text"].strip()
        if not w:
            continue
        t0 = seg["offsets"]["from"] / 1000 + s
        t1 = seg["offsets"]["to"] / 1000 + s
        words.append({"w": w, "t0": round(t0, 2), "t1": round(t1, 2)})
    text = " ".join(x["w"] for x in words)
    near = " ".join(x["w"] for x in words if c["t"] - 6 <= x["t0"] <= c["t"] + 6)
    c = dict(c)
    c.update({
        "win": [round(s, 1), round(s + d, 1)],
        "words": words,
        "text": text,
        "react_hits": len(REACT.findall(near)),
        "words_per_s_near": round(len(near.split()) / 12, 2),
    })
    os.remove(wav)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("workdir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--before", type=float, default=25)
    ap.add_argument("--after", type=float, default=15)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if not os.path.exists(a.model):
        sys.exit(f"whisper model not found: {a.model}")
    os.makedirs(os.path.join(a.workdir, "cand"), exist_ok=True)
    sig = json.load(open(os.path.join(a.workdir, "signals.json")))
    timeline(a.audio, a.workdir)           # decode once, before the parallel workers
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda c: run_one(c, a.audio, a.workdir, a.model, a.before, a.after), sig["candidates"]))
    json.dump(res, open(os.path.join(a.workdir, "candidates.json"), "w"), indent=1)

    lines = [f"# Candidates ({len(res)}) from {sig['duration_s']/3600:.1f} h of stream",
             "Each block: stream time of the spike, how much louder than the streamer's usual level,",
             "reaction words near the spike, then the 40 s transcript around it (>>> marks the spike).", ""]
    for c in res:
        marked = []
        done = False
        for w in c["words"]:
            if not done and w["t0"] >= c["t"]:
                marked.append(">>>")
                done = True
            marked.append(w["w"])
        chat = f" · chat x{c['chat_x']}" if c.get("chat_x") is not None else ""
        lines.append(f"## {fmt(c['t'])}  (t={c['t']})  +{c['excess_db']} dB{chat} · reactions {c['react_hits']}")
        lines.append(" ".join(marked))
        lines.append("")
    open(os.path.join(a.workdir, "candidates.md"), "w").write("\n".join(lines))
    print(f"transcribed {len(res)} candidates -> candidates.md")


if __name__ == "__main__":
    main()
