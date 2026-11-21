#!/usr/bin/env bash
# One-time setup for stream-clipper (macOS + Homebrew). Linux: install ffmpeg, yt-dlp, whisper.cpp and uv
# with your package manager, then run this script with SKIP_BREW=1.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${SKIP_BREW:-0}" != "1" ]; then
  brew install ffmpeg yt-dlp whisper-cpp uv
fi

# main environment: download, signals, layout, rendering
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

# alignment environment: clean voice (Demucs) + exact word timing (wav2vec2). Python 3.12 for PyTorch wheels.
uv venv -q -p 3.12 .align-venv
uv pip install -q --python .align-venv/bin/python -r requirements-align.txt

mkdir -p models
get() { [ -s "models/$1" ] || curl -L --fail -o "models/$1" "$2"; }
HF=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
get ggml-base.en.bin        "$HF/ggml-base.en.bin"          # 148 MB, quick candidate transcripts
get ggml-large-v3-q5_0.bin  "$HF/ggml-large-v3-q5_0.bin"    # 1.1 GB, caption text
get face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# first run of the aligner and Demucs downloads their weights (~1.3 GB + 80 MB) into the torch cache
echo "Setup done. Try: .venv/bin/python scripts/fetch.py audio <stream_url> work/test/audio.mp4"
