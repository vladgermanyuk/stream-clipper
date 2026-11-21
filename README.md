# stream-clipper

A Claude Code skill that turns a stream into ready-to-post vertical clips. It works on a recording
(Twitch, YouTube or Kick VOD) or **live, while the stream is on**.

You give Claude a link. It finds the moments where the streamer suddenly gets loud and chat explodes,
reads what was said, judges which moments a stranger would stop scrolling for, and renders 1080x1920
Shorts / TikToks / Reels in the format top clipping channels use:

- webcam big on top, gameplay below, hard seam
- one word at a time on the seam, timed to the voice within a few hundredths of a second
- jump cuts through dead air, the payoff always plays in full
- a short red flash on the payoff, `*ACTION*` captions over silent action
- swears masked, slurs never shown

Then it checks every clip (sync, captions, word timing) before you see it.

> **Clip only streamers who allow it** (an official clipping campaign or written permission), and follow
> the campaign's rules. Posting is up to you.

## How it works

```
stream link
  -> audio, downloaded in parallel pieces (a 9-hour VOD in ~40 s)
  -> loudness spikes vs the streamer's own level (+ chat speed, live)
  -> 40 s transcript around each spike
  -> Claude scores each moment 0-10 and picks start / payoff / end
  -> webcam found automatically (the face that stays put)
  -> clean voice (Demucs) -> whisper large-v3 -> wav2vec2 forced alignment = exact captions
  -> render: split layout, jump cuts, captions, flash
  -> check: sync, overlaps, every word on the voice
```

Live mode does the same on a rolling 20-minute buffer: from moment to finished clip in about a minute.

## Install

Requirements: macOS (Apple Silicon recommended) or Linux, Python 3.10+, ~3 GB of disk for models,
[Claude Code](https://claude.com/claude-code).

```bash
git clone https://github.com/vladgermanyuk/stream-clipper ~/.claude/skills/stream-clipper
cd ~/.claude/skills/stream-clipper
./setup.sh
```

`setup.sh` installs ffmpeg, yt-dlp, whisper.cpp and uv (Homebrew), creates two Python environments
(`.venv` for the pipeline, `.align-venv` with PyTorch for voice separation and alignment) and downloads
the models. On Linux, install those four tools with your package manager and run `SKIP_BREW=1 ./setup.sh`.

## Use

In Claude Code:

```
Clip this stream with the stream-clipper skill and make the top 3 clips: https://www.twitch.tv/videos/...
```

```
Watch <streamer>'s live stream on Twitch and clip the best moments for the next 30 minutes.
```

Clips land in `work/<name>/clips/`. Every step is also a plain script you can run yourself; see `SKILL.md`.

| Script | Does |
|---|---|
| `scripts/fetch.py` | fast parallel download of audio or a video section |
| `scripts/signals.py` | loudness spikes (ignores sleep/AFK hours of marathons) |
| `scripts/transcribe_candidates.py` | 40 s transcript around each spike |
| `scripts/find_layout.py` | finds webcam and face, writes `layout.json` |
| `scripts/render.py` | renders the picks |
| `scripts/check_clip.py` | checks finished clips |
| `scripts/live.py` | live mode |
| `scripts/render_candidate.py` | renders a live near-miss by hand |
| `scripts/align_words.py` | exact word timing (runs in `.align-venv`) |

## Speed (Apple M-series)

| Step | Time |
|---|---|
| Audio of a 9-hour VOD | ~40 s |
| Spikes | ~20 s |
| 60 candidate transcripts | ~1 min |
| One clip with exact captions | ~15 s |
| Live: moment to clip | ~1-1.5 min |

## Limits

- Twitch serves anonymous viewers ads; in live mode, moments during an ad break are lost.
- English captions only (the aligner model is English).
- The judge is Claude: it needs Claude Code (`claude -p` in live mode).
- Captions are as good as what can be heard: the clean-voice step helps a lot, but shouting over loud
  game audio can still be misheard. `check_clip.py` lists every word so you can read it before posting.

## What the scripts touch (security notes)

Skill scanners flag this skill for running programs and using the network; that is what it does, so
here is the full list:

- **Network:** stream playlists and video pieces from Twitch / YouTube / Kick (via yt-dlp), public
  Twitch chat in live mode (anonymous read-only IRC login, nothing is ever sent to chat), model downloads
  in `setup.sh` (Hugging Face, GitHub opencv_zoo) and on first use (PyTorch weights for wav2vec2 and Demucs).
- **Programs it runs:** `ffmpeg`, `ffprobe`, `yt-dlp`, `whisper-cli`, the two local Python environments,
  and in live mode `claude -p` (to score moments; it is allowed only the Read tool).
- All commands are passed as argument lists, never through a shell. No credentials, tokens or cookies are
  read or stored. Everything it writes stays in `work/`.

## Credits

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) (MIT) and OpenAI Whisper models (MIT),
[Demucs](https://github.com/facebookresearch/demucs) (MIT), wav2vec2 via
[torchaudio](https://github.com/pytorch/audio) (BSD), forced-alignment approach from
[WhisperX](https://github.com/m-bain/whisperX), [YuNet](https://github.com/opencv/opencv_zoo) face
detector (MIT), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [FFmpeg](https://ffmpeg.org).

## License

MIT, see `LICENSE`.
