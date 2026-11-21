---
name: stream-clipper
description: Turn a Twitch, YouTube or Kick stream (a recording, or live while it is on) into ready-to-post vertical clips. Finds the moments where the streamer suddenly gets loud and chat explodes, scores them, and renders 1080x1920 Shorts/TikToks in the proven clipping-channel format - webcam big on top, gameplay below, one word at a time on the seam with exact word timing, jump cuts through dead air. Use for "clip this stream", "find the best moments in this VOD", "make shorts from this stream", "watch this stream live and clip it", clipping campaigns.
---

# stream-clipper

One stream link in, a handful of clips worth posting out. Scripts do the heavy lifting: download,
spike finding, transcription, layout, rendering, checks. You do the judging: which moments are funny
or shocking to a stranger, where the clip starts and ends, and whether the finished clip is good.
Run everything from this folder with `.venv/bin/python`.

**Rules first.** Clip only streamers who allow it: an official clipping campaign or written permission.
A campaign license usually covers posting the campaign's clips only, not reusing the footage elsewhere.
Read the campaign rules before posting (name in the caption, hashtag, submission window, no cutscenes,
no logos). Some campaigns ban "AI-generated videos": these clips are cut from real footage and nothing
is generated, but if a campaign is unclear, tell the user to ask in its Discord. Posting is the user's job.

## Setup (once)

`./setup.sh` (macOS with Homebrew; see README for Linux/Windows). It installs ffmpeg, yt-dlp,
whisper.cpp and uv, creates two Python environments and downloads the models:

| What | Used for |
|---|---|
| `.venv` (Python 3.10+): pillow, numpy, opencv-python-headless | everything except alignment |
| `.align-venv` (Python 3.12): torch, torchaudio, soundfile, demucs | clean voice + exact word timing |
| `models/ggml-base.en.bin` (148 MB) | quick transcripts of 60 candidate moments |
| `models/ggml-large-v3-q5_0.bin` (1.1 GB) | caption text |
| `models/face_detection_yunet.onnx` | finding the webcam |

The renderer falls back step by step when something is missing (no `.align-venv` -> no voice separation
and no alignment; no large-v3 -> turbo/small/base), so a half setup still works, with worse captions.

## The run (a recording)

Work folder per stream: `work/<name>/`. Note the start time; the user will ask how long it took.

1. **Audio** (about 40 s for a 9-hour Twitch VOD; pieces download in parallel):
   `scripts/fetch.py audio <url> work/<name>/audio.mp4`
   Chat: YouTube replays may have one (`yt-dlp --skip-download --write-subs --sub-langs live_chat`).
   Twitch has no official chat download; use a chat file only if the user exported it themselves.
2. **Spikes** (about 20 s per 9 h): `scripts/signals.py work/<name>/audio.mp4 work/<name> [--chat chat.json]`
   The 60 biggest moments that are loud against the streamer's own 10-minute level AND loud in absolute
   terms. Sleep and AFK hours of marathons are ignored (`--all-hours` turns that off).
3. **Transcripts** (about 1 min): `scripts/transcribe_candidates.py work/<name>/audio.mp4 work/<name>`
   40 s around each spike -> `candidates.md`. The stream is decoded once into an exact timeline
   (`audio16k.s16`), so this also works on 40+ hour marathons. Game dialogue gets mixed in; that is fine.
4. **Judge. This is your job.** Read `candidates.md` and score every moment 0-10:
   - 9-10: a stranger scrolling stops and gets it instantly (a clear funny or shocking event and the
     streamer's reaction to it).
   - 7-8: strong reaction with a clear cause.
   - 4-6: loud but no context (yelling at chat, music, donation sounds, inside jokes).
   - 0-3: ads, menus, story cutscenes, game TV, dark scenes, the streamer asleep.
   Write `scores.md` (every moment, score, one-line reason) and `picks.json` for the 7+ ones:
   `{"id": "c1-crash", "start": <first second of the hook line>, "peak": <the payoff>, "end": <right after the punchline>, "title": "...", "beats": [{"t": <stream second>, "text": "*CHECKING*"}]}`
   Times are stream seconds. Aim for 8-20 s after jump cuts. Start on the line or action that hooks
   (a question, a dare, "watch this"), end right after the payoff, no trailing chatter. `beats` are
   optional one- or two-word action captions for silent action the viewer must follow.
   Title (for the post caption, not drawn on the clip): streamer's name first, what happens, plain
   words, under 55 characters.
5. **Look before you render.** For each pick, preview the picture:
   `scripts/fetch.py video <url> <start> <end> work/<name>/prev-<id>.mp4 --format "480p/360p"`
   then a frame strip (`ffmpeg -i prev.mp4 -vf "fps=1/3,scale=240:135,tile=15x1" -frames:v 1 strip.jpg`).
   Drop the pick if the payoff is not visible (dark scene, happens off screen, cutscene). Candidate
   timings can be 2-3 s off; set `peak` where the payoff actually is on screen.
6. **Layout:** `scripts/find_layout.py <url> work/<name> [--at <a second inside the stream>]`
   Finds the webcam (the face that stays in one place; game faces move) and writes `layout.json`. Look
   at the probe frame. If the streamer moves the camera, put `"facecam"`/`"face"` into that pick.
7. **Render** (about 15 s per clip): `scripts/render.py <url> work/<name> [--only c1-crash]` -> `work/<name>/clips/`
8. **Check:** `scripts/check_clip.py work/<name>`. Hard checks (fail = fix before showing anyone):
   audio and video start together, no two captions at once. It also lists every word with its time and
   flags words that do not sit on the streamer's voice, and writes `check/<id>.jpg` (the caption band at
   5 fps). Read the words, look at the sheet, fix picks and re-render with `--only`.
9. **Report** in plain words: stream length, moments found, kept, minutes taken, the clips. Remind the
   user of the campaign's caption, hashtag and submission window.

## Clip style (`split`, default)

Copied from a top clipping-channel Short (399K views, 6.9 s), measured frame by frame:
- 1080x1920. Webcam on top, 796 px (41%), cropped tight on the face (face about a third of the width,
  `face_zoom` 3.4). Game below, 1124 px, centre crop zoomed 1.2x. Hard seam, no border.
- Captions: one word at a time, held until the next word, caps, narrow bold sans (DIN Condensed Bold;
  Impact as fallback), white with a thick black outline and a hard black shadow, centred on the seam.
- `*ACTION*` captions only in real silences of 0.7 s or more.
- Jump cuts: silent stretches over 1.2 s shrink to two 0.35 s pieces, except around the payoff, which
  always plays in full. A 0.13 s red flash on the loudest moment near the peak.
- If the game goes to a black loading screen ("wasted", "busted"), the last game frame is frozen over it.
- No title card, no repeated hook: the clip starts on its own hook line.
- `"style": "classic"` (in layout.json or a pick) gives the older look: title card, peak shown first.

## Captions: exact words

Captions are burned in, so they must be right. Pipeline (render.py `words_of`, about 15 s per clip):
1. **Clean voice.** Demucs (htdemucs) separates the streamer's voice from game sound and music. Speech
   over a game is what makes whisper mishear ("five" -> "well").
2. **Text.** whisper.cpp large-v3 (q5_0) with DTW token timing (`-dtw large.v3 -nfa`; DTW needs flash
   attention off) on the clean voice. Turbo, small and base dropped or misheard words in testing.
3. **Exact timing.** `scripts/align_words.py` (in `.align-venv`): wav2vec2 forced alignment
   (torchaudio WAV2VEC2_ASR_LARGE_LV60K_960H, the WhisperX method) pins every word to the clean voice.
   Each phrase is aligned only in its own window; where the aligner and whisper disagree by more than
   0.35 s (it can latch onto the wrong one of "okay okay okay"), whisper's time is kept.
4. **Safety.** Stacked or zero-length timings are spread out; sound tags `(...)`, `[...]`, `*...*`, `♪`
   are dropped; never two captions on screen; slurs are dropped and swears masked (F*CK, SH*T). The
   audio itself is left as is.
Measured on test clips: whisper timing alone ran 0.1-0.26 s late; after alignment captions land within
a few hundredths of a second of the voice.

## Live mode (clip while the stream is on)

`scripts/live.py <twitch_channel> work/live-<name> --minutes 60 [--min-score 7] [--model sonnet]`

- Pulls the live video in 2-second pieces (old MPEG-TS and newer fMP4 Twitch streams both work), keeps
  the last 20 minutes, reads the public chat anonymously (read-only IRC), and watches two signals:
  chat much faster than its own 10-minute normal, and the streamer much louder than usual.
- A moment needs chat AND loudness (or a flood of "clip" in chat). It cuts 25 s before and 22 s after,
  transcribes, and asks headless `claude -p` (Sonnet) to open four frames and give a 0-10 score, clip
  bounds, beats and a title. 7+ is rendered straight away into `clips/`. Everything is logged to
  `events.log` (`MOMENT`, `SCORE`, `CLIP`), one line each, for a monitor to follow.
- A verdict reason like "I didn't view the frames" means its beats are guesses: drop them.
- Near misses (score 6) can be rendered by hand: `scripts/render_candidate.py work/live-<name> m07 m12`.
- First 2 minutes are calibration. Anonymous viewers get Twitch ad breaks: the stream is not served
  during an ad, so moments in that gap are lost. Sleep streams produce nothing; check the stream is active.
- Moment to finished clip: about 1 minute (plus ~15 s for exact captions). Mac and Claude Code must stay on.

## Known pitfalls (all handled in the scripts; keep them handled)

- Twitch pieces start audio and video at different times: cut sections with output seeking
  (`-i joined -ss off`), never input seeking, or video lands 1.5 s off the audio.
- Seeking inside a stream stitched from HLS pieces drifts after ~20 h: use the decoded timeline.
- whisper.cpp small/base word timestamps collapse (several words on one time); never caption with them
  without alignment.
- A face in the game is not the webcam: pick the face that stays put across frames.
- Judge frames must be absolute paths (`claude -p` runs with cwd = the candidate folder).
