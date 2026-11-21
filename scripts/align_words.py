#!/usr/bin/env python3
"""Exact word timestamps: forced alignment of a known transcript to the audio (the WhisperX method).

whisper gives the words; wav2vec2 (torchaudio WAV2VEC2_ASR_BASE_960H, CTC over letters) pins every word
to the audio, accurate to ~20-40 ms. Runs in its own environment (.align-venv, Python 3.12 + torch).

    .align-venv/bin/python scripts/align_words.py <audio.wav> <words_in.json> <words_out.json>

words_in.json: [[word, start, end], ...] (rough times, e.g. whisper DTW)
words_out.json: same list with aligned start/end; words with no letters (numbers, symbols) keep their rough
times, squeezed between their aligned neighbours.
"""
import json, re, sys
import torch, torchaudio

wav_path, in_path, out_path = sys.argv[1:4]
words = json.load(open(in_path))
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H   # large model: steadier on fast repeated words
labels = bundle.get_labels()                    # ('-', '|', 'E', 'T', ...): '-' blank, '|' word separator
index = {c: i for i, c in enumerate(labels)}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = bundle.get_model().to(dev).eval()

import soundfile as sf
data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
wave = torch.from_numpy(data.mean(1)).unsqueeze(0)
if sr != bundle.sample_rate:
    wave = torchaudio.functional.resample(wave, sr, bundle.sample_rate)
with torch.inference_mode():
    emission, _ = model(wave.to(dev))
emission = torch.log_softmax(emission, dim=-1).cpu()
sec_per_frame = wave.shape[1] / bundle.sample_rate / emission.shape[1]


def norm(w):
    return re.sub(r"[^A-Z']", "", w.upper().replace("’", "'"))


def align_chunk(idx, t0, t1):
    """Align words[idx] inside audio [t0, t1] (seconds). Returns {word_index: (start, end)}."""
    f0, f1 = int(t0 / sec_per_frame), min(emission.shape[1], int(t1 / sec_per_frame) + 1)
    em = emission[:, f0:f1]
    keys_ = [(i, norm(words[i][0])) for i in idx]
    keys_ = [(i, k) for i, k in keys_ if k]
    if not keys_:
        return {}
    tokens = []
    for _, k in keys_:
        tokens += [index[c] for c in k if c in index] + [index["|"]]
    tokens = tokens[:-1]
    if em.shape[1] < len(tokens) + 2:
        return {}
    ali, scores = torchaudio.functional.forced_align(em, torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp())
    groups, cur = [], []
    for sp in spans:
        if sp.token == index["|"]:
            if cur:
                groups.append(cur); cur = []
        else:
            cur.append(sp)
    if cur:
        groups.append(cur)
    return {i: ((f0 + g[0].start) * sec_per_frame, (f0 + g[-1].end) * sec_per_frame) for (i, _), g in zip(keys_, groups)}


# phrases = runs of words whisper heard less than 0.8 s apart; each is aligned only in its own window,
# so noise before a phrase cannot pull its words away
out = [list(w) for w in words]
chunks, cur = [], [0] if words else []
for i in range(1, len(words)):
    if words[i][1] - words[i - 1][2] > 0.8:
        chunks.append(cur); cur = []
    cur.append(i)
if cur:
    chunks.append(cur)
total = wave.shape[1] / bundle.sample_rate
done = 0
for ch in chunks:
    t0 = max(0.0, words[ch[0]][1] - 0.6); t1 = min(total, words[ch[-1]][2] + 0.6)
    got = align_chunk(ch, t0, t1)
    for i in ch:
        if i not in got:
            continue
        a, b = got[i]
        # two independent clocks: whisper DTW and the letter aligner. Agree within 0.35 s -> the aligner
        # (sharper); disagree -> the aligner has latched onto the wrong one of several identical words
        # ("okay okay okay"), keep whisper's time.
        if abs(a - words[i][1]) <= 0.35 and b - a >= 0.04:
            out[i][1], out[i][2] = round(a, 3), round(b, 3); done += 1
        else:
            out[i][1] = round(max(0.0, words[i][1] - 0.04), 3)
keys = [norm(w[0]) for w in words]
# words without letters: keep order, sit between their neighbours
for i, k in enumerate(keys):
    if not k:
        lo = out[i - 1][2] if i else 0.0
        hi = next((out[j][1] for j in range(i + 1, len(out)) if keys[j]), out[i][2])
        out[i][1] = max(lo, min(out[i][1], hi - 0.1)); out[i][2] = max(out[i][1] + 0.1, min(out[i][2], hi))
json.dump(out, open(out_path, "w"))
print(f"aligned {done}/{len(words)} words exactly")
