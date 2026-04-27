# VoiceClone Setup

Get the server running, then (optionally) replace the bundled voice with
your own. The repo ships with a public-domain LibriVox reference, so you
can skip straight to the *Run the Server* section if you just want to try
it out.

---

## 1. Prerequisites

### Common

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — handles
  all Python deps on the fly.
- `ffmpeg`, `sox` — for recording and audio cleanup.

```bash
# macOS
brew install ffmpeg sox

# Debian / Ubuntu
sudo apt install ffmpeg sox
```

### Hugging Face access

F5-TTS_v1 is gated. Visit
[huggingface.co/SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS),
accept the license, then log in:

```bash
huggingface-cli login
```

The first server start will download the model (~1.5 GB) into the HF
cache.

### Apple Silicon (MPS / CPU)

No extra setup — `serve.sh` calls `uv` which installs `f5-tts`, `torch`,
and `torchaudio` for you.

### NVIDIA / DGX (CUDA)

`serve-dgx.sh` defaults to the PyTorch nightly cu130 wheel index. Override
for a different CUDA version:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu121 ./serve-dgx.sh
```

---

## 2. Run the Server

```bash
./serve.sh                 # Apple Silicon / CPU (auto-picks mps > cpu)
./serve-dgx.sh             # NVIDIA / CUDA

./serve.sh --port 9000     # custom port
./serve.sh --device cpu    # force CPU
```

It opens `http://localhost:8765` once the model is loaded. Paste in text,
hit Generate, and audio streams back chunk by chunk in the bundled
LibriVox voice.

---

## 3. (Optional) Use Your Own Voice

The bundled reference is fine for testing, but the whole point of voice
cloning is to use *your* voice. Three steps:

### 3a. Record a reference clip

```bash
./record_and_transcribe.sh 20
```

Records 20 s from your default mic, saves to `reference_voice.wav`,
auto-transcribes via Whisper, writes `transcript.txt`. Adjust the duration
as needed (aim for 6–15 s of actual speech).

What to say — anything natural and conversational, single speaker, no
music or other voices. Example:

> "This is a recording of my voice. My name is [your name]. I'm recording
> this sample so that a voice cloning model can learn what I sound like."

### 3b. Review the transcript

Whisper is good but not perfect. Open `transcript.txt` and fix anything it
got wrong — F5-TTS aligns the reference audio to this text, so accuracy
matters.

### 3c. Clean the audio

```bash
./clean_reference.sh
```

Reads `reference_voice.wav`, writes `reference_clean.wav`. Pipeline:
spectral noise reduction → fade-in → silence trim → highpass 80 Hz →
lowpass 6 kHz → compress → normalize → resample to 24 kHz mono.

Common options:

```bash
./clean_reference.sh reference_voice.wav reference_clean.wav --trim-start 0.5
./clean_reference.sh reference_voice.wav reference_clean.wav --denoise 0.3
./clean_reference.sh reference_voice.wav reference_clean.wav --no-denoise
./clean_reference.sh reference_voice.wav reference_clean.wav --no-trim
```

Verify:

```bash
afplay reference_clean.wav   # macOS
play reference_clean.wav     # sox / Linux
```

Aim for 6–15 s of clean speech.

### 3d. Restart the server

```bash
./serve.sh
```

---

## 4. Tips for a Good Reference Recording

- **Duration**: 6–15 s of actual speech. Shorter ⇒ too little to clone from. Longer doesn't help much.
- **Quiet room**: background noise is the #1 killer of clone quality.
- **Mic distance**: 6–12 inches. Too close ⇒ plosives. Too far ⇒ room echo.
- **Speak naturally**: your everyday voice, not a "radio voice."
- **Single speaker only**: no other people, no TV, no music.
- **Avoid clipping**: if the audio hits max volume, back away from the mic.

---

## 5. Troubleshooting

**"F5-TTS checkpoint appears gated"**
Accept the license at huggingface.co/SWivid/F5-TTS and run
`huggingface-cli login`.

**"--device cuda requested but no CUDA device available"**
PyTorch was installed without CUDA. Use `serve-dgx.sh` (which pulls the
CUDA wheels) or run `./serve.sh --device mps` (Apple Silicon) /
`--device cpu`.

**Server slow to start the first time**
First launch downloads F5-TTS_v1 + Vocos (~1.5 GB) into the HF cache. Be
patient. Subsequent launches load from cache in seconds.

**Generation is slow on Mac**
F5-TTS on MPS runs around 1.5–2× realtime — that's expected. CUDA is much
faster. The web UI streams chunks as they finish, so long-form generation
doesn't block playback.

**Recording captures no audio (Mac)**
`record_and_transcribe.sh` uses avfoundation device `:1`. List devices
with:

```bash
ffmpeg -f avfoundation -list_devices true -i "" 2>&1
```

Edit the `-i ":1"` line in the script to match your mic's index.

**Port already in use**

```bash
./serve.sh --port 9000
```
