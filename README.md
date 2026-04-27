# VoiceClone

A self-hosted, zero-shot voice cloning TTS web app powered by
[F5-TTS](https://github.com/SWivid/F5-TTS). Drop in a short reference clip,
paste in text, and stream back audio in that voice — with a chunk-level
inspector for cleanup, regen, trimming, and export.

Runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU.

## Quick Start

```bash
./serve.sh                  # Apple Silicon / CPU (auto-picks mps > cpu)
./serve-dgx.sh              # NVIDIA CUDA (DGX, workstation, etc.)
```

Both open `http://localhost:8765` once the model is loaded. The repo ships
with a public-domain LibriVox reference voice (~9s of *The Raven* read by
Chris Goringe), so it works out of the box — no recording required to try
it out.

To use your own voice instead, see [SETUP.md](SETUP.md).

## Features

- **Streaming generation** (SSE) with per-chunk progress
- **Resumable generation** — pending chunks survive cancel, page reload, and session export
- **Chunk inspector**: play raw vs. trimmed, regenerate, edit text, interactive trim selection
- **Pitch-preserving speed control** (0.50× – 1.50×)
- **Server-side audio pipeline**: silence trim, fade, highpass, normalize, pause compression
- **Session persistence** (IndexedDB) + export/import as `.vcsession` files

## Hardware

| Backend       | Device | RTF (rough)        | Notes                          |
|---------------|--------|--------------------|--------------------------------|
| Apple Silicon | MPS    | ~1.5–2× realtime   | M-series, 16+ GB unified RAM   |
| NVIDIA        | CUDA   | well under realtime | DGX Spark, 24+ GB GPU          |
| Any           | CPU    | many× realtime     | Last-resort fallback           |

## Requirements

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — installs all Python deps on the fly
- `ffmpeg`, `sox` — `brew install ffmpeg sox` (Mac) / your distro's package manager
- For Apple Silicon: macOS 14+
- For CUDA: PyTorch CUDA wheels (`serve-dgx.sh` defaults to the cu130 nightly index; override with `TORCH_INDEX=…`)

The first launch downloads the F5-TTS_v1 model (~1.5 GB) into the HuggingFace cache.

## Architecture

A single GPU worker thread feeds a priority queue:
- priority 0 (regen) jumps ahead of priority 1 (batch).
- All `MODEL.generate()` calls happen on that one thread; HTTP handlers never touch GPU state.

See [CLAUDE.md](CLAUDE.md) for deeper notes on threading, the audio post-processing pipeline, and common issues.

## Repo Layout

```
.
├── serve.sh                 # F5-TTS launcher (Apple Silicon / CPU)
├── serve-dgx.sh             # F5-TTS launcher (NVIDIA / CUDA)
├── web_tts_server.py        # F5-TTS server
├── web_tts.html             # frontend (player, inspector, sessions)
├── reference_clean.wav      # default reference voice (LibriVox, public domain)
├── transcript.txt           # transcript of reference audio
├── clean_reference.sh       # mic recording → clean 24kHz WAV
├── record_and_transcribe.sh # record + Whisper transcribe
└── SETUP.md                 # onboarding guide
```

## Default Voice Sample

`reference_clean.wav` is a 9-second excerpt from the LibriVox recording of
*The Raven* read by Chris Goringe
([archive.org/details/raven](https://archive.org/details/raven)).
LibriVox recordings are dedicated to the public domain.

## License

[MIT](LICENSE).
