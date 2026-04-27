# VoiceClone — F5-TTS Zero-Shot Voice Cloning TTS

## Quick Start

```bash
./serve.sh              # Apple Silicon (MPS) / CPU
./serve-dgx.sh          # NVIDIA CUDA (DGX Spark, etc.)
./serve.sh --port 9000  # custom port (same flag on both)
```

Requires: `uv`. The serve scripts pull `f5-tts`, `torch`, `torchaudio` via
`uv run`. CUDA wheels come from the index in `serve-dgx.sh`
(`https://download.pytorch.org/whl/nightly/cu130` by default; override with
`TORCH_INDEX=…`).

## Backend

[F5-TTS](https://github.com/SWivid/F5-TTS) (`SWivid/F5-TTS_v1`) via the
`f5_tts.api.F5TTS` high-level wrapper. It clones a 24 kHz reference voice
without fine-tuning. Runs on:

- `cuda` — fastest (well under realtime on a modern NVIDIA GPU)
- `mps` — Apple Silicon, ~1.5–2× realtime
- `cpu` — last-resort fallback, many× realtime

Device is auto-picked (`cuda > mps > cpu`); override with `--device`.

F5-TTS does not expose `temperature` / `top_p` / `top_k`, so those rows are
hidden in the frontend tuning panel. The server still accepts them in
request bodies (silently ignored) for protocol stability.

## Architecture

### Server (`web_tts_server.py`)

- Python HTTP server with **ThreadingMixIn** for concurrent request handling.
- **Single GPU worker thread** (`_gen_worker`) — the ONLY thread that ever
  calls `MODEL.generate()` or touches the GPU.
  - PyTorch on MPS / CUDA tolerates concurrent calls better than MLX/Metal,
    but the single-worker invariant is preserved for predictable memory and
    deterministic ordering.
  - All jobs go through a `PriorityQueue`: priority 0 (regen) runs before
    priority 1 (batch).
  - Worker converts torch tensors → numpy on the worker thread before
    signaling the handler (mirrors the original MLX→numpy pattern).
  - Periodic `torch.cuda.empty_cache()` every 10 jobs (no-op on MPS/CPU).
- **SSE streaming** for batch generation (`/generate`) — streams `info`
  (with `chunk_texts`), `chunk`, `error`, `done` events.
- **Synchronous regen** (`/regenerate`) — returns JSON with the regenerated
  chunk (also used by resume).
- **Text splitting** (`/split`) — lightweight text-only endpoint, no GPU;
  returns `{ chunks, total }`.
- **Apply cuts** (`/apply-cuts`) — applies user-modified cuts to raw audio,
  returns processed WAV.
- **Cancel** (`/cancel`) — sets per-request `threading.Event` checked every
  500 ms during generation.
- **Status** (`/status`) — `ready`, `model`, `defaults`, `param_ui` for
  frontend label/visibility overrides, plus `ref_name` / `ref_text_full`
  describing the active reference voice.
- **Reference upload** (`POST /reference`) — multipart `audio` blob +
  `transcript` field. Decodes via soundfile (with ffmpeg fallback for
  MP3/M4A), resamples to 24 kHz mono, rejects clips longer than ~12s,
  runs the standard `trim_audio` cleanup pass (silence trim, fades,
  80 Hz highpass, -1 dB normalize) so uploads land on the same audio
  footing as the bundled sample, writes `reference_uploaded.wav`, and
  atomically swaps `MODEL.ref_path` / `MODEL.ref_text` under
  `_ref_lock`. Returns 409 while a generation job is in flight. No
  persistence across restarts.

### Frontend (`web_tts.html`)

- Single-file HTML/CSS/JS — no build system, no dependencies.
- Web Audio API for playback with chunk-level seeking.
- Canvas waveform rendering with color-coded cut regions.
- Chunk Inspector modal: play trimmed/raw, view cuts, edit text, regenerate, delete.
- **Interactive trim selection**: click-drag on inspector waveform to select regions, then Trim/Untrim/Reset.
- Regen queue: multiple regens can be queued, processed sequentially, with log badges showing queue status.
- **Session persistence** (IndexedDB): auto-saves after each chunk complete, regen, or delete.
  - Save As / Load / Delete / New session management UI.
  - Export to gzipped `.vcsession` file for portability.
  - Import `.vcsession` files to restore sessions.
- **Silence threshold slider**: -50 dB to -25 dB (default -40 dB) in main tuning and inspector panels.
- **Output volume normalization**: -1 dB (default), -3 dB, -6 dB, -12 dB, or Off for final download.
- **Pitch-preserving speed control** (0.50× – 1.50×) via `<audio preservesPitch>`.
- **Resumable generation**: chunks are pre-staged with text before audio generates.
  - Cancel mid-generation → pending chunks persist with warm yellow waveform indicators.
  - Resume bar appears with count of pending chunks + Resume/Stop buttons.
  - Pending state survives page reload (IndexedDB), export/import (`.vcsession`), and session load.
  - Uses `/regenerate` endpoint sequentially for each pending chunk.
- **Dynamic tuning UI**: labels, slider ranges, and visibility are populated
  from the `/status` response's `param_ui` block on first load.

### Audio Processing Pipeline (server-side, numpy)

1. **Artifact skip** — spectral scan removes tinny/metallic prefix (legacy from CSM-1B; F5-TTS rarely needs it but the step is harmless and cheap).
2. **Silence trim** — leading/trailing silence removal (threshold from `silence_db` param, default -40 dB).
3. **Fade** — 15 ms fade-in, 5 ms fade-out.
4. **Highpass** — 80 Hz single-pole filter (rumble removal).
5. **Noise gate** — threshold = `silence_db + 5 dB`, 5 ms attack, 50 ms release.
6. **Normalize** — -1 dB peak.
7. **Pause compress** — silence gaps shortened to max 300 ms (threshold = `silence_db + 4 dB`).
8. **Garbage detection** — auto-retry on tin-can/static/silence/clipping (up to 2 retries; mainly defensive — F5-TTS output is reliable).

## Key Files

| File | Description |
|------|-------------|
| `web_tts_server.py` | F5-TTS server — generation, audio processing, SSE streaming |
| `web_tts.html` | Frontend — player, inspector, waveform, regen queue, sessions |
| `serve.sh` | Apple Silicon / CPU launcher (installs deps via uv, opens browser) |
| `serve-dgx.sh` | NVIDIA CUDA launcher (CUDA torch wheels via uv) |
| `SETUP.md` | Onboarding guide — recording reference audio, setup, tips |
| `reference_clean.wav` | Reference audio for voice cloning (24 kHz, public-domain LibriVox sample by default) |
| `transcript.txt` | Transcript of reference audio (F5-TTS uses this for the reference text) |
| `clean_reference.sh` | Audio cleanup for reference recordings |
| `record_and_transcribe.sh` | Record + Whisper transcribe reference |

## Threading Model

```
HTTP threads (ThreadingMixIn)          GPU Worker Thread
─────────────────────────────          ─────────────────
/generate handler ──► PriorityQueue ──► _gen_worker()
  submits jobs          (pri=1)           │
  waits on Event                          ├─ MODEL.generate() (F5TTS.infer)
                                          ├─ tensor.detach().cpu().numpy()
/regenerate handler ──► PriorityQueue     ├─ torch.cuda.synchronize() (CUDA only)
  submits jobs          (pri=0)           └─ result_event.set()
  waits on Event
```

**Critical rule**: No thread other than `_gen_worker` may ever call
`MODEL.generate()` or touch GPU tensors directly.

## Common Issues

- **`torch.cuda.is_available()` is False on the DGX**: PyTorch was installed
  without CUDA — re-run via `serve-dgx.sh` so `uv` resolves against the
  CUDA wheel index (`TORCH_INDEX`).
- **F5-TTS gated checkpoint**: visit
  https://huggingface.co/SWivid/F5-TTS, accept the license, and ensure
  `HF_TOKEN` is set (or `huggingface-cli login`).
- **Generation blocks regen**: Regen should jump the queue (priority=0).
  If not happening, check that the regen path passes `priority=True` to
  `generate_chunk()`.
- **First launch is slow**: F5-TTS_v1 + Vocos download (~1.5 GB) on first
  start; subsequent runs hit the cache.

## Model

- **F5-TTS_v1** (SWivid) via `f5_tts.api.F5TTS`.
- Zero-shot voice cloning from a single reference audio + transcript.
- Sample rate: 24 kHz.
- Inference-only — fine-tuning is upstream's concern, not handled here.
