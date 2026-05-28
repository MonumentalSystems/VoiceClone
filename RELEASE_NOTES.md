# Release Notes

Newest first. Each tagged release (`vMAJOR.MINOR.PATCH`) gets an entry here.

## v0.6.0 — 2026-05-28

**Batch TTS daemon work orders (MVP floor: WO-1 + WO-3).**

Adds the two server-side primitives the Hyades durable batch-narration job grain
(`IBatchTtsJobGrain`) needs to drive long, segment-by-segment synthesis. See
`~/hyades/docs/TTS_BATCH_DESIGN.md` §4. Both endpoints are **additive** — the
existing synchronous/streaming paths (`/tts_stream`, `/generate`, `/regenerate`)
are unchanged.

- **`POST /segment` (WO-1)** — server-side segmentation helper. Text-only, no
  GPU. `{ text, maxChars?, minChunk? }` → `{ segments: [{index, text}], total }`.
  Same `split_text()` chunking as `/split`, reshaped into the indexed segment
  plan a batch driver stores. Accepts camelCase or snake_case knobs.

- **`POST /tts/segment` (WO-3)** — per-segment synthesis primitive.
  `{ job_id?, index, text, f5_speed?, f5_cfg_strength?, f5_nfe_step?,
  f5_output_gain_db?, trim_pauses?, silence_db? }` →
  `{ job_id, index, audio_b64, audio_ms, sample_rate, gen_time }`.
  - Idempotent on `(job_id, index)`: holds no accumulating daemon state, so a
    crash-recovery re-call simply re-synthesizes (the latest result is
    authoritative — "re-synthesizing overwrites"; subsumes WO-7 idempotency).
  - Bytes are **not** cached daemon-side — memory stays bounded to one segment,
    mirroring `/regenerate`. Job durability/storage stays Hyades-side for this MVP.
  - Enters the worker queue at **batch priority (1)** so interactive regen/TTS
    (priority 0) jumps ahead — the daemon-side half of fill-idle yielding.
  - Defaults mirror the `/tts_stream` read-aloud profile; all overridable.

Not in this release (deferred, stay Hyades-side per the design): WO-2 (daemon
durable jobs), WO-4 (status polling), WO-5 (abort), WO-6 (daemon WAV store +
assembled output), WO-7 (resume — subsumed by WO-3 idempotency).

_Note: first tracked git tag for this repo. Versioning starts at v0.6.0 —
semver minor bump over the prior `/tts_stream` feature work; the "0.5.x" numbers
referenced in the Hyades design doc are Hyades' TTS subsystem, not this repo._
