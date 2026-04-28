#!/usr/bin/env python3
"""
Streaming TTS Web Server

Accepts text via HTTP, chunks it, generates voice-cloned audio,
and streams WAV chunks back via chunked transfer encoding.

The frontend buffers N chunks ahead so playback stays smooth
even while generation continues.

Usage:
    uv run --with f5-tts --with torch --with torchaudio \
        python3 web_tts_server.py [--port 8765] [--model F5-TTS]

This is the F5-TTS variant of the CUDA / PyTorch port (DGX Spark). The
HTTP/SSE/audio post-processing layers are unchanged; only the model load
and `_gen_worker` body were swapped to use F5-TTS via the
`f5_tts.api.F5TTS` high-level API on CUDA. The single-worker-thread
invariant is preserved.
"""

import argparse
import io
import json
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import soundfile as sf

# Globals set at startup
MODEL = None
REF_AUDIO_DATA = None
REF_TEXT = ""
REF_NAME = ""  # display name for the active reference (filename only)
SAMPLE_RATE = 24000
MAX_CHARS = 375
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lock for swapping the active reference voice. Held briefly while
# MODEL.ref_path / MODEL.ref_text are mutated; the gen worker reads
# REF_AUDIO_DATA / REF_TEXT under no lock but each .infer() call is
# atomic w.r.t. the swap because the worker is single-threaded.
_ref_lock = threading.Lock()
MAX_REF_DURATION_S = 12.0
MAX_REF_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB cap on raw upload payload

# Per-request cancellation: maps request_id → threading.Event (set = cancelled)
_cancel_events = {}  # type: dict[str, threading.Event]
_cancel_lock = threading.Lock()

# ── Single-Worker Generation Queue ───────────────────────────────────
# MLX/Metal is NOT thread-safe at any level — even a Python threading.Lock
# isn't enough because Metal command buffers crash if two threads have
# ever touched the model, even sequentially.  The only safe approach is
# a SINGLE dedicated thread that owns all MODEL.generate() calls.
#
# Design: a PriorityQueue feeds jobs to one worker thread.
#   • priority=0  → regeneration  (high priority, runs first)
#   • priority=1  → batch chunk   (low priority)
# Each job is (priority, sequence_number, args_dict, result_event).
# The caller blocks on result_event until the worker finishes.

_gen_queue = queue.PriorityQueue()
_gen_seq = 0        # tie-breaker for same-priority jobs (FIFO within level)
_gen_seq_lock = threading.Lock()


_gen_worker_count = 0  # jobs completed by worker (for periodic cache clear)
# Cap on how many jobs the worker batches in one F5 forward pass. Larger =
# better aggregate throughput (more GPU parallelism) but each job's wall
# time is bounded by the longest in the batch, so don't go too high.
MAX_BATCH = int(os.environ.get('GEN_MAX_BATCH', '8'))


def _drain_same_priority(first):
    """Pull first item plus up to MAX_BATCH-1 *same-priority* jobs from the
    queue without blocking. If a different-priority job is encountered it's
    put back so the queue's priority ordering still holds."""
    batch = [first]
    target_pri = first[0]
    while len(batch) < MAX_BATCH:
        try:
            nxt = _gen_queue.get_nowait()
        except queue.Empty:
            break
        if nxt[0] != target_pri:
            _gen_queue.put(nxt)
            break
        batch.append(nxt)
    return batch


def _normalize_audio_arrays(results, torch_mod):
    """Convert MODEL.generate(...) result list to numpy float32 mono."""
    out = []
    for r in results:
        audio = r.audio
        if hasattr(audio, 'detach'):
            audio = audio.detach().to('cpu', dtype=torch_mod.float32).numpy()
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        out.append(np.asarray(audio, dtype=np.float32).flatten())
    return out


def _gen_worker():
    """Dedicated worker thread — only thread that ever calls MODEL.generate().
    On the DGX/CUDA build, PyTorch is thread-safer than MLX, so we *can*
    batch up to MAX_BATCH same-priority jobs in a single call when MODEL
    exposes ``generate_batch``. The single-worker invariant still holds —
    we just process up to N jobs per pickup instead of strictly one.
    """
    import torch
    global _gen_worker_count
    print(f"  [gen_worker] started on thread {threading.current_thread().name}")
    while True:
        first = _gen_queue.get()
        batch = _drain_same_priority(first) if MAX_BATCH > 1 else [first]
        n = len(batch)
        priority = batch[0][0]
        seqs = [b[1] for b in batch]
        pri_label = "REGEN" if priority == 0 else "BATCH"
        if n == 1:
            print(f"  [gen_worker] picked up {pri_label} job seq={seqs[0]}")
        else:
            print(f"  [gen_worker] picked up {pri_label} ×{n} jobs seqs={seqs}")
        try:
            if n > 1 and hasattr(MODEL, 'generate_batch'):
                # Batched path — single F5 invocation runs N gen_texts in
                # parallel via ThreadPoolExecutor under the hood.
                texts = [b[2]['text'] for b in batch]
                results = MODEL.generate_batch(
                    texts,
                    ref_audio=REF_AUDIO_DATA,
                    ref_text=REF_TEXT,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Each result corresponds to one job in input order.
                for (_, _, job, _), r in zip(batch, results):
                    job['_audio_arrays'] = _normalize_audio_arrays([r], torch)
            else:
                # Single-job path (or fallback when no batch API)
                for _, _, job, _ in batch:
                    results = MODEL.generate(
                        text=job['text'],
                        ref_audio=REF_AUDIO_DATA,
                        ref_text=REF_TEXT,
                        temperature=job['temperature'],
                        top_p=job['top_p'],
                        top_k=job['top_k'],
                        verbose=False,
                    )
                    job['_audio_arrays'] = _normalize_audio_arrays(results, torch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            _gen_worker_count += n
            if _gen_worker_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # On batch failure, mark every job in the batch with the error
            # so each waiter sees a clean failure (don't silently drop jobs).
            for _, _, job, _ in batch:
                if '_audio_arrays' not in job:
                    job['_error'] = e
            print(f"  [gen_worker] ERROR ({n}-way): {e}")
        finally:
            for _, _, _, ev in batch:
                ev.set()
                _gen_queue.task_done()
            if n == 1:
                print(f"  [gen_worker] done with {pri_label} job seq={seqs[0]}")
            else:
                print(f"  [gen_worker] done with {pri_label} ×{n} (seqs={seqs})")


def _submit_generate(text, temp, top_p, top_k, priority=False):
    """Submit a generation job and block until the worker completes it.
    priority=True → runs before batch chunks.
    Returns (results_list) or raises on error."""
    global _gen_seq
    with _gen_seq_lock:
        seq = _gen_seq
        _gen_seq += 1

    job = {'text': text, 'temperature': temp, 'top_p': top_p, 'top_k': top_k}
    result_event = threading.Event()
    pri = 0 if priority else 1
    _gen_queue.put((pri, seq, job, result_event))

    # Block until worker finishes this job — caller can poll result_event
    # externally for cancel checks
    return job, result_event


def prefilter_text(text):
    """Strip symbols and formatting that won't generate well as speech."""
    # Remove horizontal rules / separator lines  (===, ---, ***, ___)
    text = re.sub(r'[=\-\*_]{3,}', ' ', text)

    # Remove markdown headers (# ## ### etc) but keep the text
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)

    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove image tags ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove code fences and inline code
    text = re.sub(r'```[^`]*```', ' code block ', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove bullet markers (-, *, numbered lists)
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove repeated punctuation (... is OK, but !!!!! or ???? not great)
    text = re.sub(r'([!?]){2,}', r'\1', text)
    text = re.sub(r'\.{4,}', '...', text)

    # Remove pipe tables  |col|col|
    text = re.sub(r'\|[^\n]+\|', ' ', text)

    # Remove standalone special chars that aren't speech
    text = re.sub(r'[~^\\|@#$%&{}\[\]<>]', ' ', text)

    # Normalize smart quotes to ASCII equivalents (model knows these)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Normalize ASCII double-dash to proper em dash (what LLMs see in training)
    text = re.sub(r'(?<!\-)--(?!\-)', '\u2014', text)

    # Collapse whitespace within lines, but preserve newlines as split boundaries
    text = re.sub(r'[^\S\n]+', ' ', text)      # collapse spaces/tabs (not newlines)
    text = re.sub(r' *\n *', '\n', text)        # clean whitespace around newlines
    text = text.strip()

    return text


MIN_CHUNK = 150  # minimum chars per chunk — short chunks cause voice drift


def split_text(text, max_chars=MAX_CHARS, min_chunk=MIN_CHUNK):
    """Split text into chunks at sentence boundaries.

    Strategy:
      1. Split on double-newlines into paragraphs (hard boundaries)
      2. Single newlines within a paragraph are treated as spaces
      3. Split paragraphs into sentences at . ! ? (and em-dash/colon boundaries)
      4. Merge consecutive sentences until chunk reaches min_chunk
      5. Don't exceed max_chars — split long sentences at commas
      6. Short paragraphs are merged with neighbors to avoid tiny chunks

    This keeps chunks substantial enough for the voice model to stay
    consistent, while still respecting natural breaks.
    """
    text = prefilter_text(text)

    # Split on double-newlines (hard paragraph boundaries)
    # Single newlines are folded into spaces within each paragraph
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = []
    for rp in raw_paragraphs:
        # Fold single newlines to spaces within paragraph
        folded = re.sub(r'\n', ' ', rp).strip()
        if folded:
            paragraphs.append(folded)

    # Extract all sentences across all paragraphs, with paragraph break markers
    sentences = []  # list of (text, is_para_start)
    for pi, para in enumerate(paragraphs):
        sents = re.split(r'(?<=[.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        for si, s in enumerate(sents):
            sentences.append((s, si == 0 and pi > 0))  # para_start for first sent of each para (except first)

    chunks = []
    current = ''

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ''

    for s, is_para_start in sentences:
        # At paragraph boundary: flush only if current is substantial enough
        if is_para_start and current and len(current) >= min_chunk:
            flush()

        # If this sentence alone exceeds max_chars, split at commas
        if len(s) > max_chars:
            flush()
            parts = re.split(r'(?<=,)\s+', s)
            sub = ''
            for p in parts:
                if len(sub) + len(p) + 1 > max_chars and sub:
                    chunks.append(sub.strip())
                    sub = p
                else:
                    sub = sub + ' ' + p if sub else p
            if sub.strip():
                current = sub
            continue

        # Would adding this sentence exceed max_chars?
        if current and len(current) + len(s) + 1 > max_chars:
            flush()

        # If current chunk is below minimum, keep merging
        if current and len(current) < min_chunk:
            current = current + ' ' + s
            continue

        # If adding still fits within max_chars, merge
        if current and len(current) + len(s) + 1 <= max_chars:
            current = current + ' ' + s
            continue

        # Current chunk is substantial enough — flush and start new
        if current:
            flush()
        current = s

    flush()

    # Merge any trailing runt chunk with previous
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk:
        if len(chunks[-2]) + len(chunks[-1]) + 1 <= max_chars:
            chunks[-2] = chunks[-2] + ' ' + chunks[-1]
            chunks.pop()

    # Ensure every chunk ends with terminal punctuation — CSM-1B needs it
    # to know the utterance is complete (otherwise it trails off / adds pauses)
    for i in range(len(chunks)):
        if chunks[i] and not chunks[i][-1] in '.!?':
            chunks[i] += '.'

    return chunks


def audio_to_wav_bytes(audio_np, sample_rate):
    """Convert numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def wav_bytes_to_audio(wav_bytes):
    """Convert WAV bytes to numpy float32 audio array."""
    buf = io.BytesIO(wav_bytes)
    audio_np, sr = sf.read(buf, dtype='float32')
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # mono
    return audio_np


def noise_gate(audio_np, sample_rate, threshold_db=-40, attack_ms=5, release_ms=50):
    """Noise gate to suppress low-level noise/static between speech.
    Uses vectorized RMS envelope for speed."""
    threshold = 10 ** (threshold_db / 20)
    attack_samples = max(1, int(sample_rate * attack_ms / 1000))
    release_samples = max(1, int(sample_rate * release_ms / 1000))

    # Compute RMS envelope in 10ms windows (vectorized)
    win_size = int(sample_rate * 0.01)
    n_windows = len(audio_np) // win_size
    if n_windows == 0:
        return audio_np
    trimmed_len = n_windows * win_size
    reshaped = audio_np[:trimmed_len].reshape(n_windows, win_size)
    rms_per_window = np.sqrt(np.mean(reshaped ** 2, axis=1))

    # Expand RMS back to sample-level envelope
    envelope = np.repeat(rms_per_window, win_size)
    if len(envelope) < len(audio_np):
        envelope = np.concatenate([envelope, np.full(len(audio_np) - len(envelope), rms_per_window[-1])])

    # Create gate mask and smooth with attack/release
    gate = (envelope > threshold).astype(np.float64)
    smoothed = np.zeros_like(gate)
    current = 0.0
    attack_rate = 1.0 / attack_samples
    release_rate = 1.0 / release_samples
    # Process per-window for speed (not per-sample)
    for w in range(n_windows):
        g = float(gate[w * win_size])
        if g > current:
            current = min(1.0, current + attack_rate * win_size)
        else:
            current = max(0.0, current - release_rate * win_size)
        smoothed[w * win_size:(w + 1) * win_size] = current
    if len(smoothed) > trimmed_len:
        smoothed[trimmed_len:] = current

    return audio_np * smoothed


def highpass_simple(audio_np, sample_rate, cutoff=80):
    """Simple single-pole highpass filter to remove rumble."""
    rc = 1.0 / (2 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    # Vectorized via diff
    out = np.empty_like(audio_np)
    out[0] = audio_np[0]
    for i in range(1, len(audio_np)):
        out[i] = alpha * (out[i-1] + audio_np[i] - audio_np[i-1])
    return out


def fade_samples(audio_np, n_samples, fade_in=True):
    """Apply a linear fade to the start or end of audio."""
    if n_samples <= 0 or len(audio_np) == 0:
        return audio_np
    n_samples = min(n_samples, len(audio_np))
    ramp = np.linspace(0.0, 1.0, n_samples)
    audio_np = audio_np.copy()
    if fade_in:
        audio_np[:n_samples] *= ramp
    else:
        audio_np[-n_samples:] *= ramp[::-1]
    return audio_np


def trim_audio(audio_np, sample_rate, silence_db=-40):
    """Trim leading/trailing silence + artifact zone, apply noise gate, normalize.

    Returns (audio_np, cuts) where cuts is a list of
    {'start_ms': float, 'end_ms': float, 'reason': str} regions removed
    (in terms of the original raw audio timeline).

    Args:
      silence_db: silence threshold in dB (default -40). Lower = keep quieter audio.

    Pipeline:
      1. Smart artifact skip — scan entire audio for tinny/metallic prefix
      2. Trim leading silence using absolute threshold
      3. Trim trailing silence
      4. Fade in 15ms / fade out 5ms (kills breathing/click at start)
      5. Highpass 80Hz (rumble)
      6. Noise gate (silence_db + 5dB headroom)
      7. Normalize to -1dB peak
    """
    ABS_THRESH = 10 ** (silence_db / 20)  # convert dB to linear amplitude
    original_len = len(audio_np)
    cuts = []
    offset = 0  # tracks cumulative sample offset into original audio

    # Note: the original CSM-1B build did a spectral "artifact skip" here to
    # cut its tinny/metallic prefix. F5-TTS doesn't have that artifact, and
    # the scan was eating real consonants on clean output, so it's omitted.

    if len(audio_np) == 0:
        return audio_np, cuts

    # 2. Trim leading silence
    abs_audio = np.abs(audio_np)
    above = np.where(abs_audio > ABS_THRESH)[0]
    if len(above) == 0:
        cuts.append({'start_ms': round(offset / sample_rate * 1000, 1),
                     'end_ms': round((offset + len(audio_np)) / sample_rate * 1000, 1),
                     'reason': 'silence'})
        return audio_np[:1], cuts

    lead_in = int(sample_rate * 0.01)
    start_idx = max(0, above[0] - lead_in)

    if start_idx > 0:
        cuts.append({'start_ms': round(offset / sample_rate * 1000, 1),
                     'end_ms': round((offset + start_idx) / sample_rate * 1000, 1),
                     'reason': 'leading silence'})

    # 3. Trim trailing silence — keep 30ms after last speech
    trail_out = int(sample_rate * 0.03)
    end_idx = min(len(audio_np), above[-1] + trail_out)

    if end_idx < len(audio_np):
        cuts.append({'start_ms': round((offset + end_idx) / sample_rate * 1000, 1),
                     'end_ms': round((offset + len(audio_np)) / sample_rate * 1000, 1),
                     'reason': 'trailing silence'})

    offset += start_idx
    audio_np = audio_np[start_idx:end_idx]

    if len(audio_np) == 0:
        return audio_np, cuts

    # 4. Fade in 15ms (kills breathing/click), fade out 5ms
    audio_np = fade_samples(audio_np, int(sample_rate * 0.015), fade_in=True)
    audio_np = fade_samples(audio_np, int(sample_rate * 0.005), fade_in=False)

    # 5. Highpass to remove rumble
    audio_np = highpass_simple(audio_np, sample_rate, cutoff=80)

    # (No noise gate for F5-TTS — output is clean; gating chops quiet
    # consonants and is perceived as flutter/echo. Was a CSM-1B defense.)

    # 7. Normalize to -1dB peak
    peak = np.max(np.abs(audio_np))
    if peak > 0.01:
        target = 10 ** (-1.0 / 20)  # -1dB
        audio_np = audio_np * (target / peak)

    return audio_np, cuts


def compress_pauses(audio_np, sample_rate, max_pause_ms=500, raw_offset_ms=0, silence_db=-40):
    """Shorten internal silence gaps to max_pause_ms.

    Returns (audio_np, cuts) where cuts is a list of
    {'start_ms': float, 'end_ms': float, 'reason': str} regions removed.
    start_ms/end_ms are relative to the raw audio (raw_offset_ms added).

    Args:
      silence_db: silence threshold in dB (default -40). Used with +4dB headroom for pause detection.
    """
    THRESH = 10 ** ((silence_db + 4) / 20)  # silence_db + 4dB headroom for pause detection
    win_samples = int(sample_rate * 0.01)  # 10ms analysis window
    max_silence = int(sample_rate * max_pause_ms / 1000)

    # Find per-window RMS
    n_windows = len(audio_np) // win_samples
    if n_windows < 3:
        return audio_np, []

    # Detect silent regions
    silent = np.zeros(len(audio_np), dtype=bool)
    for w in range(n_windows):
        s = w * win_samples
        e = s + win_samples
        rms = np.sqrt(np.mean(audio_np[s:e] ** 2))
        if rms < THRESH:
            silent[s:e] = True

    # Find contiguous silent runs
    segments = []
    in_silence = False
    sil_start = 0
    for i in range(len(silent)):
        if silent[i] and not in_silence:
            in_silence = True
            sil_start = i
        elif not silent[i] and in_silence:
            in_silence = False
            sil_len = i - sil_start
            if sil_len > max_silence:
                segments.append((sil_start, i, sil_len))
    # Handle trailing silence run
    if in_silence:
        sil_len = len(silent) - sil_start
        if sil_len > max_silence:
            segments.append((sil_start, len(silent), sil_len))

    if not segments:
        return audio_np, []

    # Build output by copying audio and shortening long pauses
    pieces = []
    cuts = []
    prev_end = 0
    total_removed = 0
    fade_n = int(sample_rate * 0.003)  # 3ms crossfade at splice points

    for sil_start, sil_end, sil_len in segments:
        # Keep audio before this silence gap
        pieces.append(audio_np[prev_end:sil_start])
        # Keep only max_silence worth of the gap (centered)
        keep = max_silence
        margin = (sil_len - keep) // 2
        kept_start = sil_start + margin
        kept_end = kept_start + keep
        kept = audio_np[kept_start:kept_end].copy()
        # Crossfade at splice points
        if fade_n > 0 and len(kept) > fade_n * 2:
            kept[:fade_n] *= np.linspace(0, 1, fade_n)
            kept[-fade_n:] *= np.linspace(1, 0, fade_n)
        pieces.append(kept)

        # Record the portions that were removed (before and after the kept center)
        removed_before = margin
        removed_after = sil_len - keep - margin
        if removed_before > 0:
            cuts.append({
                'start_ms': round((sil_start / sample_rate * 1000) + raw_offset_ms, 1),
                'end_ms': round((kept_start / sample_rate * 1000) + raw_offset_ms, 1),
                'reason': 'pause',
            })
        if removed_after > 0:
            cuts.append({
                'start_ms': round((kept_end / sample_rate * 1000) + raw_offset_ms, 1),
                'end_ms': round((sil_end / sample_rate * 1000) + raw_offset_ms, 1),
                'reason': 'pause',
            })

        total_removed += sil_len - keep
        prev_end = sil_end

    pieces.append(audio_np[prev_end:])
    result = np.concatenate(pieces)

    if total_removed > 0:
        removed_ms = total_removed / sample_rate * 1000
        print(f"    Compressed pauses: removed {removed_ms:.0f}ms of silence")

    return result, cuts


def audio_is_garbage(audio_np, sample_rate):
    """Detect degenerate CSM-1B output (tin-can, static, silence).

    Returns (is_bad, reason) tuple.

    Checks:
      1. Too short (< 0.3s of actual content)
      2. Too quiet (all silence / near-zero)
      3. Excessive high-frequency energy (tin-can / metallic artifact)
      4. Clipping / constant max amplitude
      5. No variation (flat/stuck decoder)
    """
    if len(audio_np) < int(sample_rate * 0.3):
        return True, "too short"

    peak = np.max(np.abs(audio_np))
    if peak < 0.005:
        return True, "silence"

    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 0.002:
        return True, "near-silence"

    # Check for excessive clipping (>5% of samples at max)
    clip_count = np.sum(np.abs(audio_np) > 0.99)
    if clip_count > len(audio_np) * 0.05:
        return True, f"clipping ({clip_count} samples)"

    # Check spectral balance — tin-can output has abnormally high HF energy
    # Compare energy above 3kHz vs below 1kHz
    fft = np.fft.rfft(audio_np)
    freqs = np.fft.rfftfreq(len(audio_np), 1 / sample_rate)
    mag = np.abs(fft)
    lo_mask = freqs < 1000
    hi_mask = freqs > 3000
    if lo_mask.any() and hi_mask.any():
        lo_energy = np.sqrt(np.mean(mag[lo_mask] ** 2))
        hi_energy = np.sqrt(np.mean(mag[hi_mask] ** 2))
        if lo_energy > 0 and hi_energy / lo_energy > 2.0:
            return True, f"metallic (HF/LF ratio {hi_energy/lo_energy:.1f})"

    # Check for flat/stuck output — very low variance
    if np.std(audio_np) < 0.003:
        return True, "flat/stuck"

    return False, "ok"


MAX_RETRIES = 2  # retry up to 2 times on garbage output

# Track generation times to detect runaway chunks
_gen_times = []  # recent gen times in seconds


def _estimate_timeout(text_len):
    """Estimate max allowed generation time for a chunk.

    Uses rolling average of recent chunks × 3, with a hard floor/ceiling.
    First chunk gets a generous default since we have no history.
    """
    MIN_TIMEOUT = 45   # never timeout before 45s (model needs warmup)
    MAX_TIMEOUT = 180  # never wait more than 3 min
    if _gen_times:
        avg = sum(_gen_times) / len(_gen_times)
        timeout = max(MIN_TIMEOUT, avg * 3)
    else:
        # First chunk — estimate from text length (~0.3s gen per char is generous)
        timeout = max(MIN_TIMEOUT, text_len * 0.3)
    return min(timeout, MAX_TIMEOUT)


def generate_chunk(text, chunk_idx, total, temperature=0.5, top_p=0.95, top_k=50, trim_pauses=True, silence_db=-40, cancel_event=None, priority=False):
    """Generate a single audio chunk with quality check, timeout, and auto-retry.
    priority=True → job jumps the queue (used for regeneration).
    Returns (wav_bytes, duration, raw_wav_bytes, raw_duration, cuts) or None."""

    timeout = _estimate_timeout(len(text))

    for attempt in range(1 + MAX_RETRIES):
        # Check cancellation before each attempt
        if cancel_event and cancel_event.is_set():
            return None

        try:
            # Nudge temperature slightly on retries to escape bad states
            temp = temperature + (attempt * 0.02)

            # Submit to the single-worker generation queue
            gen_start = time.time()
            job, result_event = _submit_generate(text, temp, top_p, top_k, priority=priority)

            # Wait for the worker to finish, checking cancel periodically
            should_retry = False
            while not result_event.wait(timeout=0.5):
                if cancel_event and cancel_event.is_set():
                    print(f"  Chunk {chunk_idx}/{total} cancel requested — waiting for worker...")
                    result_event.wait()  # must wait — only one worker thread
                    print(f"  Chunk {chunk_idx}/{total} cancelled (worker done after {time.time()-gen_start:.1f}s)")
                    return None
                if time.time() - gen_start > timeout:
                    print(f"  Chunk {chunk_idx}/{total} TIMEOUT — waiting for worker...")
                    result_event.wait()
                    if attempt < MAX_RETRIES:
                        print(f"  Chunk {chunk_idx}/{total} timed out after {time.time()-gen_start:.0f}s, retrying ({attempt+1})...")
                        should_retry = True
                    else:
                        print(f"  Chunk {chunk_idx}/{total} timed out after all retries")
                        return None
                    break

            if should_retry:
                continue

            gen_elapsed = time.time() - gen_start

            # Check cancel again after worker finished
            if cancel_event and cancel_event.is_set():
                return None

            if '_error' in job:
                raise job['_error']

            # Worker already converted MLX arrays → numpy (thread-safe)
            all_audio = job.get('_audio_arrays', [])

            if not all_audio:
                if attempt < MAX_RETRIES:
                    print(f"  Chunk {chunk_idx}/{total} empty, retrying ({attempt+1})...")
                    continue
                return None

            combined = np.concatenate(all_audio)

            # Check cancellation before post-processing
            if cancel_event and cancel_event.is_set():
                return None

            # Quality check before trimming
            is_bad, reason = audio_is_garbage(combined, SAMPLE_RATE)
            if is_bad:
                if attempt < MAX_RETRIES:
                    print(f"  Chunk {chunk_idx}/{total} garbage ({reason}), retrying ({attempt+1}, temp={temp:.2f})...")
                    continue
                else:
                    print(f"  Chunk {chunk_idx}/{total} garbage after {MAX_RETRIES} retries ({reason}), using anyway")

            # Encode raw audio for inspector before any processing
            raw_wav_bytes = audio_to_wav_bytes(combined, SAMPLE_RATE)
            raw_duration = len(combined) / SAMPLE_RATE

            trimmed, trim_cuts = trim_audio(combined, SAMPLE_RATE, silence_db=silence_db)

            # Compute offset where trim_audio kept audio starts (for compress_pauses mapping)
            # The first non-cut region starts after artifact skip + leading silence
            kept_start_ms = 0
            for c in trim_cuts:
                if c['reason'] in ('artifact', 'leading silence'):
                    kept_start_ms = c['end_ms']

            pause_cuts = []
            if trim_pauses:
                trimmed, pause_cuts = compress_pauses(
                    trimmed, SAMPLE_RATE, max_pause_ms=500,
                    raw_offset_ms=kept_start_ms, silence_db=silence_db)
            all_cuts = trim_cuts + pause_cuts

            duration = len(trimmed) / SAMPLE_RATE
            wav_bytes = audio_to_wav_bytes(trimmed, SAMPLE_RATE)

            # Track generation time for timeout estimation
            _gen_times.append(gen_elapsed)
            if len(_gen_times) > 10:
                _gen_times.pop(0)  # keep rolling window of 10

            if attempt > 0:
                print(f"  Chunk {chunk_idx}/{total} OK on retry {attempt}")
            return (wav_bytes, duration, raw_wav_bytes, raw_duration, all_cuts)

        except Exception as e:
            print(f"  Chunk {chunk_idx}/{total} failed: {e}")
            if attempt < MAX_RETRIES:
                continue
            return None

    return None


def _parse_multipart(body, content_type):
    """Parse a multipart/form-data body into a dict of {name: (filename, bytes)}.
    Text fields have filename=None and bytes are utf-8 encoded text.
    Uses email.parser so we don't depend on the deprecated cgi module.
    """
    from email.parser import BytesParser
    from email.policy import default
    header_blob = b'MIME-Version: 1.0\r\nContent-Type: ' + content_type.encode('latin-1') + b'\r\n\r\n'
    msg = BytesParser(policy=default).parsebytes(header_blob + body)
    parts = {}
    for part in msg.iter_parts():
        cd = part.get('content-disposition', '')
        if 'form-data' not in cd.lower():
            continue
        name = part.get_param('name', header='content-disposition')
        if not name:
            continue
        filename = part.get_param('filename', header='content-disposition')
        payload = part.get_payload(decode=True) or b''
        parts[name] = (filename, payload)
    return parts


def _decode_audio_to_24k_mono(raw_bytes, filename_hint=None):
    """Decode arbitrary audio bytes (WAV/FLAC/MP3/M4A/OGG…) to numpy float32 mono @ 24 kHz.
    Tries soundfile first; falls back to ffmpeg piping for compressed formats.
    Returns (audio_np, duration_s) or raises ValueError.
    """
    # First attempt: soundfile (handles WAV/FLAC/OGG and many others)
    try:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype='float32', always_2d=False)
    except Exception:
        # Fallback: ffmpeg → s16le wav on stdout
        try:
            proc = subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0',
                 '-ac', '1', '-ar', '24000', '-f', 'wav', 'pipe:1'],
                input=raw_bytes, capture_output=True, check=True, timeout=30,
            )
        except FileNotFoundError:
            raise ValueError("Could not decode audio: install ffmpeg for MP3/M4A support, "
                             "or upload WAV/FLAC.")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"ffmpeg failed to decode audio: {e.stderr.decode('utf-8', 'replace')[:200]}")
        audio, sr = sf.read(io.BytesIO(proc.stdout), dtype='float32', always_2d=False)

    # Mix down to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32, copy=False)

    # Resample to 24 kHz if needed (linear interp is fine for a reference clip)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / float(sr)
        new_len = int(round(len(audio) * ratio))
        if new_len <= 0:
            raise ValueError("Audio is too short after decoding.")
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
        audio = np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

    duration = len(audio) / SAMPLE_RATE
    return audio, duration


class TTSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html_path = os.path.join(SCRIPT_DIR, 'web_tts.html')
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())

        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'ready': MODEL is not None,
                'model': 'F5-TTS' if MODEL else None,
                'ref_text': REF_TEXT[:80],
                'ref_name': REF_NAME,
                'ref_text_full': REF_TEXT,
                'defaults': {
                    'temperature': 0.5,
                    'top_p': 0.95,
                    'top_k': 50,
                    'max_chars': MAX_CHARS,
                    'min_chunk': MIN_CHUNK,
                    'trim_pauses': True,
                },
                # F5-TTS does not expose temperature/top_p/top_k via its API,
                # so hide those rows in the frontend tuning panel.
                'param_ui': {
                    'temperature': {'visible': False},
                    'top_p':       {'visible': False},
                    'top_k':       {'visible': False},
                },
            }).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
            text = data.get('text', '')

            if not text.strip():
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No text provided'}).encode())
                return

            # Read tuning params from request (use defaults if not provided)
            temperature = float(data.get('temperature', 0.5))
            top_p = float(data.get('top_p', 0.95))
            top_k = int(data.get('top_k', 50))
            max_chars = int(data.get('max_chars', MAX_CHARS))
            min_chunk = int(data.get('min_chunk', MIN_CHUNK))
            trim_pauses = bool(data.get('trim_pauses', True))
            silence_db = float(data.get('silence_db', -40))

            # Create a cancel event for this request
            import uuid
            request_id = data.get('request_id', str(uuid.uuid4()))
            cancel_event = threading.Event()
            with _cancel_lock:
                # Cancel any previous generation first
                for rid, evt in list(_cancel_events.items()):
                    evt.set()
                _cancel_events.clear()
                _cancel_events[request_id] = cancel_event

            chunks = split_text(text, max_chars=max_chars, min_chunk=min_chunk)
            total = len(chunks)

            print(f"  Params: temp={temperature}, top_p={top_p}, top_k={top_k}, max_chars={max_chars}, min_chunk={min_chunk}, trim_pauses={trim_pauses}, silence_db={silence_db}")
            print(f"  Request ID: {request_id}")

            # Stream chunks as server-sent events
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Send initial info (include request_id so client can cancel)
            import base64
            self._send_sse('info', {
                'total_chunks': total,
                'sample_rate': SAMPLE_RATE,
                'request_id': request_id,
                'chunk_texts': chunks,
            })

            cancelled = False
            # Submit all chunks to the gen queue concurrently so the worker
            # can pull a batch of same-priority jobs in one F5 forward pass.
            # We still emit SSE events in chunk-index order — completed
            # chunks queue up in `buf` and drain as the contiguous prefix
            # fills in.
            import concurrent.futures
            parallel = min(MAX_BATCH, total) if total > 1 else 1

            def _process_chunk(idx, ctext):
                t0 = time.time()
                if cancel_event.is_set():
                    return idx, None, 0.0
                r = generate_chunk(
                    ctext, idx, total,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    trim_pauses=trim_pauses,
                    silence_db=silence_db,
                    cancel_event=cancel_event,
                    priority=False,
                )
                return idx, r, time.time() - t0

            buf = {}            # idx -> (result, gen_time)
            next_to_send = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_process_chunk, i, ct) for i, ct in enumerate(chunks)]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        idx, result, gen_time = fut.result()
                    except Exception as e:
                        print(f"  [chunk worker] ERROR: {e}")
                        continue
                    buf[idx] = (result, gen_time)

                    # Drain any contiguous completed chunks in order
                    while next_to_send in buf:
                        result, gen_time = buf.pop(next_to_send)
                        chunk_text = chunks[next_to_send]
                        i = next_to_send

                        if cancel_event.is_set():
                            print(f"  [{i+1}/{total}] ⊘ Cancelled (in-order drain)")
                            cancelled = True
                            break

                        if result:
                            wav_bytes, duration, raw_wav_bytes, raw_duration, cuts = result
                            b64_audio = base64.b64encode(wav_bytes).decode('ascii')
                            b64_raw = base64.b64encode(raw_wav_bytes).decode('ascii')
                            if not self._send_sse('chunk', {
                                'index': i,
                                'total': total,
                                'text': chunk_text,
                                'audio_b64': b64_audio,
                                'duration': round(duration, 2),
                                'gen_time': round(gen_time, 2),
                                'audio_raw_b64': b64_raw,
                                'duration_raw': round(raw_duration, 2),
                                'cuts': cuts,
                            }):
                                print(f"  [{i+1}/{total}] ⊘ Client disconnected")
                                cancelled = True
                                break
                            print(f"  [{i+1}/{total}] ✓ {duration:.1f}s audio in {gen_time:.1f}s | {chunk_text[:50]}...")
                        else:
                            if not self._send_sse('error', {
                                'index': i,
                                'total': total,
                                'text': chunk_text,
                                'message': 'Generation failed',
                            }):
                                cancelled = True
                                break
                            print(f"  [{i+1}/{total}] ✗ Failed")

                        next_to_send += 1

                    if cancelled:
                        # Signal remaining in-flight chunks to bail
                        cancel_event.set()
                        for f in futures:
                            f.cancel()
                        break


            if cancelled:
                self._send_sse('cancelled', {'total': total})
                print(f"  Generation cancelled")
            else:
                self._send_sse('done', {'total': total})
                print(f"  Generation complete: {total} chunks")

            # Cleanup cancel event
            with _cancel_lock:
                _cancel_events.pop(request_id, None)

        elif self.path == '/regenerate':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
            chunk_text = data.get('text', '')
            chunk_index = int(data.get('index', 0))
            total = int(data.get('total', 1))

            if not chunk_text.strip():
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No text provided'}).encode())
                return

            temperature = float(data.get('temperature', 0.5))
            top_p = float(data.get('top_p', 0.95))
            top_k = int(data.get('top_k', 50))
            trim_pauses = bool(data.get('trim_pauses', True))
            silence_db = float(data.get('silence_db', -40))

            print(f"  Regenerating chunk {chunk_index} (PRIORITY): temp={temperature}, top_p={top_p}, top_k={top_k}, silence_db={silence_db}")

            import base64
            start = time.time()
            result = generate_chunk(chunk_text, chunk_index, total,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    trim_pauses=trim_pauses,
                                    silence_db=silence_db,
                                    priority=True)
            gen_time = time.time() - start

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            if result:
                wav_bytes, duration, raw_wav_bytes, raw_duration, cuts = result
                b64_audio = base64.b64encode(wav_bytes).decode('ascii')
                b64_raw = base64.b64encode(raw_wav_bytes).decode('ascii')
                self.wfile.write(json.dumps({
                    'index': chunk_index,
                    'total': total,
                    'text': chunk_text,
                    'audio_b64': b64_audio,
                    'duration': round(duration, 2),
                    'gen_time': round(gen_time, 2),
                    'audio_raw_b64': b64_raw,
                    'duration_raw': round(raw_duration, 2),
                    'cuts': cuts,
                }).encode())
                print(f"  Regen [{chunk_index+1}/{total}] ✓ {duration:.1f}s audio in {gen_time:.1f}s")
            else:
                self.wfile.write(json.dumps({
                    'error': 'Regeneration failed',
                    'index': chunk_index,
                }).encode())
                print(f"  Regen [{chunk_index+1}/{total}] ✗ Failed")

        elif self.path == '/apply-cuts':
            # Apply user-modified cuts to raw audio and return processed WAV
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            audio_raw_b64 = data.get('audio_raw_b64', '')
            cuts = data.get('cuts', [])
            silence_db = float(data.get('silence_db', -40))

            if not audio_raw_b64:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No audio provided'}).encode())
                return

            import base64

            # Decode raw audio
            raw_bytes = base64.b64decode(audio_raw_b64)
            raw_np = wav_bytes_to_audio(raw_bytes)
            raw_duration = len(raw_np) / SAMPLE_RATE

            # Apply cuts: remove cut regions from raw audio
            if cuts:
                sorted_cuts = sorted(cuts, key=lambda c: c['start_ms'])
                pieces = []
                prev_end_sample = 0
                fade_n = int(SAMPLE_RATE * 0.003)  # 3ms crossfade

                for c in sorted_cuts:
                    cut_start = int(c['start_ms'] / 1000 * SAMPLE_RATE)
                    cut_end = int(c['end_ms'] / 1000 * SAMPLE_RATE)
                    cut_start = max(0, min(cut_start, len(raw_np)))
                    cut_end = max(cut_start, min(cut_end, len(raw_np)))

                    if cut_start > prev_end_sample:
                        piece = raw_np[prev_end_sample:cut_start].copy()
                        # Apply crossfade at edges
                        if len(piece) > fade_n and pieces:
                            piece[:fade_n] *= np.linspace(0, 1, fade_n)
                        if len(piece) > fade_n:
                            piece[-fade_n:] *= np.linspace(1, 0, fade_n)
                        pieces.append(piece)
                    prev_end_sample = cut_end

                # Remaining audio after last cut
                if prev_end_sample < len(raw_np):
                    piece = raw_np[prev_end_sample:].copy()
                    if len(piece) > fade_n and pieces:
                        piece[:fade_n] *= np.linspace(0, 1, fade_n)
                    pieces.append(piece)

                if pieces:
                    trimmed = np.concatenate(pieces)
                else:
                    trimmed = np.array([], dtype=np.float32)
            else:
                trimmed = raw_np.copy()

            # Apply DSP pipeline: highpass, noise gate, normalize
            if len(trimmed) > 0:
                trimmed = highpass_simple(trimmed, SAMPLE_RATE, cutoff=80)
                gate_db = silence_db + 5
                trimmed = noise_gate(trimmed, SAMPLE_RATE, threshold_db=gate_db)
                # Normalize to -1dB peak
                peak = np.max(np.abs(trimmed))
                if peak > 0.01:
                    target = 10 ** (-1.0 / 20)
                    trimmed = trimmed * (target / peak)

            duration = len(trimmed) / SAMPLE_RATE
            wav_bytes = audio_to_wav_bytes(trimmed, SAMPLE_RATE)
            b64_audio = base64.b64encode(wav_bytes).decode('ascii')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'audio_b64': b64_audio,
                'duration': round(duration, 2),
            }).encode())
            print(f"  Apply-cuts: {len(cuts)} cuts, {raw_duration:.1f}s raw -> {duration:.1f}s trimmed")

        elif self.path == '/split':
            # Lightweight text splitting — no GPU, no generation
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
            text = data.get('text', '')

            if not text.strip():
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No text provided'}).encode())
                return

            max_chars = int(data.get('max_chars', MAX_CHARS))
            min_chunk = int(data.get('min_chunk', MIN_CHUNK))
            chunks = split_text(text, max_chars=max_chars, min_chunk=min_chunk)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'chunks': chunks,
                'total': len(chunks),
            }).encode())

        elif self.path == '/reference':
            global REF_AUDIO_DATA, REF_TEXT, REF_NAME
            content_type = self.headers.get('Content-Type', '')
            content_len = int(self.headers.get('Content-Length', 0))

            def _err(code, msg):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': msg}).encode())

            if content_len <= 0 or content_len > MAX_REF_UPLOAD_BYTES:
                _err(400, f"Upload missing or too large (>{MAX_REF_UPLOAD_BYTES // (1024*1024)} MB).")
                return
            if 'multipart/form-data' not in content_type.lower():
                _err(400, "Expected multipart/form-data with 'audio' and 'transcript' fields.")
                return

            # Reject if a generation is currently in flight (queue not drained)
            in_flight = _gen_queue.unfinished_tasks
            if in_flight > 0:
                _err(409, f"Cannot swap voice while {in_flight} generation job(s) are in flight. Stop or wait, then retry.")
                return

            body = self.rfile.read(content_len)
            try:
                parts = _parse_multipart(body, content_type)
            except Exception as e:
                _err(400, f"Could not parse multipart body: {e}")
                return

            if 'audio' not in parts or not parts['audio'][1]:
                _err(400, "Missing 'audio' file field.")
                return
            if 'transcript' not in parts:
                _err(400, "Missing 'transcript' text field.")
                return

            audio_filename, audio_bytes = parts['audio']
            transcript = parts['transcript'][1].decode('utf-8', errors='replace').strip()
            if not transcript:
                _err(400, "Transcript is required.")
                return

            try:
                audio_np, duration = _decode_audio_to_24k_mono(audio_bytes, audio_filename)
            except ValueError as e:
                _err(400, str(e))
                return
            except Exception as e:
                _err(400, f"Audio decode failed: {e}")
                return

            if duration > MAX_REF_DURATION_S:
                _err(400, f"Reference clip is {duration:.1f}s; max allowed is {MAX_REF_DURATION_S:.0f}s.")
                return

            # Run the same cleanup pipeline used per-chunk: leading/trailing
            # silence trim, fades, 80 Hz highpass, -1 dB normalize. Keeps
            # uploaded references on the same audio footing as the bundled
            # LibriVox sample without shelling out to sox.
            try:
                cleaned, _cuts = trim_audio(audio_np, SAMPLE_RATE, silence_db=-40)
            except Exception as e:
                _err(400, f"Audio cleanup failed: {e}")
                return
            cleaned_duration = len(cleaned) / SAMPLE_RATE
            if cleaned_duration < 0.5:
                _err(400, f"Reference clip is too short after silence trim ({cleaned_duration:.1f}s); upload at least 0.5s of speech.")
                return
            print(f"  Reference cleanup: {duration:.2f}s raw → {cleaned_duration:.2f}s trimmed")
            audio_np = cleaned
            duration = cleaned_duration

            out_path = os.path.join(SCRIPT_DIR, "reference_uploaded.wav")
            try:
                sf.write(out_path, audio_np, SAMPLE_RATE, subtype='PCM_16')
            except Exception as e:
                _err(500, f"Could not save uploaded reference: {e}")
                return

            # Atomic-ish swap of the active reference. Worker only reads
            # REF_AUDIO_DATA/REF_TEXT at the start of each .infer() call,
            # so taking the lock here is enough.
            with _ref_lock:
                REF_AUDIO_DATA = out_path
                REF_TEXT = transcript
                REF_NAME = os.path.basename(out_path)
                if MODEL is not None:
                    MODEL.ref_path = out_path
                    MODEL.ref_text = transcript

            print(f"  Reference voice swapped: {REF_NAME} ({duration:.1f}s, transcript {len(transcript)} chars)")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'ok': True,
                'ref_name': REF_NAME,
                'ref_text': REF_TEXT[:80],
                'ref_text_full': REF_TEXT,
                'duration': round(duration, 2),
            }).encode())

        elif self.path == '/cancel':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len) if content_len > 0 else b'{}'
            data = json.loads(body) if body else {}
            request_id = data.get('request_id', None)

            cancelled = False
            with _cancel_lock:
                if request_id and request_id in _cancel_events:
                    _cancel_events[request_id].set()
                    cancelled = True
                    print(f"  Cancel requested for {request_id}")
                else:
                    # Cancel all active generations
                    for rid, evt in _cancel_events.items():
                        evt.set()
                        cancelled = True
                    if cancelled:
                        print(f"  Cancel requested for all active generations")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'cancelled': cancelled}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight for POST endpoints."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_sse(self, event, data):
        """Send a server-sent event. Returns True on success, False if client disconnected."""
        try:
            msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(msg.encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def log_message(self, format, *args):
        # Quiet down request logging
        pass


def main():
    global MODEL, REF_AUDIO_DATA, REF_TEXT, REF_NAME, SAMPLE_RATE

    parser = argparse.ArgumentParser(description="Streaming TTS Web Server")
    parser.add_argument("--port", type=int, default=8765)
    # `--model` is accepted for CLI compatibility but F5-TTS's high-level API
    # always loads the default F5-TTS_v1 checkpoint; we surface the value in
    # logs and ignore it otherwise.
    parser.add_argument("--model", default="F5-TTS")
    parser.add_argument("--ref-audio", default=os.path.join(SCRIPT_DIR, "reference_clean.wav"))
    parser.add_argument("--ref-text-file", default=os.path.join(SCRIPT_DIR, "transcript.txt"))
    parser.add_argument("--device", default="auto",
                        help="auto|cuda|mps|cpu (auto picks cuda > mps > cpu)")
    args = parser.parse_args()

    # Load reference text
    if os.path.exists(args.ref_text_file):
        with open(args.ref_text_file) as f:
            REF_TEXT = f.read().strip()
        print(f"Reference text: {REF_TEXT[:60]}...")
    else:
        print(f"Warning: no transcript file at {args.ref_text_file}")

    # Load model (F5-TTS via the f5_tts.api high-level wrapper, which
    # internally instantiates the default F5-TTS_v1 checkpoint on CUDA).
    print(f"Loading model: {args.model} (F5-TTS_v1)...")
    try:
        import torch
        from f5_tts.api import F5TTS
    except ImportError as e:
        print(f"ERROR: missing dep — {e}", file=sys.stderr)
        print("Run via ./serve-dgx.sh which installs f5-tts/torch via uv.", file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    else:
        DEVICE = args.device

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but no CUDA device available.", file=sys.stderr)
        sys.exit(1)
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        print("ERROR: --device mps requested but MPS not available on this system.", file=sys.stderr)
        sys.exit(1)
    print(f"Device: {DEVICE}")

    try:
        f5_model = F5TTS(device=DEVICE)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "access" in msg.lower():
            print(f"ERROR: F5-TTS checkpoint appears gated. Accept the license at "
                  f"https://huggingface.co/SWivid/F5-TTS and ensure HF_TOKEN is set "
                  f"(check ~/.cache/huggingface/token).", file=sys.stderr)
        raise

    SAMPLE_RATE = 24000

    # F5-TTS expects the *path* to the reference audio (it does its own
    # loading/resampling). We just verify it exists.
    ref_audio_path = None
    if os.path.exists(args.ref_audio):
        ref_audio_path = os.path.abspath(args.ref_audio)
        print(f"Reference audio: {ref_audio_path}")
    else:
        print(f"Warning: no reference audio at {args.ref_audio}")

    # Wrapper exposing the same MODEL.generate(...) API the rest of the
    # server (specifically `_gen_worker`) expects. Returns a list with a
    # single object whose `.audio` is a numpy float32 mono @ 24 kHz array.
    class _GenResult:
        __slots__ = ('audio',)
        def __init__(self, audio):
            self.audio = audio

    class _F5TtsWrapper:
        sample_rate = SAMPLE_RATE
        model_type = "f5-tts"
        def __init__(self, tts, ref_path, ref_text, device):
            self.tts = tts
            self.ref_path = ref_path
            self.ref_text = ref_text
            self.device = device
            # Cache key for the prepared reference audio tensor — invalidated
            # when ref_path changes (e.g. via /reference upload).
            self._ref_cache_key = None
            self._ref_audio_tensor = None
            self._ref_audio_len = None

        def _ensure_ref_tensor(self, ref_file):
            """Lazy-load+cache the reference audio tensor for batched calls.
            F5's internal `infer_process` reloads ref audio every call; for
            batching we want it once, then re-use across all gen_texts."""
            import torch
            import torchaudio
            from f5_tts.infer.utils_infer import target_sample_rate, hop_length
            if self._ref_cache_key == ref_file and self._ref_audio_tensor is not None:
                return self._ref_audio_tensor, self._ref_audio_len
            audio, sr = torchaudio.load(ref_file)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            # Match F5's RMS normalisation logic so batched output matches single
            target_rms = 0.1
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                audio = resampler(audio)
            audio = audio.to(self.device)
            self._ref_audio_tensor = audio
            self._ref_audio_len = audio.shape[-1] // hop_length
            self._ref_cache_key = ref_file
            return audio, self._ref_audio_len

        def generate(self, text, ref_audio=None, ref_text="", temperature=0.9,
                     top_p=0.95, top_k=50, verbose=False):
            # F5-TTS does not use temperature/top_p/top_k from the API
            # request; accept them silently for caller compatibility.
            # `ref_audio` arg is the path string (server passes through
            # REF_AUDIO_DATA which we set to the path). Fall back to the
            # one captured at startup if missing.
            ref_file = ref_audio if isinstance(ref_audio, str) and ref_audio else self.ref_path
            r_text = ref_text or self.ref_text or ""
            if not ref_file:
                raise RuntimeError("F5-TTS requires a reference audio file path; none configured.")

            wav, sr, _spec = self.tts.infer(
                ref_file=ref_file,
                ref_text=r_text,
                gen_text=text,
                nfe_step=32,
                cfg_strength=2.0,
                speed=1.0,
            )
            # F5 returns a 24 kHz waveform (numpy or torch tensor depending
            # on version). Normalize to numpy float32 mono.
            if hasattr(wav, 'detach'):
                wav = wav.detach().to('cpu', dtype=torch.float32).numpy()
            audio = np.asarray(wav, dtype=np.float32).flatten()
            return [_GenResult(audio)]

        def generate_batch(self, texts, ref_audio=None, ref_text="",
                           nfe_step=32, cfg_strength=2.0, speed=1.0):
            """Generate audio for *N* gen_texts in parallel using F5's
            underlying flow-matching model + a ThreadPoolExecutor. Each
            text gets its own model.sample() call but they overlap on the
            GPU, amortising prefill / kernel launch overhead.

            Returns a list of [_GenResult(...)] in the same order as `texts`.
            """
            import torch
            from concurrent.futures import ThreadPoolExecutor
            from f5_tts.infer.utils_infer import (
                target_sample_rate, hop_length, convert_char_to_pinyin,
            )

            ref_file = ref_audio if isinstance(ref_audio, str) and ref_audio else self.ref_path
            r_text = ref_text or self.ref_text or ""
            if not ref_file:
                raise RuntimeError("F5-TTS requires a reference audio file path; none configured.")

            audio, ref_audio_len = self._ensure_ref_tensor(ref_file)
            target_rms = 0.1
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            ref_text_safe = r_text + (" " if r_text and len(r_text[-1].encode("utf-8")) == 1 else "")

            def _gen_one(gen_text):
                local_speed = speed
                if len(gen_text.encode("utf-8")) < 10:
                    local_speed = 0.3
                text_list = [ref_text_safe + gen_text]
                final_text_list = convert_char_to_pinyin(text_list)
                ref_text_len = max(len(ref_text_safe.encode("utf-8")), 1)
                gen_text_len = len(gen_text.encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
                with torch.inference_mode():
                    generated, _ = self.tts.ema_model.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=-1,
                    )
                    generated = generated.to(torch.float32)
                    generated = generated[:, ref_audio_len:, :]
                    generated = generated.permute(0, 2, 1)
                    if self.tts.mel_spec_type == "vocos":
                        wave = self.tts.vocoder.decode(generated)
                    elif self.tts.mel_spec_type == "bigvgan":
                        wave = self.tts.vocoder(generated)
                    else:
                        wave = self.tts.vocoder.decode(generated)
                    if rms < target_rms:
                        wave = wave * rms / target_rms
                    return wave.squeeze().cpu().numpy().astype(np.float32)

            with ThreadPoolExecutor(max_workers=max(1, len(texts))) as ex:
                waves = list(ex.map(_gen_one, texts))
            return [_GenResult(np.asarray(w, dtype=np.float32).flatten()) for w in waves]

    MODEL = _F5TtsWrapper(f5_model, ref_audio_path, REF_TEXT, DEVICE)
    # The worker passes REF_AUDIO_DATA through to MODEL.generate as
    # `ref_audio`; for F5 that's the path string.
    REF_AUDIO_DATA = ref_audio_path
    REF_NAME = os.path.basename(ref_audio_path) if ref_audio_path else ""
    print(f"Loaded F5-TTS on {DEVICE}")

    # Start the single generation worker thread (only thread that touches MODEL)
    worker = threading.Thread(target=_gen_worker, daemon=True)
    worker.start()
    print(f"Generation worker started (single-threaded GPU access)")

    # Start server
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True   # threads die when main thread exits

    server = ThreadedHTTPServer(('0.0.0.0', args.port), TTSHandler)
    print(f"\n=== TTS Server ready ===")
    print(f"http://localhost:{args.port}")
    print(f"Model: {args.model}")
    print(f"Ctrl-C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
