#!/bin/bash
# Start the Voice Clone TTS web server (DGX / NVIDIA CUDA, F5-TTS build).
#
# Usage: ./serve-dgx.sh [options]
#
# Options:
#   --port N        Port to listen on (default: 8765)
#   --model NAME    HF model id (default: F5-TTS — informational only;
#                   F5TTS() always loads the default F5-TTS_v1 checkpoint)
#   --ref FILE      Reference audio file (default: reference_clean.wav)
#   --transcript F  Transcript file (default: transcript.txt)
#
# Installs f5-tts (which brings its own transformers/vocos/etc.),
# torch and torchaudio (CUDA wheels) via uv.

set -e
cd "$(dirname "$0")"

# uv lives at ~/.local/bin; non-interactive shells skip ~/.bashrc PATH setup.
export PATH="$HOME/.local/bin:$PATH"

PORT=8765
MODEL="F5-TTS"
REF="reference_clean.wav"
TRANSCRIPT="transcript.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --ref) REF="$2"; shift ;;
        --transcript) TRANSCRIPT="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "$REF" ]; then
    echo "Error: Reference audio not found: $REF"
    echo "Record one with: ./record_and_transcribe.sh, then ./clean_reference.sh"
    exit 1
fi

if [ ! -f "$TRANSCRIPT" ]; then
    echo "Error: Transcript not found: $TRANSCRIPT"
    exit 1
fi

# Pick a CUDA wheel index for PyTorch. CUDA 13.x → nightly cu130 wheels.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/nightly/cu130}"

echo "=== Voice Clone TTS Server (CUDA / DGX, F5-TTS) ==="
echo "Reference:  $REF"
echo "Transcript: $(cat "$TRANSCRIPT")"
echo "Model:      $MODEL"
echo "Port:       $PORT"
echo "Torch idx:  $TORCH_INDEX"
echo
echo "Open http://localhost:$PORT in your browser once you see 'TTS Server ready'."
echo

# Force Python's stdout/stderr to be line-buffered so per-chunk progress
# logs from the generation worker show up live in tee/journald, not in
# 4 KB block-buffered chunks.
export PYTHONUNBUFFERED=1

exec uv run \
    --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    --prerelease allow \
    --with f5-tts \
    --with torch \
    --with torchaudio \
    python3 web_tts_server.py \
    --port "$PORT" \
    --model "$MODEL" \
    --ref-audio "$REF" \
    --ref-text-file "$TRANSCRIPT"
