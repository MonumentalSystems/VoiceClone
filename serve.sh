#!/bin/bash
# Start the Voice Clone TTS web server (F5-TTS, Apple Silicon / CPU build).
#
# Usage: ./serve.sh [options]
#
# Options:
#   --port N        Port to listen on (default: 8765)
#   --device D      auto|mps|cpu (auto picks mps > cpu) (default: auto)
#   --ref FILE      Reference audio file (default: reference_clean.wav)
#   --transcript F  Transcript file (default: transcript.txt)
#
# For NVIDIA / DGX use ./serve-dgx.sh instead — it pulls the CUDA torch wheels.
# Opens http://localhost:8765 in your browser automatically.

set -e
cd "$(dirname "$0")"

PORT=8765
DEVICE="auto"
REF="reference_clean.wav"
TRANSCRIPT="transcript.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
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

echo "=== Voice Clone TTS Server (F5-TTS) ==="
echo "Reference:  $REF ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REF" 2>/dev/null)s)"
echo "Transcript: $(cat "$TRANSCRIPT")"
echo "Device:     $DEVICE"
echo "Port:       $PORT"
echo
echo "Open http://localhost:$PORT once you see 'TTS Server ready'."
echo

(sleep 6 && open "http://localhost:$PORT" 2>/dev/null || true) &

export PYTHONUNBUFFERED=1

exec uv run --python 3.11 \
    --with f5-tts --with torch --with torchaudio --with faster-whisper \
    python3 web_tts_server.py \
    --port "$PORT" \
    --device "$DEVICE" \
    --ref-audio "$REF" \
    --ref-text-file "$TRANSCRIPT"
