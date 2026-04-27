#!/bin/bash
# Record voice and auto-transcribe with Whisper
#
# Usage: ./record_and_transcribe.sh [seconds]
#   seconds: recording duration (default: 20)
#
# Output:
#   reference_voice.wav  — raw recording
#   transcript.txt       — whisper transcription (edit if needed)
#
# Workflow:
#   1. ./record_and_transcribe.sh 20        # record + transcribe
#   2. Edit transcript.txt if whisper got anything wrong
#   3. (Optional) Manually trim reference_voice.wav in Audacity/ocenaudio
#   4. ./clean_reference.sh reference_voice.wav   # clean + prepare for cloning
#   5. ./serve.sh                                  # start the web app

set -e
cd "$(dirname "$0")"

DURATION=${1:-20}
RAW="reference_voice.wav"
TRANSCRIPT="transcript.txt"

echo "=== Voice Recording ==="
echo "Recording for ${DURATION}s — speak clearly into your mic."
echo "Press Ctrl-C to stop early."
echo

# Record from webcam mic (device :1)
ffmpeg -y -f avfoundation -i ":1" -ar 24000 -ac 1 -t "$DURATION" "$RAW" 2>/dev/null

echo
echo "Saved: $RAW"

# Show stats
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$RAW")
echo "Duration: ${DUR}s"
echo

# Transcribe with Whisper (on raw audio — don't need clean for transcription)
echo "Transcribing with Whisper..."
uvx --from mlx-audio --with phonemizer --with pip \
    mlx_audio.stt.generate \
    --audio "$RAW" \
    --output-path /tmp/whisper_out \
    --format txt \
    --language en \
    --model mlx-community/whisper-large-v3-turbo 2>&1

# Copy transcript
if [ -f /tmp/whisper_out.txt ]; then
    cp /tmp/whisper_out.txt "$TRANSCRIPT"
else
    echo "(Whisper output not found — enter transcript manually)"
    touch "$TRANSCRIPT"
fi

echo
echo "=== Transcript ==="
cat "$TRANSCRIPT"
echo
echo "=== Next Steps ==="
echo "1. Edit transcript.txt if Whisper got anything wrong"
echo "2. (Optional) Trim reference_voice.wav if there's dead space at start/end"
echo "     e.g: sox reference_voice.wav trimmed.wav trim 0.5 15"
echo "     or open in Audacity / ocenaudio"
echo "3. Run: ./clean_reference.sh [input.wav]"
echo "4. Run: ./serve.sh   (then open http://localhost:8765)"
