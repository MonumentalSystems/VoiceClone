#!/bin/bash
# Clean and prepare a reference audio file for voice cloning
#
# Usage: ./clean_reference.sh [input.wav] [output.wav] [options]
#   input:  raw or manually trimmed recording (default: reference_voice.wav)
#   output: cleaned file ready for cloning    (default: reference_clean.wav)
#
# Options:
#   --no-trim       Skip silence trimming (if you already manually trimmed)
#   --no-denoise    Skip spectral noise reduction
#   --denoise N     Noise reduction sensitivity 0.0-1.0 (default: 0.21, higher = more aggressive)
#   --trim-start N  Trim N seconds from the start (e.g. --trim-start 0.5)
#   --trim-end N    Trim N seconds from the end
#   --fade N        Fade-in duration in seconds (default: 0.05, kills mic pops)
#   --lowpass N     Lowpass cutoff in Hz (default: 6000, lower = less hiss/noise)
#   --highpass N    Highpass cutoff in Hz (default: 80, higher = less rumble/hum)
#   --norm N        Normalize peak level in dB (default: -3, less hot = less static)
#
# What it does:
#   1. Optional start/end trim
#   2. Spectral noise reduction (profiles quietest segment, subtracts noise floor)
#   3. Short fade-in to kill mic pop/click at start
#   4. Gentle silence trim (only true silence, not quiet speech)
#   5. Highpass 130Hz (remove rumble + 120Hz hum)
#   6. Lowpass 3500Hz (remove hiss/whine from webcam mic)
#   7. Compression (even out loud/quiet parts)
#   8. Normalize to -0.5dB peak
#   9. Resample to 24kHz mono
#
# Examples:
#   ./clean_reference.sh                                  # defaults
#   ./clean_reference.sh my_recording.wav                 # custom input
#   ./clean_reference.sh ref.wav clean.wav --trim-start 0.5
#   ./clean_reference.sh ref.wav clean.wav --no-trim --fade 0.1
#   ./clean_reference.sh ref.wav clean.wav --no-denoise   # skip noise reduction
#   ./clean_reference.sh ref.wav clean.wav --denoise 0.3  # more aggressive

set -e
cd "$(dirname "$0")"

INPUT="${1:-reference_voice.wav}"
OUTPUT="${2:-reference_clean.wav}"
NO_TRIM=false
NO_DENOISE=false
DENOISE="0.21"
TRIM_START=""
TRIM_END=""
FADE="0.05"
LOWPASS="6000"
HIGHPASS="80"
NORM="-3"

# Parse options (skip first two positional args)
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --no-trim) NO_TRIM=true ;;
        --no-denoise) NO_DENOISE=true ;;
        --denoise) DENOISE="$2"; shift ;;
        --trim-start) TRIM_START="$2"; shift ;;
        --trim-end) TRIM_END="$2"; shift ;;
        --fade) FADE="$2"; shift ;;
        --lowpass) LOWPASS="$2"; shift ;;
        --highpass) HIGHPASS="$2"; shift ;;
        --norm) NORM="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

echo "=== Cleaning Reference Audio ==="
echo "Input:  $INPUT"

# Show input stats
echo
echo "--- Input stats ---"
sox "$INPUT" -n stats 2>&1 | head -8
echo

IN_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
echo "Input duration: ${IN_DUR}s"

# Step 1: If manual trim requested, do that first to a temp file
WORKING="$INPUT"
TEMPS=()

if [ -n "$TRIM_START" ] || [ -n "$TRIM_END" ]; then
    TEMP_TRIM=$(mktemp /tmp/ref_trim_XXXX.wav)
    TEMPS+=("$TEMP_TRIM")
    TRIM_ARGS="trim"
    if [ -n "$TRIM_START" ]; then
        TRIM_ARGS="$TRIM_ARGS $TRIM_START"
        echo "Trimming ${TRIM_START}s from start"
    else
        TRIM_ARGS="$TRIM_ARGS 0"
    fi
    if [ -n "$TRIM_END" ]; then
        KEEP_DUR=$(echo "$IN_DUR - ${TRIM_START:-0} - $TRIM_END" | bc 2>/dev/null || python3 -c "print($IN_DUR - ${TRIM_START:-0} - $TRIM_END)")
        TRIM_ARGS="$TRIM_ARGS $KEEP_DUR"
        echo "Trimming ${TRIM_END}s from end (keeping ${KEEP_DUR}s)"
    fi
    sox "$INPUT" "$TEMP_TRIM" $TRIM_ARGS 2>/dev/null
    WORKING="$TEMP_TRIM"
fi

# Step 2: Spectral noise reduction (profile quietest segment, subtract)
if [ "$NO_DENOISE" = false ]; then
    echo
    echo "Applying spectral noise reduction (sensitivity=${DENOISE})..."

    # Find quietest 0.5s segment using sox's stat on windows
    NOISE_PROF=$(mktemp /tmp/ref_noiseprof_XXXX.prof)
    TEMP_DENOISED=$(mktemp /tmp/ref_denoised_XXXX.wav)
    TEMPS+=("$NOISE_PROF" "$TEMP_DENOISED")

    # Use python to find quietest window, fall back to midpoint
    QUIET_POS=$(python3 -c "
import struct, wave
with wave.open('$WORKING', 'rb') as w:
    sr = w.getframerate()
    n = w.getnframes()
    raw = w.readframes(n)
    samples = struct.unpack('<' + 'h' * n, raw)
    win = int(sr * 0.5)
    best_rms, best_pos = float('inf'), 0
    step = win // 4
    for i in range(0, n - win, step):
        chunk = samples[i:i+win]
        rms = (sum(s*s for s in chunk) / win) ** 0.5
        if rms < best_rms:
            best_rms, best_pos = rms, i
    print(f'{best_pos / sr:.3f}')
" 2>/dev/null || echo "$(echo "$IN_DUR / 2" | bc -l 2>/dev/null || echo '5')")

    echo "  Noise profile from ${QUIET_POS}s (quietest 0.5s segment)"
    sox "$WORKING" -n trim "$QUIET_POS" 0.5 noiseprof "$NOISE_PROF" 2>/dev/null
    sox "$WORKING" "$TEMP_DENOISED" noisered "$NOISE_PROF" "$DENOISE" 2>/dev/null
    WORKING="$TEMP_DENOISED"
fi

# Step 3: Build main sox command (filters + normalize)
SOX_ARGS=()

# Fade-in to kill mic pop/click at start
if [ "$FADE" != "0" ]; then
    SOX_ARGS+=(fade t "$FADE")
fi

# Gentle silence trim
if [ "$NO_TRIM" = false ]; then
    SOX_ARGS+=(silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse)
fi

# Filters + compress + normalize
SOX_ARGS+=(
    highpass "$HIGHPASS"
    lowpass "$LOWPASS"
    compand 0.005,0.1 -70,-70,-50,-20,-30,-10,-20,-6,-6,-3,0,-1 -5 0 0.05
    norm "$NORM"
    rate 24000
)

echo
echo "Processing (fade=${FADE}s, trim=$([ "$NO_TRIM" = true ] && echo 'off' || echo 'on'), denoise=$([ "$NO_DENOISE" = true ] && echo 'off' || echo "${DENOISE}"), hp=${HIGHPASS}Hz, lp=${LOWPASS}Hz, norm=${NORM}dB)..."
sox "$WORKING" "$OUTPUT" "${SOX_ARGS[@]}" 2>/dev/null

# Clean up temp files
for f in "${TEMPS[@]}"; do
    rm -f "$f"
done

OUT_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT")
echo "Output: $OUTPUT (${OUT_DUR}s)"

echo
echo "--- Output stats ---"
sox "$OUTPUT" -n stats 2>&1 | head -8

# Warn if too short
DUR_INT=$(echo "$OUT_DUR" | cut -d. -f1)
if [ "$DUR_INT" -lt 4 ]; then
    echo
    echo "WARNING: Reference is only ${OUT_DUR}s — aim for 6-15s for best cloning"
fi

echo
echo "=== Done ==="
echo "Next: ./serve.sh   (then open http://localhost:8765)"
