#!/bin/bash
# Real-time CNC Phase Detection and Anomaly Monitoring
# Usage: ./monitor.sh [options]

set -e

# Default values
CHECKPOINT="${CHECKPOINT:-outputs/models/MM_DTAE_LSTM_20251116_165433_best.pt}"
INPUT_CSV="${INPUT_CSV:-data/test_001_aligned.csv}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
STRIDE="${STRIDE:-1}"
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-0.3}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/realtime_monitoring}"

# Export PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run monitoring
.venv/bin/python src/miracle/inference/realtime_monitor.py \
    --checkpoint "$CHECKPOINT" \
    --input-csv "$INPUT_CSV" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --anomaly-threshold "$ANOMALY_THRESHOLD" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
