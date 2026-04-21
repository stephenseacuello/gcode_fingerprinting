#!/bin/bash
set -euo pipefail

BASE="outputs/anomaly20260319"
SCRIPTS="scripts/anomaly"
LOG="$BASE/logs"
SESSION="anomaly"

mkdir -p "$LOG"

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session
tmux new-session -d -s "$SESSION" -n "phase0"

# Phase 0: Cache + Attacks (sequential, GPU)
tmux send-keys -t "$SESSION:phase0" \
  "cd /home/seacuello/Documents/gcode_fingerprinting && \
   echo '=== Phase 0: Cache Inference ===' && \
   python3 $SCRIPTS/cache_inference.py --output-dir $BASE/cached_inference 2>&1 | tee $LOG/cache_inference.log && \
   python3 $SCRIPTS/generate_attacks.py --output-dir $BASE/attacks 2>&1 | tee $LOG/generate_attacks.log && \
   echo '=== Phase 0 DONE ===' && \
   bash $SCRIPTS/launch_groups.sh" C-m

echo "tmux session '$SESSION' created."
echo "Attach: tmux attach -t $SESSION"
