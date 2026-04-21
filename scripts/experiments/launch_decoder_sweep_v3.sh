#!/bin/bash
# Launch V3 sweep: 2 encoder configs × 5 folds, 500 epochs, patience 100
set -e
cd "$(dirname "$0")/../.."

mkdir -p outputs/decoder20260304/sweep_v3

echo "Creating V3 wandb sweep..."
wandb sweep sweep_config_v3.yaml 2>&1 | tee /tmp/sweep_v3_init.log

SWEEP_ID=$(grep "wandb agent" /tmp/sweep_v3_init.log | grep -oP '[^\s]+$')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to extract sweep ID"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Launching 2000 trials in tmux session 'sweep'..."

tmux new-session -d -s sweep \
    "cd $(pwd) && wandb agent --count 2000 $SWEEP_ID 2>&1 | tee outputs/decoder20260304/sweep_v3_log.txt"

echo ""
echo "V3 sweep launched in tmux session 'sweep'"
echo "  Reattach:  tmux attach -t sweep"
echo "  Dashboard: https://wandb.ai (project: decoder-sweep-v3)"
echo "  Trials:    2000"
echo "  Epochs:    500 (patience 100)"
echo "  Encoders:  f110_w256_s64, f98_w64_s16"
echo "  Folds:     1-5"
