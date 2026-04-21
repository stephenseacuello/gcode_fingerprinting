#!/bin/bash
# Launch wandb decoder hyperparameter sweep
# Creates sweep and launches agent in tmux session

set -e

cd "$(dirname "$0")/../.."

# Create output directory
mkdir -p outputs/decoder20260304

# Create sweep and capture sweep ID
echo "Creating wandb sweep..."
wandb sweep sweep_config.yaml 2>&1 | tee /tmp/sweep_init.log

SWEEP_ID=$(grep "wandb agent" /tmp/sweep_init.log | grep -oP '[^\s]+$')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to extract sweep ID from wandb output"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Launching agent with 350 trials in tmux session 'sweep'..."

tmux new-session -d -s sweep \
    "cd $(pwd) && wandb agent --count 350 $SWEEP_ID 2>&1 | tee outputs/decoder20260304/sweep_log.txt"

echo ""
echo "Sweep launched in tmux session 'sweep'"
echo "  Reattach:  tmux attach -t sweep"
echo "  Dashboard: https://wandb.ai (project: decoder-sweep)"
echo "  Log:       outputs/decoder20260304/sweep_log.txt"
