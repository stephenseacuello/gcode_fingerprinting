#!/bin/bash
# Wrapper script for wandb agent to ensure PYTHONPATH is set

# Set PYTHONPATH
export PYTHONPATH=src

# Enable MPS fallback for Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Filter out boolean flags with values (wandb passes them as --use-wandb=True)
# and add them back as simple flags
filtered_args=()
for arg in "$@"; do
    # Skip --use-wandb=True and --wandb-project=... since we'll add them properly
    if [[ ! "$arg" =~ ^--use-wandb= ]] && [[ ! "$arg" =~ ^--wandb-project= ]]; then
        filtered_args+=("$arg")
    fi
done

# Run the training script with filtered arguments and correct flags
.venv/bin/python "${filtered_args[@]}" --use-wandb --wandb-project gcode-fingerprinting
