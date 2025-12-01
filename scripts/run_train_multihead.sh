#!/bin/bash
# Wrapper script to run train_multihead.py with correct environment
# This ensures W&B sweeps use the virtual environment Python

cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=src:$PYTHONPATH

.venv/bin/python scripts/train_multihead.py "$@"
