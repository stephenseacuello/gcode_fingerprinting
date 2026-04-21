#!/bin/bash
# Hybrid Position-Gated Decoding evaluation
# No training — inference only, ~2-3 minutes total
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

python3 scripts/evaluation/run_hybrid_position_decoding.py \
    --folds 1 2 3 4 5 \
    --vocab data/gcode_vocab_712.json \
    --beam_width 1 \
    --device auto \
    "$@"
