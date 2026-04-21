#!/bin/bash
# Launch V7 sweep v2 (reduced: 500 epochs, 120 patience, 1500 trials)
set -e
cd /home/seacuello/Documents/gcode_fingerprinting
export PYTHONPATH="src:$PYTHONPATH"

mkdir -p outputs/decoder20260304/sweep_v7
echo "Starting V7 sweep v2 at $(date)"
echo "Samples: 1500, Epochs: 500, Patience: 120, GPUs/trial: 0.14, Max concurrent: 14"
echo ""

python3 scripts/experiments/run_v7_ray_sweep.py \
    --num_samples 1500 \
    --gpus_per_trial 0.14 \
    --max_concurrent 14 \
    2>&1 | tee outputs/decoder20260304/sweep_v7/v7_sweep_v2.log
