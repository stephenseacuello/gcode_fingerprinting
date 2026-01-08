#!/bin/bash
# =============================================================================
# Comprehensive Local Training Script
# =============================================================================
# Full-featured training run with all the new Phase 1 features enabled
# Expected duration: ~2-4 hours on Mac (M1/M2)
#
# Usage:
#   ./scripts/train_comprehensive_local.sh
# =============================================================================

set -e

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "COMPREHENSIVE LOCAL TRAINING"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/comprehensive_run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training with all features enabled
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/processed_v3 \
    --vocab-path data/vocabulary_4digit_full.json \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output-dir "$OUTPUT_DIR" \
    \
    `# === Architecture (larger model) ===` \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 5 \
    --dropout 0.25 \
    --embed-dropout 0.15 \
    --d-ff-multiplier 4.0 \
    --sensor-dim 128 \
    \
    `# === Training Duration ===` \
    --max-epochs 100 \
    --patience 20 \
    --batch-size 32 \
    \
    `# === Optimizer ===` \
    --optimizer adamw \
    --learning-rate 0.001 \
    --weight-decay 0.05 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad-clip 1.0 \
    \
    `# === LR Scheduler ===` \
    --lr-scheduler warmup_cosine \
    --warmup-epochs 5 \
    --cosine-t-max 95 \
    \
    `# === Data Augmentation (enabled) ===` \
    --augment \
    --augment-prob 0.35 \
    --noise-level 0.015 \
    --shift-range 2 \
    --scale-range-min 0.97 \
    --scale-range-max 1.03 \
    --time-warp-sigma 0.12 \
    --feature-dropout-prob 0.08 \
    --cutout-length 3 \
    --jitter-sigma 0.008 \
    \
    `# === Curriculum Learning ===` \
    --curriculum \
    --curriculum-phases 3 \
    --curriculum-epochs-per-phase 30 \
    --curriculum-structure-weight 1.0 \
    --curriculum-digit-weight-p2 0.5 \
    --curriculum-digit-weight-p3 1.0 \
    \
    `# === Scheduled Sampling ===` \
    --scheduled-sampling \
    --teacher-forcing-start 1.0 \
    --teacher-forcing-end 0.2 \
    --teacher-forcing-decay-epochs 60 \
    --teacher-forcing-decay cosine \
    \
    `# === Focal Loss ===` \
    --use-focal-loss \
    --focal-gamma 2.5 \
    --focal-alpha 0.3 \
    --digit-focal-gamma 5.0 \
    --command-focal-gamma-boost 1.0 \
    \
    `# === Loss Weights ===` \
    --type-weight 1.0 \
    --command-weight 1.5 \
    --param-type-weight 1.0 \
    --digit-weight 1.2 \
    \
    `# === Grammar Constraints ===` \
    --use-grammar-constraints \
    --grammar-weight 0.15 \
    --constraint-arc-radius-weight 1.0 \
    --constraint-modal-group-weight 0.6 \
    --constraint-cmd-param-weight 0.4 \
    \
    `# === WandB Logging ===` \
    --use-wandb \
    --wandb-project gcode-comprehensive-local \
    --run-name "comprehensive_${TIMESTAMP}" \
    \
    `# === Evaluation ===` \
    --val-interval 1 \
    --track-metric token \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/training.log | tail -50"
echo ""
