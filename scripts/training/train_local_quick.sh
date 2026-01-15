#!/bin/bash
# =============================================================================
# Quick Local Training - Mac M1/M2 Optimized
# =============================================================================
# Uses latest stratified splits with 2K vocab and 96 window size
# Expected duration: ~1-2 hours on Mac
# =============================================================================

set -e

cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "QUICK LOCAL TRAINING - Latest Splits"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/local_train_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Use latest splits and vocab
SPLIT_DIR="outputs/stratified_splits_2k_vocab_w96"
VOCAB_PATH="data/vocabulary_4digit_full.json"
ENCODER_PATH="outputs/large_models/xlarge_v1/best_model.pt"

echo "Split dir: $SPLIT_DIR"
echo "Vocab: $VOCAB_PATH"
echo "Encoder: $ENCODER_PATH"
echo ""

# Run training
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
    --split-dir "$SPLIT_DIR" \
    --vocab-path "$VOCAB_PATH" \
    --encoder-path "$ENCODER_PATH" \
    --output-dir "$OUTPUT_DIR" \
    \
    `# === Architecture (moderate size for local) ===` \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --dropout 0.2 \
    --embed-dropout 0.1 \
    --d-ff-multiplier 4.0 \
    --sensor-dim 296 \
    \
    `# === Training Duration ===` \
    --max-epochs 50 \
    --patience 15 \
    --batch-size 16 \
    \
    `# === Optimizer ===` \
    --optimizer adamw \
    --learning-rate 0.0005 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad-clip 1.0 \
    \
    `# === LR Scheduler ===` \
    --lr-scheduler cosine \
    --warmup-epochs 3 \
    --cosine-t-max 47 \
    \
    `# === Light Augmentation ===` \
    --augment \
    --augment-prob 0.2 \
    --noise-level 0.01 \
    \
    `# === Focal Loss ===` \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --focal-alpha 0.25 \
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
echo "Model checkpoint: $OUTPUT_DIR/best_model.pt"
echo ""
