#!/bin/bash
# =============================================================================
# Local MultiHead Training - End-to-End (No Pre-trained Encoder Required)
# =============================================================================
# Uses train_multihead.py which trains the full model from scratch
# Expected duration: ~2-3 hours on Mac M1/M2
# =============================================================================

set -e

cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "LOCAL MULTIHEAD TRAINING - End-to-End"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/multihead_local_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Use latest splits and vocab
SPLIT_DIR="outputs/stratified_splits_2k_vocab_w96"
VOCAB_PATH="data/vocabulary_4digit_full.json"

echo "Output: $OUTPUT_DIR"
echo "Split dir: $SPLIT_DIR"
echo "Vocab: $VOCAB_PATH (2498 tokens)"
echo ""

# Run training
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/training/train_multihead.py \
    --data-dir "$SPLIT_DIR" \
    --split-dir "$SPLIT_DIR" \
    --vocab-path "$VOCAB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    \
    `# === Architecture ===` \
    --hidden-dim 256 \
    --num-heads 8 \
    --num-layers 4 \
    --dropout 0.2 \
    --embed-dropout 0.1 \
    --sensor-input-dim 296 \
    \
    `# === Training ===` \
    --max-epochs 100 \
    --patience 20 \
    --batch-size 16 \
    \
    `# === Optimizer ===` \
    --optimizer adamw \
    --learning-rate 0.0003 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad-clip 1.0 \
    \
    `# === LR Scheduler ===` \
    --lr-scheduler cosine \
    --warmup-epochs 5 \
    --cosine-t-max 95 \
    \
    `# === Loss Weights ===` \
    --command-weight 1.5 \
    --param-type-weight 1.0 \
    --param-value-weight 1.2 \
    --label-smoothing 0.1 \
    \
    `# === Focal Loss ===` \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --focal-alpha 0.25 \
    \
    `# === Augmentation ===` \
    --augmentation \
    --noise-std 0.01 \
    \
    `# === Digit Value Head (better numeric accuracy) ===` \
    --use-digit-value-head \
    --digit-loss-weight 1.0 \
    \
    --track-metric composite_acc \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Model: $OUTPUT_DIR/best_model.pt"
echo ""
