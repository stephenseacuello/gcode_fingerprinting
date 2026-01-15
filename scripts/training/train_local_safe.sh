#!/bin/bash
# =============================================================================
# Local MultiHead Training - Safe Configuration
# =============================================================================
# Uses conservative hyperparameters to avoid NaN loss issues
# =============================================================================

set -e

cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "LOCAL MULTIHEAD TRAINING - Safe Config"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/multihead_safe_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Use latest splits and vocab
SPLIT_DIR="outputs/stratified_splits_2k_vocab_w96"
VOCAB_PATH="data/vocabulary_4digit_full.json"

echo "Output: $OUTPUT_DIR"
echo "Split dir: $SPLIT_DIR"
echo "Vocab: $VOCAB_PATH (2498 tokens)"
echo ""

# Run training with safe config
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/training/train_multihead.py \
    --data-dir "$SPLIT_DIR" \
    --split-dir "$SPLIT_DIR" \
    --vocab-path "$VOCAB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    \
    `# === Architecture (smaller for stability) ===` \
    --hidden-dim 128 \
    --num-heads 4 \
    --num-layers 2 \
    --dropout 0.3 \
    --embed-dropout 0.2 \
    --sensor-input-dim 296 \
    \
    `# === Training ===` \
    --max-epochs 50 \
    --patience 15 \
    --batch-size 32 \
    \
    `# === Optimizer (lower LR for stability) ===` \
    --optimizer adamw \
    --learning-rate 0.0001 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad-clip 0.5 \
    \
    `# === LR Scheduler ===` \
    --lr-scheduler cosine \
    --warmup-epochs 3 \
    --cosine-t-max 47 \
    \
    `# === Loss Weights (simpler) ===` \
    --command-weight 1.0 \
    --param-type-weight 1.0 \
    --param-value-weight 1.0 \
    --label-smoothing 0.1 \
    \
    `# === NO Focal Loss (for stability) ===` \
    \
    `# === Augmentation ===` \
    --augmentation \
    --noise-std 0.005 \
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
