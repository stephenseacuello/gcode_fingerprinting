#!/bin/bash
################################################################################
# Train V8 decoder: Multi-line tokenization with max_token_len=64
# Uses same hyperparameters as best_config_5fold but with:
#   - V8 preprocessed data (multi-line G-code per window)
#   - max_token_len=64 (vs 16 in V7)
#   - max_seq_len=128 in decoder (safety margin)
#
# Output: outputs/anomaly20260319/decoder_v8_multiline/fold_{1-5}/
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

V8_DATA="outputs/anomaly20260319/preprocessed_v8_multiline"
ENCODER_BASE="outputs/experiments_2026_02_25/full_w256_s64_cv"
VOCAB="data/gcode_vocab_712.json"
OUTPUT_BASE="outputs/anomaly20260319/decoder_v8_multiline"

MAX_TOKEN_LEN=64
EPOCHS=500
PATIENCE=100
BATCH_SIZE=32
D_MODEL=384
N_LAYERS=8
N_HEADS=8
DROPOUT=0.1
DIGIT_WEIGHT=15.0
LEGACY_WEIGHT=1.0
LABEL_SMOOTHING=0.2
SCHEDULED_SAMPLING=0.5
FOCAL_GAMMA=2.0
LR=0.0005
WEIGHT_DECAY=0.1
WARMUP=20

echo "================================================================================"
echo "V8 DECODER TRAINING: Multi-line tokenization, max_token_len=${MAX_TOKEN_LEN}"
echo "================================================================================"
echo "V8 data: $V8_DATA"
echo "Output: $OUTPUT_BASE"
echo "Started at: $(date)"
echo ""

FOLD=${1:-0}  # Pass specific fold as argument, or 0 for all

train_fold() {
    local f=$1
    echo ""
    echo "================ FOLD ${f} / 5 ================"

    python3 scripts/evaluation/run_decoder_quick_test.py \
        --data_dir "${V8_DATA}/fold_${f}" \
        --encoder_ckpt "${ENCODER_BASE}/fold_${f}/encoder/checkpoint/best_model.pt" \
        --vocab "$VOCAB" \
        --output_dir "${OUTPUT_BASE}/fold_${f}" \
        --max_token_len "$MAX_TOKEN_LEN" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch_size "$BATCH_SIZE" \
        --d_model "$D_MODEL" \
        --n_layers "$N_LAYERS" \
        --n_heads "$N_HEADS" \
        --dropout "$DROPOUT" \
        --digit_weight "$DIGIT_WEIGHT" \
        --legacy_weight "$LEGACY_WEIGHT" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --scheduled_sampling "$SCHEDULED_SAMPLING" \
        --focal_gamma "$FOCAL_GAMMA" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --warmup_epochs "$WARMUP" \
        --seed 42 \
        2>&1 | tee "${OUTPUT_BASE}/fold_${f}_training.log"

    echo "Fold $f complete at $(date)"
}

mkdir -p "$OUTPUT_BASE"

if [ "$FOLD" -eq 0 ]; then
    # Train all folds sequentially
    for f in 1 2 3 4 5; do
        train_fold $f
    done
else
    train_fold $FOLD
fi

echo ""
echo "================================================================================"
echo "V8 DECODER TRAINING COMPLETE: $(date)"
echo "Output: $OUTPUT_BASE"
echo "================================================================================"
