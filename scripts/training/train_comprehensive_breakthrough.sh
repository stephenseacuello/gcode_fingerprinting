#!/bin/bash
# Comprehensive Breakthrough Training Script
# Runs training with ALL breakthrough phases enabled (1-6)
#
# Phases:
#   1. Type-constrained decoding (--use-type-constraint)
#   2. Sensor value prior (--use-sensor-prior)
#   3. Position-weighted loss (--use-position-weights)
#   4. Error-focused sampling (--use-error-sampling)
#   5. Larger model capacity (d_model=256, n_layers=5)
#   6. Ensemble training (run 3 seeds)
#
# Usage:
#   ./scripts/train_comprehensive_breakthrough.sh [output_dir] [epochs]
#
# Example:
#   ./scripts/train_comprehensive_breakthrough.sh outputs/breakthrough_final 100

set -e

# Configuration
OUTPUT_BASE="${1:-outputs/breakthrough_comprehensive}"
MAX_EPOCHS="${2:-100}"
DATA_DIR="outputs/stratified_splits_v2"
ENCODER_PATH="outputs/mm_dtae_lstm_v2/best_model.pt"
VOCAB_PATH="data/vocabulary_4digit_hybrid.json"

# Seeds for ensemble training
SEEDS=(42 123 456)

echo "=============================================="
echo "COMPREHENSIVE BREAKTHROUGH TRAINING"
echo "=============================================="
echo "Output base: ${OUTPUT_BASE}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Ensemble seeds: ${SEEDS[*]}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Function to train a single model
train_model() {
    local seed=$1
    local output_dir="${OUTPUT_BASE}/model_seed_${seed}"

    echo "=============================================="
    echo "Training model with seed ${seed}"
    echo "Output: ${output_dir}"
    echo "=============================================="

    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
        --split-dir "${DATA_DIR}" \
        --encoder-path "${ENCODER_PATH}" \
        --vocab-path "${VOCAB_PATH}" \
        --output-dir "${output_dir}" \
        --seed "${seed}" \
        \
        --d-model 256 \
        --n-layers 5 \
        --n-heads 8 \
        --dropout 0.35 \
        --embed-dropout 0.15 \
        --d-ff-multiplier 4.0 \
        \
        --batch-size 24 \
        --max-epochs "${MAX_EPOCHS}" \
        --patience 30 \
        --track-metric token \
        \
        --optimizer adamw \
        --learning-rate 1.5e-4 \
        --weight-decay 0.05 \
        --lr-scheduler cosine \
        --warmup-epochs 10 \
        --grad-clip 1.0 \
        \
        --use-type-constraint \
        \
        --use-sensor-prior \
        --sensor-prior-weight 0.5 \
        \
        --use-position-weights \
        --position-weight-scale 3.0 \
        \
        --use-error-sampling \
        --error-boost 2.0 \
        \
        --use-focal-loss \
        --focal-gamma 3.0 \
        --label-smoothing 0.1 \
        \
        --curriculum \
        --curriculum-phases 3 \
        --curriculum-epochs-per-phase 30 \
        \
        --scheduled-sampling \
        --teacher-forcing-start 1.0 \
        --teacher-forcing-end 0.5 \
        --teacher-forcing-decay cosine \
        \
        --use-grammar-constraints \
        --grammar-weight 0.1 \
        \
        --command-weight 2.5 \
        --param-type-weight 1.5 \
        --digit-weight 1.0 \
        \
        --use-wandb \
        --wandb-project "gcode-breakthrough" \
        --run-name "breakthrough_seed${seed}" \
        \
        --print-every 10 \
        --num-samples 3

    echo "Model with seed ${seed} completed!"
    echo ""
}

# Train all ensemble models
echo "Starting ensemble training with ${#SEEDS[@]} seeds..."
echo ""

for seed in "${SEEDS[@]}"; do
    train_model "${seed}"
done

echo "=============================================="
echo "ALL MODELS TRAINED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Trained models:"
for seed in "${SEEDS[@]}"; do
    echo "  ${OUTPUT_BASE}/model_seed_${seed}/best_model.pt"
done
echo ""
echo "Next steps:"
echo "  1. Run ensemble evaluation:"
echo "     PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble.py \\"
echo "       --checkpoints ${OUTPUT_BASE}/model_seed_*/best_model.pt \\"
echo "       --data-dir ${DATA_DIR} \\"
echo "       --vocab-path ${VOCAB_PATH} \\"
echo "       --output reports/ensemble_evaluation.json"
echo ""
echo "  2. Run error analysis on best model:"
echo "     PYTHONPATH=src .venv/bin/python scripts/error_analysis.py \\"
echo "       --checkpoint ${OUTPUT_BASE}/model_seed_42/best_model.pt \\"
echo "       --output reports/breakthrough_error_analysis.json"
echo ""
