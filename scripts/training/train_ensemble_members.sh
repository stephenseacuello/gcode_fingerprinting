#!/bin/bash
# Train 3 ensemble members with SAME seed (for encoder compatibility)
# but different training dynamics for diversity
#
# All models share the same encoder initialization (seed=123)
# Diversity comes from:
#   - Different data shuffling (via different run order)
#   - Different dropout realizations
#   - Slightly different hyperparameters (learning rate, warmup)
#
# Usage: ./scripts/train_ensemble_members.sh

set -e

# Base configuration (from best_config_100ep)
BASE_SEED=123
EPOCHS=150
PATIENCE=40

# Required paths
SPLIT_DIR="outputs/stratified_splits_full_vocab"
VOCAB_PATH="data/vocabulary_4digit_full.json"
ENCODER_PATH="outputs/mm_dtae_lstm_v2/best_model.pt"

echo "=============================================="
echo "Training 3 Ensemble Members (Same Seed=$BASE_SEED)"
echo "=============================================="
echo ""

# Ensemble Member 1: Base configuration
echo ">>> Training Ensemble Member 1 (base config)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_m1 \
    --seed $BASE_SEED \
    --max-epochs $EPOCHS \
    --patience $PATIENCE \
    --d-model 320 \
    --n-heads 8 \
    --n-layers 5 \
    --dropout 0.32 \
    --embed-dropout 0.17 \
    --learning-rate 0.00024 \
    --weight-decay 0.052 \
    --warmup-epochs 12 \
    --lr-scheduler cosine_restarts \
    --restart-period 15 \
    --label-smoothing 0.1 \
    --use-sensor-prior \
    --sensor-prior-weight 0.51 \
    --drop-path-rate 0.11 \
    --use-enhanced-encoder \
    --use-multihead-pooling \
    --pooling-n-heads 4 \
    --pooling-n-queries 8 \
    --augment \
    --augment-prob 0.3 \
    --progressive-augmentation \
    --use-error-sampling \
    --error-boost 1.5 \
    --use-position-weights \
    --position-weight-scale 3.0 \
    --track-metric token \
    --use-wandb \
    --wandb-project gcode-ensemble \
    --run-name ensemble-m1-base

echo ""
echo ">>> Ensemble Member 1 complete!"
echo ""

# Ensemble Member 2: Slightly higher learning rate, different warmup
echo ">>> Training Ensemble Member 2 (higher LR, shorter warmup)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_m2 \
    --seed $BASE_SEED \
    --max-epochs $EPOCHS \
    --patience $PATIENCE \
    --d-model 320 \
    --n-heads 8 \
    --n-layers 5 \
    --dropout 0.32 \
    --embed-dropout 0.17 \
    --learning-rate 0.00028 \
    --weight-decay 0.052 \
    --warmup-epochs 8 \
    --lr-scheduler cosine_restarts \
    --restart-period 20 \
    --label-smoothing 0.1 \
    --use-sensor-prior \
    --sensor-prior-weight 0.51 \
    --drop-path-rate 0.11 \
    --use-enhanced-encoder \
    --use-multihead-pooling \
    --pooling-n-heads 4 \
    --pooling-n-queries 8 \
    --augment \
    --augment-prob 0.35 \
    --progressive-augmentation \
    --use-error-sampling \
    --error-boost 1.5 \
    --use-position-weights \
    --position-weight-scale 3.0 \
    --track-metric token \
    --use-wandb \
    --wandb-project gcode-ensemble \
    --run-name ensemble-m2-higherLR

echo ""
echo ">>> Ensemble Member 2 complete!"
echo ""

# Ensemble Member 3: Lower learning rate, more regularization
echo ">>> Training Ensemble Member 3 (lower LR, more regularization)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_m3 \
    --seed $BASE_SEED \
    --max-epochs $EPOCHS \
    --patience $PATIENCE \
    --d-model 320 \
    --n-heads 8 \
    --n-layers 5 \
    --dropout 0.35 \
    --embed-dropout 0.20 \
    --learning-rate 0.00020 \
    --weight-decay 0.06 \
    --warmup-epochs 15 \
    --lr-scheduler cosine_restarts \
    --restart-period 12 \
    --label-smoothing 0.12 \
    --use-sensor-prior \
    --sensor-prior-weight 0.51 \
    --drop-path-rate 0.13 \
    --use-enhanced-encoder \
    --use-multihead-pooling \
    --pooling-n-heads 4 \
    --pooling-n-queries 8 \
    --augment \
    --augment-prob 0.25 \
    --progressive-augmentation \
    --use-error-sampling \
    --error-boost 1.5 \
    --use-position-weights \
    --position-weight-scale 3.0 \
    --track-metric token \
    --use-wandb \
    --wandb-project gcode-ensemble \
    --run-name ensemble-m3-moreReg

echo ""
echo "=============================================="
echo "All 3 Ensemble Members Trained!"
echo "=============================================="
echo ""
echo "Run ensemble evaluation with:"
echo "PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble_simple.py \\"
echo "    --checkpoints outputs/ensemble_m1/best_model.pt \\"
echo "                  outputs/ensemble_m2/best_model.pt \\"
echo "                  outputs/ensemble_m3/best_model.pt \\"
echo "    --output reports/ensemble_final.json"
