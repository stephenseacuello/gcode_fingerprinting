#!/bin/bash
# Approach A: Same architecture, different seeds
# Each model gets its own encoder initialization, but encoder states are saved
# This enables valid ensemble evaluation with saved encoder states
#
# Seeds: 123, 42, 456
#
# Usage: ./scripts/train_ensemble_diverse_seeds.sh

set -e

# Shared configuration
EPOCHS=150
PATIENCE=40

# Required paths
SPLIT_DIR="outputs/stratified_splits_full_vocab"
VOCAB_PATH="data/vocabulary_4digit_full.json"
ENCODER_PATH="outputs/mm_dtae_lstm_v2/best_model.pt"

echo "=============================================="
echo "Approach A: Same Architecture, Different Seeds"
echo "=============================================="
echo "Seeds: 123, 42, 456"
echo "Encoder states will be SAVED in checkpoints"
echo ""

# Model 1: Seed 123
echo ">>> Training Model A1 (seed=123)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_seedA1 \
    --seed 123 \
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
    --wandb-project gcode-ensemble-seeds \
    --run-name seedA1-s123

echo ""
echo ">>> Model A1 complete!"
echo ""

# Model 2: Seed 42
echo ">>> Training Model A2 (seed=42)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_seedA2 \
    --seed 42 \
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
    --wandb-project gcode-ensemble-seeds \
    --run-name seedA2-s42

echo ""
echo ">>> Model A2 complete!"
echo ""

# Model 3: Seed 456
echo ">>> Training Model A3 (seed=456)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_seedA3 \
    --seed 456 \
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
    --wandb-project gcode-ensemble-seeds \
    --run-name seedA3-s456

echo ""
echo "=============================================="
echo "Approach A Complete!"
echo "=============================================="
echo ""
echo "Models saved to:"
echo "  - outputs/ensemble_seedA1/best_model.pt (seed=123)"
echo "  - outputs/ensemble_seedA2/best_model.pt (seed=42)"
echo "  - outputs/ensemble_seedA3/best_model.pt (seed=456)"
echo ""
echo "Evaluate with:"
echo "PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble_simple.py \\"
echo "    --checkpoints outputs/ensemble_seedA1/best_model.pt \\"
echo "                  outputs/ensemble_seedA2/best_model.pt \\"
echo "                  outputs/ensemble_seedA3/best_model.pt \\"
echo "    --output reports/ensemble_diverse_seeds.json"
