#!/bin/bash
# Approach B: Different architectures, same seed
# All models share the same encoder (seed=123) but have different decoder architectures
# This provides diversity through architecture variation
#
# Architectures:
#   - B1: Small  (d_model=256, n_layers=4) - faster, may generalize differently
#   - B2: Medium (d_model=320, n_layers=5) - baseline architecture
#   - B3: Large  (d_model=384, n_layers=6) - more capacity
#
# Usage: ./scripts/train_ensemble_diverse_arch.sh

set -e

# Shared configuration
SEED=123
EPOCHS=150
PATIENCE=40

# Required paths
SPLIT_DIR="outputs/stratified_splits_full_vocab"
VOCAB_PATH="data/vocabulary_4digit_full.json"
ENCODER_PATH="outputs/mm_dtae_lstm_v2/best_model.pt"

echo "=============================================="
echo "Approach B: Different Architectures, Same Seed"
echo "=============================================="
echo "Seed: $SEED (shared encoder)"
echo "Architectures: Small/Medium/Large decoders"
echo ""

# Model B1: Small architecture
echo ">>> Training Model B1 (small: d_model=256, n_layers=4)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_archB1 \
    --seed $SEED \
    --max-epochs $EPOCHS \
    --patience $PATIENCE \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --dropout 0.30 \
    --embed-dropout 0.15 \
    --learning-rate 0.00028 \
    --weight-decay 0.05 \
    --warmup-epochs 10 \
    --lr-scheduler cosine_restarts \
    --restart-period 15 \
    --label-smoothing 0.1 \
    --use-sensor-prior \
    --sensor-prior-weight 0.51 \
    --drop-path-rate 0.10 \
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
    --wandb-project gcode-ensemble-arch \
    --run-name archB1-small

echo ""
echo ">>> Model B1 complete!"
echo ""

# Model B2: Medium architecture (baseline)
echo ">>> Training Model B2 (medium: d_model=320, n_layers=5)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_archB2 \
    --seed $SEED \
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
    --wandb-project gcode-ensemble-arch \
    --run-name archB2-medium

echo ""
echo ">>> Model B2 complete!"
echo ""

# Model B3: Large architecture
echo ">>> Training Model B3 (large: d_model=384, n_layers=6)..."
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir $SPLIT_DIR \
    --vocab-path $VOCAB_PATH \
    --encoder-path $ENCODER_PATH \
    --output-dir outputs/ensemble_archB3 \
    --seed $SEED \
    --max-epochs $EPOCHS \
    --patience $PATIENCE \
    --d-model 384 \
    --n-heads 8 \
    --n-layers 6 \
    --dropout 0.35 \
    --embed-dropout 0.20 \
    --learning-rate 0.00020 \
    --weight-decay 0.06 \
    --warmup-epochs 15 \
    --lr-scheduler cosine_restarts \
    --restart-period 15 \
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
    --wandb-project gcode-ensemble-arch \
    --run-name archB3-large

echo ""
echo "=============================================="
echo "Approach B Complete!"
echo "=============================================="
echo ""
echo "Models saved to:"
echo "  - outputs/ensemble_archB1/best_model.pt (small: 256d, 4L)"
echo "  - outputs/ensemble_archB2/best_model.pt (medium: 320d, 5L)"
echo "  - outputs/ensemble_archB3/best_model.pt (large: 384d, 6L)"
echo ""
echo "Evaluate with:"
echo "PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble_simple.py \\"
echo "    --checkpoints outputs/ensemble_archB1/best_model.pt \\"
echo "                  outputs/ensemble_archB2/best_model.pt \\"
echo "                  outputs/ensemble_archB3/best_model.pt \\"
echo "    --output reports/ensemble_diverse_arch.json"
