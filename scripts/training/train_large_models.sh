#!/bin/bash
# ============================================================================
# Train Large Models for G-Code Fingerprinting
# ============================================================================
#
# Three parallel training tracks:
#   Track A: Single best large model (large_v1)
#   Track B: Ensemble of 3 large models with different seeds
#   Track C: Ablation study via WandB sweep
#
# Usage:
#   ./scripts/training/train_large_models.sh [track]
#
# Arguments:
#   track: 'a', 'b', 'c', or 'all' (default: all)
#
# Requirements:
#   - Pretrained encoder at outputs/production/encoder/best_model.pt
#   - Data splits at outputs/production/data_splits/
#   - Vocabulary at data/vocabulary_4digit_full.json
#
# ============================================================================

set -e

# Configuration
SPLIT_DIR="outputs/production/data_splits"
VOCAB_PATH="data/vocabulary_4digit_full.json"
ENCODER_PATH="outputs/production/encoder/best_model.pt"

# Parse argument
TRACK="${1:-all}"

echo "=============================================="
echo "G-Code Fingerprinting: Large Model Training"
echo "=============================================="
echo "Track: $TRACK"
echo ""

# Verify prerequisites
if [ ! -d "$SPLIT_DIR" ]; then
    echo "ERROR: Data splits not found at $SPLIT_DIR"
    echo "Run output organization first or check paths."
    exit 1
fi

if [ ! -f "$ENCODER_PATH" ]; then
    echo "ERROR: Encoder not found at $ENCODER_PATH"
    exit 1
fi

# ============================================================================
# Track A: Single Best Large Model
# ============================================================================
train_track_a() {
    echo "=============================================="
    echo "TRACK A: Single Best Large Model (large_v1)"
    echo "=============================================="
    echo "Config: d_model=768, n_layers=8, n_heads=24"
    echo "Encoder: hidden_dim=512, lstm_layers=3"
    echo ""

    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/large_v1 \
        --seed 123 \
        --max-epochs 200 \
        --patience 50 \
        --batch-size 16 \
        --d-model 768 \
        --n-heads 24 \
        --n-layers 8 \
        --dropout 0.38 \
        --embed-dropout 0.20 \
        --learning-rate 0.00015 \
        --weight-decay 0.06 \
        --warmup-epochs 15 \
        --lr-scheduler cosine_restarts \
        --restart-period 25 \
        --label-smoothing 0.12 \
        --use-sensor-prior \
        --sensor-prior-weight 0.55 \
        --drop-path-rate 0.15 \
        --use-enhanced-encoder \
        --use-multihead-pooling \
        --pooling-n-heads 8 \
        --pooling-n-queries 16 \
        --augment \
        --augment-prob 0.25 \
        --progressive-augmentation \
        --use-error-sampling \
        --error-boost 1.8 \
        --use-position-weights \
        --position-weight-scale 3.0 \
        --track-metric token \
        --use-wandb \
        --wandb-project gcode-large-models \
        --run-name large-v1

    echo ""
    echo "Track A complete! Model saved to outputs/large_models/large_v1/"
}

# ============================================================================
# Track B: Ensemble of Large Models (3 seeds)
# ============================================================================
train_track_b() {
    echo "=============================================="
    echo "TRACK B: Ensemble of Large Models"
    echo "=============================================="
    echo "Config: d_model=512, n_layers=7, n_heads=16"
    echo "Seeds: 123, 42, 456"
    echo ""

    for seed in 123 42 456; do
        echo ">>> Training ensemble member (seed=$seed)..."

        PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
            --split-dir $SPLIT_DIR \
            --vocab-path $VOCAB_PATH \
            --encoder-path $ENCODER_PATH \
            --output-dir outputs/large_models/ensemble_large_s${seed} \
            --seed $seed \
            --max-epochs 150 \
            --patience 40 \
            --batch-size 24 \
            --d-model 512 \
            --n-heads 16 \
            --n-layers 7 \
            --dropout 0.35 \
            --embed-dropout 0.18 \
            --learning-rate 0.00018 \
            --weight-decay 0.055 \
            --warmup-epochs 12 \
            --lr-scheduler cosine_restarts \
            --restart-period 20 \
            --label-smoothing 0.11 \
            --use-sensor-prior \
            --sensor-prior-weight 0.52 \
            --drop-path-rate 0.13 \
            --use-enhanced-encoder \
            --use-multihead-pooling \
            --pooling-n-heads 4 \
            --pooling-n-queries 8 \
            --augment \
            --augment-prob 0.28 \
            --progressive-augmentation \
            --use-error-sampling \
            --error-boost 1.6 \
            --use-position-weights \
            --position-weight-scale 3.0 \
            --track-metric token \
            --use-wandb \
            --wandb-project gcode-large-models \
            --run-name ensemble-large-s${seed}

        echo ">>> Seed $seed complete!"
        echo ""
    done

    echo "Track B complete! Models saved to outputs/large_models/ensemble_large_s*/"
    echo ""
    echo "Evaluate ensemble with:"
    echo "PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_ensemble_simple.py \\"
    echo "    --checkpoints outputs/large_models/ensemble_large_s123/best_model.pt \\"
    echo "                  outputs/large_models/ensemble_large_s42/best_model.pt \\"
    echo "                  outputs/large_models/ensemble_large_s456/best_model.pt \\"
    echo "    --output reports/large_ensemble_results.json"
}

# ============================================================================
# Track C: Ablation Study
# ============================================================================
train_track_c() {
    echo "=============================================="
    echo "TRACK C: Ablation Study"
    echo "=============================================="
    echo "Running systematic ablations via individual runs"
    echo ""

    # Baseline (current best)
    echo ">>> Ablation: baseline (d=384, L=6, H=8)"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/baseline \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 384 --n-heads 8 --n-layers 6 \
        --dropout 0.32 --learning-rate 0.00024 --weight-decay 0.052 \
        --warmup-epochs 12 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.3 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name baseline

    # Ablation: d_model 512
    echo ">>> Ablation: d_model=512"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/dmodel_512 \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 512 --n-heads 8 --n-layers 6 \
        --dropout 0.35 --learning-rate 0.00020 --weight-decay 0.055 \
        --warmup-epochs 14 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.3 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name dmodel-512

    # Ablation: d_model 768
    echo ">>> Ablation: d_model=768"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/dmodel_768 \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 24 \
        --d-model 768 --n-heads 8 --n-layers 6 \
        --dropout 0.38 --learning-rate 0.00015 --weight-decay 0.06 \
        --warmup-epochs 16 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.25 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name dmodel-768

    # Ablation: n_layers 8
    echo ">>> Ablation: n_layers=8"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/layers_8 \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 384 --n-heads 8 --n-layers 8 \
        --dropout 0.35 --learning-rate 0.00022 --weight-decay 0.055 \
        --warmup-epochs 14 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.28 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name layers-8

    # Ablation: n_layers 10
    echo ">>> Ablation: n_layers=10"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/layers_10 \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 384 --n-heads 8 --n-layers 10 \
        --dropout 0.38 --learning-rate 0.00020 --weight-decay 0.06 \
        --warmup-epochs 15 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.25 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name layers-10

    # Ablation: n_heads 16
    echo ">>> Ablation: n_heads=16"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/heads_16 \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 384 --n-heads 16 --n-layers 6 \
        --dropout 0.32 --learning-rate 0.00024 --weight-decay 0.052 \
        --warmup-epochs 12 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.3 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name heads-16

    # Ablation: d_ff_multiplier 6
    echo ">>> Ablation: d_ff_multiplier=6"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/ff_6x \
        --seed 123 --max-epochs 120 --patience 35 --batch-size 32 \
        --d-model 384 --n-heads 8 --n-layers 6 --d-ff-multiplier 6.0 \
        --dropout 0.35 --learning-rate 0.00022 --weight-decay 0.055 \
        --warmup-epochs 12 --lr-scheduler cosine_restarts --restart-period 15 \
        --use-sensor-prior --sensor-prior-weight 0.51 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.3 --progressive-augmentation \
        --use-error-sampling --error-boost 1.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name ff-6x

    # Ablation: Combined best
    echo ">>> Ablation: Combined (d=512, L=8, H=16)"
    PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
        --split-dir $SPLIT_DIR \
        --vocab-path $VOCAB_PATH \
        --encoder-path $ENCODER_PATH \
        --output-dir outputs/large_models/ablation/combined \
        --seed 123 --max-epochs 150 --patience 40 --batch-size 24 \
        --d-model 512 --n-heads 16 --n-layers 8 \
        --dropout 0.36 --learning-rate 0.00018 --weight-decay 0.055 \
        --warmup-epochs 14 --lr-scheduler cosine_restarts --restart-period 18 \
        --use-sensor-prior --sensor-prior-weight 0.53 \
        --use-enhanced-encoder --use-multihead-pooling \
        --augment --augment-prob 0.28 --progressive-augmentation \
        --use-error-sampling --error-boost 1.6 \
        --use-position-weights --position-weight-scale 3.0 \
        --track-metric token --use-wandb --wandb-project gcode-ablation --run-name combined

    echo ""
    echo "Track C complete! Results in outputs/large_models/ablation/"
}

# ============================================================================
# Run selected track(s)
# ============================================================================

case $TRACK in
    a|A)
        train_track_a
        ;;
    b|B)
        train_track_b
        ;;
    c|C)
        train_track_c
        ;;
    all)
        train_track_a
        train_track_b
        train_track_c
        ;;
    *)
        echo "Unknown track: $TRACK"
        echo "Usage: $0 [a|b|c|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results saved to outputs/large_models/"
echo ""
echo "To evaluate all models:"
echo "  PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_checkpoint.py --checkpoint outputs/large_models/large_v1/best_model.pt"
echo "  PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_ensemble_simple.py --checkpoints outputs/large_models/ensemble_large_s*/best_model.pt"
