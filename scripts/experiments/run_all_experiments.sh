#!/bin/bash
# Run All Experiments for G-code Fingerprinting
# Target: Lambda Labs GPU cluster
#
# Usage:
#   ./scripts/experiments/run_all_experiments.sh [experiment_type]
#
# experiment_type options:
#   ensemble     - Train ensemble with 3 seeds + learned weights
#   arch         - Architecture ablations
#   sensor       - Sensor ID ablations
#   modality     - Modality ablations
#   baselines    - Baseline comparisons
#   all          - Run all experiments (default)
#
# Author: Claude Code
# Date: January 2026

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Paths (adjust these for your setup)
DATA_DIR="${DATA_DIR:-outputs/processed_v3}"
ENCODER_PATH="${ENCODER_PATH:-outputs/encoder/best_model.pt}"
VOCAB_PATH="${VOCAB_PATH:-data/vocabulary_4digit_full.json}"
CONFIG_PATH="${CONFIG_PATH:-configs/best_model_config.json}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/experiments}"

# Experiment settings
MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-30}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42,123,456}"

# Parse argument
EXPERIMENT_TYPE="${1:-all}"

echo "========================================"
echo "G-code Fingerprinting Experiments"
echo "========================================"
echo "Data dir:     $DATA_DIR"
echo "Encoder:      $ENCODER_PATH"
echo "Vocab:        $VOCAB_PATH"
echo "Config:       $CONFIG_PATH"
echo "Output base:  $OUTPUT_BASE"
echo "Max epochs:   $MAX_EPOCHS"
echo "Patience:     $PATIENCE"
echo "Device:       $DEVICE"
echo "Experiment:   $EXPERIMENT_TYPE"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run ensemble training
run_ensemble() {
    echo ""
    echo "========================================"
    echo "RUNNING: Ensemble Training (3 seeds)"
    echo "========================================"

    python scripts/experiments/train_ensemble_seeds.py \
        --config "$CONFIG_PATH" \
        --data-dir "$DATA_DIR" \
        --encoder-path "$ENCODER_PATH" \
        --vocab-path "$VOCAB_PATH" \
        --output-dir "$OUTPUT_BASE/ensemble" \
        --seeds "$SEEDS" \
        --device "$DEVICE"
}

# Function to run architecture ablations
run_arch_ablations() {
    echo ""
    echo "========================================"
    echo "RUNNING: Architecture Ablations"
    echo "========================================"

    python scripts/experiments/run_architecture_ablations.py \
        --output-dir "$OUTPUT_BASE/architecture_ablations" \
        --split-dir "$DATA_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --encoder-path "$ENCODER_PATH" \
        --ablation all \
        --seed 42 \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE"
}

# Function to run sensor ablations
run_sensor_ablations() {
    echo ""
    echo "========================================"
    echo "RUNNING: Sensor ID Ablations"
    echo "========================================"

    python scripts/experiments/run_sensor_ablations.py \
        --config "$CONFIG_PATH" \
        --data-dir "$DATA_DIR" \
        --encoder-path "$ENCODER_PATH" \
        --vocab-path "$VOCAB_PATH" \
        --output-dir "$OUTPUT_BASE/sensor_ablations" \
        --ablation-type both \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --device "$DEVICE"
}

# Function to run modality ablations
run_modality_ablations() {
    echo ""
    echo "========================================"
    echo "RUNNING: Modality Ablations"
    echo "========================================"

    python scripts/experiments/run_modality_ablations.py \
        --config "$CONFIG_PATH" \
        --data-dir "$DATA_DIR" \
        --encoder-path "$ENCODER_PATH" \
        --vocab-path "$VOCAB_PATH" \
        --output-dir "$OUTPUT_BASE/modality_ablations" \
        --ablation-type groups \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --device "$DEVICE"
}

# Function to run baseline comparisons
run_baselines() {
    echo ""
    echo "========================================"
    echo "RUNNING: Baseline Comparisons"
    echo "========================================"

    python scripts/experiments/run_baseline_comparisons.py \
        --output-dir "$OUTPUT_BASE/baselines" \
        --split-dir "$DATA_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --encoder-path "$ENCODER_PATH" \
        --baseline all \
        --seeds 42 \
        --max-epochs "$MAX_EPOCHS"
}

# Run experiments based on type
case "$EXPERIMENT_TYPE" in
    ensemble)
        run_ensemble
        ;;
    arch)
        run_arch_ablations
        ;;
    sensor)
        run_sensor_ablations
        ;;
    modality)
        run_modality_ablations
        ;;
    baselines)
        run_baselines
        ;;
    all)
        run_ensemble
        run_arch_ablations
        run_sensor_ablations
        run_modality_ablations
        run_baselines
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        echo "Valid options: ensemble, arch, sensor, modality, baselines, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_BASE"
