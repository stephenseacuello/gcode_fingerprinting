#!/bin/bash
# Run missing baseline experiments for ablation study
# Usage: ./scripts/evaluation/run_missing_baselines.sh
#
# This runs 30 baseline experiments (10 models × 3 seeds) on 2 GPUs

set -e

# Configuration
DATA_DIR="outputs/7class_cascade_to_9class/9class_moddropout_final/data"
OUTPUT_BASE="outputs/ablation_study_2026_02_06/baseline"
SCRIPT="scripts/evaluation/run_baseline_models.py"

# Models to run
ML_MODELS="xgboost random_forest svm_rbf logistic_regression knn"
NN_MODELS="mlp cnn_1d lstm_simple transformer tcn"
SEEDS="42 123 456"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "============================================================"
echo "Running Missing Baseline Experiments"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo "ML models: $ML_MODELS"
echo "NN models: $NN_MODELS"
echo "Seeds: $SEEDS"
echo "============================================================"

# Check data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local model=$1
    local seed=$2
    local model_type=$3
    local gpu=$4

    local output_dir="$OUTPUT_BASE/$model_type/${model}_ds42_ms${seed}"

    if [ -f "$output_dir/results.json" ]; then
        echo "SKIP: $model seed=$seed (already completed)"
        return 0
    fi

    echo "RUN: $model seed=$seed on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" \
        --model "$model" \
        --seed "$seed"
}

# Count total experiments
total=0
for model in $ML_MODELS $NN_MODELS; do
    for seed in $SEEDS; do
        total=$((total + 1))
    done
done
echo "Total experiments: $total"
echo ""

# Run ML models (CPU-bound, run sequentially)
echo "============================================================"
echo "PHASE 1: ML Models (CPU-bound)"
echo "============================================================"
for model in $ML_MODELS; do
    for seed in $SEEDS; do
        run_experiment "$model" "$seed" "ml" "0"
    done
done

# Run NN models (GPU-bound, alternate between GPUs)
echo ""
echo "============================================================"
echo "PHASE 2: Neural Network Models (GPU-bound)"
echo "============================================================"

# Create arrays for parallel execution
declare -a jobs=()
gpu=0

for model in $NN_MODELS; do
    for seed in $SEEDS; do
        output_dir="$OUTPUT_BASE/nn/${model}_ds42_ms${seed}"

        if [ -f "$output_dir/results.json" ]; then
            echo "SKIP: $model seed=$seed (already completed)"
            continue
        fi

        echo "RUN: $model seed=$seed on GPU $gpu"

        CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT \
            --data-dir "$DATA_DIR" \
            --output-dir "$output_dir" \
            --model "$model" \
            --seed "$seed" &

        jobs+=($!)

        # Alternate GPUs
        gpu=$(( (gpu + 1) % 2 ))

        # Wait if we have 2 jobs running
        if [ ${#jobs[@]} -ge 2 ]; then
            wait ${jobs[0]}
            jobs=("${jobs[@]:1}")
        fi
    done
done

# Wait for remaining jobs
wait

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"

# Count completed
completed=$(find "$OUTPUT_BASE" -name "results.json" | wc -l)
echo "Completed: $completed / $total"

# List any failures
echo ""
echo "Checking for failures..."
for model in $ML_MODELS; do
    for seed in $SEEDS; do
        output_dir="$OUTPUT_BASE/ml/${model}_ds42_ms${seed}"
        if [ ! -f "$output_dir/results.json" ]; then
            echo "FAILED: $model seed=$seed (ml)"
        fi
    done
done
for model in $NN_MODELS; do
    for seed in $SEEDS; do
        output_dir="$OUTPUT_BASE/nn/${model}_ds42_ms${seed}"
        if [ ! -f "$output_dir/results.json" ]; then
            echo "FAILED: $model seed=$seed (nn)"
        fi
    done
done

echo ""
echo "Done!"
