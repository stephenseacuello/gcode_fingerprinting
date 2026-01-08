#!/bin/bash
# =============================================================================
# Multi-Seed Training Script for Statistical Rigor
# =============================================================================
#
# This script trains the best model configuration with multiple random seeds
# to compute confidence intervals and demonstrate reproducibility.
#
# Usage:
#   ./scripts/run_multiseed.sh                    # Run with default 5 seeds
#   ./scripts/run_multiseed.sh --seeds 42,123,456 # Custom seeds
#   ./scripts/run_multiseed.sh --config outputs/sweep_runs/best_config.json
#
# =============================================================================

set -e

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# Default Configuration
# =============================================================================

# Default seeds for reproducibility
DEFAULT_SEEDS="42,123,456,789,1337"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# Training parameters (can be overridden by config file or env vars)
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-4}"
N_HEADS="${N_HEADS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
DROPOUT="${DROPOUT:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Paths
SPLIT_DIR="${SPLIT_DIR:-outputs/processed_v3}"
VOCAB_PATH="${VOCAB_PATH:-data/vocabulary_4digit_full.json}"
ENCODER_PATH="${ENCODER_PATH:-outputs/mm_dtae_lstm_v2/best_model.pt}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/multiseed}"

# WandB settings
WANDB_PROJECT="${WANDB_PROJECT:-gcode-multiseed}"
USE_WANDB="${USE_WANDB:-true}"

# =============================================================================
# Parse Arguments
# =============================================================================

CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --no-wandb)
            USE_WANDB="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seeds SEEDS        Comma-separated list of seeds (default: $DEFAULT_SEEDS)"
            echo "  --config FILE        Load hyperparameters from JSON config file"
            echo "  --output-dir DIR     Output directory (default: outputs/multiseed)"
            echo "  --max-epochs N       Maximum epochs (default: 150)"
            echo "  --no-wandb           Disable WandB logging"
            echo ""
            echo "Environment Variables:"
            echo "  D_MODEL, N_LAYERS, N_HEADS, LEARNING_RATE, DROPOUT"
            echo "  SPLIT_DIR, VOCAB_PATH, ENCODER_PATH"
            echo "  WANDB_PROJECT, WANDB_ENTITY"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Load Config File if Provided
# =============================================================================

if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "Loading configuration from: $CONFIG_FILE"

    # Extract values using Python
    D_MODEL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('d_model', $D_MODEL))")
    N_LAYERS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('n_layers', $N_LAYERS))")
    N_HEADS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('n_heads', $N_HEADS))")
    LEARNING_RATE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('learning_rate', $LEARNING_RATE))")
    DROPOUT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('dropout', $DROPOUT))")
fi

# =============================================================================
# Create Output Directory
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "seeds": "$SEEDS",
    "d_model": $D_MODEL,
    "n_layers": $N_LAYERS,
    "n_heads": $N_HEADS,
    "learning_rate": $LEARNING_RATE,
    "dropout": $DROPOUT,
    "max_epochs": $MAX_EPOCHS,
    "patience": $PATIENCE,
    "batch_size": $BATCH_SIZE,
    "split_dir": "$SPLIT_DIR",
    "vocab_path": "$VOCAB_PATH",
    "encoder_path": "$ENCODER_PATH",
    "timestamp": "$TIMESTAMP"
}
EOF

echo "============================================================"
echo "MULTI-SEED TRAINING"
echo "============================================================"
echo "Seeds: $SEEDS"
echo "D_MODEL: $D_MODEL"
echo "N_LAYERS: $N_LAYERS"
echo "N_HEADS: $N_HEADS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "DROPOUT: $DROPOUT"
echo "MAX_EPOCHS: $MAX_EPOCHS"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# =============================================================================
# Training Loop
# =============================================================================

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
RESULTS_FILE="$OUTPUT_DIR/all_results.json"

# Initialize results file
echo "[]" > "$RESULTS_FILE"

for seed in "${SEED_ARRAY[@]}"; do
    echo ""
    echo "============================================================"
    echo "Training with seed: $seed"
    echo "============================================================"

    SEED_DIR="$OUTPUT_DIR/seed_$seed"
    mkdir -p "$SEED_DIR"

    # Build command
    CMD=".venv/bin/python scripts/train_sensor_multihead.py"
    CMD="$CMD --seed $seed"
    CMD="$CMD --split-dir $SPLIT_DIR"
    CMD="$CMD --vocab-path $VOCAB_PATH"
    CMD="$CMD --encoder-path $ENCODER_PATH"
    CMD="$CMD --output-dir $SEED_DIR"
    CMD="$CMD --d-model $D_MODEL"
    CMD="$CMD --n-layers $N_LAYERS"
    CMD="$CMD --n-heads $N_HEADS"
    CMD="$CMD --dropout $DROPOUT"
    CMD="$CMD --learning-rate $LEARNING_RATE"
    CMD="$CMD --max-epochs $MAX_EPOCHS"
    CMD="$CMD --patience $PATIENCE"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --use-enhanced-encoder"
    CMD="$CMD --use-multiscale-encoder"
    CMD="$CMD --use-multihead-pooling"
    CMD="$CMD --augment"
    CMD="$CMD --augment-prob 0.3"
    CMD="$CMD --use-focal-loss"
    CMD="$CMD --focal-gamma 2.0"
    CMD="$CMD --curriculum"

    if [[ "$USE_WANDB" == "true" ]]; then
        CMD="$CMD --use-wandb"
        CMD="$CMD --wandb-project $WANDB_PROJECT"
        CMD="$CMD --run-name seed_${seed}"
    fi

    # Run training
    echo "Running: $CMD"
    eval "$CMD" 2>&1 | tee "$SEED_DIR/training.log"

    # Check if results exist
    if [[ -f "$SEED_DIR/results.json" ]]; then
        echo "Seed $seed completed successfully"
    else
        echo "WARNING: Seed $seed may have failed - no results.json found"
    fi
done

# =============================================================================
# Compute Statistics
# =============================================================================

echo ""
echo "============================================================"
echo "Computing Statistics"
echo "============================================================"

python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
from scipy import stats
from pathlib import Path
import os

output_dir = os.environ.get('OUTPUT_DIR', 'outputs/multiseed')
seeds_str = os.environ.get('SEEDS', '42,123,456,789,1337')
seeds = [int(s.strip()) for s in seeds_str.split(',')]

results = []
token_accs = []
seq_accs = []

for seed in seeds:
    results_path = Path(output_dir) / f'seed_{seed}' / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            r = json.load(f)
            results.append({
                'seed': seed,
                'token_accuracy': r.get('val_token_accuracy', r.get('token_accuracy', 0)),
                'sequence_accuracy': r.get('val_sequence_accuracy', r.get('sequence_accuracy', 0)),
            })
            token_accs.append(results[-1]['token_accuracy'])
            seq_accs.append(results[-1]['sequence_accuracy'])
    else:
        print(f"Warning: No results for seed {seed}")

if len(token_accs) < 2:
    print("Not enough results to compute statistics")
    exit(0)

# Compute statistics
def compute_ci(values, confidence=0.95):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * se
    return mean, std, ci

token_mean, token_std, token_ci = compute_ci(token_accs)
seq_mean, seq_std, seq_ci = compute_ci(seq_accs)

print(f"\n{'='*60}")
print("MULTI-SEED RESULTS")
print('='*60)

print(f"\nToken Accuracy:")
print(f"  Mean: {token_mean*100:.2f}%")
print(f"  Std:  {token_std*100:.2f}%")
print(f"  95% CI: ±{token_ci*100:.2f}%")
print(f"  Range: [{(token_mean-token_ci)*100:.2f}%, {(token_mean+token_ci)*100:.2f}%]")

print(f"\nSequence Accuracy:")
print(f"  Mean: {seq_mean*100:.2f}%")
print(f"  Std:  {seq_std*100:.2f}%")
print(f"  95% CI: ±{seq_ci*100:.2f}%")
print(f"  Range: [{(seq_mean-seq_ci)*100:.2f}%, {(seq_mean+seq_ci)*100:.2f}%]")

print(f"\nPer-seed results:")
for r in results:
    print(f"  Seed {r['seed']}: token={r['token_accuracy']*100:.2f}%, seq={r['sequence_accuracy']*100:.2f}%")

# Save statistics
stats_results = {
    'n_seeds': len(results),
    'seeds': seeds,
    'per_seed_results': results,
    'token_accuracy': {
        'mean': token_mean,
        'std': token_std,
        'ci_95': token_ci,
        'values': token_accs,
    },
    'sequence_accuracy': {
        'mean': seq_mean,
        'std': seq_std,
        'ci_95': seq_ci,
        'values': seq_accs,
    },
    'summary': f"{token_mean*100:.2f}% ± {token_ci*100:.2f}% (95% CI, n={len(results)})",
}

with open(Path(output_dir) / 'statistics.json', 'w') as f:
    json.dump(stats_results, f, indent=2)

print(f"\nStatistics saved to {output_dir}/statistics.json")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Multi-seed training complete!"
echo "Results in: $OUTPUT_DIR"
echo "============================================================"
