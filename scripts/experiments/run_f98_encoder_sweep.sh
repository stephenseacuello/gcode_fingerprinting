#!/bin/bash
################################################################################
# F98 Encoder Hyperparameter Sweep
#
# Trains f98_w256_s64 encoders with various hyperparameter combinations.
# The default encoder used d_model=256, lr=1e-3, dropout=0.2, n_heads=4.
# This sweep explores d_model={256,384}, lr={5e-4,1e-3,2e-3},
# dropout={0.1,0.2,0.3}, n_heads={4,8} = 36 configs x 5 folds.
#
# Output: outputs/decoder20260304/encoder_sweep_f98/
# Each config: {d_model}_{lr}_{dropout}_{n_heads}/fold_{1-5}/
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="src:$PYTHONPATH"

PREPROCESS_BASE="outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv"
OUTPUT_BASE="outputs/decoder20260304/encoder_sweep_f98"
RESULTS_CSV="${OUTPUT_BASE}/sweep_results.csv"

mkdir -p "$OUTPUT_BASE"

echo "================================================================================"
echo "F98 ENCODER HYPERPARAMETER SWEEP"
echo "================================================================================"
echo "Started at: $(date)"
echo "Output: $OUTPUT_BASE"
echo ""

# Initialize results CSV
echo "d_model,n_heads,lr,dropout,fold,val_acc,test_acc,train_time_s" > "$RESULTS_CSV"

# Sweep grid
D_MODELS=(256 384)
N_HEADS_LIST=(4 8)
LRS=(0.0005 0.001 0.002)
DROPOUTS=(0.1 0.2 0.3)
MAX_PARALLEL=5  # run 5 folds in parallel per config

config_idx=0
total_configs=$(( ${#D_MODELS[@]} * ${#N_HEADS_LIST[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} ))

for D_MODEL in "${D_MODELS[@]}"; do
  for N_HEADS in "${N_HEADS_LIST[@]}"; do
    # Skip invalid combos (d_model must be divisible by n_heads)
    if (( D_MODEL % N_HEADS != 0 )); then
      echo "SKIP: d_model=$D_MODEL not divisible by n_heads=$N_HEADS"
      continue
    fi

    for LR in "${LRS[@]}"; do
      for DROPOUT in "${DROPOUTS[@]}"; do
        config_idx=$((config_idx + 1))
        CONFIG_NAME="dm${D_MODEL}_nh${N_HEADS}_lr${LR}_do${DROPOUT}"
        CONFIG_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

        echo ""
        echo "[$config_idx/$total_configs] Config: $CONFIG_NAME"
        echo "  d_model=$D_MODEL, n_heads=$N_HEADS, lr=$LR, dropout=$DROPOUT"

        # Run all 5 folds in parallel
        pids=()
        for FOLD in 1 2 3 4 5; do
          DATA_DIR="${PREPROCESS_BASE}/fold_${FOLD}/preprocessed"
          FOLD_OUT="${CONFIG_DIR}/fold_${FOLD}"

          # Skip if already completed
          if [ -f "${FOLD_OUT}/checkpoint/best_model.pt" ]; then
            echo "  Fold $FOLD: already done, skipping"
            continue
          fi

          mkdir -p "$FOLD_OUT"

          python3 scripts/evaluation/run_9class_direct.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$FOLD_OUT" \
            --iteration "sweep_${CONFIG_NAME}_fold${FOLD}" \
            --d_model "$D_MODEL" \
            --n_heads "$N_HEADS" \
            --lr "$LR" \
            --dropout "$DROPOUT" \
            --max_epochs 200 \
            --patience 40 \
            --seed 42 \
            > "${FOLD_OUT}/train.log" 2>&1 &

          pids+=($!)
        done

        # Wait for all folds of this config
        for pid in "${pids[@]}"; do
          wait "$pid" || echo "  WARNING: Process $pid failed"
        done

        # Collect results
        for FOLD in 1 2 3 4 5; do
          FOLD_OUT="${CONFIG_DIR}/fold_${FOLD}"
          RESULTS_JSON="${FOLD_OUT}/results.json"
          if [ -f "$RESULTS_JSON" ]; then
            VAL_ACC=$(python3 -c "import json; r=json.load(open('$RESULTS_JSON')); print(r['val_best_test']['accuracy'])" 2>/dev/null || echo "N/A")
            TEST_ACC=$(python3 -c "import json; r=json.load(open('$RESULTS_JSON')); print(r['test_best_test']['accuracy'])" 2>/dev/null || echo "N/A")
            TRAIN_TIME=$(python3 -c "import json; r=json.load(open('$RESULTS_JSON')); print(r.get('train_time_seconds', 'N/A'))" 2>/dev/null || echo "N/A")
            echo "$D_MODEL,$N_HEADS,$LR,$DROPOUT,$FOLD,$VAL_ACC,$TEST_ACC,$TRAIN_TIME" >> "$RESULTS_CSV"
            echo "  Fold $FOLD: val_acc=$VAL_ACC, test_acc=$TEST_ACC"
          else
            echo "  Fold $FOLD: FAILED (no results.json)"
            echo "$D_MODEL,$N_HEADS,$LR,$DROPOUT,$FOLD,FAILED,FAILED,FAILED" >> "$RESULTS_CSV"
          fi
        done

      done
    done
  done
done

echo ""
echo "================================================================================"
echo "SWEEP COMPLETE: $(date)"
echo "Results: $RESULTS_CSV"
echo "================================================================================"

# Summary: find best config by mean val_acc across folds
python3 -c "
import csv
from collections import defaultdict

results = defaultdict(list)
with open('$RESULTS_CSV') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['val_acc'] == 'FAILED' or row['val_acc'] == 'N/A':
            continue
        key = f\"dm{row['d_model']}_nh{row['n_heads']}_lr{row['lr']}_do{row['dropout']}\"
        results[key].append(float(row['val_acc']))

print('\nTop 10 configs by mean val_acc (val-best checkpoint):')
print(f'{\"Config\":<40} {\"Mean\":>8} {\"Min\":>8} {\"Max\":>8} {\"N\":>4}')
print('-' * 68)
ranked = sorted(results.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
for cfg, accs in ranked[:10]:
    mean_acc = sum(accs)/len(accs)
    print(f'{cfg:<40} {mean_acc:>8.4f} {min(accs):>8.4f} {max(accs):>8.4f} {len(accs):>4}')
"
