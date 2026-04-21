#!/bin/bash
# ── Decoder Ablations (POST_SWEEP_PLAN Sections 2A, 2B, 2C) ──
#
# Runs decoder ablation experiments across all 5 folds.
# Each ablation toggles one feature from V7 best config.
# Uses run_decoder_quick_test.py with V7 args + one override.
#
# Usage:
#   bash scripts/experiments/run_decoder_ablations.sh [--gpu GPU_ID] [--ablations 2A 2B 2C]
#
# Author: Claude Code
# Date: March 2026

set -euo pipefail

# ── Parse args ──
GPU_ID=${GPU_ID:-0}
ABLATIONS="${ABLATIONS:-2A 2B 2C}"
FOLDS="${FOLDS:-1 2 3 4 5}"
MAX_PARALLEL=${MAX_PARALLEL:-2}
DRY_RUN=${DRY_RUN:-0}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --ablations) ABLATIONS="$2"; shift 2 ;;
        --folds) FOLDS="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DECODER_SCRIPT="$REPO_ROOT/scripts/evaluation/run_decoder_quick_test.py"
OUTPUT_BASE="$REPO_ROOT/outputs/decoder20260304/ablations"
VOCAB="$REPO_ROOT/data/gcode_vocab_712.json"
LOG_FILE="$OUTPUT_BASE/ablation_runs.log"

mkdir -p "$OUTPUT_BASE"

# ── V7 Base Config (from fold 5 best trial) ──
# These are the V7 defaults that every ablation starts from.
V7_BASE=(
    --encoder_config v7_f110_w256_s64
    --vocab "$VOCAB"
    --epochs 500
    --patience 120
    --batch_size 32
    --lr 0.0003
    --max_token_len 16
    --seed 42
    --curriculum none
    --digit_weight 15.0
    --legacy_weight 1.0
    --label_smoothing 0.1
    --scheduled_sampling 0.5
    --focal_gamma 2.0
    --d_model 384
    --n_layers 8
    --n_heads 12
    --dropout 0.1
    --memory_pos_encoding true
    --grammar_constraint true
    --multi_window_context 2
    --weight_decay 0.1
    --warmup_epochs 10
    --use_window_position true
    --use_sensor_prior true
    --window_dropout 0.1
    --beam_width 0
)

# ── Ablation Definitions ──
# Each ablation: NAME OVERRIDE_ARGS...
# 2A: Data pipeline ablations
ABLATION_2A_1="2A1_no_window_pos --use_window_position false"
ABLATION_2A_2="2A2_no_mwc_neighbors --multi_window_context 0"
ABLATION_2A_3="2A3_no_scheduled_sampling --scheduled_sampling 0.0"
ABLATION_2A_4="2A4_all_data_off --use_window_position false --multi_window_context 0"

# 2B: Architecture ablations
ABLATION_2B_1="2B1_no_mpe --memory_pos_encoding false"
ABLATION_2B_2="2B2_no_sensor_prior --use_sensor_prior false"
ABLATION_2B_3="2B3_no_focal --focal_gamma 0.0"
ABLATION_2B_4="2B4_no_grammar --grammar_constraint false"
ABLATION_2B_5="2B5_no_label_smoothing --label_smoothing 0.0"

# 2C: Capacity ablations
ABLATION_2C_1="2C1_shallow --n_layers 2"
ABLATION_2C_2="2C2_narrow --d_model 192 --n_heads 6"
ABLATION_2C_3="2C3_fewer_heads --n_heads 4"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_ablation() {
    local name="$1"
    shift
    local overrides=("$@")
    local fold="$FOLD"
    local out_dir="$OUTPUT_BASE/${name}/fold_${fold}"

    # Skip if already done
    if [[ -f "$out_dir/results/metrics.json" ]]; then
        log_msg "SKIP $name fold $fold (already done)"
        return 0
    fi

    log_msg "START $name fold $fold"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  DRY RUN: CUDA_VISIBLE_DEVICES=$GPU_ID python $DECODER_SCRIPT ${V7_BASE[*]} --fold $fold --output_dir $out_dir ${overrides[*]}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID /home/seacuello/Documents/venv/bin/python "$DECODER_SCRIPT" \
        "${V7_BASE[@]}" \
        --fold "$fold" \
        --output_dir "$out_dir" \
        "${overrides[@]}" \
        2>&1 | tail -5

    if [[ -f "$out_dir/results/metrics.json" ]]; then
        local seq_acc=$(python3 -c "import json; m=json.load(open('$out_dir/results/metrics.json')); print(f\"{m['test_metrics']['sequence_accuracy']:.1%}\")")
        log_msg "DONE $name fold $fold: seq_acc=$seq_acc"
    else
        log_msg "FAIL $name fold $fold (no metrics.json)"
    fi
}

# ── Build ablation list ──
ABLATION_LIST=()
if [[ "$ABLATIONS" == *"2A"* ]]; then
    ABLATION_LIST+=("$ABLATION_2A_1" "$ABLATION_2A_2" "$ABLATION_2A_3" "$ABLATION_2A_4")
fi
if [[ "$ABLATIONS" == *"2B"* ]]; then
    ABLATION_LIST+=("$ABLATION_2B_1" "$ABLATION_2B_2" "$ABLATION_2B_3" "$ABLATION_2B_4" "$ABLATION_2B_5")
fi
if [[ "$ABLATIONS" == *"2C"* ]]; then
    ABLATION_LIST+=("$ABLATION_2C_1" "$ABLATION_2C_2" "$ABLATION_2C_3")
fi

log_msg "="
log_msg "Decoder Ablation Suite"
log_msg "Ablations: ${#ABLATION_LIST[@]} conditions × $(echo $FOLDS | wc -w) folds"
log_msg "GPU: $GPU_ID, Max parallel: $MAX_PARALLEL"
log_msg "="

# ── Run ──
RUNNING_PIDS=()

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; do
        local new_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        RUNNING_PIDS=("${new_pids[@]}")
        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; then
            sleep 10
        fi
    done
}

for FOLD in $FOLDS; do
    for ablation_def in "${ABLATION_LIST[@]}"; do
        # Parse: first word is name, rest are overrides
        read -r name rest <<< "$ablation_def"
        # shellcheck disable=SC2086
        overrides=($rest)

        wait_for_slot
        run_ablation "$name" "${overrides[@]}" &
        RUNNING_PIDS+=($!)

        # Small delay to stagger GPU memory allocation
        sleep 2
    done
done

# Wait for all remaining
wait

# ── Aggregate Results ──
log_msg ""
log_msg "AGGREGATING RESULTS..."
python3 - <<'PYEOF'
import json
import os
from pathlib import Path
from collections import defaultdict

base = Path("outputs/decoder20260304/ablations")
results = defaultdict(dict)

for condition_dir in sorted(base.iterdir()):
    if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
        continue
    for fold_dir in sorted(condition_dir.iterdir()):
        if not fold_dir.is_dir():
            continue
        metrics_path = fold_dir / 'results' / 'metrics.json'
        if metrics_path.exists():
            m = json.load(open(metrics_path))
            fold = fold_dir.name.replace('fold_', '')
            results[condition_dir.name][fold] = m.get('test_metrics', {})

# Print summary table
print(f"\n{'Condition':<30} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8} {'Mean':>8} {'Std':>8}")
print("-" * 100)
for name in sorted(results.keys()):
    folds = results[name]
    seq_accs = [folds[str(f)].get('sequence_accuracy', 0) for f in range(1,6) if str(f) in folds]
    fold_strs = []
    for f in range(1, 6):
        if str(f) in folds:
            fold_strs.append(f"{folds[str(f)].get('sequence_accuracy', 0):.1%}")
        else:
            fold_strs.append("  -   ")
    if seq_accs:
        import numpy as np
        mean = np.mean(seq_accs)
        std = np.std(seq_accs)
        print(f"{name:<30} {' '.join(f'{s:>8}' for s in fold_strs)} {mean:>7.1%} {std:>7.1%}")

# Save aggregated JSON
summary = {}
for name, folds in results.items():
    seq_accs = [folds[str(f)].get('sequence_accuracy', 0) for f in range(1,6) if str(f) in folds]
    tok_accs = [folds[str(f)].get('token_accuracy', 0) for f in range(1,6) if str(f) in folds]
    if seq_accs:
        import numpy as np
        summary[name] = {
            'mean_seq': float(np.mean(seq_accs)),
            'std_seq': float(np.std(seq_accs)),
            'mean_tok': float(np.mean(tok_accs)),
            'folds': {k: v for k, v in folds.items()},
        }

with open(base / 'ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {base / 'ablation_summary.json'}")
PYEOF

log_msg "ALL DONE"
