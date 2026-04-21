#!/bin/bash
# =============================================================================
# Decoder Sweep: 3 feature configs × 5 folds = 15 runs
# Trains SensorMultiHeadDecoder on frozen 20260225 encoders
#
# Usage:
#   bash scripts/experiments/run_decoder_sweep.sh
#   # Or in tmux for persistence:
#   tmux new-session -d -s decoder_sweep 'bash scripts/experiments/run_decoder_sweep.sh 2>&1 | tee outputs/decoder20260303/sweep_log.txt'
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to repo root

VOCAB="data/gcode_vocab_712.json"
EPOCHS=150
PATIENCE=20
BATCH=32
BASE="outputs/experiments_2026_02_25"
OUT="outputs/decoder20260303"

# Feature configurations
declare -A CONFIGS
CONFIGS[feat110]="full_w128_s32_cv"
CONFIGS[feat98]="no_proximity_no_pressure_w64_s16_cv"
CONFIGS[feat56]="no_proximity_no_pressure_no_color_no_magnetometer_w64_s16_cv"

# Feature counts for display
declare -A FEAT_COUNTS
FEAT_COUNTS[feat110]=110
FEAT_COUNTS[feat98]=98
FEAT_COUNTS[feat56]=56

echo "============================================================"
echo "Decoder Sweep: 3 configs × 5 folds = 15 runs"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Vocab: $VOCAB"
echo "Epochs: $EPOCHS, Patience: $PATIENCE, Batch: $BATCH"
echo "============================================================"
echo ""

TOTAL=15
DONE=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)

for config_key in feat110 feat98 feat56; do
    config_name="${CONFIGS[$config_key]}"
    feat_count="${FEAT_COUNTS[$config_key]}"

    echo "──────────────────────────────────────────────────────────"
    echo "Config: ${config_key} (${feat_count} features)"
    echo "Encoder: ${config_name}"
    echo "──────────────────────────────────────────────────────────"

    for fold in 1 2 3 4 5; do
        output_dir="$OUT/${config_key}/fold_${fold}"
        run_label="${config_key}/fold_${fold}"

        # Skip if already completed
        if [ -f "$output_dir/results/metrics.json" ]; then
            echo "[SKIP] $run_label (already done)"
            SKIPPED=$((SKIPPED + 1))
            DONE=$((DONE + 1))
            continue
        fi

        echo ""
        echo "[RUN] $run_label ($((DONE + 1))/$TOTAL)"
        echo "  Data: $BASE/$config_name/fold_$fold/preprocessed"
        echo "  Output: $output_dir"

        RUN_START=$(date +%s)

        if python3 scripts/evaluation/run_decoder_quick_test.py \
            --data_dir "$BASE/$config_name/fold_$fold/preprocessed" \
            --encoder_ckpt "$BASE/$config_name/fold_$fold/encoder/checkpoint/best_model.pt" \
            --vocab "$VOCAB" \
            --output_dir "$output_dir" \
            --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH; then

            RUN_END=$(date +%s)
            RUN_ELAPSED=$((RUN_END - RUN_START))
            echo "[DONE] $run_label (${RUN_ELAPSED}s)"
            DONE=$((DONE + 1))
        else
            echo "[FAIL] $run_label"
            FAILED=$((FAILED + 1))
            DONE=$((DONE + 1))
        fi
    done
    echo ""
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "Sweep Complete"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m)"
echo "Completed: $((DONE - SKIPPED - FAILED)), Skipped: $SKIPPED, Failed: $FAILED"
echo "============================================================"
