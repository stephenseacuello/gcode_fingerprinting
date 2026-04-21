#!/bin/bash
# =============================================================================
# Curriculum Learning Test: 3-phase on 110_w128_s32 × 5 folds
# Compares to baseline (no curriculum) which achieved 81.8% token acc.
#
# Usage:
#   bash scripts/experiments/run_curriculum_test.sh
#   # Or in tmux:
#   tmux new-session -d -s curriculum 'bash scripts/experiments/run_curriculum_test.sh 2>&1 | tee outputs/decoder20260303/curriculum_log.txt'
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to repo root

VOCAB="data/gcode_vocab_712.json"
EPOCHS=150
PATIENCE=20
BATCH=32
BASE="outputs/experiments_2026_02_25"
OUT="outputs/decoder20260303/curriculum_3phase"
CONFIG_NAME="full_w128_s32_cv"

echo "============================================================"
echo "Curriculum Learning Test: 3-phase × 5 folds"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Encoder: ${CONFIG_NAME} (110 features)"
echo "Vocab: $VOCAB"
echo "Epochs: $EPOCHS, Patience: $PATIENCE, Batch: $BATCH"
echo "Curriculum: 3phase"
echo "  Phase 1 (epochs 1-30): structure only"
echo "  Phase 2 (epochs 31-80): + digit loss"
echo "  Phase 3 (epochs 81+): + legacy loss"
echo "============================================================"
echo ""

TOTAL=5
DONE=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)

for fold in 1 2 3 4 5; do
    output_dir="$OUT/fold_${fold}"
    run_label="curriculum_3phase/fold_${fold}"

    # Skip if already completed
    if [ -f "$output_dir/results/metrics.json" ]; then
        echo "[SKIP] $run_label (already done)"
        SKIPPED=$((SKIPPED + 1))
        DONE=$((DONE + 1))
        continue
    fi

    echo ""
    echo "[RUN] $run_label ($((DONE + 1))/$TOTAL)"
    echo "  Data: $BASE/$CONFIG_NAME/fold_$fold/preprocessed"
    echo "  Output: $output_dir"

    RUN_START=$(date +%s)

    if python3 scripts/evaluation/run_decoder_quick_test.py \
        --data_dir "$BASE/$CONFIG_NAME/fold_$fold/preprocessed" \
        --encoder_ckpt "$BASE/$CONFIG_NAME/fold_$fold/encoder/checkpoint/best_model.pt" \
        --vocab "$VOCAB" \
        --output_dir "$output_dir" \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH \
        --curriculum 3phase; then

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

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Curriculum Test Complete"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m)"
echo "Completed: $((DONE - SKIPPED - FAILED)), Skipped: $SKIPPED, Failed: $FAILED"
echo "============================================================"

# Print summary of all fold results
echo ""
echo "Per-fold results:"
for fold in 1 2 3 4 5; do
    metrics="$OUT/fold_${fold}/results/metrics.json"
    if [ -f "$metrics" ]; then
        echo "  Fold $fold: $(python3 -c "
import json
m = json.load(open('$metrics'))['test_metrics']
print(f\"token={m['token_accuracy']:.1%} seq={m['sequence_accuracy']:.1%} type={m['type_accuracy']:.1%} cmd={m['command_accuracy']:.1%} param={m['param_type_accuracy']:.1%}\")
")"
    fi
done
