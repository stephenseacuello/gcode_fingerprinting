#!/usr/bin/env bash
# Phase-5 5-fold sweep on V8 per_row data (no shortcuts).
#
# Produces per-fold metrics so the manuscript can quote mean ± std instead of
# a single fold-1 point estimate.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/per_row_5fold}"

for F in 1 2 3 4 5; do
  DATA_DIR="outputs/decoder20260511/preprocessed_f98/per_row/fold_${F}"
  OUT="${OUT_ROOT}/fold_${F}"
  echo
  echo "=================================================="
  echo "  V8 per_row no_shortcuts: fold ${F} (${EPOCHS} epochs)"
  echo "  out -> ${OUT}"
  echo "=================================================="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${DATA_DIR}" \
    --encoder_config f98_w256_s64 \
    --fold "${F}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --max_token_len 32 \
    --seed 42 \
    --d_model 384 \
    --n_layers 8 \
    --n_heads 12 \
    --dropout 0.1 \
    --memory_pos_encoding true \
    --use_sensor_prior true \
    --grammar_constraint true \
    --use_window_position false \
    --multi_window_context 0 \
    --window_dropout 0.0 \
    --legacy_weight 1.0 \
    --digit_weight 1.0 \
    --device auto 2>&1 | tail -5
done

echo
echo "=================================================="
echo "  Aggregating 5-fold results"
echo "=================================================="
python3 scripts/analysis/aggregate_v8_results.py
echo
echo "DONE."
