#!/usr/bin/env bash
# Phase B2: V8 per_row 5-fold WITH window-position shortcut enabled.
#
# This is the cleanest "is the shortcut really contributing?" experiment.
# Same data, same model, only difference is use_window_position=true.
# Compare against per_row_5fold (use_window_position=false) row-by-row.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/per_row_5fold_with_shortcuts}"

for F in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${F}"
  echo
  echo "=== V8 per_row WITH shortcuts fold ${F} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/per_row/fold_${F}" \
    --encoder_config f98_w256_s64 \
    --fold "${F}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE:-75}" \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --max_token_len 32 \
    --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 \
    --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position true \
    --multi_window_context 2 \
    --window_dropout 0.1 \
    --legacy_weight 1.0 --digit_weight 1.0 \
    --device auto 2>&1 | tail -5
done

echo
echo "=== with_shortcuts 5-fold COMPLETE ==="
python3 scripts/analysis/aggregate_v8_results.py
