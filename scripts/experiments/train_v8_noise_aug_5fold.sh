#!/usr/bin/env bash
# Phase B3: V8 per_row 5-fold WITH noise augmentation.
#
# Tests whether sensor noise + token masking improves numeric accuracy
# beyond the no-shortcut baseline.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/per_row_5fold_noise_aug}"

for F in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${F}"
  echo
  echo "=== V8 per_row no_shortcuts + noise_aug fold ${F} ==="
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
    --dropout 0.15 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false \
    --multi_window_context 0 \
    --window_dropout 0.0 \
    --noise_scale 0.02 \
    --oversample_rare true \
    --legacy_weight 1.0 --digit_weight 1.0 \
    --device auto 2>&1 | tail -5
done

echo
echo "=== noise_aug 5-fold COMPLETE ==="
python3 scripts/analysis/aggregate_v8_results.py
