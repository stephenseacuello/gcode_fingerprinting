#!/usr/bin/env bash
# Phase B4: cross-fold sensor ablation for the top-2 modalities (gyro + color).
#
# We already have fold-1 ablations for all 7 groups. To get error bars
# for the manuscript, run folds 2-5 for the two most-impactful groups.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"

for MODALITY in gyroscope color; do
  for F in 2 3 4 5; do
    OUT="outputs/decoder20260511/ablations/sensor/zero_${MODALITY}/fold_${F}"
    if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
      echo "  skip ${MODALITY} fold ${F}: already trained"
      continue
    fi
    echo
    echo "=== sensor ablation: zero ${MODALITY} fold ${F} ==="
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
      --use_window_position false \
      --multi_window_context 0 --window_dropout 0.0 \
      --zero_modality_groups "${MODALITY}" \
      --legacy_weight 1.0 --digit_weight 1.0 \
      --device auto 2>&1 | tail -5
  done
done

echo
echo "=== cross-fold sensor ablation COMPLETE ==="
