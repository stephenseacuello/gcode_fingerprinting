#!/usr/bin/env bash
# Cross-fold sensor ablation for the 5 modalities that only had V7-legacy
# coverage so far (accelerometer, magnetometer, environmental, rms, electrical).
# gyroscope and color already have V8 folds 2-5 from the existing
# run_sensor_ablation_v8_cross_fold.sh; here we close the gap so the manuscript
# can report full 5-fold cross-fold ablation for every modality group.
#
# Each cell trains the per_row baseline config with the specified modality
# zeroed at the encoder input.
#
# Fold order: 1 first (fold-1 pilots for the 5 missing modalities), then
# 2-5 for the same modalities so the table can be filled incrementally.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"

# These 5 only have V7-legacy folders so far.
MODALITIES=(accelerometer magnetometer environmental rms electrical)

# fold 1 first so the manuscript gets fold-1 pilots quickly,
# then folds 2-5 for the same modalities.
for F in 1 2 3 4 5; do
  for MODALITY in "${MODALITIES[@]}"; do
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
      --epochs "${EPOCHS}" --patience "${PATIENCE}" \
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
echo "=== sensor ablation missing-modalities COMPLETE ==="
