#!/usr/bin/env bash
# Close the sensor-ablation matrix: gyroscope and color have V8 folds
# 2-5 from the cross-fold run but are missing fold_1 (the missing-
# modalities script only targets the other 5). This fills those 2 cells
# so every modality has a complete fold-1..5 V8 sweep.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"

for MODALITY in gyroscope color; do
  OUT="outputs/decoder20260511/ablations/sensor/zero_${MODALITY}/fold_1"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
    echo "  skip ${MODALITY} fold 1: already trained"
    continue
  fi
  echo
  echo "=== sensor ablation: zero ${MODALITY} fold 1 ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/per_row/fold_1" \
    --encoder_config f98_w256_s64 \
    --fold 1 \
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

echo
echo "=== gyro/color fold-1 COMPLETE — sensor matrix closed ==="
