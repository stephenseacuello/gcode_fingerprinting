#!/usr/bin/env bash
# All 7 sensor modalities × 5 folds, on TRUE V8 per_row data.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"

MODALITIES=(accelerometer gyroscope magnetometer environmental color rms electrical)
for MOD in "${MODALITIES[@]}"; do
  for F in 1 2 3 4 5; do
    OUT="outputs/decoder20260511/ablations/sensor/zero_${MOD}/fold_${F}"
    if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
      echo "  skip zero_${MOD}/fold_${F}: already trained"
      continue
    fi
    echo
    echo "=== sensor ablation: zero ${MOD} fold ${F} ==="
    python3 scripts/evaluation/run_decoder_quick_test.py \
      --data_dir "outputs/decoder20260511/preprocessed_f98/per_row/fold_${F}" \
      --encoder_config f98_w256_s64 \
      --fold "${F}" \
      --vocab "${VOCAB}" \
      --output_dir "${OUT}" \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --batch_size 64 --lr 1e-4 --weight_decay 0.05 \
      --max_token_len 32 --seed 42 \
      --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
      --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
      --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
      --zero_modality_groups "${MOD}" \
      --legacy_weight 1.0 --digit_weight 1.0 \
      --device auto 2>&1 | tail -3
  done
done

echo "=== sensor ablation COMPLETE ==="
