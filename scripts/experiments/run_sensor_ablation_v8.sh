#!/usr/bin/env bash
# Phase-6 (decoder20260511) sensor ablation: leave-one-modality-out.
#
# For each available sensor modality group, train a V8 decoder with that group
# zeroed at encoder input. Compare against the f98 baseline (all groups
# active). Modalities are zeroed at the SENSOR INPUT before the frozen encoder
# pass, which is why we don't need to retrain the encoder.
#
# Note: the f98 encoder excluded proximity + pressure at TRAINING time, so the
# 'environmental' group here is only Temperature (proximity/pressure not in data).
#
# Output: outputs/decoder20260511/ablations/sensor/{group}/fold_1/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-outputs/decoder20260511/preprocessed_f98/per_row/fold_1}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-30}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/ablations/sensor}"

# Groups present in the f98 encoder. 'environmental' contains only Temperature
# in this configuration (proximity + pressure excluded at preprocessing).
MODALITIES=(accelerometer gyroscope magnetometer environmental color rms electrical)

for MODALITY in "${MODALITIES[@]}"; do
  OUT="${OUT_ROOT}/zero_${MODALITY}/fold_1"
  echo
  echo "=================================================="
  echo "  Sensor ablation: zero=${MODALITY}"
  echo "  out -> ${OUT}"
  echo "=================================================="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${DATA_DIR}" \
    --encoder_config f98_w256_s64 \
    --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" \
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
    --zero_modality_groups "${MODALITY}" \
    --device auto 2>&1 | tail -3
done

echo
echo "=== ALL ABLATIONS COMPLETE ==="
echo "Results at: ${OUT_ROOT}"
