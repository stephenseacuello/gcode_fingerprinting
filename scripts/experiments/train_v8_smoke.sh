#!/usr/bin/env bash
# Phase-5 smoke train: 1-epoch sanity check on V8 fold 1 per_row.
#
# Purpose: confirm the pipeline end-to-end runs without exceptions and the
# loss decreases — BEFORE committing to a multi-hour 5-fold sweep.
#
# Uses:
#   - V8 NPZ at outputs/decoder20260511/preprocessed_f98/per_row/fold_1
#     (requires the f98 exclusion preprocessing run, see preprocess_v8_f98.sh)
#   - The available f98 frozen encoder (no_proximity_no_pressure_w256_s64_cv)
#   - configs/decoder_v8_no_shortcuts.json defaults (no window position shortcut)
#   - V8 vocab 2418 tokens
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-outputs/decoder20260511/preprocessed_f98/per_row/fold_1}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
OUT="${OUT:-outputs/decoder20260511/checkpoints/smoke/per_row_fold_1}"
EPOCHS="${EPOCHS:-1}"

mkdir -p "${OUT}"

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
  --device auto 2>&1 | tee "${OUT}/smoke.log"
