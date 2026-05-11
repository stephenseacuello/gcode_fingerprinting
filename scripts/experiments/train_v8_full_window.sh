#!/usr/bin/env bash
# Phase-5 full_window training driver.
#
# Full_window mode emits one sample per 256-sample window with a multi-line
# target that preserves every distinct G-code line that fired during the
# window. V8 fold 1 train has token sequences up to 1,339 tokens long, so
# the decoder needs max_seq_len >= 1400 (positional encoding capacity).
#
# Memory note: self-attention scales as O(L^2). At L=1400 with batch=4 and
# d_model=384 the per-batch attention activation is ~350MB plus encoder-
# decoder cross-attention ~70MB. Fits comfortably on A6000 (48GB).
#
# Default: 1 epoch smoke. Override with EPOCHS=N env var for longer runs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

FOLD="${FOLD:-1}"
DATA_DIR="${DATA_DIR:-outputs/decoder20260511/preprocessed_f98/full_window/fold_${FOLD}}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
OUT="${OUT:-outputs/decoder20260511/checkpoints/full_window/fold_${FOLD}}"
EPOCHS="${EPOCHS:-1}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
BATCH="${BATCH:-4}"

mkdir -p "${OUT}"

python3 scripts/evaluation/run_decoder_quick_test.py \
  --data_dir "${DATA_DIR}" \
  --encoder_config f98_w256_s64 \
  --fold "${FOLD}" \
  --vocab "${VOCAB}" \
  --output_dir "${OUT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr 5e-5 \
  --weight_decay 0.05 \
  --max_token_len "${MAX_TOKEN_LEN}" \
  --seed 42 \
  --d_model 256 \
  --n_layers 4 \
  --n_heads 8 \
  --dropout 0.1 \
  --memory_pos_encoding true \
  --use_sensor_prior true \
  --grammar_constraint true \
  --use_window_position false \
  --multi_window_context 0 \
  --window_dropout 0.0 \
  --legacy_weight 1.0 \
  --digit_weight 1.0 \
  --device auto 2>&1 | tee "${OUT}/train.log"
