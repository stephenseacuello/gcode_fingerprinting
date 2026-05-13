#!/usr/bin/env bash
# Phase B1: V8 full_window 5-fold sweep (no shortcuts) — predicts the
# full multi-line G-code per window. Per the 2026-05-12 encoder audit,
# this is the manuscript's headline experiment (per_row was the pilot).
#
# Config = composite-winner from HP sweep stages 1+2:
#   - Stage 1 winner architecture: d_model=384, n_layers=8, n_heads=12
#   - Stage 1 winner loss weights:  legacy_weight=3.0, digit_weight=1.0
#   - Stage 1 winner training:      lr=5e-5, warmup_epochs=10, weight_decay=0.05
#   - Stage 2 add-on:               scheduled_sampling=0.5 (lifted per_row cmd +17 pp;
#                                                          keep for now, ablate later)
# Full_window specifics:
#   - max_token_len=1400 (data max is 1339)
#   - batch_size=4        (long sequences → memory-bound)
#   - 303 train / 110 val / 132 test per fold (one window = one sample)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/full_window_5fold}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
BATCH="${BATCH:-4}"
USE_SS="${USE_SS:-1}"          # 1 → scheduled_sampling=0.5, 0 → no ss
SS_VALUE="${SS_VALUE:-0.5}"

EXTRA_FLAGS=()
if [ "${USE_SS}" = "1" ]; then
  EXTRA_FLAGS+=(--scheduled_sampling "${SS_VALUE}")
fi

for F in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${F}"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ] || \
     [ -f "${OUT}"/*/decoder_checkpoint/best_decoder.pt ]; then
    echo "  skip fold ${F}: already trained"
    continue
  fi
  echo
  echo "=== V8 full_window winner-config fold ${F} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_${F}" \
    --encoder_config f98_w256_s64 \
    --fold "${F}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size "${BATCH}" \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --max_token_len "${MAX_TOKEN_LEN}" \
    --seed 42 \
    --d_model 384 \
    --n_layers 8 \
    --n_heads 12 \
    --dropout 0.1 \
    --legacy_weight 3.0 \
    --digit_weight 1.0 \
    --memory_pos_encoding true \
    --use_sensor_prior true \
    --grammar_constraint true \
    --use_window_position false \
    --multi_window_context 0 \
    --window_dropout 0.0 \
    --wandb --wandb_project gcode-decoder-2026 \
    --device auto \
    "${EXTRA_FLAGS[@]}" 2>&1 | tail -5
done

echo
echo "=== full_window 5-fold COMPLETE ==="
python3 scripts/analysis/aggregate_v8_results.py
