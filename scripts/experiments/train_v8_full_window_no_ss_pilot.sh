#!/usr/bin/env bash
# Phase B (post-full_window): V8 full_window fold-1 pilot WITHOUT
# scheduled sampling. Disentangles the +27pp command lift between
# the full_window mode and the ss=0.5 component.
#
# Composite winner = full_window + ss=0.5; this cell asks: how much of the
# 0.78 cmd comes from full_window alone? If cmd stays ~0.78, full_window
# is the lever and ss is auxiliary. If cmd drops to ~0.32, the +27pp
# emerged from the interaction of both.
#
# Resolves recommendation #17 in notes.md.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
OUT="${OUT:-outputs/decoder20260511/checkpoints/full_window_no_ss_pilot/fold_1}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"

echo "=== V8 full_window WITHOUT scheduled sampling (fold-1 pilot) ==="
python3 scripts/evaluation/run_decoder_quick_test.py \
  --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_1" \
  --encoder_config f98_w256_s64 \
  --fold 1 \
  --vocab "${VOCAB}" \
  --output_dir "${OUT}" \
  --epochs "${EPOCHS}" --patience "${PATIENCE}" \
  --batch_size 4 \
  --lr 5e-5 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  --max_token_len "${MAX_TOKEN_LEN}" \
  --seed 42 \
  --d_model 384 --n_layers 8 --n_heads 12 \
  --dropout 0.1 \
  --legacy_weight 3.0 --digit_weight 1.0 \
  --memory_pos_encoding true \
  --use_sensor_prior true \
  --grammar_constraint true \
  --use_window_position false \
  --multi_window_context 0 \
  --window_dropout 0.0 \
  --wandb --wandb_project gcode-decoder-2026 \
  --device auto 2>&1 | tail -8
