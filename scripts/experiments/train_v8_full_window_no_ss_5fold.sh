#!/usr/bin/env bash
# Phase B (extension): V8 full_window 5-fold WITHOUT scheduled sampling.
#
# Motivation: fold-1 no_ss pilot lifted numeric accuracy from 0.42 -> 0.835
# (+41pp) over the ss=0.5 baseline fold-1. If the 5-fold extension confirms,
# scheduled sampling at 0.5 may have been actively hurting per-row numeric
# recovery for the full_window formulation.
#
# Runs sequentially on a dedicated CUDA device (set GPU env var).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/full_window_no_ss_5fold}"

# Fold 1 already exists in full_window_no_ss_pilot — link it in so the
# aggregator picks up all five.
mkdir -p "${OUT_ROOT}"
if [[ ! -d "${OUT_ROOT}/fold_1" && -d outputs/decoder20260511/checkpoints/full_window_no_ss_pilot/fold_1 ]]; then
  ln -s "$(pwd)/outputs/decoder20260511/checkpoints/full_window_no_ss_pilot/fold_1" "${OUT_ROOT}/fold_1"
  echo "linked fold_1 from no_ss_pilot"
fi

for FOLD in 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${FOLD}"
  if [[ -d "${OUT}" && -n "$(find "${OUT}" -name metrics.json -print -quit 2>/dev/null)" ]]; then
    echo "=== fold ${FOLD} already complete, skipping ==="
    continue
  fi
  echo "=== V8 full_window NO_SS — fold ${FOLD} (GPU ${GPU}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_${FOLD}" \
    --encoder_config f98_w256_s64 \
    --fold "${FOLD}" \
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
done

echo "=== no_ss 5-fold complete ==="
