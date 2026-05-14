#!/usr/bin/env bash
# Phase F-lite step 2: retrain the decoder using the Phase F-lite encoder
# on V8 full_window 5-fold. This is what produces the result we compare
# against the headline ("did the encoder retrain lift the decoder?").
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-1}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
ENC_ROOT="${ENC_ROOT:-outputs/decoder20260511/encoder_phase_f_lite}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/full_window_5fold_phase_f_lite}"

for FOLD in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${FOLD}"
  ENC_CKPT="${ENC_ROOT}/fold_${FOLD}/checkpoint/best_val.pt"
  if [[ ! -f "${ENC_CKPT}" ]]; then
    echo "fold_${FOLD}: encoder checkpoint missing at ${ENC_CKPT}, skipping"; continue
  fi
  if [[ -d "${OUT}" && -n "$(find "${OUT}" -name metrics.json -print -quit 2>/dev/null)" ]]; then
    echo "fold_${FOLD}: decoder already trained, skipping"; continue
  fi
  echo "--- fold ${FOLD} decoder retrain on Phase F-lite encoder ---"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_${FOLD}" \
    --encoder_ckpt "${ENC_CKPT}" \
    --fold "${FOLD}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size 4 --lr 5e-5 --warmup_epochs 10 --weight_decay 0.05 \
    --max_token_len "${MAX_TOKEN_LEN}" --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 \
    --dropout 0.1 --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --scheduled_sampling 0.0 \
    --wandb --wandb_project gcode-decoder-2026 --device auto 2>&1 | tail -8
done
echo "=== Phase F-lite decoder retrain complete ==="
