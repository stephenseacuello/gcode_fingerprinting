#!/usr/bin/env bash
# Free-running AR categorical measurement: re-evaluate the released
# full_window_5fold checkpoints with --ar_categorical, so the categorical
# heads (type/command/param-type/sign/digit) are scored from a forward pass
# conditioned on the GENERATED token stream (greedy AR + FSM, deployment
# default) instead of the teacher-forced pass. Closes the "free-running
# categorical accuracy unmeasured" gap flagged in panel round 4 (2026-08-14).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-1}"
VOCAB="data/gcode_vocab_v8.json"

for FOLD in 1 2 3 4 5; do
  CKPT="outputs/decoder20260511/checkpoints/full_window_5fold/fold_${FOLD}/decoder_checkpoint/best_decoder.pt"
  [ -f "$CKPT" ] || CKPT=$(find outputs/decoder20260511/checkpoints/full_window_5fold -path "*fold_${FOLD}*/decoder_checkpoint/best_decoder.pt" | head -1)
  DATA_DIR="outputs/decoder20260511/preprocessed_f98/full_window/fold_${FOLD}"
  OUT_DIR="outputs/decoder20260511/checkpoints/full_window_5fold/fold_${FOLD}_ar_categorical"
  if [ -f "${OUT_DIR}/results/beam_1_metrics.json" ]; then
    echo "skip fold ${FOLD}: exists"; continue
  fi
  mkdir -p "${OUT_DIR}/results"
  echo "=== AR-categorical re-eval: fold ${FOLD} (GPU ${GPU}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_decoder_quick_test.py \
    --eval_only \
    --checkpoint "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --encoder_config f98_w256_s64 --fold "${FOLD}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT_DIR}" \
    --max_token_len 1400 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --device auto --beam_widths 1 \
    --fsm_grammar --ar_categorical 2>&1 | tail -4
done
echo "=== AR-categorical batch complete ==="
