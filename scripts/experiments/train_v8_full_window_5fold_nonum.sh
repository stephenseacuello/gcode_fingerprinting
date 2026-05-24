#!/usr/bin/env bash
# Design B: V8 full_window 5-fold "no-numbers" retrain + eval.
#
# Action item 2 (2026-05-21 follow-up). Identical to
# train_v8_full_window_5fold.sh EXCEPT:
#   - vocab:   gcode_vocab_v8_nonum.json (config.collapse_numeric=true -> the
#              tokenizer emits one <NUM> token for every coordinate value)
#   - data:    preprocessed_f98_nonum/  (sensor data identical; gcode_texts
#              identical; only the cached token ids collapsed)
#   - output:  full_window_5fold_nonum/
#   - digit_weight 0 (no digits to predict; <NUM> -> TYPE_SPECIAL, so
#     DigitByDigitLoss returns 0.0 cleanly)
#   - no wandb
# Every other hyperparameter, the frozen encoder, and seed 42 are unchanged,
# so this is a clean A/B against the current model.
#
# Phase 1 trains all 5 folds; Phase 2 runs --eval_only on each to emit
# beam_0 (teacher-forced) and beam_1 (autoregressive) predictions.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8_nonum.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/full_window_5fold_nonum}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
BATCH="${BATCH:-4}"
SS_VALUE="${SS_VALUE:-0.5}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-1 2 3 4 5}"   # override e.g. FOLDS=1 for a single-fold diagnostic
DATA_ROOT="${DATA_ROOT:-outputs/decoder20260511/preprocessed_f98_nonum/full_window}"

ARCH=(--d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1
      --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true
      --use_window_position false --multi_window_context 0)

# ---- Phase 1: train ----------------------------------------------------------
for F in ${FOLDS}; do
  OUT="${OUT_ROOT}/fold_${F}"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
    echo "  skip train fold ${F}: already trained"
    continue
  fi
  echo
  echo "=== Design B TRAIN fold ${F} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${DATA_ROOT}/fold_${F}" \
    --encoder_config f98_w256_s64 --fold "${F}" \
    --vocab "${VOCAB}" --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size "${BATCH}" --lr 5e-5 --warmup_epochs 10 --weight_decay 0.05 \
    --max_token_len "${MAX_TOKEN_LEN}" --seed "${SEED}" \
    "${ARCH[@]}" \
    --legacy_weight 3.0 --digit_weight 0 \
    --window_dropout 0.0 --scheduled_sampling "${SS_VALUE}" \
    --device auto 2>&1 | tail -8
done

# ---- Phase 2: eval-only -> beam_0 (TF) + beam_1 (AR) predictions --------------
for F in ${FOLDS}; do
  OUT="${OUT_ROOT}/fold_${F}"
  echo
  echo "=== Design B EVAL fold ${F} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${DATA_ROOT}/fold_${F}" \
    --encoder_config f98_w256_s64 --fold "${F}" \
    --vocab "${VOCAB}" --output_dir "${OUT}" \
    --eval_only --beam_widths 0,1 \
    --batch_size "${BATCH}" --max_token_len "${MAX_TOKEN_LEN}" --seed "${SEED}" \
    "${ARCH[@]}" \
    --legacy_weight 3.0 --digit_weight 0 \
    --device auto 2>&1 | tail -8
done

echo
echo "=== Design B full_window 5-fold COMPLETE ==="
