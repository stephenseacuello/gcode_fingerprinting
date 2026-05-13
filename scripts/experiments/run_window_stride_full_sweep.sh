#!/usr/bin/env bash
# Phase B+ : full window/stride ablation that RETRAINS THE ENCODER per cell
# (no-proximity-no-pressure feature set, same recipe as the published
# encoder paper) and then retrains the decoder on the new encoder.
#
# The previous run_window_stride_sweep.sh used a mismatched single encoder
# (f98_w256_s64) for every (W, S) combination, which is invalid because the
# frozen encoder consumes a fixed-shape input. This version does the right
# thing per the 2026-05-13 user direction:
#   - For each (W, S) cell, preprocess with --exclude-proximity
#     --exclude-pressure (98 features → matches the encoder-paper recipe).
#   - Train a new encoder for that cell using run_9class_direct.py (same
#     loss / hyperparameters as the published encoder paper).
#   - Train the decoder on the new encoder.
# Fold 1 only (8 cells × 2 trainings ≈ 6--8 hours total).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"

run_cell() {
  local TAG="$1"
  local WIN="$2"
  local STR="$3"
  local PREPROC="outputs/decoder20260511/preprocessed_f98_${TAG}/per_row/fold_1"
  local ENC_OUT="outputs/decoder20260511/encoders_window_stride/${TAG}/fold_1"
  local DEC_OUT="outputs/decoder20260511/checkpoints/window_stride_v2/${TAG}/fold_1"

  if [ -f "${DEC_OUT}/decoder_checkpoint/best_decoder.pt" ] || \
     [ -f "${DEC_OUT}"/*/decoder_checkpoint/best_decoder.pt ]; then
    echo "  skip ${TAG}: already trained"
    return
  fi
  echo
  echo "=============================================================="
  echo "  cell ${TAG}  win=${WIN}  stride=${STR}"
  echo "=============================================================="

  # ----- (1) preprocess (no_proximity, no_pressure → 98 features) -----
  if [ ! -f "${PREPROC}/train_sequences.npz" ]; then
    echo "  [1/3] preprocessing..."
    python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
      --data-dir data_clean \
      --output-dir "outputs/decoder20260511/preprocessed_f98_${TAG}/per_row" \
      --vocab-path "${VOCAB}" \
      --fold 1 --n-folds 5 \
      --window-size "${WIN}" --stride "${STR}" \
      --label-mode per_row \
      --exclude-proximity --exclude-pressure 2>&1 | tail -3
  fi

  # ----- (2) train encoder using the encoder-paper recipe -----
  if [ ! -f "${ENC_OUT}/encoder/checkpoint/best_model.pt" ]; then
    echo "  [2/3] training encoder (${TAG})..."
    python3 scripts/evaluation/run_9class_direct.py \
      --data-dir "${PREPROC}" \
      --output-dir "${ENC_OUT}/encoder" \
      --iteration "cv_fold_1_${TAG}" \
      --lr 1e-3 --label_smoothing 0.1 --dropout 0.2 \
      --max_epochs 200 --patience 40 --seed 42 \
      --modality_dropout 0.1 --d_model 256 --n_heads 4 --lstm_layers 2 \
      --ablation full 2>&1 | tail -5
  fi

  # ----- (3) train decoder on the new encoder -----
  echo "  [3/3] training decoder..."
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${PREPROC}" \
    --encoder_ckpt "${ENC_OUT}/encoder/checkpoint/best_model.pt" \
    --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${DEC_OUT}" \
    --epochs "${EPOCHS:-300}" --patience "${PATIENCE:-75}" \
    --batch_size 64 --lr 5e-5 --warmup_epochs 10 \
    --weight_decay 0.05 --max_token_len 32 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true --use_sensor_prior true \
    --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --scheduled_sampling 0.5 \
    --wandb --wandb_project gcode-decoder-2026 \
    --device auto 2>&1 | tail -5
}

# Fractional design (7 cells; baseline w256_s64 already trained as full_window_5fold)
run_cell "w64_s16"    64    16
run_cell "w64_s32"    64    32
run_cell "w128_s16"   128   16
run_cell "w128_s32"   128   32
run_cell "w128_s64"   128   64
run_cell "w256_s128"  256   128
run_cell "w512_s128"  512   128

echo
echo "=== window/stride v2 sweep COMPLETE ==="
python3 scripts/analysis/aggregate_v8_results.py \
  --sweep-name window_stride_v2 || true
