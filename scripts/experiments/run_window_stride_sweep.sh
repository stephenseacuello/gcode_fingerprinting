#!/usr/bin/env bash
# Phase B+: window/stride fractional sweep on per_row.
#
# Tests whether the V7-inherited window=256, stride=64 is optimal for the V8
# per_row formulation. Fractional design instead of full 4x4 grid:
#
#   Cell  Window  Stride  Note
#    A     64       16    Smallest — high temporal resolution
#    B     128      32
#    C     128      64
#    D     256      64    <- baseline (current)
#    E     256      128   Less overlap
#    F     512      128   Largest
#    G     128      16    Heavy overlap
#    H     64       32    Tiny windows
#
# Each cell: re-preprocess + 5-fold train. Each preprocessing takes ~5 min,
# each training fold ~5-10 min depending on sample count. Fold 1 only to
# stay under 4 hours total.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"

run_cell() {
  local TAG="$1"
  local WIN="$2"
  local STR="$3"
  local PREPROC="outputs/decoder20260511/preprocessed_f98_${TAG}/per_row/fold_1"
  local OUT="outputs/decoder20260511/checkpoints/window_stride/${TAG}/fold_1"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
    echo "  skip ${TAG}: already trained"
    return
  fi
  echo
  echo "=== window/stride sweep: ${TAG} (win=${WIN} str=${STR}) ==="

  # Preprocess if needed
  if [ ! -f "${PREPROC}/train_sequences.npz" ]; then
    python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
      --data-dir data_clean \
      --output-dir "${PREPROC}" \
      --vocab-path "${VOCAB}" \
      --fold 1 --n-folds 5 \
      --window-size "${WIN}" --stride "${STR}" \
      --label-mode per_row \
      --exclude-proximity --exclude-pressure 2>&1 | tail -3
  fi

  # Pick an encoder config that has matching window size (if available)
  local ENC_CFG
  case "${WIN}_${STR}" in
    256_64)  ENC_CFG="f98_w256_s64" ;;
    64_16)   ENC_CFG="f98_w64_s16" ;;
    *)       ENC_CFG="f98_w256_s64" ;;   # default — will mismatch but model accepts
  esac

  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${PREPROC}" \
    --encoder_config "${ENC_CFG}" --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size 64 --lr 1e-4 --weight_decay 0.05 \
    --max_token_len 32 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --legacy_weight 1.0 --digit_weight 1.0 \
    --device auto 2>&1 | tail -5
}

# Fractional design (8 cells)
run_cell "w64_s16"    64    16
run_cell "w64_s32"    64    32
run_cell "w128_s16"   128   16
run_cell "w128_s32"   128   32
run_cell "w128_s64"   128   64
# w256_s64 is the baseline — already trained in checkpoints/per_row_5fold/fold_1
run_cell "w256_s128"  256   128
run_cell "w512_s128"  512   128

echo
echo "=== window/stride sweep COMPLETE ==="
