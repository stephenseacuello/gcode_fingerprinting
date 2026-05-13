#!/usr/bin/env bash
# Round-2 Phase A: re-run --eval_only on every V8 checkpoint to capture
# per-class precision/recall/F1 via the extended evaluate().
#
# After my edit to evaluate() (run_decoder_quick_test.py line ~1218), the
# test_metrics dict now carries a "per_class" key with full P/R/F1 +
# confusion matrices per head. Re-running --eval_only saves that to
# beam_0_metrics.json in each results/ dir.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="data/gcode_vocab_v8.json"

run_one() {
  local DATA_DIR="$1"
  local OUT="$2"
  local CKPT="${OUT}/decoder_checkpoint/best_decoder.pt"
  if [ ! -f "${CKPT}" ]; then
    echo "  SKIP ${OUT}: no checkpoint at ${CKPT}"
    return
  fi
  echo
  echo "=== refresh ${OUT} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${DATA_DIR}" \
    --encoder_config f98_w256_s64 \
    --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs 0 --batch_size 64 --max_token_len 32 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --eval_only --beam_width 0 \
    --checkpoint "${CKPT}" 2>&1 | tail -10
}

# 5-fold per_row sweep
for F in 1 2 3 4 5; do
  run_one "outputs/decoder20260511/preprocessed_f98/per_row/fold_${F}" \
          "outputs/decoder20260511/checkpoints/per_row_5fold/fold_${F}"
done

# 50ep single-fold per_row baseline (legacy, for reference)
run_one "outputs/decoder20260511/preprocessed_f98/per_row/fold_1" \
        "outputs/decoder20260511/checkpoints/per_row_50ep/fold_1"

# Sensor ablations (zero each modality)
for MOD in accelerometer gyroscope magnetometer environmental color rms electrical; do
  OUT="outputs/decoder20260511/ablations/sensor/zero_${MOD}/fold_1"
  CKPT="${OUT}/decoder_checkpoint/best_decoder.pt"
  if [ ! -f "${CKPT}" ]; then
    echo "  SKIP ${OUT}: no checkpoint"
    continue
  fi
  echo
  echo "=== refresh ${OUT} (ablation: zero ${MOD}) ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/per_row/fold_1" \
    --encoder_config f98_w256_s64 \
    --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs 0 --batch_size 64 --max_token_len 32 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --zero_modality_groups "${MOD}" \
    --eval_only --beam_width 0 \
    --checkpoint "${CKPT}" 2>&1 | tail -10
done

echo
echo "=== ALL --eval_only REFRESHES COMPLETE ==="
echo "Per-class metrics now in each fold's results/beam_0_metrics.json"
