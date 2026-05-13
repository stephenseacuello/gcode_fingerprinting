#!/usr/bin/env bash
# Phase B7: pattern-aware decoder pilot.
#
# The existing model already has a `sequence_classifier` head that predicts
# which whole G-code line is being executed. This biases the token-level
# logits toward that line's tokens — exactly the "pattern prior" Dr. Sodhi
# described in the 2026-04-28 meeting. Just enable the head.
#
# Fold 1 only (pilot). If it lifts numeric accuracy meaningfully, run 5-fold.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
OUT="${OUT:-outputs/decoder20260511/checkpoints/per_row_pattern_aware/fold_1}"

echo "=== V8 per_row + pattern-aware (sequence_classifier) fold 1 ==="
python3 scripts/evaluation/run_decoder_quick_test.py \
  --data_dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1 \
  --encoder_config f98_w256_s64 \
  --fold 1 \
  --vocab "${VOCAB}" \
  --output_dir "${OUT}" \
  --epochs "${EPOCHS}" --patience "${PATIENCE:-75}" \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --max_token_len 32 \
  --seed 42 \
  --d_model 384 --n_layers 8 --n_heads 12 \
  --dropout 0.1 \
  --memory_pos_encoding true \
  --use_sensor_prior true \
  --grammar_constraint true \
  --use_window_position false \
  --multi_window_context 0 \
  --window_dropout 0.0 \
  --use_sequence_classifier true \
  --sequence_class_weight 0.5 \
  --legacy_weight 1.0 \
  --digit_weight 1.0 \
  --device auto 2>&1 | tail -10

echo
echo "=== pattern-aware pilot COMPLETE ==="
