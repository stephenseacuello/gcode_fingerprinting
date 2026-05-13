#!/usr/bin/env bash
# Phase B+: nested ablations — 2x2 and 2x2x2 crossings.
#
# Existing main effects we've measured:
#   - shortcuts (use_window_position) ON/OFF
#   - gyroscope ON/OFF
#   - color ON/OFF
#   - pattern-aware (use_sequence_classifier) ON/OFF
#
# Nested ablations test for INTERACTIONS:
#   - shortcuts × gyroscope: does the position-leak compensate when gyro is removed?
#   - shortcuts × color: same question for color
#   - pattern × shortcuts: does pattern-aware bias compose with positional info?
#   - pattern × no-color: does pattern-aware help more when color is removed?
#
# Each on fold 1 only (pilot). 300 epochs / 75 patience.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"

run_one() {
  local TAG="$1"
  local OUT="outputs/decoder20260511/checkpoints/nested/${TAG}/fold_1"
  shift
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
    echo "  skip ${TAG}: already trained"
    return
  fi
  echo
  echo "=== nested: ${TAG} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1 \
    --encoder_config f98_w256_s64 --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size 64 --lr 1e-4 --weight_decay 0.05 \
    --max_token_len 32 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --multi_window_context 0 --window_dropout 0.0 \
    --legacy_weight 1.0 --digit_weight 1.0 \
    "$@" --device auto 2>&1 | tail -5
}

# 2x2 interaction: shortcuts × gyroscope
run_one "shortcuts_gyro"     --use_window_position true
run_one "noshortcuts_nogyro" --use_window_position false --zero_modality_groups gyroscope
run_one "shortcuts_nogyro"   --use_window_position true  --zero_modality_groups gyroscope
# (baseline noshortcuts_gyro is already in checkpoints/per_row_5fold/fold_1)

# 2x2 interaction: shortcuts × color
run_one "shortcuts_color"      --use_window_position true
run_one "noshortcuts_nocolor"  --use_window_position false --zero_modality_groups color
run_one "shortcuts_nocolor"    --use_window_position true  --zero_modality_groups color

# pattern × shortcuts
run_one "pattern_shortcuts"   --use_window_position true  --use_sequence_classifier true --sequence_class_weight 0.5
run_one "pattern_noshortcuts" --use_window_position false --use_sequence_classifier true --sequence_class_weight 0.5
# (pattern + baseline modality is in per_row_pattern_aware)

# pattern × no-color
run_one "pattern_nocolor" --use_window_position false --use_sequence_classifier true --sequence_class_weight 0.5 --zero_modality_groups color

echo
echo "=== nested ablations COMPLETE ==="
