#!/usr/bin/env bash
# Phase B6: vocab precision sweep — rebuild vocab with bucket-digits=2,
# re-preprocess, retrain on per_row fold 1.
#
# Hypothesis: the 4-digit vocab over-quantizes; collapsing to 2 digits
# should reduce NUM_X cardinality from ~1048 to ~200ish, possibly
# improving numeric accuracy at the cost of value precision.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# Step 1: Build 2-digit vocab
echo "=== Rebuilding 2-digit vocab ==="
PYTHONPATH=src python3 scripts/data/rebuild_vocabulary.py \
  --data-dir data_clean \
  --output-vocab data/gcode_vocab_v8_2digit.json \
  --mode hybrid \
  --bucket-digits 2 \
  --vocab-size 5000 \
  --min-freq 1 2>&1 | tail -10

# Step 2: Re-preprocess fold 1 with new vocab
echo
echo "=== Re-preprocessing per_row fold 1 with 2-digit vocab ==="
python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
  --data-dir data_clean \
  --output-dir outputs/decoder20260511/preprocessed_f98_2digit/per_row/fold_1 \
  --vocab-path data/gcode_vocab_v8_2digit.json \
  --fold 1 --n-folds 5 \
  --window-size 256 --stride 64 \
  --label-mode per_row \
  --exclude-proximity --exclude-pressure 2>&1 | tail -10

# Step 3: Train
echo
echo "=== V8 2-digit per_row fold 1 ==="
python3 scripts/evaluation/run_decoder_quick_test.py \
  --data_dir outputs/decoder20260511/preprocessed_f98_2digit/per_row/fold_1 \
  --encoder_config f98_w256_s64 --fold 1 \
  --vocab data/gcode_vocab_v8_2digit.json \
  --output_dir outputs/decoder20260511/checkpoints/per_row_vocab2digit/fold_1 \
  --epochs 300 --patience 75 \
  --batch_size 64 --lr 1e-4 --weight_decay 0.05 \
  --max_token_len 32 --seed 42 \
  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
  --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
  --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
  --legacy_weight 1.0 --digit_weight 1.0 \
  --device auto 2>&1 | tail -10

echo
echo "=== vocab 2-digit COMPLETE ==="
