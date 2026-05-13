#!/usr/bin/env bash
# Phase B5: LOCO (leave-one-class-out) on V8 per_row data.
# For each of 9 operation classes, train without it and test on it.
# Tests true cross-class generalization. Fold 1 only for time.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
LOCO_ROOT="outputs/decoder20260511/preprocessed_f98/per_row_loco"

# Step 1: derive LOCO NPZ subsets
echo "=== Deriving LOCO NPZ subsets ==="
python3 scripts/preprocessing/make_loco_subset.py \
  --src-dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1 \
  --dst-root "${LOCO_ROOT}" 2>&1 | tail -20

# Step 2: train one decoder per holdout class
for CLS in adaptive adaptive150025 face face150025 pocket pocket150025 damageadaptive damageface damagepocket; do
  SRC="${LOCO_ROOT}/holdout_${CLS}"
  OUT="outputs/decoder20260511/checkpoints/loco/holdout_${CLS}"
  if [ ! -f "${SRC}/test_sequences.npz" ]; then
    echo "  SKIP ${CLS}: missing subset"
    continue
  fi
  # Skip if test set is empty
  python3 -c "
import numpy as np
d = np.load('${SRC}/test_sequences.npz', allow_pickle=True)
import sys
sys.exit(0 if len(d['continuous']) > 0 else 1)
" 2>/dev/null || { echo "  SKIP ${CLS}: empty test"; continue; }

  echo
  echo "=== LOCO holdout=${CLS} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "${SRC}" \
    --encoder_config f98_w256_s64 \
    --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE:-75}" \
    --batch_size 64 \
    --lr 1e-4 --weight_decay 0.05 \
    --max_token_len 32 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --legacy_weight 1.0 --digit_weight 1.0 \
    --device auto 2>&1 | tail -5
done

echo
echo "=== LOCO COMPLETE ==="
python3 scripts/analysis/aggregate_v8_results.py
