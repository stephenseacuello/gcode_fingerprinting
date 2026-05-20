#!/usr/bin/env bash
# Autoregressive (beam_width=1) re-evaluation of the 9 leave-one-class-out
# (LOCO) per_row decoder checkpoints, so the tamper-injection audit can be
# run under the open-vocabulary regime (referee R3 M5 / R3 M1).
# LOCO training was per_row; checkpoints are flat (no wandb subdir).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-1}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
CLASSES="adaptive adaptive150025 face face150025 pocket pocket150025 damageadaptive damageface damagepocket"

for CLS in $CLASSES; do
  CKPT="outputs/decoder20260511/checkpoints/loco/holdout_${CLS}/decoder_checkpoint/best_decoder.pt"
  CKPT_PARENT="outputs/decoder20260511/checkpoints/loco/holdout_${CLS}"
  DATA_DIR="outputs/decoder20260511/preprocessed_f98/per_row_loco/holdout_${CLS}"
  if [ ! -f "$CKPT" ]; then echo "skip ${CLS}: no checkpoint"; continue; fi
  if [ -f "${CKPT_PARENT}/results/beam_1_all_predictions.json" ]; then
    echo "skip ${CLS}: AR predictions already exist"; continue
  fi
  echo
  echo "=== LOCO AR re-eval: holdout_${CLS} (GPU ${GPU}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_decoder_quick_test.py \
    --eval_only \
    --checkpoint "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --encoder_config f98_w256_s64 --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${CKPT_PARENT}" \
    --max_token_len 32 --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --device auto --beam_widths 1 2>&1 | tail -5
done

echo
echo "=== LOCO AR re-eval complete ==="
