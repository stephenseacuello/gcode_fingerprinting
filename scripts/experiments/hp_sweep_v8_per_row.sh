#!/usr/bin/env bash
# HP sweep on V8 per_row fold 1, designed from the observed fold-1 trajectory:
#   - Val loss spikes (137→552) suggest LR too high
#   - Val tok stuck at 0.74 plateau from epoch 31 onward (no real improvement after that)
#   - Sequence acc stuck at 0.008
#   - Type/param at 0.98 (saturated, not the bottleneck)
#
# Each cell: fold 1 only, 150 epochs cap, patience 40 for fast turnaround.
# 8 cells × ~30 min = ~4 hours total.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-40}"
SWEEP_ROOT="outputs/decoder20260511/checkpoints/hp_sweep"

run_cell() {
  local TAG="$1"; shift
  local OUT="${SWEEP_ROOT}/${TAG}/fold_1"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ]; then
    echo "  skip ${TAG}: already trained"
    return
  fi
  echo
  echo "=== HP cell: ${TAG} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1 \
    --encoder_config f98_w256_s64 --fold 1 \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --max_token_len 32 --seed 42 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --device auto \
    "$@" 2>&1 | tail -3
}

# Baseline (matches our current Phase A config) — for reference
run_cell "baseline_1e-4_b64_d384_w1-1"   --lr 1e-4 --batch_size 64  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 1.0 --warmup_epochs 0
# Lower LR — should reduce gradient explosions
run_cell "lr_5e-5_b64_d384_w1-1"         --lr 5e-5 --batch_size 64  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 1.0 --warmup_epochs 10
run_cell "lr_2e-5_b64_d384_w1-1"         --lr 2e-5 --batch_size 64  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 1.0 --warmup_epochs 10
# Larger batch — smoother gradients
run_cell "lr_5e-5_b128_d384_w1-1"        --lr 5e-5 --batch_size 128 --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 1.0 --warmup_epochs 10
# Bigger model — more capacity for 19k samples × 2418 vocab
run_cell "lr_5e-5_b64_d512_w1-1"         --lr 5e-5 --batch_size 64  --d_model 512 --n_layers 8 --n_heads 16 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 1.0 --warmup_epochs 10
# Heavier digit loss — try to lift numeric_accuracy
run_cell "lr_5e-5_b64_d384_w1-3"         --lr 5e-5 --batch_size 64  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 3.0 --warmup_epochs 10
# Heavier legacy loss — pull on token-level
run_cell "lr_5e-5_b64_d384_w3-1"         --lr 5e-5 --batch_size 64  --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1  --weight_decay 0.05 --legacy_weight 3.0 --digit_weight 1.0 --warmup_epochs 10
# Larger model + heavier digit (combined best guess)
run_cell "lr_5e-5_b64_d512_w1-3"         --lr 5e-5 --batch_size 64  --d_model 512 --n_layers 8 --n_heads 16 --dropout 0.1  --weight_decay 0.05 --legacy_weight 1.0 --digit_weight 3.0 --warmup_epochs 10

echo
echo "=== HP sweep COMPLETE ==="
python3 scripts/analysis/aggregate_hp_sweep.py 2>&1 | tail -20 || true
