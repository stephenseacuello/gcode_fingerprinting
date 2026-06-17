#!/usr/bin/env bash
# B3 — patched-target 5-fold decoder retrain (float-epsilon fix).
# decompose_value_to_digits (src/miracle/dataset/target_utils.py) was patched to
# round-to-scaled-integer BEFORE this run, so digit targets are now exact.
# Writes to a NEW checkpoint dir — the released full_window_5fold is untouched.
# Config mirrors scripts/experiments/train_v8_full_window_5fold.sh (headline winner config).
set -uo pipefail
REPO=/home/seacuello/Documents/gcode_fingerprinting
cd "$REPO"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

OUT_ROOT=outputs/decoder20260511/checkpoints/full_window_5fold_patched
VOCAB=data/gcode_vocab_v8.json
EPOCHS=300; PATIENCE=75; BATCH=4; MAXLEN=1400; SEED=42
mkdir -p "$OUT_ROOT"
MASTER="$OUT_ROOT/MASTER.log"
echo "[$(date)] launching patched 5-fold retrain (folds 1,3,5->GPU0  2,4->GPU1)" | tee "$MASTER"

run_fold () {
  local F=$1; local GPU=$2
  local OUT="$OUT_ROOT/fold_${F}"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] START fold $F on GPU $GPU" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_${F}" \
    --encoder_config f98_w256_s64 \
    --fold "$F" \
    --vocab "$VOCAB" \
    --output_dir "$OUT" \
    --epochs $EPOCHS --patience $PATIENCE \
    --batch_size $BATCH \
    --lr 5e-5 --warmup_epochs 10 --weight_decay 0.05 \
    --max_token_len $MAXLEN --seed $SEED \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --scheduled_sampling 0.5 \
    --device auto > "$OUT/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE fold $F (exit $?)" | tee -a "$MASTER"
}

# All five folds concurrently: 3 on GPU0, 2 on GPU1 (small model, ~2GB each).
run_fold 1 0 &
run_fold 2 1 &
run_fold 3 0 &
run_fold 4 1 &
run_fold 5 0 &
wait
echo "[$(date)] ALL FOLDS COMPLETE" | tee -a "$MASTER"
python3 scripts/analysis/aggregate_v8_results.py --root "$OUT_ROOT" 2>&1 | tail -30 | tee -a "$MASTER" || \
  echo "[note] aggregate script needs manual root arg; per-fold metrics.json are in $OUT_ROOT/fold_*/" | tee -a "$MASTER"
