#!/usr/bin/env bash
# Flat-vs-structured ablation, v7 recipe, current code, cached frozen-encoder memory.
#   Config A (full)      : structural_weight=1.0, digit_weight=15.0, legacy_weight=1.0  (reproduces v7 ~94.7%)
#   Config B (flat-only) : structural_weight=0,   digit_weight=0,    legacy_weight=1.0  (flat-712 head alone)
set -uo pipefail
REPO=/home/seacuello/Documents/gcode_fingerprinting; cd "$REPO"
EXP=outputs/decoder20260511/two_paper_split/_flatbaseline
M="$EXP/MASTER.log"; : > "$M"
run () {  # $1=config $2=fold $3=gpu $4=structural_w $5=digit_w
  local CFG=$1 F=$2 GPU=$3 SW=$4 DW=$5
  local OUT="$EXP/config${CFG}/fold_${F}"; mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] START config$CFG fold$F gpu$GPU (struct=$SW digit=$DW)" >> "$M"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir outputs/decoder20260304/preprocessed_v7/fold_${F} \
    --encoder_ckpt DUMMY_USING_CACHED_MEMORY \
    --cached_memory_dir outputs/decoder20260304/v7_best_5fold/fold_${F}/encoder_memory \
    --fold ${F} --vocab data/gcode_vocab_712.json --output_dir "$OUT" \
    --epochs 1000 --patience 200 --batch_size 32 --lr 3e-4 --max_token_len 16 --seed 42 --curriculum none \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --digit_weight ${DW} --legacy_weight 1.0 --structural_weight ${SW} \
    --label_smoothing 0.1 --scheduled_sampling 0.5 --focal_gamma 2.0 \
    --multi_window_context 2 --weight_decay 0.1 --warmup_epochs 10 \
    --memory_pos_encoding true --grammar_constraint true \
    --use_window_position true --use_sensor_prior true \
    --window_dropout 0.1 --grad_accum 2 --device auto > "$OUT/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  config$CFG fold$F (exit $?)" >> "$M"
}
for F in 1 2 3 4 5; do run A $F 0 1.0 15.0 & done
for F in 1 2 3 4 5; do run B $F 1 0   0    & done
wait
echo "[$(date +%H:%M:%S)] ALL 10 RUNS COMPLETE" >> "$M"
