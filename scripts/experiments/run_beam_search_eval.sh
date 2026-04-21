#!/bin/bash
# Run beam search evaluation on existing checkpoints
# V8 f98 pilot + V7 f110 sweep best (per fold)
set -e
cd /home/seacuello/Documents/gcode_fingerprinting
export PYTHONPATH="src:$PYTHONPATH"

echo "=== Beam Search Evaluation ==="
echo "Started: $(date)"

# Common decoder args (must match V7 best training config)
COMMON="--d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
  --memory_pos_encoding True --use_window_position True \
  --use_sensor_prior True --grammar_constraint False \
  --multi_window_context 2 --use_pointer_network False \
  --drop_path_rate 0.0 --focal_gamma 2.0 --label_smoothing 0.1 \
  --digit_weight 15 --max_token_len 16"

mkdir -p outputs/decoder20260304/v8_beam

# === V8 f98 Pilot ===
echo ""
echo "=== V8 f98 Pilot Beam Search ==="
for FOLD in 1 2 3 4 5; do
    CKPT="outputs/decoder20260304/v8_pilot/fold_${FOLD}/decoder_checkpoint/best_decoder.pt"
    OUTDIR="outputs/decoder20260304/v8_beam/f98_fold_${FOLD}"
    mkdir -p "$OUTDIR/results"
    echo "  f98 Fold $FOLD..."
    python3 scripts/evaluation/run_decoder_quick_test.py \
        --encoder_config v8_f98_w256_s64 --fold $FOLD \
        --vocab data/gcode_vocab_712.json \
        --output_dir "$OUTDIR" \
        --checkpoint "$CKPT" \
        --eval_only --beam_width 1 \
        $COMMON \
        2>&1 | tee "${OUTDIR}/beam_eval.log" &
done
wait
echo "V8 f98 beam search complete"

# === V7 f110 Sweep Best (per fold) ===
echo ""
echo "=== V7 f110 Sweep Best Beam Search ==="

# Best V7 sweep checkpoint per fold
V7_FOLD1="outputs/decoder20260304/sweep_v7/784c43e0a2/aaw0ez60/decoder_checkpoint/best_decoder.pt"
V7_FOLD2="outputs/decoder20260304/sweep_v7/6cd9cb3cdd/wp1edmba/decoder_checkpoint/best_decoder.pt"
V7_FOLD3="outputs/decoder20260304/sweep_v7/7b24e99e90/oufw903c/decoder_checkpoint/best_decoder.pt"
V7_FOLD4="outputs/decoder20260304/sweep_v7/085435298f/r5tcwf1p/decoder_checkpoint/best_decoder.pt"
V7_FOLD5="outputs/decoder20260304/sweep_v7/c3355bff40/i0rc1805/decoder_checkpoint/best_decoder.pt"

declare -A V7_CKPTS
V7_CKPTS[1]="$V7_FOLD1"
V7_CKPTS[2]="$V7_FOLD2"
V7_CKPTS[3]="$V7_FOLD3"
V7_CKPTS[4]="$V7_FOLD4"
V7_CKPTS[5]="$V7_FOLD5"

for FOLD in 1 2 3 4 5; do
    CKPT="${V7_CKPTS[$FOLD]}"
    OUTDIR="outputs/decoder20260304/v8_beam/f110_fold_${FOLD}"
    mkdir -p "$OUTDIR/results"

    if [ -f "$CKPT" ]; then
        # Read the trial's training config to get correct decoder architecture
        TRIAL_DIR=$(dirname $(dirname "$CKPT"))
        METRICS="$TRIAL_DIR/results/metrics.json"

        echo "  f110 Fold $FOLD..."
        python3 scripts/evaluation/run_decoder_quick_test.py \
            --encoder_config v7_f110_w256_s64 --fold $FOLD \
            --vocab data/gcode_vocab_712.json \
            --output_dir "$OUTDIR" \
            --checkpoint "$CKPT" \
            --eval_only --beam_width 1 \
            $COMMON \
            2>&1 | tee "${OUTDIR}/beam_eval.log" &
    else
        echo "  f110 Fold $FOLD: checkpoint not found at $CKPT"
    fi
done
wait
echo "V7 f110 beam search complete"

echo ""
echo "=== All Beam Search Evaluations Complete ==="
echo "Finished: $(date)"
