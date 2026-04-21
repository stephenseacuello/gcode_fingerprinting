#!/bin/bash
# Launch decoder training for window size ablation
# Usage: bash launch_window_ablation.sh <gpu_id> <config>
# Configs: full_w128_s32, full_w64_s16
set -e

GPU=${1:-0}
CONFIG=${2:-"full_w128_s32"}
PROJECT="/home/seacuello/Documents/gcode_fingerprinting"

echo "============================================================"
echo "WINDOW ABLATION: $CONFIG on GPU $GPU"
echo "============================================================"

for FOLD in 1 2 3 4 5; do
    OUTPUT="$PROJECT/outputs/decoder20260304/window_ablation/$CONFIG/decoder/fold_$FOLD"

    if [ -f "$OUTPUT/results/metrics.json" ]; then
        echo "  Fold $FOLD: already done, skipping"
        continue
    fi

    ENCODER_CKPT="$PROJECT/outputs/experiments_2026_02_25/${CONFIG}_cv/fold_$FOLD/encoder/checkpoint/best_model.pt"
    DATA_DIR="$PROJECT/outputs/decoder20260304/window_ablation/$CONFIG/fold_$FOLD"

    if [ ! -f "$ENCODER_CKPT" ]; then
        echo "  Fold $FOLD: NO encoder checkpoint, skipping"
        continue
    fi

    echo "  --- Fold $FOLD ---"
    CUDA_VISIBLE_DEVICES=$GPU python3 "$PROJECT/scripts/evaluation/run_decoder_quick_test.py" \
        --vocab "$PROJECT/data/gcode_vocab_712.json" \
        --encoder_ckpt "$ENCODER_CKPT" \
        --data_dir "$DATA_DIR" \
        --fold "$FOLD" \
        --epochs 500 \
        --patience 150 \
        --d_model 384 \
        --n_layers 8 \
        --n_heads 12 \
        --lr 0.0003 \
        --batch_size 32 \
        --dropout 0.1 \
        --digit_weight 15.0 \
        --label_smoothing 0.1 \
        --scheduled_sampling 0.5 \
        --focal_gamma 2.0 \
        --weight_decay 0.1 \
        --warmup_epochs 10 \
        --legacy_weight 1.0 \
        --memory_pos_encoding True \
        --grammar_constraint True \
        --multi_window_context 2 \
        --use_window_position True \
        --use_sensor_prior True \
        --use_pointer_network False \
        --use_regression_head False \
        --hierarchical False \
        --drop_path_rate 0.0 \
        --oversample_rare False \
        --window_dropout 0.1 \
        --noise_scale 0.0 \
        --use_sequence_classifier False \
        --use_distillation False \
        --pointer_weight 2.0 \
        --axis_value_tables "$PROJECT/data/axis_value_tables.json" \
        --seed 42 \
        --output_dir "$OUTPUT" \
        2>&1 | tail -3

    echo "  Fold $FOLD done"
done
echo "$CONFIG complete"
