#!/bin/bash
# Approach A: End-to-end encoder-decoder training on f98 encoder
# Decoder loss backpropagates through encoder (unlike Approach B which used CLS loss)
#
# Strategy:
#   - Fold 1 first (the bottleneck at 66.7% frozen baseline)
#   - If fold 1 > 75%, run all 5 folds
#   - batch_size=16 + grad_accum=2 = effective batch 32 (fits both models in GPU)
#   - encoder_lr = 3e-5 (1/10th of decoder lr)
#   - aux CLS loss (0.1 weight) prevents catastrophic forgetting
#   - No MWC (simplifies live encoder forward)
#
# Expected GPU memory: ~2-3x current (both models in graph)
# Expected time: ~5-8s/epoch (vs ~3.5s frozen), ~20-40 min per fold with early stopping

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

VOCAB="data/gcode_vocab_712.json"
OUTPUT_BASE="outputs/decoder20260304/e2e_pilot"

# V7 best decoder config + e2e flags
COMMON_ARGS="
    --vocab $VOCAB
    --epochs 300
    --batch_size 32
    --lr 3e-4
    --patience 100
    --d_model 384
    --n_layers 8
    --n_heads 12
    --dropout 0.1
    --digit_weight 15.0
    --legacy_weight 1.0
    --label_smoothing 0.1
    --focal_gamma 2.0
    --scheduled_sampling 0.5
    --memory_pos_encoding true
    --use_window_position true
    --use_sensor_prior true
    --weight_decay 0.1
    --warmup_epochs 10
    --beam_width 1
    --e2e
    --grad_accum 2
    --encoder_lr 3e-5
    --e2e_cls_weight 0.1
    --e2e_recache_interval 20
    --multi_window_context 0
"

run_fold() {
    local fold=$1
    local encoder_ckpt="outputs/decoder20260304/encoder_sweep_f98/dm256_nh4_lr0.0005_do0.3/fold_${fold}/checkpoint/best_model.pt"
    local data_dir="outputs/decoder20260304/preprocessed_v8_f98/fold_${fold}"
    local output_dir="${OUTPUT_BASE}/fold_${fold}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting E2E fold $fold"
    echo "  Encoder: $encoder_ckpt"
    echo "  Data: $data_dir"
    echo "  Output: $output_dir"

    python3 scripts/evaluation/run_decoder_quick_test.py \
        --data_dir "$data_dir" \
        --encoder_ckpt "$encoder_ckpt" \
        --output_dir "$output_dir" \
        --seed 42 \
        $COMMON_ARGS

    # Extract result
    if [ -f "$output_dir/results/metrics.json" ]; then
        local seq_acc=$(python3 -c "import json; r=json.load(open('$output_dir/results/metrics.json')); print(f\"{r['test_metrics']['sequence_accuracy']:.4f}\")")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE fold $fold: seq_acc=$seq_acc"
    fi
}

echo "================================================================"
echo "APPROACH A: End-to-End Encoder-Decoder Training (f98)"
echo "================================================================"
echo "Started: $(date)"
echo ""

# Phase 1: Fold 1 only (the bottleneck)
echo "PHASE 1: Fold 1 pilot"
echo "  Frozen baseline: 66.7% seq_acc"
echo "  Target: > 75% (proceed to all folds)"
echo ""
run_fold 1

# Check fold 1 result
FOLD1_SEQ=$(python3 -c "import json; r=json.load(open('${OUTPUT_BASE}/fold_1/results/metrics.json')); print(f\"{r['test_metrics']['sequence_accuracy']:.4f}\")" 2>/dev/null || echo "0.0000")
echo ""
echo "Fold 1 result: seq_acc=${FOLD1_SEQ}"

FOLD1_PCT=$(python3 -c "print(int(float('${FOLD1_SEQ}') * 100))")
if [ "$FOLD1_PCT" -ge 70 ]; then
    echo "Fold 1 >= 70% -- running all remaining folds"
    echo ""

    # Phase 2: Remaining folds
    for fold in 2 3 4 5; do
        run_fold $fold
    done

    # Aggregate
    echo ""
    echo "================================================================"
    echo "E2E RESULTS SUMMARY"
    echo "================================================================"
    for fold in 1 2 3 4 5; do
        f="${OUTPUT_BASE}/fold_${fold}/results/metrics.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
r = json.load(open('$f'))
t = r['test_metrics']
frozen = {1: 0.6667, 2: 0.8091, 3: 0.8381, 4: 0.8981, 5: 0.9140}
delta = (t['sequence_accuracy'] - frozen[$fold]) * 100
print(f'  Fold $fold: seq={t[\"sequence_accuracy\"]:.1%} ({delta:+.1f}pp vs frozen) tok={t[\"token_accuracy\"]:.1%} cmd={t[\"command_accuracy\"]:.1%} num={t[\"numeric_accuracy\"]:.1%} (best ep {r[\"best_epoch\"]})')
"
        fi
    done
    python3 -c "
import json, numpy as np
seqs = []
for fold in range(1, 6):
    f = '${OUTPUT_BASE}/fold_{}/results/metrics.json'.format(fold)
    try:
        r = json.load(open(f))
        seqs.append(r['test_metrics']['sequence_accuracy'])
    except: pass
if seqs:
    print(f'  Mean: {np.mean(seqs):.1%} +/- {np.std(seqs):.1%}')
    print(f'  Frozen baseline mean: 82.5%')
    print(f'  V7 f110 mean: 86.0%')
"
else
    echo "Fold 1 < 70% -- stopping. E2E Approach A did not meet threshold."
fi

echo ""
echo "================================================================"
echo "E2E PILOT COMPLETE -- $(date)"
echo "================================================================"
