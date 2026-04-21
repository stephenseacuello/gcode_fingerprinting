#!/bin/bash
################################################################################
# V8 Phase 3a: Encoder Fine-Tuning Pilot (Fold 1)
#
# Tests 4 fine-tuning variants on fold 1 (the bottleneck at 66.7% seq_acc).
# Uses Approach B: re-cache encoder memory between rounds.
# Encoder trains on CLS + 0.1*DTAE recon loss, NOT decoder loss.
#
# Baseline (frozen optimized f98): fold 1 = 66.7% seq_acc
# Target: >75% to proceed to Phase 3b (all 5 folds)
#
# Variants:
#   3a-1: lstm_only, lr=3e-5, from epoch 0, recache every 50
#   3a-2: last_layer, lr=3e-5, from epoch 0, recache every 50
#   3a-3: lstm_only, lr=3e-5, staged (after epoch 50), recache every 25
#   3a-4: all layers, lr=1e-5, from epoch 0, recache every 50
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="src:$PYTHONPATH"

ENCODER_SWEEP_DIR="outputs/decoder20260304/encoder_sweep_f98/dm256_nh4_lr0.0005_do0.3"
DATA_BASE="outputs/decoder20260304/preprocessed_v8_f98"
OUTPUT_BASE="outputs/decoder20260304/v8_finetune"
VOCAB="data/gcode_vocab_712.json"
FOLD=1

# Common decoder args (identical to V8 optimized encoder pilot)
COMMON_ARGS=(
    --data_dir "${DATA_BASE}/fold_${FOLD}"
    --encoder_ckpt "${ENCODER_SWEEP_DIR}/fold_${FOLD}/checkpoint/best_model.pt"
    --vocab "$VOCAB"
    --epochs 500
    --batch_size 32
    --lr 0.0003
    --patience 120
    --seed 42
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
    --multi_window_context 2
    --use_window_position true
    --use_sensor_prior true
    --weight_decay 0.1
    --warmup_epochs 10
    --beam_width 1
    --finetune_encoder
)

echo "================================================================================"
echo "V8 PHASE 3a: ENCODER FINE-TUNING PILOT (FOLD 1)"
echo "================================================================================"
echo "Encoder: ${ENCODER_SWEEP_DIR}"
echo "Data:    ${DATA_BASE}/fold_${FOLD}"
echo "Output:  ${OUTPUT_BASE}"
echo "Baseline (frozen): fold 1 = 66.7% seq_acc"
echo "Started: $(date)"
echo ""

run_variant() {
    local NAME="$1"
    local LAYERS="$2"
    local ENC_LR="$3"
    local AFTER_EPOCH="$4"
    local RECACHE="$5"
    local FOLD_OUT="${OUTPUT_BASE}/fold_${FOLD}_${NAME}"

    # Skip if already completed
    if [ -f "${FOLD_OUT}/results/metrics.json" ]; then
        echo "[$NAME] Already done, skipping"
        python3 -c "
import json
r = json.load(open('${FOLD_OUT}/results/metrics.json'))
t = r['test_metrics']
print(f'  seq_acc={t[\"sequence_accuracy\"]:.4f}  tok_acc={t[\"token_accuracy\"]:.4f}  cmd={t[\"command_accuracy\"]:.4f}  num={t[\"numeric_accuracy\"]:.4f}')
"
        echo ""
        return
    fi

    echo "[$NAME] Starting: finetune_layers=$LAYERS, encoder_lr=$ENC_LR, after_epoch=$AFTER_EPOCH, recache=$RECACHE"

    python3 scripts/evaluation/run_decoder_quick_test.py \
        "${COMMON_ARGS[@]}" \
        --output_dir "$FOLD_OUT" \
        --finetune_layers "$LAYERS" \
        --encoder_lr "$ENC_LR" \
        --finetune_after_epoch "$AFTER_EPOCH" \
        --recache_interval "$RECACHE"

    # Report result
    if [ -f "${FOLD_OUT}/results/metrics.json" ]; then
        python3 -c "
import json
r = json.load(open('${FOLD_OUT}/results/metrics.json'))
t = r['test_metrics']
v = r['val_metrics']
print(f'  [$NAME] DONE (best epoch {r[\"best_epoch\"]}):')
print(f'    Test:  seq_acc={t[\"sequence_accuracy\"]:.4f}  tok_acc={t[\"token_accuracy\"]:.4f}  cmd={t[\"command_accuracy\"]:.4f}  num={t[\"numeric_accuracy\"]:.4f}')
print(f'    Val:   seq_acc={v[\"sequence_accuracy\"]:.4f}  tok_acc={v[\"token_accuracy\"]:.4f}')
"
    else
        echo "  [$NAME] FAILED (no metrics.json)"
    fi
    echo ""
}

# Run 4 variants sequentially
run_variant "3a1_lstm_only"  "lstm_only"  "3e-5"  "0"   "50"
run_variant "3a2_last_layer" "last_layer" "3e-5"  "0"   "50"
run_variant "3a3_staged"     "lstm_only"  "3e-5"  "50"  "25"
run_variant "3a4_all"        "all"        "1e-5"  "0"   "50"

echo "================================================================================"
echo "PHASE 3a COMPLETE: $(date)"
echo "================================================================================"

# Summary
echo ""
echo "SUMMARY (fold 1, baseline frozen = 66.7%):"
echo "-------------------------------------------"
for VARIANT in 3a1_lstm_only 3a2_last_layer 3a3_staged 3a4_all; do
    FOLD_OUT="${OUTPUT_BASE}/fold_${FOLD}_${VARIANT}"
    if [ -f "${FOLD_OUT}/results/metrics.json" ]; then
        python3 -c "
import json
r = json.load(open('${FOLD_OUT}/results/metrics.json'))
t = r['test_metrics']
delta = t['sequence_accuracy'] - 0.6667
print(f'  ${VARIANT}: seq_acc={t[\"sequence_accuracy\"]:.4f} ({delta:+.1%}pp vs frozen)  tok={t[\"token_accuracy\"]:.4f}  num={t[\"numeric_accuracy\"]:.4f}')
"
    else
        echo "  ${VARIANT}: NOT COMPLETED"
    fi
done
