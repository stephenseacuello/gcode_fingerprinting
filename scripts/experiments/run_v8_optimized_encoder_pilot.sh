#!/bin/bash
################################################################################
# V8 Decoder Pilot with Optimized Encoder
#
# Uses the best encoder from the f98 encoder sweep (dm256_nh4_lr0.0005_do0.3)
# with the same decoder hyperparameters as the V8 pilot, to test whether
# the +2.4pp classification gain translates to reconstruction improvement.
#
# V8 pilot (default f98 encoder): 81.1% mean seq_acc (-4.9pp vs V7)
# Question: does the optimized encoder close this gap?
#
# Decoder config: d_model=384, n_layers=8, n_heads=12, dropout=0.1
# (identical to v8_pilot)
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="src:$PYTHONPATH"

# Encoder sweep best config
ENCODER_SWEEP_DIR="outputs/decoder20260304/encoder_sweep_f98/dm256_nh4_lr0.0005_do0.3"
DATA_BASE="outputs/decoder20260304/preprocessed_v8_f98"
OUTPUT_BASE="outputs/decoder20260304/v8_optimized_encoder"
VOCAB="data/gcode_vocab_712.json"

# Which folds to run (start with fold 5 for quick signal)
FOLDS="${1:-5}"

echo "================================================================================"
echo "V8 DECODER PILOT WITH OPTIMIZED ENCODER"
echo "================================================================================"
echo "Encoder: ${ENCODER_SWEEP_DIR}"
echo "Data:    ${DATA_BASE}"
echo "Output:  ${OUTPUT_BASE}"
echo "Folds:   ${FOLDS}"
echo "Started: $(date)"
echo ""

for FOLD in $FOLDS; do
    ENCODER_CKPT="${ENCODER_SWEEP_DIR}/fold_${FOLD}/checkpoint/best_model.pt"
    DATA_DIR="${DATA_BASE}/fold_${FOLD}"
    FOLD_OUT="${OUTPUT_BASE}/fold_${FOLD}"

    if [ ! -f "$ENCODER_CKPT" ]; then
        echo "ERROR: Encoder checkpoint not found: $ENCODER_CKPT"
        continue
    fi

    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        continue
    fi

    # Skip if already completed
    if [ -f "${FOLD_OUT}/results/metrics.json" ]; then
        echo "Fold $FOLD: already done, skipping"
        echo "  $(python3 -c "import json; r=json.load(open('${FOLD_OUT}/results/metrics.json')); print(f\"seq_acc={r['test_metrics']['sequence_accuracy']:.4f}, tok_acc={r['test_metrics']['token_accuracy']:.4f}\")")"
        continue
    fi

    echo "[$FOLD] Starting fold $FOLD..."
    echo "  Encoder: $ENCODER_CKPT"
    echo "  Data:    $DATA_DIR"

    python3 scripts/evaluation/run_decoder_quick_test.py \
        --data_dir "$DATA_DIR" \
        --encoder_ckpt "$ENCODER_CKPT" \
        --vocab "$VOCAB" \
        --output_dir "$FOLD_OUT" \
        --epochs 500 \
        --batch_size 32 \
        --lr 0.0003 \
        --patience 120 \
        --seed 42 \
        --d_model 384 \
        --n_layers 8 \
        --n_heads 12 \
        --dropout 0.1 \
        --digit_weight 15.0 \
        --legacy_weight 1.0 \
        --label_smoothing 0.1 \
        --focal_gamma 2.0 \
        --scheduled_sampling 0.5 \
        --memory_pos_encoding true \
        --multi_window_context 2 \
        --use_window_position true \
        --use_sensor_prior true \
        --weight_decay 0.1 \
        --warmup_epochs 10 \
        --beam_width 1

    # Report result
    if [ -f "${FOLD_OUT}/results/metrics.json" ]; then
        python3 -c "
import json
r = json.load(open('${FOLD_OUT}/results/metrics.json'))
t = r['test_metrics']
v = r['val_metrics']
print(f'  Fold $FOLD DONE (best epoch {r[\"best_epoch\"]}):')
print(f'    Test:  seq_acc={t[\"sequence_accuracy\"]:.4f}  tok_acc={t[\"token_accuracy\"]:.4f}  cmd={t[\"command_accuracy\"]:.4f}  num={t[\"numeric_accuracy\"]:.4f}')
print(f'    Val:   seq_acc={v[\"sequence_accuracy\"]:.4f}  tok_acc={v[\"token_accuracy\"]:.4f}')
"
    else
        echo "  Fold $FOLD FAILED (no metrics.json)"
    fi
    echo ""
done

echo "================================================================================"
echo "COMPLETE: $(date)"
echo "================================================================================"

# If all 5 folds are done, summarize
if [ -f "${OUTPUT_BASE}/fold_1/results/metrics.json" ] && \
   [ -f "${OUTPUT_BASE}/fold_2/results/metrics.json" ] && \
   [ -f "${OUTPUT_BASE}/fold_3/results/metrics.json" ] && \
   [ -f "${OUTPUT_BASE}/fold_4/results/metrics.json" ] && \
   [ -f "${OUTPUT_BASE}/fold_5/results/metrics.json" ]; then
    echo ""
    echo "ALL 5 FOLDS COMPLETE — Summary:"
    python3 -c "
import json
import numpy as np

seq_accs = []
tok_accs = []
for fold in range(1, 6):
    r = json.load(open(f'${OUTPUT_BASE}/fold_{fold}/results/metrics.json'))
    t = r['test_metrics']
    seq_accs.append(t['sequence_accuracy'])
    tok_accs.append(t['token_accuracy'])
    print(f'  Fold {fold}: seq_acc={t[\"sequence_accuracy\"]:.4f}  tok_acc={t[\"token_accuracy\"]:.4f}')

print(f'')
print(f'  Mean seq_acc: {np.mean(seq_accs):.4f} +/- {np.std(seq_accs):.4f}')
print(f'  Mean tok_acc: {np.mean(tok_accs):.4f} +/- {np.std(tok_accs):.4f}')
print(f'')
print(f'  Compare to V8 pilot (default encoder): 81.1% mean seq_acc')
print(f'  Compare to V7 (f110):                  85.9% mean seq_acc')
"
fi
