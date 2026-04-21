#!/bin/bash
################################################################################
# V8 Preprocessing: Multi-line G-code tokenization
#   - Same as V7 (window=256, stride=64, 110 features) BUT
#   - Tokenizes ALL unique G-code lines per window, not just most-frequent
#   - This produces ~20-60 tokens per window instead of ~3-6
#
# Output: outputs/anomaly20260319/preprocessed_v8_multiline/fold_{1-5}/
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data_clean"
VOCAB_PATH="data/gcode_vocab_712.json"
SENSOR_REPORT="outputs/sensor_consistency_report.json"
OUTPUT_BASE="outputs/anomaly20260319/preprocessed_v8_multiline"
WINDOW_SIZE=256
STRIDE=64
N_FOLDS=5

echo "================================================================================"
echo "V8 PREPROCESSING: Multi-line G-code tokenization"
echo "================================================================================"
echo "Data source: $DATA_DIR"
echo "Vocab: $VOCAB_PATH"
echo "Window: ${WINDOW_SIZE}, Stride: ${STRIDE}"
echo "Output: $OUTPUT_BASE"
echo "Started at: $(date)"
echo ""

for FOLD in $(seq 1 $N_FOLDS); do
    echo ""
    echo "================================================================================"
    echo "FOLD ${FOLD} / ${N_FOLDS}"
    echo "================================================================================"

    PREPROCESS_DIR="${OUTPUT_BASE}/fold_${FOLD}"
    mkdir -p "$PREPROCESS_DIR"

    python3 romesh_changes/run_preprocessing_cv_fold.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PREPROCESS_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --sensor-report "$SENSOR_REPORT" \
        --threshold 93.0 \
        --fold "$FOLD" \
        --n-folds "$N_FOLDS" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE"

    echo "Fold $FOLD complete."
done

echo ""
echo "================================================================================"
echo "V8 PREPROCESSING COMPLETE: $(date)"
echo "Output: $OUTPUT_BASE"
echo "================================================================================"

# Verification
echo ""
echo "Verifying NPZ contents..."
python3 -c "
import numpy as np
from pathlib import Path
base = Path('$OUTPUT_BASE')
for fold in range(1, 6):
    for split in ['train', 'val', 'test']:
        npz = base / f'fold_{fold}' / f'{split}_sequences.npz'
        d = np.load(npz, allow_pickle=True)
        n = d['continuous'].shape[0]
        tok_shape = d['tokens'].shape
        avg_len = sum(1 for row in d['tokens'] for t in row if t != 0) / n
        print(f'  fold_{fold}/{split}: {n} samples, tokens={tok_shape}, avg_tokens={avg_len:.1f}')
print('Verification done.')
"
