#!/bin/bash
################################################################################
# V8 Preprocessing: f98 w64_s16 for decoder
#
# Re-preprocesses all 5 folds from data_clean/ with:
#   - f98 features (no proximity, no pressure)
#   - window=64, stride=16 (matching the best f98 encoder)
#   - window_index, total_windows, source_file metadata in NPZ
#   - Stratified splits with sequence coverage repair
#
# Output: outputs/decoder20260304/preprocessed_v8_f98_w64/fold_{1-5}/
# Encoder checkpoints: outputs/experiments_2026_02_25/no_proximity_no_pressure_w64_s16_cv/
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data_clean"
VOCAB_PATH="data/gcode_vocab_712.json"
SENSOR_REPORT="outputs/sensor_consistency_report.json"
OUTPUT_BASE="outputs/decoder20260304/preprocessed_v8_f98_w64"
WINDOW_SIZE=64
STRIDE=16
N_FOLDS=5

echo "================================================================================"
echo "V8 PREPROCESSING: f98 w64_s16 for decoder"
echo "================================================================================"
echo "Data source: $DATA_DIR"
echo "Vocab: $VOCAB_PATH"
echo "Window: ${WINDOW_SIZE}, Stride: ${STRIDE}"
echo "Features: f98 (no proximity, no pressure)"
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
        --stride "$STRIDE" \
        --exclude-proximity \
        --exclude-pressure

    echo "Fold $FOLD complete."
done

echo ""
echo "================================================================================"
echo "V8 PREPROCESSING (f98 w64_s16) COMPLETE: $(date)"
echo "Output: $OUTPUT_BASE"
echo "================================================================================"

# Quick verification
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
        w = d['continuous'].shape[1]
        f = d['continuous'].shape[2]
        has_wi = 'window_index' in d
        has_sf = 'source_file' in d
        print(f'  fold_{fold}/{split}: {n} samples, window={w}, features={f}, window_index={has_wi}, source_file={has_sf}')
print('Verification done.')
"
