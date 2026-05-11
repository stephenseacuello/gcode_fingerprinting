#!/usr/bin/env bash
# Re-preprocess V8 with f98 (no proximity, no pressure) so the existing
# frozen encoder (no_proximity_no_pressure_w256_s64_cv) can consume it.
#
# Output: outputs/decoder20260511/preprocessed_f98/{full_window,per_row}/fold_{1..5}/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-data_clean}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/preprocessed_f98}"

for MODE in full_window per_row; do
  for FOLD in 1 2 3 4 5; do
    OUT="${OUT_ROOT}/${MODE}/fold_${FOLD}"
    echo
    echo "=== V8 f98 preprocess: mode=${MODE} fold=${FOLD} ==="
    python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUT}" \
      --vocab-path "${VOCAB}" \
      --fold "${FOLD}" --n-folds 5 \
      --window-size 256 --stride 64 \
      --label-mode "${MODE}" \
      --exclude-proximity --exclude-pressure
  done
done

echo "=== diagnostics ==="
PYTHONPATH=src python3 -m miracle.dataset.preprocessing_diagnostics \
  --input-dir "${OUT_ROOT}" \
  --output outputs/decoder20260511/audit/diagnostics_v8_f98.json
