#!/usr/bin/env bash
# Run V8 preprocessing for both label modes across all 5 CV folds.
#
# Output structure:
#   outputs/decoder20260511/preprocessed/
#     full_window/fold_1..5/{train,val,test}_sequences.npz + metadata
#     per_row/fold_1..5/{train,val,test}_sequences.npz + metadata
#
# Phase-2 of the decoder20260511 remediation. See plan and AUDIT_REPORT.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-data_clean}"   # V7 was preprocessed from data_clean (120 files)
VOCAB="${VOCAB:-data/gcode_vocab_712.json}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/preprocessed}"
WINDOW="${WINDOW:-256}"
STRIDE="${STRIDE:-64}"

for MODE in full_window per_row; do
  for FOLD in 1 2 3 4 5; do
    OUT="${OUT_ROOT}/${MODE}/fold_${FOLD}"
    echo
    echo "=================================================="
    echo "  V8 preprocess: mode=${MODE} fold=${FOLD}"
    echo "  out -> ${OUT}"
    echo "=================================================="
    python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUT}" \
      --vocab-path "${VOCAB}" \
      --fold "${FOLD}" \
      --n-folds 5 \
      --window-size "${WINDOW}" \
      --stride "${STRIDE}" \
      --label-mode "${MODE}"
  done
done

echo
echo "Done. Running diagnostic probe..."
python3 scripts/analysis/diagnose_decoder_npz.py \
  --input-dir "${OUT_ROOT}" \
  --output outputs/decoder20260511/audit/diagnostics_v8.json \
  --markdown outputs/decoder20260511/audit/diagnostics_v8.md \
  --max-token-len 64
echo "Diagnostics: outputs/decoder20260511/audit/diagnostics_v8.{json,md}"
