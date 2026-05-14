#!/usr/bin/env bash
# Phase F-lite: retrain the encoder on V8 per_row data (65x more samples)
# with reconstruction weight bumped from 0.1 -> 1.0 to balance with the
# classification objective. Same architecture as the published encoder.
#
# This addresses two of the three Phase F design findings (data scale,
# loss balance) without needing the auxiliary row-level heads. A full
# Phase F auxiliary-head retrain remains a future-work direction.
#
# Cost: ~5-10 min per fold x 5 folds = ~30-60 min on one A6000.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/encoder_phase_f_lite}"
DATA_ROOT="${DATA_ROOT:-outputs/decoder20260511/preprocessed_f98/per_row}"
RECON_WEIGHT="${RECON_WEIGHT:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
PATIENCE="${PATIENCE:-15}"

mkdir -p "${OUT_ROOT}"
echo "=== Phase F-lite encoder retrain ==="
echo "data:    ${DATA_ROOT}"
echo "output:  ${OUT_ROOT}"
echo "recon_w: ${RECON_WEIGHT} (encoder paper used 0.1)"
echo "GPU:     ${GPU}"
echo

for FOLD in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${FOLD}"
  if [[ -d "${OUT}" && -f "${OUT}/checkpoint/best_val.pt" ]]; then
    echo "fold_${FOLD}: already trained, skipping"
    continue
  fi
  echo "--- fold ${FOLD} ---"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_9class_direct.py \
    --data-dir "${DATA_ROOT}/fold_${FOLD}" \
    --output-dir "${OUT}" \
    --iteration "phase_f_lite_fold${FOLD}" \
    --lr 1e-3 \
    --label_smoothing 0.1 \
    --dropout 0.2 \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --modality_dropout 0.1 \
    --recon_weight "${RECON_WEIGHT}" \
    --d_model 256 --n_heads 4 --lstm_layers 2 \
    --ablation full --seed 42 2>&1 | tail -8
done

echo "=== Phase F-lite encoder retrain complete ==="
echo "Next: train decoder against the new encoder via"
echo "  bash scripts/experiments/phase_f_decoder_retrain.sh"
