#!/usr/bin/env bash
# AR re-eval WITH --fsm_grammar (higher-order grammar rules enforced at
# inference). Writes beam_1_metrics.json to a separate fsm_results/ subdir
# so the bigram-only AR results we already have aren't overwritten.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-1}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
SWEEPS="${SWEEPS:-full_window_5fold full_window_5fold_with_shortcuts}"

for SWEEP in $SWEEPS; do
  for FOLD in 1 2 3 4 5; do
    CKPT=$(find outputs/decoder20260511/checkpoints/${SWEEP} -path "*fold_${FOLD}*/decoder_checkpoint/best_decoder.pt" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then echo "skip ${SWEEP}/fold_${FOLD}: no checkpoint"; continue; fi

    CKPT_PARENT=$(dirname $(dirname "$CKPT"))
    OUT_DIR="${CKPT_PARENT}_fsm"

    if [ -f "${OUT_DIR}/results/beam_1_metrics.json" ]; then
      echo "skip ${SWEEP}/fold_${FOLD}: FSM AR already exists at ${OUT_DIR}/results/beam_1_metrics.json"
      continue
    fi

    DATA_DIR="outputs/decoder20260511/preprocessed_f98/full_window/fold_${FOLD}"
    WIN_POS=false; MWC=0; WIN_DRP=0.0
    if [[ "$SWEEP" == *"with_shortcuts"* ]]; then
      WIN_POS=true; MWC=2; WIN_DRP=0.1
    fi
    MAX_TOK=1400
    [[ "$SWEEP" == *"per_row"* ]] && MAX_TOK=32

    mkdir -p "${OUT_DIR}/results"
    echo
    echo "=== FSM AR re-eval: ${SWEEP} fold ${FOLD} (GPU ${GPU}) ==="
    CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_decoder_quick_test.py \
      --eval_only \
      --checkpoint "${CKPT}" \
      --data_dir "${DATA_DIR}" \
      --encoder_config f98_w256_s64 --fold "${FOLD}" \
      --vocab "${VOCAB}" \
      --output_dir "${OUT_DIR}" \
      --max_token_len ${MAX_TOK} --seed 42 \
      --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
      --legacy_weight 3.0 --digit_weight 1.0 \
      --memory_pos_encoding true --use_sensor_prior true --grammar_constraint true \
      --use_window_position ${WIN_POS} --multi_window_context ${MWC} --window_dropout ${WIN_DRP} \
      --device auto --beam_widths 1 \
      --fsm_grammar 2>&1 | tail -6
  done
done

echo
echo "=== FSM AR re-eval batch complete ==="
