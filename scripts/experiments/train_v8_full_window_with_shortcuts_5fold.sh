#!/usr/bin/env bash
# Phase B (post-full_window): V8 full_window 5-fold WITH positional metadata
# exposed to the decoder. Tests the 2x2 ablation cell missing from the
# original plan: (full_window, shortcuts).
#
# Same config as train_v8_full_window_5fold.sh except --use_window_position=true,
# --multi_window_context=2, --window_dropout=0.1.
#
# Resolves recommendation #16 in notes.md.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder20260511/checkpoints/full_window_5fold_with_shortcuts}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
BATCH="${BATCH:-4}"

for F in 1 2 3 4 5; do
  OUT="${OUT_ROOT}/fold_${F}"
  if [ -f "${OUT}/decoder_checkpoint/best_decoder.pt" ] || \
     [ -f "${OUT}"/*/decoder_checkpoint/best_decoder.pt ]; then
    echo "  skip fold ${F}: already trained"
    continue
  fi
  echo
  echo "=== V8 full_window + shortcuts fold ${F} ==="
  python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "outputs/decoder20260511/preprocessed_f98/full_window/fold_${F}" \
    --encoder_config f98_w256_s64 \
    --fold "${F}" \
    --vocab "${VOCAB}" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch_size "${BATCH}" \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --max_token_len "${MAX_TOKEN_LEN}" \
    --seed 42 \
    --d_model 384 --n_layers 8 --n_heads 12 \
    --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 \
    --memory_pos_encoding true \
    --use_sensor_prior true \
    --grammar_constraint true \
    --use_window_position true \
    --multi_window_context 2 \
    --window_dropout 0.1 \
    --scheduled_sampling 0.5 \
    --wandb --wandb_project gcode-decoder-2026 \
    --device auto 2>&1 | tail -5
done

echo
echo "=== full_window + shortcuts 5-fold COMPLETE ==="
