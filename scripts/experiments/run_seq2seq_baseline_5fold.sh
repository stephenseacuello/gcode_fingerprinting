#!/usr/bin/env bash
# =============================================================================
# 5-fold sweep of the from-scratch seq2seq baseline (addresses R1 reviewer
# concern about architectural novelty). Two-GPU fold-sharded execution:
#   Worker A : GPU0, folds 1 2 3   (3 cells)
#   Worker B : GPU1, folds 4 5     (2 cells)
#
# Per cell ~1-2 h on RTX A6000; full sweep ~5-7 h wall.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="outputs/decoder20260511/seq2seq_baseline"
DATA="outputs/decoder20260511/preprocessed_f98/full_window"
VOCAB="data/gcode_vocab_v8.json"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"

mkdir -p "$ROOT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_one() {
  local GPU="$1" FOLD="$2"
  local OUT="${ROOT}/fold_${FOLD}"
  mkdir -p "$OUT"
  if [[ -s "${OUT}/results/metrics.json" ]]; then
    echo "[$(ts)] SKIP fold=${FOLD} (metrics.json already exists)"
    return 0
  fi
  echo "[$(ts)] START GPU${GPU} fold=${FOLD}"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/evaluation/run_seq2seq_baseline.py \
    --data_dir "${DATA}/fold_${FOLD}" --vocab "$VOCAB" --fold "$FOLD" \
    --output_dir "$OUT" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch_size 8 --lr 1e-4 --warmup_epochs 5 --weight_decay 0.05 \
    --d_model 256 --n_layers_enc 4 --n_layers_dec 4 --n_heads 8 --ff_dim 1024 \
    --dropout 0.1 --max_token_len 1400 --seed 42 \
    --device "cuda:0" --num_workers 2 \
    > "${OUT}/training.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then echo "[$(ts)] DONE GPU${GPU} fold=${FOLD}"
  else echo "[$(ts)] FAIL GPU${GPU} fold=${FOLD} rc=${rc}"; fi
}

worker_A() { for f in 1 2 3; do run_one 0 "$f"; done; }
worker_B() { for f in 4 5;   do run_one 1 "$f"; done; }

echo "[$(ts)] launching worker A (GPU0, folds 1 2 3) and worker B (GPU1, folds 4 5)"
worker_A > "${ROOT}/_workerA.log" 2>&1 &
PA=$!
worker_B > "${ROOT}/_workerB.log" 2>&1 &
PB=$!
wait "$PA" || echo "[$(ts)] worker A non-zero exit"
wait "$PB" || echo "[$(ts)] worker B non-zero exit"
echo "[$(ts)] 5-fold sweep complete. Summary:"
for f in 1 2 3 4 5; do
  M="${ROOT}/fold_${f}/results/metrics.json"
  if [[ -s "$M" ]]; then
    python3 -c "
import json
d = json.load(open('${M}'))['test_metrics']
print(f'  fold ${f}: command={d.get(\"command_accuracy\",0):.4f} token={d.get(\"token_accuracy\",0):.4f} command_ar={d.get(\"command_accuracy_ar\",0):.4f}')
"
  else
    echo "  fold ${f}: MISSING"
  fi
done
