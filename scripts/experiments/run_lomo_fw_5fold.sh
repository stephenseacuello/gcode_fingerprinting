#!/usr/bin/env bash
# =============================================================================
# Full_window LOMO study -- folds 2-5 (fold 1 is the already-completed pilot).
# 8 modalities x 4 folds = 32 cells, ~1h each, ~16h on two GPUs.
#
#   Worker A : GPU0, folds 2 3  -- starts immediately (GPU0 is free)
#   Worker B : GPU1, folds 4 5  -- starts once the fold-1 audio cell frees GPU1
#
# Each fold is owned end-to-end by one worker (its baseline cell is built first
# and is the gate reference for that fold's modality cells) -> the two workers
# are fully independent. Aggregates all 5 folds at the end.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="outputs/decoder20260511/lomo_fw"
DRIVER="scripts/experiments/run_lomo_encoder_study.sh"
MODS="baseline accelerometer gyroscope magnetometer color temperature audio electrical"
AUD="${ROOT}/lomo/audio/fold_1/decoder/results/metrics.json"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] workerA: GPU0, folds 2 3 -> starting now"
( GPU=0 MODALITIES="$MODS" FOLDS="2 3" SUMMARY_TAG=f23 bash "$DRIVER" ) > "${ROOT}/_f23.log" 2>&1 &
PA=$!

echo "[$(ts)] waiting for the fold-1 audio cell to free GPU1 (cap 120 min)..."
for _ in $(seq 1 120); do
  [ -f "$AUD" ] && { echo "[$(ts)] audio fold-1 done -> GPU1 free"; break; }
  sleep 60
done
echo "[$(ts)] workerB: GPU1, folds 4 5 -> starting"
( GPU=1 MODALITIES="$MODS" FOLDS="4 5" SUMMARY_TAG=f45 bash "$DRIVER" ) > "${ROOT}/_f45.log" 2>&1 &
PB=$!

wait "$PA" || echo "[$(ts)] workerA exited non-zero"
echo "[$(ts)] workerA (folds 2 3) done"
wait "$PB" || echo "[$(ts)] workerB exited non-zero"
echo "[$(ts)] workerB (folds 4 5) done"

echo "[$(ts)] ==== 5-FOLD FULL_WINDOW LOMO COMPLETE ===="
{ head -1 "${ROOT}"/summary_baseline.tsv 2>/dev/null
  grep -hv '^modality' "${ROOT}"/summary_*.tsv 2>/dev/null | sort -u; } > "${ROOT}/summary_ALL.tsv"
column -t -s$'\t' "${ROOT}/summary_ALL.tsv" 2>/dev/null || cat "${ROOT}/summary_ALL.tsv"
echo
echo "[$(ts)] aggregating 5-fold attribution + stats..."
python3 scripts/analysis/aggregate_lomo_results.py --root "$ROOT" 2>&1 | tail -32
echo "[$(ts)] done. Review, then re-run aggregator with --write-table --write-figure to fill the paper."
