#!/usr/bin/env bash
# =============================================================================
# 5-fold leave-one-SENSOR-out study. 7 conditions (baseline + 6 physical
# sensors) x 5 folds = 35 cells. Fold-sharded across 2 GPUs:
#   Worker A : GPU0, folds 1 2 3  (21 cells)
#   Worker B : GPU1, folds 4 5    (14 cells)
# Each fold is owned end-to-end by one worker (its baseline cell is built
# first and is the gate reference for that fold's sensor cells), so the two
# workers are fully independent. ~1.3 h/cell -> ~27 h wall.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="outputs/decoder20260511/lomo_sensor"
DRIVER="scripts/experiments/run_lomo_sensor_study.sh"
SENSORS="baseline frame_l2 frame_l3 frame_r2 spindle2 y_bed__3 y_bed__4"
mkdir -p "$ROOT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] workerA: GPU0, folds 1 2 3 (21 cells)"
( GPU=0 SENSORS="$SENSORS" FOLDS="1 2 3" SUMMARY_TAG=f123 bash "$DRIVER" ) > "${ROOT}/_f123.log" 2>&1 &
PA=$!
echo "[$(ts)] workerB: GPU1, folds 4 5 (14 cells)"
( GPU=1 SENSORS="$SENSORS" FOLDS="4 5" SUMMARY_TAG=f45 bash "$DRIVER" ) > "${ROOT}/_f45.log" 2>&1 &
PB=$!

wait "$PA" || echo "[$(ts)] workerA exited non-zero"
echo "[$(ts)] workerA (folds 1 2 3) done"
wait "$PB" || echo "[$(ts)] workerB exited non-zero"
echo "[$(ts)] workerB (folds 4 5) done"

echo "[$(ts)] ==== 5-FOLD LEAVE-ONE-SENSOR-OUT COMPLETE ===="
{ head -1 "${ROOT}"/summary_f123.tsv 2>/dev/null
  grep -hv '^sensor' "${ROOT}"/summary_*.tsv 2>/dev/null | sort -u; } > "${ROOT}/summary_ALL.tsv"
column -t -s$'\t' "${ROOT}/summary_ALL.tsv" 2>/dev/null || cat "${ROOT}/summary_ALL.tsv"
echo "[$(ts)] next: sensor-aware aggregation (aggregate_lomo_results.py adapted for --root $ROOT)"
