#!/usr/bin/env bash
# =============================================================================
# 5-fold NESTED leave-one-(sensor, modality)-out study.
# 37 conditions (baseline + 6 sensors x 6 modalities) x 5 folds = 185 cells.
# Fold-sharded across 2 GPUs:
#   Worker A : GPU0, folds 1 2 3  (111 cells)
#   Worker B : GPU1, folds 4 5    ( 74 cells)
# Each fold is owned end-to-end by one worker (its baseline cell is built
# first and is the gate reference for that fold's nested cells), so the two
# workers are fully independent. ~50 min/cell on the 2-GPU shard.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="outputs/decoder20260511/lomo_nested"
DRIVER="scripts/experiments/run_lomo_nested_study.sh"

SENSORS="frame_l2 frame_l3 frame_r2 spindle2 y_bed__3 y_bed__4"
MODALITIES="accelerometer gyroscope magnetometer color temperature audio"
CELLS="baseline"
for s in $SENSORS; do for m in $MODALITIES; do CELLS+=" ${s}__${m}"; done; done

mkdir -p "$ROOT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] workerA: GPU0, folds 1 2 3 (111 cells)"
( GPU=0 CELLS="$CELLS" FOLDS="1 2 3" SUMMARY_TAG=f123 bash "$DRIVER" ) > "${ROOT}/_f123.log" 2>&1 &
PA=$!
echo "[$(ts)] workerB: GPU1, folds 4 5 (74 cells)"
( GPU=1 CELLS="$CELLS" FOLDS="4 5" SUMMARY_TAG=f45 bash "$DRIVER" ) > "${ROOT}/_f45.log" 2>&1 &
PB=$!

wait "$PA" || echo "[$(ts)] workerA exited non-zero"
echo "[$(ts)] workerA (folds 1 2 3) done"
wait "$PB" || echo "[$(ts)] workerB exited non-zero"
echo "[$(ts)] workerB (folds 4 5) done"

echo "[$(ts)] ==== 5-FOLD NESTED LEAVE-ONE-(SENSOR,MODALITY)-OUT COMPLETE ===="
{ head -1 "${ROOT}"/summary_f123.tsv 2>/dev/null
  grep -hv '^cell' "${ROOT}"/summary_*.tsv 2>/dev/null | sort -u; } > "${ROOT}/summary_ALL.tsv"
column -t -s$'\t' "${ROOT}/summary_ALL.tsv" 2>/dev/null || cat "${ROOT}/summary_ALL.tsv"
echo "[$(ts)] next: nested aggregation -> python3 scripts/analysis/aggregate_lomo_results.py --root $ROOT --group-kind nested --write-figure"
