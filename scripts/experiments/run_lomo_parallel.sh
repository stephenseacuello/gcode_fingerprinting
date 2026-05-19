#!/usr/bin/env bash
# =============================================================================
# Two-GPU parallel orchestrator for the LOMO encoder study.
#
# Fold-sharded so the two workers are FULLY independent (no cross-worker
# dependency): each fold is owned end-to-end by one worker, including that
# fold's baseline reference cell that its modality gates check against.
#
#   Worker A  -> GPU $GA, folds 1 2 3, lomo arm   (baseline + 7 mod = 24 cells)
#   Worker B  -> GPU $GB, folds 4 5,   lomo arm   (baseline + 7 mod = 16 cells)
#                          then        suffic arm (7 mod x fold1   =  7 cells)
#
# This script is the single long-lived process: it launches both workers and
# `wait`s, so the harness tracks it for the whole run and notifies on real
# completion (no nested-nohup foot-gun). Per-worker tagged summaries avoid the
# shared-file clobber; merged at the end.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${ROOT:-outputs/decoder20260511/lomo}"
DRIVER="scripts/experiments/run_lomo_encoder_study.sh"
mkdir -p "$ROOT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# --- pick two free GPUs (mem.used < 2 GiB); degrade to 1 worker if only one --
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', *' '{ if (($2+0) < 2000) print $1 }')
echo "[$(ts)] free GPUs: ${FREE[*]:-NONE}"
if [ "${#FREE[@]}" -ge 2 ]; then
  GA="${FREE[0]}"; GB="${FREE[1]}"; MODE="2-GPU parallel"
elif [ "${#FREE[@]}" -eq 1 ]; then
  GA="${FREE[0]}"; GB="${FREE[0]}"; MODE="1-GPU (only ${GA} free) - serial fallback"
else
  echo "[$(ts)] NO free GPU -> abort"; exit 2
fi
echo "[$(ts)] ${MODE}: workerA=GPU${GA} (folds 1 2 3)  workerB=GPU${GB} (folds 4 5 + suffic)"

# --- Worker A: GPU GA, folds 1-3, lomo arm --------------------------------
( GPU="$GA" FOLDS="1 2 3" SUMMARY_TAG="A_f123" bash "$DRIVER" ) \
  > "${ROOT}/_workerA.log" 2>&1 &
PA=$!
echo "[$(ts)] workerA pid=$PA -> ${ROOT}/_workerA.log"

# --- Worker B: GPU GB, folds 4-5 lomo, THEN suffic fold-1 (sequential) -----
( GPU="$GB" FOLDS="4 5" SUMMARY_TAG="B_f45" bash "$DRIVER" \
  && GPU="$GB" SUMMARY_TAG="suffic" bash "$DRIVER" --suffic ) \
  > "${ROOT}/_workerB.log" 2>&1 &
PB=$!
echo "[$(ts)] workerB pid=$PB -> ${ROOT}/_workerB.log"

# --- block until BOTH finish (keeps harness tracking the whole run) --------
rc=0
wait "$PA" || { echo "[$(ts)] workerA exited rc=$?"; rc=1; }
wait "$PB" || { echo "[$(ts)] workerB exited rc=$?"; rc=1; }

# --- merge per-worker summaries -------------------------------------------
MERGED="${ROOT}/lomo_run_summary_ALL.tsv"
{ head -1 "$(ls -1 ${ROOT}/lomo_run_summary_*.tsv 2>/dev/null | head -1)" 2>/dev/null
  grep -hv '^modality' ${ROOT}/lomo_run_summary_*.tsv 2>/dev/null | sort -u
} > "$MERGED" 2>/dev/null
echo "[$(ts)] ALL WORKERS DONE (rc=$rc). Merged summary:"
column -t -s$'\t' "$MERGED" 2>/dev/null || cat "$MERGED" 2>/dev/null
echo "[$(ts)] free disk now: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')GB"
exit "$rc"
