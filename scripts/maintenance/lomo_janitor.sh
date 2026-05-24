#!/usr/bin/env bash
# =============================================================================
# LOMO janitor — non-disruptive disk cleanup for cells whose decoder finished.
#
# Each completed decoder dir holds ~14.8 GB of regenerable artifacts (encoder
# memory caches + per-step training_history). At 47 cells x ~14.8 GB the
# sweep would trip the 80 GB disk guard ~halfway through. This janitor removes
# only the SAFE-TO-DROP files from FINISHED cells while the sweep is running.
#
# Safety predicates (ALL must hold before a cell is touched):
#   (1) `decoder/results/metrics.json` exists  (decoder finished writing)
#   (2) No live python process has this cell's per_row dir on its cmdline
#       (defense in depth against any weird mid-write state)
#
# Per eligible cell, DELETE:
#   - decoder/encoder_memory/{train,test,val,}memory.pt        (~13.4 GB)
#   - decoder/results/training_history.json                    (~1.3 GB)
#   - decoder/decoder_checkpoint/final_decoder.pt              (~120 MB)
# KEEP:
#   - decoder/results/metrics.json                 (attribution-table input)
#   - decoder/results/predictions.npz              (test predictions)
#   - decoder/results/{training_log.txt,sample_generations.txt}
#   - decoder/decoder_checkpoint/best_decoder.pt   (re-eval / verification)
#   - decoder/encoder_memory/*_op_pred.pt          (tiny, may be useful)
#   - everything else under encoder/ and per_row/  (already managed by driver)
#
# Exits when the orchestrator process is gone (sweep done) -> self-terminates.
# Poll interval: 600s (cheap; freed space helps the NEXT cell's PRE/data NPZ).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
LR="outputs/decoder20260511/lomo"
LOG="${LR}/_janitor.log"
mkdir -p "$LR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

POLL_S="${POLL_S:-600}"
DRY_RUN="${DRY_RUN:-0}"   # set DRY_RUN=1 to list what WOULD be deleted, no rm

say "janitor start (poll=${POLL_S}s, dry_run=${DRY_RUN})"

cell_in_use() {  # $1 = cell dir; 0 if a python proc references its per_row
  local cell="$1"
  pgrep -fa python3 2>/dev/null | grep -qF "$cell/per_row"
}

clean_cell() {  # $1 = cell dir (.../<modality>/fold_N)
  local cell="$1"
  local dec="$cell/decoder"
  [ -f "$dec/results/metrics.json" ] || return 0      # not finished
  cell_in_use "$cell" && return 0                      # still in use (paranoid)
  local em="$dec/encoder_memory"
  local victims=(
    "$em/train_memory.pt"
    "$em/memory.pt"
    "$em/test_memory.pt"
    "$em/val_memory.pt"
    "$dec/results/training_history.json"
    "$dec/decoder_checkpoint/final_decoder.pt"
  )
  local freed_kb=0 f
  for f in "${victims[@]}"; do
    [ -f "$f" ] || continue
    local kb; kb=$(du -k "$f" 2>/dev/null | cut -f1); kb=${kb:-0}
    freed_kb=$((freed_kb + kb))
    if [ "$DRY_RUN" = 1 ]; then
      say "  WOULD rm ($(numfmt --to=iec --from-unit=1024 "$kb")B) $f"
    else
      rm -f "$f" && say "  rm ($(numfmt --to=iec --from-unit=1024 "$kb")B) $f"
    fi
  done
  if [ "$freed_kb" -gt 0 ]; then
    say "cleaned $cell (~$(numfmt --to=iec --from-unit=1024 "$freed_kb")B freed)"
  fi
}

pass() {
  local n=0 mb_before mb_after
  mb_before=$(df --output=avail "$LR" 2>/dev/null | tail -1)
  while IFS= read -r cell; do
    [ -d "$cell/decoder" ] || continue
    clean_cell "$cell"
    n=$((n+1))
  done < <(find "$LR"/{lomo,suffic} -mindepth 2 -maxdepth 2 -type d 2>/dev/null)
  mb_after=$(df --output=avail "$LR" 2>/dev/null | tail -1)
  local diff_kb=$((mb_after - mb_before))
  say "pass done: scanned $n cells; free-space delta ${diff_kb}KB; total free now $(df -BG --output=avail "$LR" | tail -1 | tr -dc 0-9)GB"
}

while :; do
  pgrep -f '[r]un_lomo_parallel\.sh' >/dev/null 2>&1 || { say "orchestrator gone -> janitor exit"; break; }
  pass
  sleep "$POLL_S"
done
say "janitor end"
