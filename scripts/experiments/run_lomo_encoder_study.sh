#!/usr/bin/env bash
# =============================================================================
# Leave-One-Modality-Out (LOMO) ENCODER study driver.
#
# Answers the question inference-time zeroing structurally cannot (paper #6):
# the encoder is RETRAINED with a modality removed, so it never learns to
# depend on it, then a fresh decoder is trained on that encoder.
#
# Hazard this guards against: encoder side (run_9class_direct.py) and decoder
# side (run_preprocessing_v8_cv_fold.py) historically used different modality
# taxonomies -> mismatched channels -> a confident, WRONG attribution table
# (the silent-bug class, cf. AUDIT_REPORT.md). Mitigations baked in here:
#   (a) ONE preprocessing source feeds both encoder and decoder.
#   (b) The encoder's actually-consumed column list is captured by re-calling
#       prepare_data_9class itself (its own code, no duplication).
#   (c) lomo_channel_identity_check.py runs as a HARD per-cell gate; a cell
#       that fails the gate is SKIPPED -- no GPU is spent on a bad cell.
#
# Canonical LOMO set == the paper's 7-modality ANOVA framing, on the
# established 98-feature base (proximity+pressure already dropped):
#   accelerometer gyroscope magnetometer color temperature audio electrical
#
# Modes:
#   --smoke      1 cell (gyroscope, fold 1), 2 decoder epochs. THE GATE.
#                Run and inspect this BEFORE the full sweep.
#   (default)    full 7 modalities x 5 folds.
#   --suffic     single-modality "sufficiency" arm (encoder on ONLY that
#                modality), fold 1 only -- the interpretability companion.
#
# Usage:
#   scripts/experiments/run_lomo_encoder_study.sh --smoke
#   scripts/experiments/run_lomo_encoder_study.sh            # full sweep
#   MODALITIES="gyroscope color" FOLDS="1 2" scripts/.../run_lomo_encoder_study.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- config (env-overridable) ----------------------------------------------
export PYTHONHASHSEED=0   # belt-and-suspenders determinism (the split fix uses
                          # sorted(), but pin this for any other set iteration)
DATA_DIR="${DATA_DIR:-data_clean}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
ROOT="${ROOT:-outputs/decoder20260511/lomo}"
BASE_F98="${BASE_F98:-outputs/decoder20260511/preprocessed_f98/per_row}"  # 98-ch baseline, per fold
GPU="${GPU:-0}"
SEED="${SEED:-42}"
N_FOLDS=5
MIN_FREE_GB="${MIN_FREE_GB:-80}"   # abort the sweep if free disk drops below this

# 'baseline' (no extra exclusion) is processed FIRST and is the gate's
# reference + the no-modality-removed row of the attribution table.
MODALITIES="${MODALITIES:-baseline accelerometer gyroscope magnetometer color temperature audio electrical}"
FOLDS="${FOLDS:-1 2 3 4 5}"
DEC_EPOCHS="${DEC_EPOCHS:-50}"
DEC_PATIENCE="${DEC_PATIENCE:-12}"
SMOKE=false
SUFFIC=false

for arg in "$@"; do
  case "$arg" in
    --smoke)  SMOKE=true;  MODALITIES="baseline gyroscope"; FOLDS="1"; DEC_EPOCHS=2; DEC_PATIENCE=2 ;;
    --suffic) SUFFIC=true; FOLDS="1" ;;
    *) echo "unknown arg: $arg" >&2; exit 64 ;;
  esac
done

# modality name -> the run_preprocessing_v8_cv_fold.py exclude flag
flag_for() { case "$1" in
  accelerometer) echo "--exclude-accelerometer";; gyroscope) echo "--exclude-gyroscope";;
  magnetometer)  echo "--exclude-magnetometer";;  color)     echo "--exclude-color";;
  temperature)   echo "--exclude-temperature";;   audio)     echo "--exclude-audio";;
  electrical)    echo "--exclude-electrical";;
  *) echo "BAD_MODALITY_$1"; return 1;; esac; }

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Parallel-safe: each worker instance writes its OWN tagged summary so two
# concurrent drivers don't clobber one shared file.
SUMMARY="${ROOT}/lomo_run_summary${SUMMARY_TAG:+_${SUMMARY_TAG}}.tsv"
mkdir -p "$ROOT"; : > "$SUMMARY"
echo -e "modality\tfold\tarm\tgate\tencoder_ckpt\tdecoder_out\tstatus" >> "$SUMMARY"

disk_free_gb() { df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'; }
disk_guard() {  # hard stop before a cell if disk is low -> never cause a 100% crash
  local free; free=$(disk_free_gb)
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then
    log "DISK GUARD: only ${free}GB free (< ${MIN_FREE_GB}GB) -> ABORT sweep cleanly"
    echo -e "ABORT\t-\t-\tdisk_${free}GB\t-\t-\tdisk_abort" >> "$SUMMARY"
    exit 3
  fi
}

run_cell() {  # $1=modality $2=fold $3=arm(lomo|suffic)
  local m="$1" f="$2" arm="$3"
  local cell="${ROOT}/${arm}/${m}/fold_${f}"
  local PRE="${cell}/per_row" ENC="${cell}/encoder" DEC="${cell}/decoder"
  mkdir -p "$PRE" "$ENC" "$DEC"
  # delete the multi-GB regenerable input NPZ on EVERY exit path of this cell
  # (success or any early return). Keeps metadata/splits/ckpt/decoder outputs.
  # EXCEPTION: a baseline cell's per_row IS the gate reference for every
  # modality cell of that fold -> keep it; only drop run_9class's duplicate.
  if [[ "$m" == "baseline" ]]; then
    trap 'rm -f "${ENC}"/data/*sequences*.npz 2>/dev/null' RETURN
  else
    trap 'rm -f "${PRE}"/*sequences*.npz "${ENC}"/data/*sequences*.npz 2>/dev/null' RETURN
  fi
  disk_guard
  log "==== ${arm} | modality=${m} | fold=${f} ===="

  local is_base=false; [[ "$m" == "baseline" ]] && is_base=true
  # gate reference = the fixed-code baseline cell of THIS arm/fold (NOT the
  # stale pre-determinism-fix preprocessed_f98).
  local BASE_DIR="${ROOT}/${arm}/baseline/fold_${f}/per_row"

  # --- 1. ONE preprocessing source (decoder+encoder share it) -------------
  local excl
  if $is_base; then
    excl=""                                   # baseline = full 98-ch reference
  elif [[ "$arm" == "suffic" ]]; then
    # sufficiency: keep ONLY this modality -> exclude every OTHER one
    excl=""
    for other in accelerometer gyroscope magnetometer color temperature audio electrical; do
      [[ "$other" != "$m" ]] && excl="$excl $(flag_for "$other")"
    done
  else
    excl="$(flag_for "$m")"
  fi
  log "preprocess: --exclude-proximity --exclude-pressure ${excl}"
  python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
    --data-dir "$DATA_DIR" --output-dir "$PRE" --vocab-path "$VOCAB" \
    --fold "$f" --n-folds "$N_FOLDS" --window-size 256 --stride 64 \
    --label-mode per_row --exclude-proximity --exclude-pressure $excl \
    > "${cell}/preprocess.log" 2>&1 \
    || { log "preprocess errored (see ${cell}/preprocess.log) -> skip cell"; \
         echo -e "${m}\t${f}\t${arm}\t-\t-\t-\tpreprocess_error" >> "$SUMMARY"; return 0; }

  # --- 2. capture encoder's ACTUALLY-consumed columns (its own code) ------
  python3 - "$PRE" "${ENC}/consumed_columns.json" <<'PY' || true
import sys, json, tempfile
from scripts.evaluation.run_9class_direct import prepare_data_9class
pre, out = sys.argv[1], sys.argv[2]
with tempfile.TemporaryDirectory() as td:
    keep_columns, *_ = prepare_data_9class(pre, td)
json.dump(list(keep_columns), open(out, "w"))
print(f"encoder will consume {len(keep_columns)} columns -> {out}")
PY
  if [[ ! -s "${ENC}/consumed_columns.json" ]]; then
    log "consumed-columns capture failed -> skip cell (cross-path proof impossible)"
    echo -e "${m}\t${f}\t${arm}\t-\t-\t-\tcols_error" >> "$SUMMARY"; return 0
  fi

  # --- 3. HARD GATE: cross-path channel identity --------------------------
  # baseline IS the reference -> nothing excluded -> no gate (it defines the
  # split every modality cell is checked against).
  local gate="${cell}/channel_identity.json"
  if $is_base; then
    log "BASELINE cell -> no gate (defines reference split)"
  elif python3 scripts/analysis/lomo_channel_identity_check.py \
        --modality "$m" --baseline-dir "$BASE_DIR" \
        --excluded-dir "$PRE" --encoder-cols "${ENC}/consumed_columns.json" \
        --out "$gate"; then
    log "GATE PASS"
  else
    log "GATE FAIL -> SKIPPING cell (no GPU spent). See ${gate}"
    echo -e "${m}\t${f}\t${arm}\tFAIL\t-\t-\tskipped_gate" >> "$SUMMARY"
    return 0
  fi

  # --- 4. encoder retrain (per_row ~17min train + eval) -------------------
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluation/run_9class_direct.py \
    --data-dir "$PRE" --output-dir "$ENC" \
    --iteration "lomo_${arm}_${m}_fold${f}" \
    --max_epochs 200 --patience 40 --seed "$SEED" \
    > "${cell}/encoder.log" 2>&1 \
    || { log "encoder errored (see ${cell}/encoder.log) -> skip cell"; \
         echo -e "${m}\t${f}\t${arm}\tPASS\t-\t-\tencoder_error" >> "$SUMMARY"; return 0; }
  local ckpt="${ENC}/checkpoint/best_model.pt"
  [[ -f "$ckpt" ]] || ckpt="${ENC}/best_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    log "ENCODER produced no checkpoint -> skip"; \
    echo -e "${m}\t${f}\t${arm}\tPASS\t-\t-\tno_encoder_ckpt" >> "$SUMMARY"; return 0
  fi

  # --- 5. decoder retrain on that encoder (phase_f param block) -----------
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "$PRE" --encoder_ckpt "$ckpt" --fold "$f" --vocab "$VOCAB" \
    --output_dir "$DEC" --epochs "$DEC_EPOCHS" --patience "$DEC_PATIENCE" \
    --batch_size 4 --lr 5e-5 --warmup_epochs 10 --weight_decay 0.05 \
    --seed 42 --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 --memory_pos_encoding true \
    --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 \
    --window_dropout 0.0 --scheduled_sampling 0.0 --device auto \
    > "${cell}/decoder.log" 2>&1 || { log "decoder run errored (see ${cell}/decoder.log)"; \
       echo -e "${m}\t${f}\t${arm}\tPASS\t${ckpt}\t-\tdecoder_error" >> "$SUMMARY"; return 0; }

  log "CELL DONE -> ${DEC}"
  echo -e "${m}\t${f}\t${arm}\tPASS\t${ckpt}\t${DEC}\tok" >> "$SUMMARY"
}

ARM=$([[ "$SUFFIC" == true ]] && echo suffic || echo lomo)
log "LOMO study start | arm=${ARM} | modalities=[${MODALITIES}] | folds=[${FOLDS}] | smoke=${SMOKE}"
for m in $MODALITIES; do for f in $FOLDS; do run_cell "$m" "$f" "$ARM"; done; done
log "ALL CELLS DONE. Summary: ${SUMMARY}"
column -t -s$'\t' "$SUMMARY" || cat "$SUMMARY"
