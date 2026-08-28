#!/usr/bin/env bash
# =============================================================================
# Nested leave-one-(sensor, modality)-out encoder study (full_window).
# Companion to the per-modality and per-sensor LOMO studies; cuts the interior
# of the modality x sensor grid: 6 physical sensors x 6 channel-types
# = 36 cells (electrical excluded because it is a global, not per-sensor,
# modality and already characterised by the modality LOMO).
#
# Each cell drops 1-4 specific channels (e.g., frame_l2 x accelerometer =
# {frame_l2.Ax, frame_l2.Ay, frame_l2.Az}), retrains the encoder from scratch,
# and trains a fresh decoder under the headline recipe. Same pipeline as
# run_lomo_encoder_study.sh / run_lomo_sensor_study.sh; only the exclusion
# specification differs.
#
# Cell directory layout (matches aggregate_lomo_results.py --group-kind nested):
#   ${ROOT}/lomo/<sensor>__<modality>/fold_<f>/{full_window,encoder,decoder}
#   ${ROOT}/lomo/baseline/fold_<f>/...                       (shared baseline)
#
# Env: GPU, CELLS (space-separated list of <sensor>__<modality> tokens, plus
# 'baseline' once), FOLDS, EPOCHS, PATIENCE, SUMMARY_TAG.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0

DATA_DIR="${DATA_DIR:-data_clean}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
ROOT="${ROOT:-outputs/decoder20260511/lomo_nested}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
N_FOLDS=5
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
SCHED_SAMPLING="${SCHED_SAMPLING:-0.5}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"

# Default cell list: baseline first, then all 36 (sensor, modality) pairs.
SENSORS_DEF="frame_l2 frame_l3 frame_r2 spindle2 y_bed__3 y_bed__4"
MODALITIES_DEF="accelerometer gyroscope magnetometer color temperature audio"
if [[ -z "${CELLS:-}" ]]; then
  CELLS="baseline"
  for s in $SENSORS_DEF; do for m in $MODALITIES_DEF; do CELLS+=" ${s}__${m}"; done; done
fi
FOLDS="${FOLDS:-1 2 3 4 5}"

# Modality -> suffix list mapping (matches MODALITY_SUFFIXES in
# scripts/analysis/lomo_channel_identity_check.py).
suffixes_for() {
  case "$1" in
    accelerometer) echo "Ax Ay Az" ;;
    gyroscope)     echo "Gx Gy Gz" ;;
    magnetometer)  echo "Mx My Mz" ;;
    color)         echo "ColorR ColorG ColorB ColorA" ;;
    temperature)   echo "Temperature" ;;
    audio)         echo "RMS" ;;
    *) echo ""; return 1 ;;
  esac
}

log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$ROOT"
SUMMARY="${ROOT}/summary${SUMMARY_TAG:+_${SUMMARY_TAG}}.tsv"
: > "$SUMMARY"
echo -e "cell\tfold\tgate\tencoder_ckpt\tdecoder_command\tstatus" >> "$SUMMARY"

run_cell() {  # $1=cell-key (sensor__modality | baseline)  $2=fold
  local key="$1" f="$2"
  local cell="${ROOT}/lomo/${key}/fold_${f}"
  local FW="${cell}/full_window" ENC="${cell}/encoder" DEC="${cell}/decoder"
  mkdir -p "$FW" "$ENC" "$DEC"

  local is_base=false sensor="" modality="" excl=""
  if [[ "$key" == "baseline" ]]; then
    is_base=true
  else
    # Parse "<sensor>__<modality>" -- modality is the LAST __<word> suffix to
    # tolerate sensor names that themselves contain '__' (y_bed__3, y_bed__4).
    modality="${key##*__}"
    sensor="${key%__${modality}}"
    local sufs; sufs="$(suffixes_for "$modality")" || {
      log "unknown modality '$modality' in key '$key'"
      echo -e "${key}\t${f}\t-\t-\t-\tbad_key" >> "$SUMMARY"; return 0; }
    for suf in $sufs; do excl+=" --exclude-column ${sensor}.${suf}"; done
  fi

  # Idempotency: skip if a cell already has decoder metrics.
  if [[ -s "${DEC}/results/metrics.json" ]]; then
    local prev_cmd
    prev_cmd=$(python3 -c "import json;print(round(json.load(open('${DEC}/results/metrics.json'))['test_metrics']['command_accuracy'],4))" 2>/dev/null || echo "?")
    log "SKIP cell=${key} fold=${f} -- already complete (command=${prev_cmd})"
    echo -e "${key}\t${f}\tCACHED\t-\t${prev_cmd}\tskip_done" >> "$SUMMARY"
    return 0
  fi

  local BASE_FW="${ROOT}/lomo/baseline/fold_${f}/full_window"
  # Trap: clean up regenerable / redundant artifacts on cell exit:
  #   - preprocessing NPZs
  #   - decoder/encoder memory cache
  #   - the multi-GB training_history.json
  #   - final_decoder.pt (redundant with best_decoder.pt) and best_test_model.pt
  #     (redundant with best_model.pt) -- 100-250 MB each per cell
  #   - predictions.npz under decoder/results (not consumed by the aggregator)
  #   - shrink decoder metrics.json to {test_metrics, val_metrics, best_epoch}
  #     -- saves ~80-130 MB per cell while preserving every field the
  #     aggregator and per-class analyses read. Uses a helper script
  #     (scripts/analysis/shrink_decoder_metrics.py) so the trap body
  #     stays free of multi-line python that bash double-quoting mangled.
  # Note: best_decoder.pt and encoder/checkpoint/best_model.pt are also
  # removed at cell-exit because by then the cell has already written
  # metrics.json (the paper-facing artifact) and we cannot afford the
  # ~230 MB per-cell residual at the 6x6 nested sweep's scale. Re-running
  # a cell from scratch is the recovery path if those checkpoints are
  # ever needed.
  if $is_base; then
    trap "rm -f \"${ENC}\"/data/*sequences*.npz 2>/dev/null;
          rm -rf \"${DEC}/encoder_memory\" 2>/dev/null;
          rm -f  \"${DEC}/results/training_history.json\" 2>/dev/null;
          rm -f  \"${DEC}/decoder_checkpoint/final_decoder.pt\" 2>/dev/null;
          rm -f  \"${DEC}/decoder_checkpoint/best_decoder.pt\" 2>/dev/null;
          rm -f  \"${ENC}/checkpoint/best_test_model.pt\" 2>/dev/null;
          rm -f  \"${ENC}/checkpoint/best_model.pt\" 2>/dev/null;
          rm -f  \"${DEC}/results/predictions.npz\" 2>/dev/null;
          python3 scripts/analysis/shrink_decoder_metrics.py \"${DEC}/results/metrics.json\" 2>/dev/null" RETURN
  else
    trap "rm -f \"${FW}\"/*sequences*.npz \"${ENC}\"/data/*sequences*.npz 2>/dev/null;
          rm -rf \"${DEC}/encoder_memory\" 2>/dev/null;
          rm -f  \"${DEC}/results/training_history.json\" 2>/dev/null;
          rm -f  \"${DEC}/decoder_checkpoint/final_decoder.pt\" 2>/dev/null;
          rm -f  \"${DEC}/decoder_checkpoint/best_decoder.pt\" 2>/dev/null;
          rm -f  \"${ENC}/checkpoint/best_test_model.pt\" 2>/dev/null;
          rm -f  \"${ENC}/checkpoint/best_model.pt\" 2>/dev/null;
          rm -f  \"${DEC}/results/predictions.npz\" 2>/dev/null;
          python3 scripts/analysis/shrink_decoder_metrics.py \"${DEC}/results/metrics.json\" 2>/dev/null" RETURN
  fi
  log "==== cell=${key} fold=${f} (GPU${GPU}) drop=[${excl## }] ===="

  local free; free=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then
    log "DISK GUARD: ${free}GB < ${MIN_FREE_GB}GB -> abort"
    echo -e "${key}\t${f}\t-\t-\t-\tdisk_abort" >> "$SUMMARY"; exit 3
  fi

  # 1. preprocess full_window (encoder + decoder share it) -------------------
  python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
    --data-dir "$DATA_DIR" --output-dir "$FW" --vocab-path "$VOCAB" \
    --fold "$f" --n-folds "$N_FOLDS" --window-size 256 --stride 64 \
    --label-mode full_window --exclude-proximity --exclude-pressure $excl \
    > "${cell}/preprocess.log" 2>&1 \
    || { log "preprocess errored (see ${cell}/preprocess.log) -> skip"; \
         echo -e "${key}\t${f}\t-\t-\t-\tpreprocess_error" >> "$SUMMARY"; return 0; }

  # 2. encoder's actually-consumed columns -----------------------------------
  python3 - "$FW" "${ENC}/consumed_columns.json" <<'PY' || true
import sys, json, tempfile
from scripts.evaluation.run_9class_direct import prepare_data_9class
with tempfile.TemporaryDirectory() as td:
    kc, *_ = prepare_data_9class(sys.argv[1], td)
json.dump(list(kc), open(sys.argv[2], "w")); print(f"encoder consumes {len(kc)} cols")
PY
  if [[ ! -s "${ENC}/consumed_columns.json" ]]; then
    log "consumed-columns capture failed -> skip"
    echo -e "${key}\t${f}\t-\t-\t-\tcols_error" >> "$SUMMARY"; return 0
  fi

  # 3. channel-identity gate (nested mode; baseline defines the reference) ---
  local gate="${cell}/channel_identity.json"
  if $is_base; then
    log "baseline cell -> no gate (defines reference split)"
  elif python3 scripts/analysis/lomo_channel_identity_check.py \
        --modality "$modality" --group-kind nested --sensor "$sensor" \
        --baseline-dir "$BASE_FW" --excluded-dir "$FW" \
        --encoder-cols "${ENC}/consumed_columns.json" --out "$gate"; then
    log "GATE PASS"
  else
    log "GATE FAIL -> skip cell. See ${gate}"
    echo -e "${key}\t${f}\tFAIL\t-\t-\tskipped_gate" >> "$SUMMARY"; return 0
  fi

  # 4. encoder retrain on the full_window data -------------------------------
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluation/run_9class_direct.py \
    --data-dir "$FW" --output-dir "$ENC" --iteration "lomo_nested_${key}_fold${f}" \
    --max_epochs 200 --patience 40 --seed "$SEED" \
    > "${cell}/encoder.log" 2>&1 \
    || { log "encoder errored (see ${cell}/encoder.log) -> skip"; \
         echo -e "${key}\t${f}\tPASS\t-\t-\tencoder_error" >> "$SUMMARY"; return 0; }
  local ckpt="${ENC}/checkpoint/best_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    log "no encoder checkpoint -> skip"
    echo -e "${key}\t${f}\tPASS\t-\t-\tno_encoder_ckpt" >> "$SUMMARY"; return 0
  fi

  # 5. decoder retrain on full_window -- headline recipe ---------------------
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluation/run_decoder_quick_test.py \
    --data_dir "$FW" --encoder_ckpt "$ckpt" --fold "$f" --vocab "$VOCAB" \
    --output_dir "$DEC" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch_size 4 --lr 5e-5 --warmup_epochs 10 --weight_decay 0.05 \
    --max_token_len "$MAX_TOKEN_LEN" --seed "$SEED" \
    --d_model 384 --n_layers 8 --n_heads 12 --dropout 0.1 \
    --legacy_weight 3.0 --digit_weight 1.0 --memory_pos_encoding true \
    --use_sensor_prior true --grammar_constraint true \
    --use_window_position false --multi_window_context 0 --window_dropout 0.0 \
    --scheduled_sampling "$SCHED_SAMPLING" --device auto \
    > "${cell}/decoder.log" 2>&1 \
    || { log "decoder errored (see ${cell}/decoder.log)"; \
         echo -e "${key}\t${f}\tPASS\t${ckpt}\t-\tdecoder_error" >> "$SUMMARY"; return 0; }

  local cmd
  cmd=$(python3 -c "import json;print(round(json.load(open('${DEC}/results/metrics.json'))['test_metrics']['command_accuracy'],4))" 2>/dev/null || echo "?")
  log "CELL DONE -> command=${cmd}"
  echo -e "${key}\t${f}\t$( $is_base && echo NA || echo PASS )\t${ckpt}\t${cmd}\tok" >> "$SUMMARY"
}

log "LOMO nested | cells=$(echo "$CELLS"|wc -w) | folds=[${FOLDS}] | GPU=${GPU} | epochs=${EPOCHS}"
for c in $CELLS; do for f in $FOLDS; do run_cell "$c" "$f"; done; done
log "DONE. Summary: ${SUMMARY}"
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
