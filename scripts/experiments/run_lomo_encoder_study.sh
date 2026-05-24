#!/usr/bin/env bash
# =============================================================================
# LOMO encoder study -- FULL_WINDOW driver (corrected design, 2026-05-21).
#
# Why full_window: per_row command accuracy is structurally ~0.50 (a window's
# ~60 rows share one encoder memory, so a categorical head cannot disambiguate
# them) -- this is the paper's own RQ2 finding, not a bug. The 0.888 headline
# is the full_window number. To produce a modality-attribution comparable to
# the headline AND the inference-time ablation (Section RQ3), encoder AND
# decoder both use full_window data.
#
# Per cell (modality M, fold F):
#   1. preprocess full_window, excluding M's channels         (~303 windows)
#   2. capture the encoder's actually-consumed columns
#   3. channel-identity gate vs the baseline cell (skipped for baseline itself)
#   4. retrain the encoder on the full_window data            (run_9class_direct;
#      defaults lr 1e-3 / dropout 0.2 == production f98 encoder)
#   5. retrain the decoder on the full_window data            (the headline
#      recipe of train_v8_full_window_5fold.sh: 300 epochs, max_token_len 1400,
#      scheduled_sampling 0.5), but on the LOMO-retrained encoder
#
# Env knobs: GPU, MODALITIES, FOLDS, EPOCHS, PATIENCE, SUMMARY_TAG.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0   # deterministic CV split (with the sorted() repair fix)

DATA_DIR="${DATA_DIR:-data_clean}"
VOCAB="${VOCAB:-data/gcode_vocab_v8.json}"
ROOT="${ROOT:-outputs/decoder20260511/lomo_fw}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
N_FOLDS=5
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-75}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-1400}"
SCHED_SAMPLING="${SCHED_SAMPLING:-0.5}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
MODALITIES="${MODALITIES:-baseline accelerometer gyroscope magnetometer color temperature audio electrical}"
FOLDS="${FOLDS:-1}"

flag_for() { case "$1" in
  accelerometer) echo "--exclude-accelerometer";; gyroscope) echo "--exclude-gyroscope";;
  magnetometer)  echo "--exclude-magnetometer";;  color)     echo "--exclude-color";;
  temperature)   echo "--exclude-temperature";;   audio)     echo "--exclude-audio";;
  electrical)    echo "--exclude-electrical";;
  *) echo "BAD_$1"; return 1;; esac; }

log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$ROOT"
SUMMARY="${ROOT}/summary${SUMMARY_TAG:+_${SUMMARY_TAG}}.tsv"
: > "$SUMMARY"
echo -e "modality\tfold\tgate\tencoder_ckpt\tdecoder_command\tstatus" >> "$SUMMARY"

run_cell() {  # $1=modality $2=fold
  local m="$1" f="$2"
  local cell="${ROOT}/lomo/${m}/fold_${f}"
  local FW="${cell}/full_window" ENC="${cell}/encoder" DEC="${cell}/decoder"
  mkdir -p "$FW" "$ENC" "$DEC"
  local is_base=false; [[ "$m" == "baseline" ]] && is_base=true
  local BASE_FW="${ROOT}/lomo/baseline/fold_${f}/full_window"
  # cleanup on every exit path: keep baseline's full_window (it is the gate
  # reference for every modality cell of this fold); drop modality input NPZ +
  # run_9class's duplicate copy.
  if $is_base; then
    trap 'rm -f "${ENC}"/data/*sequences*.npz 2>/dev/null' RETURN
  else
    trap 'rm -f "${FW}"/*sequences*.npz "${ENC}"/data/*sequences*.npz 2>/dev/null' RETURN
  fi
  log "==== modality=${m} fold=${f} (GPU${GPU}) ===="

  local free; free=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then
    log "DISK GUARD: ${free}GB < ${MIN_FREE_GB}GB -> abort"
    echo -e "${m}\t${f}\t-\t-\t-\tdisk_abort" >> "$SUMMARY"; exit 3
  fi

  local excl=""
  if ! $is_base; then excl="$(flag_for "$m")" || { log "bad modality $m"; return 0; }; fi

  # 1. preprocess full_window (encoder + decoder share this one source) -------
  python3 scripts/preprocessing/run_preprocessing_v8_cv_fold.py \
    --data-dir "$DATA_DIR" --output-dir "$FW" --vocab-path "$VOCAB" \
    --fold "$f" --n-folds "$N_FOLDS" --window-size 256 --stride 64 \
    --label-mode full_window --exclude-proximity --exclude-pressure $excl \
    > "${cell}/preprocess.log" 2>&1 \
    || { log "preprocess errored (see ${cell}/preprocess.log) -> skip"; \
         echo -e "${m}\t${f}\t-\t-\t-\tpreprocess_error" >> "$SUMMARY"; return 0; }

  # 2. capture the encoder's actually-consumed columns (its own code) --------
  python3 - "$FW" "${ENC}/consumed_columns.json" <<'PY' || true
import sys, json, tempfile
from scripts.evaluation.run_9class_direct import prepare_data_9class
with tempfile.TemporaryDirectory() as td:
    kc, *_ = prepare_data_9class(sys.argv[1], td)
json.dump(list(kc), open(sys.argv[2], "w")); print(f"encoder consumes {len(kc)} cols")
PY
  if [[ ! -s "${ENC}/consumed_columns.json" ]]; then
    log "consumed-columns capture failed -> skip"
    echo -e "${m}\t${f}\t-\t-\t-\tcols_error" >> "$SUMMARY"; return 0
  fi

  # 3. channel-identity gate (baseline defines the reference -> no gate) ------
  local gate="${cell}/channel_identity.json"
  if $is_base; then
    log "baseline cell -> no gate (defines reference split)"
  elif python3 scripts/analysis/lomo_channel_identity_check.py \
        --modality "$m" --baseline-dir "$BASE_FW" --excluded-dir "$FW" \
        --encoder-cols "${ENC}/consumed_columns.json" --out "$gate"; then
    log "GATE PASS"
  else
    log "GATE FAIL -> skip cell (no GPU spent). See ${gate}"
    echo -e "${m}\t${f}\tFAIL\t-\t-\tskipped_gate" >> "$SUMMARY"; return 0
  fi

  # 4. encoder retrain on the full_window data (~62 s) -----------------------
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluation/run_9class_direct.py \
    --data-dir "$FW" --output-dir "$ENC" --iteration "lomo_fw_${m}_fold${f}" \
    --max_epochs 200 --patience 40 --seed "$SEED" \
    > "${cell}/encoder.log" 2>&1 \
    || { log "encoder errored (see ${cell}/encoder.log) -> skip"; \
         echo -e "${m}\t${f}\tPASS\t-\t-\tencoder_error" >> "$SUMMARY"; return 0; }
  local ckpt="${ENC}/checkpoint/best_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    log "no encoder checkpoint -> skip"
    echo -e "${m}\t${f}\tPASS\t-\t-\tno_encoder_ckpt" >> "$SUMMARY"; return 0
  fi

  # 5. decoder retrain on full_window -- headline recipe, LOMO encoder -------
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
         echo -e "${m}\t${f}\tPASS\t${ckpt}\t-\tdecoder_error" >> "$SUMMARY"; return 0; }

  local cmd
  cmd=$(python3 -c "import json;print(round(json.load(open('${DEC}/results/metrics.json'))['test_metrics']['command_accuracy'],4))" 2>/dev/null || echo "?")
  log "CELL DONE -> command=${cmd}  ->  ${DEC}"
  echo -e "${m}\t${f}\t$( $is_base && echo NA || echo PASS )\t${ckpt}\t${cmd}\tok" >> "$SUMMARY"
}

log "LOMO full_window | modalities=[${MODALITIES}] | folds=[${FOLDS}] | GPU=${GPU} | epochs=${EPOCHS}"
for m in $MODALITIES; do for f in $FOLDS; do run_cell "$m" "$f"; done; done
log "DONE. Summary: ${SUMMARY}"
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
