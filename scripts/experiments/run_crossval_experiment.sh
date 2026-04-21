#!/bin/bash
################################################################################
# Experiment: 5-Fold Cross-Validation (File-Level)
#
# Runs a full 5-fold cross-validation across the encoder and all baselines.
# Files are assigned to folds by round-robin within each class (sorted order),
# giving a deterministic, reproducible split.
#
# Per fold:
#   test  = fold k
#   val   = fold (k+1) % 5
#   train = remaining 3 folds
#
# File counts per fold per class:
#   Normal classes (20 files): 4 test, 4 val, 12 train
#   Damage classes (5 files):  1 test, 1 val,  3 train
#
# Usage:
#   ./run_crossval_experiment.sh                                             # 93 feat, w64 s16
#   ./run_crossval_experiment.sh --window-size 128 --stride 32              # 93 feat, w128 s32
#   ./run_crossval_experiment.sh --no-proximity                              # 88 feat
#   ./run_crossval_experiment.sh --no-proximity --no-color                   # 68 feat
#   ./run_crossval_experiment.sh --no-proximity --no-color --no-magnetometer # 53 feat
#
# Output naming:
#   outputs/experiments_YYYY_MM_DD/{features}_w{W}_s{S}_cv/
#   e.g. outputs/experiments_2026_02_23/full_w64_s16_cv/
#   e.g. outputs/experiments_2026_02_23/no_proximity_no_color_w128_s32_cv/
################################################################################

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
EXCLUDE_PROXIMITY=false
EXCLUDE_PRESSURE=false
EXCLUDE_COLOR=false
EXCLUDE_MAGNETOMETER=false
WINDOW_SIZE=64
STRIDE=16
N_FOLDS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-proximity)    EXCLUDE_PROXIMITY=true;    shift ;;
        --no-pressure)     EXCLUDE_PRESSURE=true;     shift ;;
        --no-color)        EXCLUDE_COLOR=true;        shift ;;
        --no-magnetometer) EXCLUDE_MAGNETOMETER=true; shift ;;
        --window-size)     WINDOW_SIZE="$2";          shift 2 ;;
        --stride)          STRIDE="$2";               shift 2 ;;
        *) echo "Unknown option: $1"
           echo "Usage: $0 [--no-proximity] [--no-pressure] [--no-color] [--no-magnetometer] [--window-size N] [--stride N]"
           exit 1 ;;
    esac
done

# ── Experiment name & output base ─────────────────────────────────────────────
DATE_TAG=$(date +%Y_%m_%d)
DIR_PARTS=()
[ "$EXCLUDE_PROXIMITY"    = true ] && DIR_PARTS+=("no_proximity")
[ "$EXCLUDE_PRESSURE"     = true ] && DIR_PARTS+=("no_pressure")
[ "$EXCLUDE_COLOR"        = true ] && DIR_PARTS+=("no_color")
[ "$EXCLUDE_MAGNETOMETER" = true ] && DIR_PARTS+=("no_magnetometer")

if [ ${#DIR_PARTS[@]} -eq 0 ]; then
    FEAT_NAME="full"
else
    IFS='_'; FEAT_NAME="${DIR_PARTS[*]}"; unset IFS
fi

EXP_NAME="${FEAT_NAME}_w${WINDOW_SIZE}_s${STRIDE}_cv"
OUTPUT_BASE="outputs/experiments_${DATE_TAG}/${EXP_NAME}"
DATA_DIR="data"
VOCAB_PATH="data/gcode_vocab_v2.json"
SENSOR_REPORT="outputs/sensor_consistency_report.json"
SEED=42

mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/cv_experiment.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "5-FOLD CROSS-VALIDATION: ${EXP_NAME}"
echo "================================================================================"
echo "Started at: $(date)"
echo "Output base: $OUTPUT_BASE"
echo "Window size: $WINDOW_SIZE  Stride: $STRIDE"
echo ""
[ "$EXCLUDE_PROXIMITY"    = true ] && echo "  Excluding: Proximity channels"
[ "$EXCLUDE_PRESSURE"     = true ] && echo "  Excluding: Pressure channels"
[ "$EXCLUDE_COLOR"        = true ] && echo "  Excluding: Color (RGBA) channels"
[ "$EXCLUDE_MAGNETOMETER" = true ] && echo "  Excluding: Magnetometer channels"
echo ""

# ── Build exclusion flags for Python scripts ───────────────────────────────────
EXCLUDE_FLAGS=""
[ "$EXCLUDE_PROXIMITY"    = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-proximity"
[ "$EXCLUDE_PRESSURE"     = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-pressure"
[ "$EXCLUDE_COLOR"        = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-color"
[ "$EXCLUDE_MAGNETOMETER" = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-magnetometer"

# ── Step 0: Sensor consistency report (once) ──────────────────────────────────
if [ ! -f "$SENSOR_REPORT" ]; then
    echo "Running sensor consistency analysis..."
    python3 scripts/analysis/identify_consistent_sensors.py \
        --data-dir "$DATA_DIR" \
        --threshold 95.0 \
        --output "$SENSOR_REPORT"
fi

# ── Main loop: folds 1..N_FOLDS ───────────────────────────────────────────────
for FOLD in $(seq 1 $N_FOLDS); do

    echo ""
    echo "================================================================================"
    echo "FOLD ${FOLD} / ${N_FOLDS}"
    echo "================================================================================"

    FOLD_DIR="$OUTPUT_BASE/fold_${FOLD}"
    PREPROCESS_DIR="$FOLD_DIR/preprocessed"
    ENCODER_DIR="$FOLD_DIR/encoder"
    BASELINE_DIR="$FOLD_DIR/baselines"

    mkdir -p "$PREPROCESS_DIR" "$ENCODER_DIR" "$BASELINE_DIR"

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    echo ""
    echo "--- [Fold $FOLD] Step 1: Preprocessing ---"
    python3 romesh_changes/run_preprocessing_cv_fold.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PREPROCESS_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --sensor-report "$SENSOR_REPORT" \
        --threshold 95.0 \
        --fold "$FOLD" \
        --n-folds "$N_FOLDS" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE" \
        $EXCLUDE_FLAGS

    # ── Step 2: MM-LSTM-DAE encoder ───────────────────────────────────────────
    echo ""
    echo "--- [Fold $FOLD] Step 2: Training MM-LSTM-DAE encoder ---"
    python3 scripts/evaluation/run_9class_direct.py \
        --data-dir "$PREPROCESS_DIR" \
        --output-dir "$ENCODER_DIR" \
        --iteration "cv_fold_${FOLD}" \
        --max_epochs 200 \
        --patience 40 \
        --seed "$SEED"

    # ── Step 3: Baselines ─────────────────────────────────────────────────────
    echo ""
    echo "--- [Fold $FOLD] Step 3: Training baselines ---"

    for MODEL in xgboost random_forest logistic_regression mlp lstm_simple; do
        echo ""
        echo "  [Fold $FOLD] $MODEL ..."
        python3 scripts/evaluation/run_baseline_models.py \
            --data-dir "$PREPROCESS_DIR" \
            --output-dir "$BASELINE_DIR/$MODEL" \
            --model "$MODEL" \
            --seed "$SEED"
    done

    echo ""
    echo "Fold $FOLD complete."

done  # end fold loop

# ── Step 4: Aggregate results ─────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "AGGREGATING CROSS-VALIDATION RESULTS"
echo "================================================================================"
python3 romesh_changes/aggregate_cv_results.py \
    --cv-dir "$OUTPUT_BASE" \
    --n-folds "$N_FOLDS" \
    --output "$OUTPUT_BASE/cv_summary.json"

echo ""
echo "================================================================================"
echo "Cross-validation complete: $(date)"
echo "Output: $OUTPUT_BASE"
echo "Log:    $LOG_FILE"
echo "================================================================================"
