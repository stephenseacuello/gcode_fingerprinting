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
#   ./run_crossval_experiment.sh                                                           # 110 features
#   ./run_crossval_experiment.sh --no-pressure                                             # 104 features
#   ./run_crossval_experiment.sh --no-pressure --no-proximity                              #  98 features
#   ./run_crossval_experiment.sh --no-pressure --no-proximity --no-color                   #  74 features
#   ./run_crossval_experiment.sh --no-pressure --no-proximity --no-color --no-magnetometer #  56 features
################################################################################

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
EXCLUDE_PRESSURE=false
EXCLUDE_PROXIMITY=false
EXCLUDE_COLOR=false
EXCLUDE_MAGNETOMETER=false
N_FOLDS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-pressure)     EXCLUDE_PRESSURE=true;     shift ;;
        --no-proximity)    EXCLUDE_PROXIMITY=true;    shift ;;
        --no-color)        EXCLUDE_COLOR=true;        shift ;;
        --no-magnetometer) EXCLUDE_MAGNETOMETER=true; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--no-pressure] [--no-proximity] [--no-color] [--no-magnetometer]"; exit 1 ;;
    esac
done

# ── Experiment name & output base ─────────────────────────────────────────────
DIR_PARTS=()
[ "$EXCLUDE_PRESSURE"     = true ] && DIR_PARTS+=("no_pressure")
[ "$EXCLUDE_PROXIMITY"    = true ] && DIR_PARTS+=("no_proximity")
[ "$EXCLUDE_COLOR"        = true ] && DIR_PARTS+=("no_color")
[ "$EXCLUDE_MAGNETOMETER" = true ] && DIR_PARTS+=("no_magnetometer")

if [ ${#DIR_PARTS[@]} -eq 0 ]; then
    EXP_NAME="file_level_cv_clip"
else
    IFS='_'; EXP_NAME="${DIR_PARTS[*]}_cv_clip_64WS_16Stride_nan_removed"; unset IFS
fi

OUTPUT_BASE="outputs_clean_data/experiments/${EXP_NAME}"
DATA_DIR="data_clean"
VOCAB_PATH="outputs/vocabulary/gcode_vocabulary_v2.json"
SENSOR_REPORT="outputs_clean_data/sensor_consistency_report_clean.json"
SEED=42

mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/cv_experiment.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "5-FOLD CROSS-VALIDATION: ${EXP_NAME}"
echo "================================================================================"
echo "Started at: $(date)"
echo "Output base: $OUTPUT_BASE"
echo ""
[ "$EXCLUDE_PRESSURE"     = true ] && echo "  Excluding: Pressure channels"
[ "$EXCLUDE_PROXIMITY"    = true ] && echo "  Excluding: Proximity channels"
[ "$EXCLUDE_COLOR"        = true ] && echo "  Excluding: Color (RGBA) channels"
[ "$EXCLUDE_MAGNETOMETER" = true ] && echo "  Excluding: Magnetometer channels"
echo ""

# ── Build exclusion flags for Python scripts ───────────────────────────────────
EXCLUDE_FLAGS=""
[ "$EXCLUDE_PRESSURE"     = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-pressure"
[ "$EXCLUDE_PROXIMITY"    = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-proximity"
[ "$EXCLUDE_COLOR"        = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-color"
[ "$EXCLUDE_MAGNETOMETER" = true ] && EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-magnetometer"

# ── Step 0: Sensor consistency report (once) ──────────────────────────────────
if [ ! -f "$SENSOR_REPORT" ]; then
    echo "Running sensor consistency analysis..."
    python scripts/analysis/identify_consistent_sensors.py \
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
    python romesh_changes/run_preprocessing_cv_fold.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PREPROCESS_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --sensor-report "$SENSOR_REPORT" \
        --threshold 95.0 \
        --fold "$FOLD" \
        --n-folds "$N_FOLDS" \
        --window-size 64 \
        --stride 16 \
        $EXCLUDE_FLAGS

    # ── Step 2: MM-LSTM-DAE encoder ───────────────────────────────────────────
    echo ""
    echo "--- [Fold $FOLD] Step 2: Training MM-LSTM-DAE encoder ---"
    python scripts/evaluation/run_9class_direct.py \
        --data-dir "$PREPROCESS_DIR" \
        --output-dir "$ENCODER_DIR" \
        --iteration "cv_fold_${FOLD}" \
        --max_epochs 100 \
        --seed "$SEED"

    # ── Step 3: Baselines ─────────────────────────────────────────────────────
    echo ""
    echo "--- [Fold $FOLD] Step 3: Training baselines ---"

    for MODEL in xgboost random_forest logistic_regression mlp lstm_simple; do
        echo ""
        echo "  [Fold $FOLD] $MODEL ..."
        python scripts/evaluation/run_baseline_models.py \
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
python romesh_changes/aggregate_cv_results.py \
    --cv-dir "$OUTPUT_BASE" \
    --n-folds "$N_FOLDS" \
    --output "$OUTPUT_BASE/cv_summary.json"

echo ""
echo "================================================================================"
echo "Cross-validation complete: $(date)"
echo "Output: $OUTPUT_BASE"
echo "Log:    $LOG_FILE"
echo "================================================================================"
