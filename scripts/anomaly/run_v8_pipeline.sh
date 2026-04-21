#!/bin/bash
# run_v8_pipeline.sh -- End-to-end V8 multi-line decoder anomaly pipeline.
#
# Runs:
#   1. cache_inference_v8.py for all 5 folds
#   2. generate_attacks_v8.py
#   3. All 18 experiments with V8 cache/attack paths
#
# Usage:
#   bash scripts/anomaly/run_v8_pipeline.sh [--device cuda|cpu] [--folds 1 2 3 4 5]
set -euo pipefail

PROJECT="/home/seacuello/Documents/gcode_fingerprinting"
BASE="outputs/anomaly20260319"
SCRIPTS="scripts/anomaly"
LOG="$BASE/logs_v8"

# V8-specific paths
V8_CACHE="$BASE/cached_inference_v8"
V8_ATTACKS="$BASE/attacks_v8"
V8_EXP_PREFIX="v8_"

# Parse optional arguments
DEVICE="cuda"
FOLDS="1 2 3 4 5"
while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --folds) shift; FOLDS=""; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do FOLDS="$FOLDS $1"; shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$PROJECT"
mkdir -p "$LOG"

echo "============================================================"
echo "V8 Multi-Line Decoder Anomaly Pipeline"
echo "============================================================"
echo "Device:  $DEVICE"
echo "Folds:   $FOLDS"
echo "Cache:   $V8_CACHE"
echo "Attacks: $V8_ATTACKS"
echo "Logs:    $LOG"
echo "============================================================"

# ============================================================
# Phase 0a: Cache V8 decoder inference for all folds
# ============================================================
echo ""
echo "=== Phase 0a: Cache V8 Inference ==="
python3 "$SCRIPTS/cache_inference_v8.py" \
    --output-dir "$V8_CACHE" \
    --folds $FOLDS \
    --device "$DEVICE" \
    2>&1 | tee "$LOG/cache_inference_v8.log"

echo "=== Phase 0a DONE ==="

# ============================================================
# Phase 0b: Generate V8 attack sequences
# ============================================================
echo ""
echo "=== Phase 0b: Generate V8 Attacks ==="
python3 "$SCRIPTS/generate_attacks_v8.py" \
    --output-dir "$V8_ATTACKS" \
    --folds $FOLDS \
    2>&1 | tee "$LOG/generate_attacks_v8.log"

echo "=== Phase 0b DONE ==="

# ============================================================
# Phase 1: Run all 18 experiments with V8 cache/attack paths
# ============================================================
echo ""
echo "=== Phase 1: Running experiments with V8 data ==="

# Common V8 experiment arguments:
#   --cache-dir / --attack-dir are not standard argparse args in the
#   existing experiment scripts. The experiment scripts read from the
#   module-level CACHE_DIR and ATTACK_DIR in anomaly_scoring_utils.py.
#
#   To redirect them to V8 paths without modifying each experiment script,
#   we set environment variables that anomaly_scoring_utils.py checks.
export ANOMALY_CACHE_DIR="$PROJECT/$V8_CACHE"
export ANOMALY_ATTACK_DIR="$PROJECT/$V8_ATTACKS"

# V8 output directories get a v8_ prefix to avoid overwriting V7 results
V8_OUT="$BASE"

run_exp() {
    local exp_num="$1"
    local exp_name="$2"
    local script="$3"
    local extra_args="${4:-}"

    local out_dir="$V8_OUT/v8_exp${exp_num}_${exp_name}"
    echo ""
    echo "--- Exp $exp_num: $exp_name ---"
    python3 "$SCRIPTS/$script" \
        --output-dir "$out_dir" \
        --folds $FOLDS \
        $extra_args \
        2>&1 | tee "$LOG/v8_exp${exp_num}_${exp_name}.log" || {
            echo "WARNING: Exp $exp_num ($exp_name) failed, continuing..."
        }
}

# Group A: No dependencies (CPU)
run_exp "01" "nll_scoring"           "run_exp01_nll_scoring.py"
run_exp "02" "calibration"           "run_exp02_calibration.py"
run_exp "03" "grammar"               "run_exp03_grammar.py"
run_exp "04" "disagreement"          "run_exp04_disagreement.py"
run_exp "11" "embedding"             "run_exp11_embedding.py"
run_exp "13" "nll_vs_argmax"         "run_exp13_nll_vs_argmax.py"
run_exp "14" "encoder_vs_decoder"    "run_exp14_encoder_vs_decoder.py"

# Group B: Depend on Exp 01 results (CPU)
run_exp "07" "cross_condition"       "run_exp07_cross_condition.py"
run_exp "08" "ensemble"              "run_exp08_ensemble.py"
run_exp "12" "graded_injection"      "run_exp12_graded_injection.py"
run_exp "18" "cusum"                 "run_exp18_cusum.py"

# Group C: GPU experiments
run_exp "05" "modality_ablation"     "run_exp05_modality_ablation.py"
run_exp "06" "sensor_failure"        "run_exp06_sensor_failure.py"
run_exp "10" "loco"                  "run_exp10_loco.py"
run_exp "15" "position_confound"     "run_exp15_position_confound.py"
run_exp "16" "feature_config"        "run_exp16_feature_config.py"

# Group D: Post-analysis (depends on all above)
run_exp "09" "error_taxonomy"        "run_exp09_error_taxonomy.py"
run_exp "17" "cost_benefit"          "run_exp17_cost_benefit.py"

echo ""
echo "============================================================"
echo "V8 Pipeline COMPLETE"
echo "============================================================"
echo "Cache:      $V8_CACHE"
echo "Attacks:    $V8_ATTACKS"
echo "Results:    $V8_OUT/v8_exp*/"
echo "Logs:       $LOG/"
echo "============================================================"
