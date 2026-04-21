#!/bin/bash
################################################################################
# Experiment: Architecture Ablation x 5-Fold Cross-Validation
#
# Runs 7 ablation conditions x 5 folds = 35 training jobs.
# Reuses pre-existing fold preprocessing from a prior full-CV run.
#
# Expects preprocessed data at:
#   $PREPROCESS_BASE/fold_{k}/preprocessed/
#
# Usage:
#   ./run_arch_ablation_cv.sh --preprocess-base outputs/experiments_2026_02_25/no_proximity_no_pressure_w64_s16_cv
#   ./run_arch_ablation_cv.sh --preprocess-base <path> --conditions "no_dtae,minimal"
#   ./run_arch_ablation_cv.sh --preprocess-base <path> --max-parallel 2
################################################################################

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
PREPROCESS_BASE=""
CONDITIONS="full,single_proj,no_gates,no_crossattn,no_denoise,no_dtae,minimal"
N_FOLDS=5
SEED=42
MAX_PARALLEL=1  # Number of concurrent jobs (adjust to GPU count)

while [[ $# -gt 0 ]]; do
    case $1 in
        --preprocess-base) PREPROCESS_BASE="$2";   shift 2 ;;
        --conditions)      CONDITIONS="$2";         shift 2 ;;
        --n-folds)         N_FOLDS="$2";            shift 2 ;;
        --seed)            SEED="$2";               shift 2 ;;
        --max-parallel)    MAX_PARALLEL="$2";       shift 2 ;;
        *) echo "Unknown option: $1"
           echo "Usage: $0 --preprocess-base <path> [--conditions cond1,cond2,...] [--max-parallel N]"
           exit 1 ;;
    esac
done

if [ -z "$PREPROCESS_BASE" ]; then
    echo "ERROR: --preprocess-base required (path to existing CV preprocessed data)"
    echo "Example: --preprocess-base outputs/experiments_2026_02_25/no_proximity_no_pressure_w64_s16_cv"
    exit 1
fi

# Verify preprocessed data exists
if [ ! -d "${PREPROCESS_BASE}/fold_1/preprocessed" ]; then
    echo "ERROR: Preprocessed data not found at ${PREPROCESS_BASE}/fold_1/preprocessed"
    echo "Run the cross-validation experiment first to generate preprocessed data."
    exit 1
fi

# ── Setup ────────────────────────────────────────────────────────────────────
DATE_TAG=$(date +%Y_%m_%d)
OUTPUT_BASE="outputs/experiments_${DATE_TAG}/arch_ablation_cv"
mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/arch_ablation.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "ARCHITECTURE ABLATION STUDY"
echo "================================================================================"
echo "Started at:       $(date)"
echo "Preprocessed:     ${PREPROCESS_BASE}"
echo "Output:           ${OUTPUT_BASE}"
echo "Conditions:       ${CONDITIONS}"
echo "Folds:            ${N_FOLDS}"
echo "Max parallel:     ${MAX_PARALLEL}"
echo "Seed:             ${SEED}"
echo "================================================================================"

IFS=',' read -ra COND_ARRAY <<< "$CONDITIONS"

# ── Main loop ────────────────────────────────────────────────────────────────
RUNNING=0
TOTAL=0
SKIPPED=0
FAILED=0

for ABLATION in "${COND_ARRAY[@]}"; do
    for FOLD in $(seq 1 $N_FOLDS); do

        PREPROCESS_DIR="${PREPROCESS_BASE}/fold_${FOLD}/preprocessed"
        FOLD_OUTPUT="${OUTPUT_BASE}/${ABLATION}/fold_${FOLD}"

        # Skip if already completed
        if [ -f "${FOLD_OUTPUT}/results.json" ]; then
            echo "[SKIP] ${ABLATION} fold ${FOLD} (results.json exists)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        mkdir -p "${FOLD_OUTPUT}"

        echo ""
        echo ">>> [${ABLATION}] fold ${FOLD} / ${N_FOLDS}"

        python3 scripts/evaluation/run_9class_direct.py \
            --data-dir "$PREPROCESS_DIR" \
            --output-dir "$FOLD_OUTPUT" \
            --iteration "arch_ablation_${ABLATION}_fold${FOLD}" \
            --ablation "$ABLATION" \
            --max_epochs 200 \
            --patience 40 \
            --seed "$SEED" &

        RUNNING=$((RUNNING + 1))
        TOTAL=$((TOTAL + 1))

        if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
            wait -n || FAILED=$((FAILED + 1))
            RUNNING=$((RUNNING - 1))
        fi

    done
done

# Wait for remaining jobs
while [ "$RUNNING" -gt 0 ]; do
    wait -n || FAILED=$((FAILED + 1))
    RUNNING=$((RUNNING - 1))
done

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "  Total:   ${TOTAL} jobs"
echo "  Skipped: ${SKIPPED} (already done)"
echo "  Failed:  ${FAILED}"
echo "================================================================================"

# ── Aggregate ────────────────────────────────────────────────────────────────
if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "Aggregating results..."
    python3 scripts/analysis/aggregate_arch_ablation.py \
        --results-dir "$OUTPUT_BASE" \
        --conditions "${CONDITIONS}" \
        --n-folds "$N_FOLDS" \
        --output "$OUTPUT_BASE/arch_ablation_summary.json"

    echo ""
    echo "================================================================================"
    echo "DONE at $(date)"
    echo "Summary: ${OUTPUT_BASE}/arch_ablation_summary.json"
    echo "Log:     ${LOG_FILE}"
    echo "================================================================================"
else
    echo ""
    echo "WARNING: ${FAILED} jobs failed. Skipping aggregation."
    echo "Fix failures and re-run (completed folds will be skipped)."
fi
