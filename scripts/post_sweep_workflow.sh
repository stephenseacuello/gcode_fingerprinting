#!/usr/bin/env bash
#
# Post-Sweep Workflow Automation Script
#
# This script automates the entire post-sweep workflow:
# 1. Analyze sweep results
# 2. Get best checkpoint from W&B
# 3. Evaluate on test set
# 4. Deploy to API (optional)
# 5. Generate comparison report
#
# Usage:
#   ./scripts/post_sweep_workflow.sh --sweep-id njo48wle --entity seacuello-university-of-rhode-island
#

set -e  # Exit on error

# Default values
SWEEP_ID=""
ENTITY=""
PROJECT="uncategorized"
METRIC="val/overall_acc"
DEPLOY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$SWEEP_ID" ]; then
    echo "❌ Error: --sweep-id is required"
    echo "Usage: $0 --sweep-id SWEEP_ID [--entity ENTITY] [--project PROJECT] [--deploy]"
    exit 1
fi

echo "================================"
echo "POST-SWEEP WORKFLOW AUTOMATION"
echo "================================"
echo "Sweep ID: $SWEEP_ID"
echo "Entity: ${ENTITY:-default}"
echo "Project: $PROJECT"
echo "Metric: $METRIC"
echo "Deploy: $DEPLOY"
echo ""

# Step 1: Analyze sweep results
echo "Step 1/5: Analyzing sweep results..."
echo "-----------------------------------"
ENTITY_FLAG=""
if [ -n "$ENTITY" ]; then
    ENTITY_FLAG="--entity $ENTITY"
fi

.venv/bin/python scripts/analyze_sweep.py \
    --sweep-id "$SWEEP_ID" \
    $ENTITY_FLAG \
    --project "$PROJECT" \
    --metric "$METRIC" \
    --output outputs/sweep_analysis

if [ $? -ne 0 ]; then
    echo "❌ Sweep analysis failed"
    exit 1
fi

echo "✓ Sweep analysis complete"
echo ""

# Step 2: Get best checkpoint from W&B
echo "Step 2/5: Getting best checkpoint from W&B..."
echo "----------------------------------------------"

.venv/bin/python scripts/get_best_checkpoint_from_sweep.py \
    --sweep-id "$SWEEP_ID" \
    $ENTITY_FLAG \
    --project "$PROJECT" \
    --metric "$METRIC" \
    --output-dir outputs/best_from_sweep

if [ $? -ne 0 ]; then
    echo "❌ Failed to get best checkpoint"
    exit 1
fi

echo "✓ Best checkpoint retrieved"
echo ""

# Step 3: Evaluate on test set
echo "Step 3/5: Evaluating checkpoint on test set..."
echo "-----------------------------------------------"

CHECKPOINT_PATH="outputs/best_from_sweep/checkpoint_best.pt"

# Check if checkpoint was found
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "⚠️  Checkpoint not found at: $CHECKPOINT_PATH"
    echo "   The checkpoint may need to be downloaded manually from W&B"
    echo "   Visit: https://wandb.ai/$ENTITY/$PROJECT/sweeps/$SWEEP_ID"
    echo ""
    echo "   Once downloaded, run:"
    echo "   cp /path/to/downloaded/checkpoint.pt $CHECKPOINT_PATH"
    echo ""
    read -p "Press Enter once checkpoint is in place, or Ctrl+C to exit..."
fi

.venv/bin/python scripts/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --test-data outputs/processed_quick/test_sequences.npz \
    --vocab-path data/vocabulary.json \
    --output outputs/evaluation_best

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi

echo "✓ Evaluation complete"
echo ""

# Step 4: Deploy checkpoint (if requested)
if [ "$DEPLOY" = true ]; then
    echo "Step 4/5: Deploying checkpoint to production..."
    echo "-----------------------------------------------"

    .venv/bin/python scripts/deploy_checkpoint.py \
        --source "$CHECKPOINT_PATH" \
        --target outputs/production/checkpoint_best.pt

    if [ $? -ne 0 ]; then
        echo "❌ Deployment failed"
        exit 1
    fi

    echo "✓ Checkpoint deployed"
    echo ""
else
    echo "Step 4/5: Skipping deployment (use --deploy to enable)"
    echo "------------------------------------------------------"
    echo ""
fi

# Step 5: Generate comparison (if multiple checkpoints exist)
echo "Step 5/5: Generating comparison report..."
echo "-----------------------------------------"

# Find all checkpoints to compare
CHECKPOINTS=()
if [ -f "outputs/training_10epoch/checkpoint_best.pt" ]; then
    CHECKPOINTS+=("outputs/training_10epoch/checkpoint_best.pt")
fi
if [ -f "outputs/training_50epoch/checkpoint_best.pt" ]; then
    CHECKPOINTS+=("outputs/training_50epoch/checkpoint_best.pt")
fi
if [ -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINTS+=("$CHECKPOINT_PATH")
fi

if [ ${#CHECKPOINTS[@]} -gt 1 ]; then
    .venv/bin/python scripts/compare_checkpoints.py \
        --checkpoints "${CHECKPOINTS[@]}" \
        --test-data outputs/processed_quick/test_sequences.npz \
        --vocab-path data/vocabulary.json \
        --output outputs/checkpoint_comparison

    if [ $? -ne 0 ]; then
        echo "⚠️  Comparison generation failed (non-fatal)"
    else
        echo "✓ Comparison report generated"
    fi
else
    echo "⚠️  Only one checkpoint found, skipping comparison"
fi

echo ""
echo "================================"
echo "WORKFLOW COMPLETE!"
echo "================================"
echo ""
echo "Results Summary:"
echo "  - Sweep analysis: outputs/sweep_analysis/"
echo "  - Best checkpoint config: outputs/best_from_sweep/best_config.json"
echo "  - Evaluation results: outputs/evaluation_best/"
echo "  - Comparison report: outputs/checkpoint_comparison/"
if [ "$DEPLOY" = true ]; then
    echo "  - Deployed checkpoint: outputs/production/checkpoint_best.pt"
fi
echo ""
echo "Next Steps:"
echo "  1. Review evaluation results: cat outputs/evaluation_best/evaluation_results.json"
echo "  2. View comparison report: cat outputs/checkpoint_comparison/comparison_report.md"
if [ "$DEPLOY" = true ]; then
    echo "  3. Restart API server to load new checkpoint"
    echo "  4. Test prediction: curl -X POST http://localhost:8000/predict -d @test_payload.json"
else
    echo "  3. Deploy checkpoint: ./scripts/post_sweep_workflow.sh --sweep-id $SWEEP_ID --deploy"
fi
echo ""
