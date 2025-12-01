#!/bin/bash
# Training Commands for G-Code Fingerprinting
# Updated: 2025-11-29 with composite_acc tracking

set -e  # Exit on error

# ============================================================================
# OPTION 1: Single Training Run (Recommended for Testing)
# ============================================================================
# Use the best configuration from experiment comparison
# Best config: hidden=256, heads=8, layers=5, lr=5.4e-05, batch=32

run_single_training() {
    echo "Starting single training run with best configuration..."

    PYTHONPATH=src python scripts/train_multihead.py \
        --use-wandb \
        --data-dir=outputs/processed_hybrid \
        --vocab-path=data/vocabulary_1digit_hybrid.json \
        --class-weights-path=outputs/class_weights_hybrid.json \
        --output-dir=outputs/best_config_run \
        --hidden_dim=256 \
        --num_heads=8 \
        --num_layers=5 \
        --learning_rate=5.415e-05 \
        --batch_size=32 \
        --weight_decay=0.05 \
        --dropout=0.15 \
        --grad-clip=1.0 \
        --lr-scheduler=cosine \
        --warmup-epochs=10 \
        --max-epochs=200 \
        --patience=15 \
        --command_weight=2.0 \
        --param_type_weight=1.0 \
        --param_value_weight=1.0 \
        --operation_weight=1.0 \
        --grammar_weight=0.05 \
        --accumulation-steps=1 \
        --oversample-factor=1

    echo "✅ Training complete! Check outputs/best_config_run/"
}

# ============================================================================
# OPTION 2: W&B Sweep (Hyperparameter Search)
# ============================================================================
# Runs Bayesian optimization to find optimal hyperparameters
# Now optimizes composite_acc and allows longer training (50+ epochs)

run_sweep() {
    echo "Starting W&B sweep with updated configuration..."
    echo "Sweep will optimize val/composite_acc and allow 50+ epoch runs"

    # Initialize the sweep
    SWEEP_ID=$(wandb sweep configs/sweep_comprehensive.yaml 2>&1 | grep -oE "[a-z0-9]{8}")

    if [ -z "$SWEEP_ID" ]; then
        echo "❌ Failed to create sweep"
        exit 1
    fi

    echo "✅ Sweep created: $SWEEP_ID"
    echo "🔗 View at: https://wandb.ai/seacuello-university-of-rhode-island/gcode-fingerprinting/sweeps/$SWEEP_ID"
    echo ""
    echo "To run sweep agents:"
    echo "  wandb agent seacuello-university-of-rhode-island/gcode-fingerprinting/$SWEEP_ID"
    echo ""
    echo "Starting 1 agent automatically..."

    # Start agent
    wandb agent "seacuello-university-of-rhode-island/gcode-fingerprinting/$SWEEP_ID"
}

# ============================================================================
# OPTION 3: Continue Existing Sweep
# ============================================================================
# If you already have a sweep running, use this to add more agents

continue_sweep() {
    SWEEP_ID="${1:-83bwwuca}"  # Default to current sweep

    echo "Continuing sweep: $SWEEP_ID"
    echo "⚠️  WARNING: This will use the OLD sweep config (not updated)"
    echo "   Consider starting a new sweep instead with run_sweep()"
    echo ""

    wandb agent "seacuello-university-of-rhode-island/gcode-fingerprinting/$SWEEP_ID"
}

# ============================================================================
# OPTION 4: Kill Current Sweep (Stop Early Termination Issues)
# ============================================================================

kill_current_sweep() {
    echo "Stopping any running sweep agents..."
    pkill -f "wandb agent" || echo "No agents running"

    # Also stop any training processes
    pkill -f "train_multihead.py" || echo "No training processes running"

    echo "✅ All processes stopped"
}

# ============================================================================
# Main Menu
# ============================================================================

show_menu() {
    echo ""
    echo "=========================================="
    echo "G-Code Fingerprinting Training"
    echo "=========================================="
    echo ""
    echo "Choose an option:"
    echo "  1) Single training run (best config)"
    echo "  2) Start NEW sweep (recommended)"
    echo "  3) Continue existing sweep"
    echo "  4) Kill current sweep/training"
    echo "  q) Quit"
    echo ""
}

# If script is run directly (not sourced)
if [ "${BASH_SOURCE[0]}" -eq "${0}" ]; then
    show_menu
    read -p "Enter choice [1-4, q]: " choice

    case $choice in
        1)
            run_single_training
            ;;
        2)
            run_sweep
            ;;
        3)
            read -p "Enter sweep ID (or press Enter for default): " sweep_id
            continue_sweep "$sweep_id"
            ;;
        4)
            kill_current_sweep
            ;;
        q|Q)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
fi
