#!/bin/bash
# Script to start wandb sweep with class weights

set -e  # Exit on error

echo "============================================================"
echo "🚀 Starting W&B Sweep with Class Weights"
echo "============================================================"

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

# 1. Check class weights exist
if [ ! -f "data/class_weights.pt" ]; then
    echo "❌ Class weights not found!"
    echo "   Generating class weights..."
    PYTHONPATH=src .venv/bin/python train_with_class_weights.py
fi
echo "✅ Class weights found"

# 2. Check preprocessed data exists
if [ ! -f "data/train_sequences.npz" ] || [ ! -f "data/val_sequences.npz" ]; then
    echo "❌ Preprocessed data not found!"
    echo ""
    echo "You need to preprocess your data first. Run:"
    echo "  PYTHONPATH=src .venv/bin/python -m miracle.dataset.preprocessing \\"
    echo "      --input-dir data \\"
    echo "      --output-dir data \\"
    echo "      --vocab-path data/vocabulary.json"
    echo ""
    exit 1
fi
echo "✅ Preprocessed data found"

# 3. Check wandb is installed
if ! .venv/bin/python -c "import wandb" 2>/dev/null; then
    echo "❌ wandb not installed!"
    echo "   Installing wandb..."
    .venv/bin/pip install wandb
fi
echo "✅ wandb installed"

# 4. Initialize sweep
echo ""
echo "🎯 Initializing sweep..."
SWEEP_ID=$(PYTHONPATH=src .venv/bin/wandb sweep sweeps/class_weight_sweep.yaml 2>&1 | grep -o "wandb agent [^ ]*" | cut -d' ' -f3)

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Failed to create sweep!"
    echo "   Trying manual sweep creation..."
    PYTHONPATH=src .venv/bin/wandb sweep sweeps/class_weight_sweep.yaml
    echo ""
    echo "⚠️  Copy the sweep ID from above and run:"
    echo "   PYTHONPATH=src .venv/bin/wandb agent <SWEEP_ID>"
    exit 1
fi

echo "✅ Sweep created: $SWEEP_ID"

# 5. Ask how many agents to run
echo ""
echo "How many parallel agents do you want to run? (1-4 recommended)"
read -p "Number of agents [1]: " NUM_AGENTS
NUM_AGENTS=${NUM_AGENTS:-1}

echo ""
echo "============================================================"
echo "🏃 Starting $NUM_AGENTS sweep agent(s)"
echo "============================================================"
echo ""
echo "Sweep ID: $SWEEP_ID"
echo "Sweep config: sweeps/class_weight_sweep.yaml"
echo "Training script: train_sweep.py"
echo ""
echo "Key metrics to watch:"
echo "  • val/g_command_acc (PRIMARY - this is what we're optimizing!)"
echo "  • val/m_command_acc"
echo "  • val/overall_acc"
echo "  • train/loss"
echo ""
echo "Press Ctrl+C to stop agents (sweep will continue on W&B)"
echo ""

# Run agents
if [ "$NUM_AGENTS" -eq 1 ]; then
    # Single agent (foreground)
    PYTHONPATH=src .venv/bin/wandb agent "$SWEEP_ID"
else
    # Multiple agents (background)
    for i in $(seq 1 $NUM_AGENTS); do
        echo "Starting agent $i/$NUM_AGENTS..."
        PYTHONPATH=src .venv/bin/wandb agent "$SWEEP_ID" &
    done
    echo ""
    echo "✅ All agents started in background"
    echo "   View progress at: https://wandb.ai"
    echo "   Stop all agents with: pkill -f 'wandb agent'"
    wait
fi
