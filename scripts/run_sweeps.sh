#!/bin/bash
# Sweep runner script for systematic hyperparameter optimization

set -e

echo "=========================================="
echo "G-Code Fingerprinting: Sweep Runner"
echo "=========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check for virtual environment and activate if exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Activated virtual environment"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Activated virtual environment"
fi

# Check if wandb is available
if ! command -v wandb &> /dev/null; then
    echo "Error: wandb not found. Install with: pip install wandb"
    echo "Make sure you've activated your virtual environment:"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Check if logged in (check for API key)
if [ -z "$WANDB_API_KEY" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "Warning: W&B may not be configured."
    echo "If sweeps fail, please login first:"
    echo "  wandb login"
    echo ""
fi

echo "✓ W&B ready"

# Parse arguments
SWEEP_NAME=""
NUM_AGENTS=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep)
            SWEEP_NAME="$2"
            shift 2
            ;;
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: ./scripts/run_sweeps.sh --sweep SWEEP_NAME [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sweep NAME      Sweep to run (required)"
            echo "  --agents N        Number of parallel agents (default: 1)"
            echo "  --dry-run         Print commands without executing"
            echo "  --help            Show this help"
            echo ""
            echo "Available sweeps:"
            echo "  basic            - Basic hyperparameter sweep (RECOMMENDED - works now!)"
            echo "  vocabulary       - Vocabulary bucketing optimization (needs preprocessing)"
            echo "  augmentation     - Data augmentation parameters (needs script updates)"
            echo "  warmup           - Warmup scheduler optimization (needs script updates)"
            echo "  architecture     - Model architecture search (needs script updates)"
            echo "  loss_weighting   - Loss weight optimization (needs script updates)"
            echo "  all              - Run all sweeps sequentially"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run_sweeps.sh --sweep vocabulary"
            echo "  ./scripts/run_sweeps.sh --sweep architecture --agents 4"
            echo "  ./scripts/run_sweeps.sh --sweep all --agents 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate sweep name
if [ -z "$SWEEP_NAME" ]; then
    echo "Error: --sweep required"
    echo "Use --help for usage information"
    exit 1
fi

# Function to get config file for sweep
get_sweep_config() {
    case "$1" in
        basic)
            echo "sweeps/phase3/basic_hyperparameter_sweep.yaml"
            ;;
        vocabulary)
            echo "sweeps/phase3/vocabulary_bucketing.yaml"
            ;;
        augmentation)
            echo "sweeps/phase3/augmentation_optimization.yaml"
            ;;
        warmup)
            echo "sweeps/phase3/warmup_optimization.yaml"
            ;;
        architecture)
            echo "sweeps/phase3/architecture_sweep.yaml"
            ;;
        loss_weighting)
            echo "sweeps/phase3/loss_weighting.yaml"
            ;;
        *)
            echo ""
            ;;
    esac
}

run_sweep() {
    local sweep_name=$1
    local config_file=$(get_sweep_config "$sweep_name")

    if [ -z "$config_file" ]; then
        echo "Error: Unknown sweep '$sweep_name'"
        return 1
    fi

    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        return 1
    fi

    echo ""
    echo "=========================================="
    echo "Running sweep: $sweep_name"
    echo "Config: $config_file"
    echo "Agents: $NUM_AGENTS"
    echo "=========================================="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  wandb sweep $config_file"
        echo "  wandb agent <sweep_id> --count $NUM_AGENTS"
        return 0
    fi

    # Create sweep
    echo "Creating sweep..."
    SWEEP_ID=$(wandb sweep "$config_file" 2>&1 | grep "wandb agent" | awk '{print $NF}')

    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Failed to create sweep"
        return 1
    fi

    echo "✓ Sweep created: $SWEEP_ID"
    echo ""

    # Run agents
    echo "Starting $NUM_AGENTS agent(s)..."
    if [ "$NUM_AGENTS" -eq 1 ]; then
        # Single agent (sequential)
        wandb agent "$SWEEP_ID"
    else
        # Multiple agents (parallel)
        for i in $(seq 1 $NUM_AGENTS); do
            echo "  Starting agent $i..."
            wandb agent "$SWEEP_ID" &
        done
        wait  # Wait for all agents to finish
    fi

    echo ""
    echo "✓ Sweep complete: $sweep_name"
    echo "  View results: https://wandb.ai/<your-entity>/gcode-fingerprinting/sweeps/$SWEEP_ID"
    echo ""
}

# Run sweep(s)
if [ "$SWEEP_NAME" = "all" ]; then
    echo "Running all sweeps sequentially..."
    for sweep in vocabulary augmentation warmup architecture loss_weighting; do
        run_sweep "$sweep"
    done
else
    run_sweep "$SWEEP_NAME"
fi

echo ""
echo "=========================================="
echo "✓ All sweeps complete!"
echo "=========================================="
