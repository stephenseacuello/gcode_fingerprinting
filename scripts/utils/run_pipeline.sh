#!/bin/bash
# Complete G-code Fingerprinting Pipeline
# Runs end-to-end from raw data to trained models

set -e  # Exit on error

echo "==========================================="
echo "G-code Fingerprinting Pipeline"
echo "==========================================="
echo

# Configuration
DATA_DIR="data"
VOCAB_V2="data/gcode_vocab_v2.json"
PROCESSED_DIR="outputs/processed_v2"
BUCKET_DIGITS=2
WINDOW_SIZE=64
STRIDE=16

# Parse command line arguments
STAGE="${1:-all}"  # all, vocab, preprocess, train, evaluate, visualize

echo "Pipeline Stage: $STAGE"
echo

# ============================================
# STAGE 1: Raw Data Analysis (optional)
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "analyze" ]; then
    echo "============================================"
    echo "STAGE 1: Raw Data Analysis"
    echo "============================================"

    if [ -f "analyze_raw_data.py" ]; then
        python analyze_raw_data.py --data-dir "$DATA_DIR"
        echo "✓ Raw data analysis complete"
    else
        echo "⚠ analyze_raw_data.py not found, skipping"
    fi
    echo
fi

# ============================================
# STAGE 2: Vocabulary Building
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "vocab" ]; then
    echo "============================================"
    echo "STAGE 2: Building Vocabulary (2-digit bucketing)"
    echo "============================================"

    PYTHONPATH=src .venv/bin/python -m miracle.utilities.gcode_tokenizer build-vocab \
        --data-dir "$DATA_DIR" \
        --output "$VOCAB_V2" \
        --vocab-size 200 \
        --min-freq 3 \
        --bucket-digits "$BUCKET_DIGITS"

    # Verify vocabulary
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$VOCAB_V2'))['vocab']))")
    echo "✓ Vocabulary built: $VOCAB_SIZE tokens"
    echo
fi

# ============================================
# STAGE 3: Data Preprocessing
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "preprocess" ]; then
    echo "============================================"
    echo "STAGE 3: Data Preprocessing"
    echo "============================================"

    PYTHONPATH=src .venv/bin/python -m miracle.dataset.preprocessing \
        --data-dir "$DATA_DIR" \
        --output-dir "$PROCESSED_DIR" \
        --vocab-path "$VOCAB_V2" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE"

    # Verify preprocessed data
    TRAIN_COUNT=$(python3 -c "import numpy as np; print(len(np.load('$PROCESSED_DIR/train_sequences.npz')['tokens']))")
    VAL_COUNT=$(python3 -c "import numpy as np; print(len(np.load('$PROCESSED_DIR/val_sequences.npz')['tokens']))")
    TEST_COUNT=$(python3 -c "import numpy as np; print(len(np.load('$PROCESSED_DIR/test_sequences.npz')['tokens']))")

    echo "✓ Data preprocessed:"
    echo "  Train: $TRAIN_COUNT sequences"
    echo "  Val:   $VAL_COUNT sequences"
    echo "  Test:  $TEST_COUNT sequences"
    echo
fi

# ============================================
# STAGE 4: Training
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
    echo "============================================"
    echo "STAGE 4: Training Models"
    echo "============================================"

    # Train using best config from Phase 1
    echo "Training Model with Vocab v2..."
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_phase1_fixed.py \
        --config configs/phase1_best.json \
        --data-dir "$PROCESSED_DIR" \
        --output-dir outputs/baseline_v2 \
        --max-epochs 50 \
        --patience 10 \
        --use-wandb \
        --run-name "baseline-vocab-v2"

    echo "✓ Model trained"
    echo

    # Note: To use augmentation, modify the dataset loading in train_phase1_fixed.py
    # or create a new training script that uses AugmentedGCodeDataset
fi

# ============================================
# STAGE 5: Evaluation
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "evaluate" ]; then
    echo "============================================"
    echo "STAGE 5: Evaluation"
    echo "============================================"

    if [ -f "outputs/baseline_v2/checkpoint_best.pt" ]; then
        echo "Evaluating baseline model..."

        if [ -f "evaluate.py" ]; then
            PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python evaluate.py \
                --checkpoint outputs/baseline_v2/checkpoint_best.pt \
                --data-dir "$PROCESSED_DIR" \
                --vocab-path "$VOCAB_V2" \
                --output-dir outputs/baseline_v2/evaluation \
                --split test

            echo "✓ Evaluation complete"
        else
            echo "⚠ evaluate.py not found, skipping"
        fi
    else
        echo "⚠ No checkpoint found, skipping evaluation"
    fi
    echo
fi

# ============================================
# STAGE 6: Visualization
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "visualize" ]; then
    echo "============================================"
    echo "STAGE 6: Visualization"
    echo "============================================"

    if [ -f "outputs/baseline_v2/checkpoint_best.pt" ]; then
        echo "Creating visualizations..."

        if [ -f "quick_visualize.py" ]; then
            python quick_visualize.py \
                --checkpoint outputs/baseline_v2/checkpoint_best.pt \
                --data-dir "$PROCESSED_DIR" \
                --vocab-path "$VOCAB_V2" \
                --output-dir outputs/baseline_v2/visualizations

            echo "✓ Visualizations created"
        else
            echo "⚠ quick_visualize.py not found, skipping"
        fi
    else
        echo "⚠ No checkpoint found, skipping visualization"
    fi
    echo
fi

# ============================================
# STAGE 7: Analysis & Organization
# ============================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "analyze_results" ]; then
    echo "============================================"
    echo "STAGE 7: Analysis & Organization"
    echo "============================================"

    if [ -f "analyze_and_organize.py" ]; then
        python analyze_and_organize.py \
            --wandb-project gcode-fingerprinting \
            --output-dir reports/analysis

        echo "✓ Analysis complete"
    else
        echo "⚠ analyze_and_organize.py not found, skipping"
    fi
    echo
fi

# ============================================
# Summary
# ============================================
echo "==========================================="
echo "Pipeline Summary"
echo "==========================================="
echo

if [ -f "$VOCAB_V2" ]; then
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$VOCAB_V2'))['vocab']))")
    echo "✓ Vocabulary: $VOCAB_SIZE tokens"
fi

if [ -d "$PROCESSED_DIR" ]; then
    TRAIN_COUNT=$(python3 -c "import numpy as np; print(len(np.load('$PROCESSED_DIR/train_sequences.npz')['tokens']))" 2>/dev/null || echo "N/A")
    echo "✓ Preprocessed: $TRAIN_COUNT train sequences"
fi

if [ -f "outputs/baseline_v2/checkpoint_best.pt" ]; then
    echo "✓ Trained: Baseline model"
fi

echo
echo "✅ Pipeline complete!"
echo
echo "Next steps:"
echo "  1. Check training logs in outputs/baseline_v2/"
echo "  2. View metrics on WandB"
echo "  3. Run hyperparameter sweeps (see PIPELINE.md)"
echo

# Usage help
if [ "$STAGE" = "help" ]; then
    echo "Usage: $0 [stage]"
    echo
    echo "Stages:"
    echo "  all           - Run complete pipeline (default)"
    echo "  analyze       - Stage 1: Raw data analysis"
    echo "  vocab         - Stage 2: Build vocabulary"
    echo "  preprocess    - Stage 3: Preprocess data"
    echo "  train         - Stage 4: Train models"
    echo "  evaluate      - Stage 5: Evaluate models"
    echo "  visualize     - Stage 6: Create visualizations"
    echo "  analyze_results - Stage 7: Analyze & organize"
    echo
    echo "Examples:"
    echo "  $0               # Run complete pipeline"
    echo "  $0 vocab         # Only build vocabulary"
    echo "  $0 train         # Only train models"
    echo
fi
