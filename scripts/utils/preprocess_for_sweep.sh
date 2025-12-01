#!/bin/bash
# Quick preprocessing script for sweep training

set -e

echo "============================================================"
echo "📦 Preprocessing Data for Training"
echo "============================================================"

# Check CSV files exist
CSV_COUNT=$(ls data/*_aligned.csv 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    echo "❌ No aligned CSV files found in data/"
    exit 1
fi

echo "✅ Found $CSV_COUNT aligned CSV files"

# Check vocabulary exists
if [ ! -f "data/vocabulary.json" ]; then
    echo "❌ Vocabulary not found at data/vocabulary.json"
    exit 1
fi
echo "✅ Vocabulary found"

# Run preprocessing
echo ""
echo "🔄 Running preprocessing..."
echo "   This may take 5-15 minutes depending on data size..."
echo ""

PYTHONPATH=src .venv/bin/python -m miracle.dataset.preprocessing \
    --input-dir data \
    --output-dir data \
    --vocab-path data/vocabulary.json \
    --window-size 32 \
    --stride 16 \
    --val-split 0.15 \
    --test-split 0.15

# Check output
echo ""
if [ -f "data/train_sequences.npz" ] && [ -f "data/val_sequences.npz" ]; then
    echo "✅ Preprocessing complete!"
    echo ""
    ls -lh data/*sequences*.npz data/*sequences*.json
    echo ""
    echo "Ready for training!"
else
    echo "❌ Preprocessing failed - output files not found"
    exit 1
fi
