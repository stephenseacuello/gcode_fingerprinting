#!/bin/bash
# Quick evaluation script for sweep 27v7pl9i with correct vocabulary

cd "$(dirname "$0")/.."

echo "=== Evaluating Sweep 27v7pl9i Best Checkpoint ==="
echo ""

# Use the hybrid 1-digit vocabulary (69 tokens) to match the checkpoint
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py \
    --checkpoint outputs/best_from_sweep_27v7pl9i/checkpoint_best.pt \
    --test-data outputs/processed_hybrid/test_sequences.npz \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output outputs/evaluation_27v7pl9i

echo ""
echo "=== Generating Visualizations ==="
PYTHONPATH=src .venv/bin/python scripts/generate_visualizations.py \
    --all \
    --use-real-data \
    --checkpoint-path outputs/best_from_sweep_27v7pl9i/checkpoint_best.pt \
    --test-data outputs/processed_hybrid/test_sequences.npz \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output figures/sweep_27v7pl9i_final

echo ""
echo "✅ Complete! Check outputs/evaluation_27v7pl9i and figures/sweep_27v7pl9i_final"
