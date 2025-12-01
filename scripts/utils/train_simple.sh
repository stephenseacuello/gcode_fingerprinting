#!/bin/bash
# Simple 50-epoch training - no wandb, no sweep, just basic training

echo "🚀 Starting simple 50-epoch training..."
echo ""

PYTHONPATH=src .venv/bin/python -m miracle.training.train \
  --data-dir outputs/processed_data_clean \
  --output-dir outputs/training_clean \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.0001 \
  --d-model 128 \
  --lstm-layers 2 \
  --optimizer adam \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --device cpu

echo ""
echo "=========================================="
echo "✅ Training complete!"
echo "Model saved to: outputs/training_clean/"
echo "=========================================="
