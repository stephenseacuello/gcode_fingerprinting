#!/bin/bash
#
# Train model with best sweep configuration (run iiv4luz5)
# Expected result: ~100% validation accuracy
#

set -e  # Exit on error

echo "========================================================================"
echo "TRAINING WITH SWEEP BEST CONFIGURATION (iiv4luz5)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - hidden_dim: 256"
echo "  - num_layers: 5"
echo "  - num_heads: 8"
echo "  - batch_size: 32"
echo "  - learning_rate: 0.000172"
echo "  - optimizer: adam + onecycle scheduler"
echo ""
echo "Expected validation accuracy: ~100%"
echo "Training time: ~2-3 hours"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Set working directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run training
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python scripts/train_multihead.py \
  --config configs/sweep_best_iiv4luz5.json \
  --data-dir outputs/processed_with_ops \
  --vocab-path data/vocabulary.json \
  --output-dir outputs/sweep_best_retrained \
  --max-epochs 15 \
  --patience 10 \
  --use-wandb \
  --wandb-project gcode-fingerprinting \
  --run-name "sweep-best-retrain" \
  --hidden_dim 256 \
  --num_layers 5 \
  --num_heads 8 \
  --batch_size 32 \
  --learning_rate 0.00017242889465641294 \
  --label_smoothing 0.1 \
  --weight_decay 0.05

echo ""
echo "========================================================================"
echo "✅ TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Model saved to: outputs/sweep_best_retrained/"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set:"
echo "     python scripts/evaluate_production_model.py \\"
echo "       --checkpoint outputs/sweep_best_retrained/checkpoint_best.pt \\"
echo "       --test-data outputs/processed_with_ops/test_sequences.npz \\"
echo "       --vocab data/vocabulary.json"
echo ""
echo "  2. Expected result: ~100% accuracy"
echo ""
