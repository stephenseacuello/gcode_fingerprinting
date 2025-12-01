#!/bin/bash
# Commands to analyze the best model

# Run comprehensive performance analysis
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/analyze_performance.py \
    --checkpoint outputs/best_config_training/checkpoint_best.pt \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output-dir outputs/analysis_20251129_135758 \
    --wandb-run 4hufje7i

# Run all visualizations
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \
    --model-path outputs/best_config_training/checkpoint_best.pt \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output-dir outputs/visualizations_20251129_135758 \
    --mode all

# Run interpretation visualization only
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \
    --model-path outputs/best_config_training/checkpoint_best.pt \
    --mode interpretation

# Run interactive visualization only
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \
    --model-path outputs/best_config_training/checkpoint_best.pt \
    --mode interactive

# Run production visualization only
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \
    --model-path outputs/best_config_training/checkpoint_best.pt \
    --mode production

