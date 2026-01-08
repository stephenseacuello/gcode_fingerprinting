# Scripts Directory

Organized scripts for training, evaluation, and analysis of G-Code fingerprinting models.

## Directory Structure

```
scripts/
├── training/         # Model training scripts
├── evaluation/       # Model evaluation and metrics
├── analysis/         # Error analysis and debugging
├── data/             # Data preprocessing and splits
├── experiments/      # Ablation studies and baselines
├── visualization/    # Figures and plots
├── utils/            # Shell utilities
└── archived/         # Deprecated/old scripts
```

## Training Scripts

| Script | Description |
|--------|-------------|
| `training/train_sensor_multihead.py` | **Main training script** - SensorMultiHeadDecoder with all features |
| `training/train_mm_dtae_lstm.py` | Train the sensor encoder (MM-DTAE-LSTM) |
| `training/train_mlp_baseline.py` | Simple MLP baseline |
| `training/train_sklearn_baselines.py` | Classical ML baselines (RF, SVM, etc.) |
| `training/train_ensemble_diverse_arch.sh` | Train 3 models with different architectures |
| `training/train_ensemble_diverse_seeds.sh` | Train 3 models with different seeds |

### Quick Start

```bash
# Train the main model
PYTHONPATH=src .venv/bin/python scripts/training/train_sensor_multihead.py \
    --split-dir outputs/production/data_splits \
    --vocab-path data/vocabulary_4digit_full.json \
    --encoder-path outputs/production/encoder/best_model.pt \
    --output-dir outputs/my_model \
    --use-wandb

# Train encoder from scratch
PYTHONPATH=src .venv/bin/python scripts/training/train_mm_dtae_lstm.py \
    --data-dir data \
    --output-dir outputs/encoder
```

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `evaluation/evaluate_checkpoint.py` | Evaluate a single model checkpoint |
| `evaluation/evaluate_ensemble_simple.py` | Evaluate ensemble of models (soft voting) |
| `evaluation/compute_calibration_metrics.py` | Compute ECE and reliability diagrams |
| `evaluation/generate_confusion_matrix.py` | Generate confusion matrices |

### Quick Start

```bash
# Evaluate single model
PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_checkpoint.py \
    --checkpoint outputs/production/best_model/best_model.pt

# Evaluate ensemble
PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_ensemble_simple.py \
    --checkpoints model1.pt model2.pt model3.pt \
    --output reports/ensemble_results.json
```

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `analysis/analyze_failure_modes.py` | Identify common prediction errors |
| `analysis/analyze_oov_tokens.py` | Check for out-of-vocabulary tokens |
| `analysis/error_analysis.py` | Detailed per-token error analysis |
| `analysis/compare_decoding_strategies.py` | Compare greedy vs beam search |

## Data Scripts

| Script | Description |
|--------|-------------|
| `data/create_multilabel_stratified_splits.py` | Create train/val/test splits |
| `data/rebuild_vocabulary.py` | Regenerate token vocabulary |
| `data/verify_vocabulary_coverage.py` | Check vocabulary coverage |
| `data/add_raw_values_to_processed.py` | Add raw sensor values to processed data |

## Experiment Scripts

| Script | Description |
|--------|-------------|
| `experiments/run_architecture_ablations.py` | Ablation study on architecture |
| `experiments/run_baseline_comparisons.py` | Compare against baselines |
| `experiments/ablate_encoder.py` | Encoder ablation experiments |
| `experiments/run_multiseed.sh` | Multi-seed training runs |

## Utility Scripts

| Script | Description |
|--------|-------------|
| `utils/cleanup.sh` | Clean temporary files |
| `utils/cleanup_artifacts.sh` | Clean WandB artifacts |
| `utils/monitor.sh` | Monitor training progress |
| `utils/run_pipeline.sh` | Full training pipeline |

## Key Command-Line Arguments

### train_sensor_multihead.py

```
Required:
  --split-dir PATH       Directory with train/val/test .npz files
  --vocab-path PATH      Path to vocabulary JSON
  --encoder-path PATH    Path to pretrained encoder checkpoint

Architecture:
  --d-model INT          Transformer dimension (default: 320)
  --n-layers INT         Number of decoder layers (default: 5)
  --n-heads INT          Attention heads (default: 8)
  --dropout FLOAT        Dropout rate (default: 0.32)

Training:
  --max-epochs INT       Maximum epochs (default: 150)
  --patience INT         Early stopping patience (default: 40)
  --learning-rate FLOAT  Initial LR (default: 0.00024)
  --lr-scheduler STR     cosine_restarts, cosine, plateau, step

Features:
  --use-sensor-prior     Enable sensor-based prior
  --augment              Enable data augmentation
  --use-enhanced-encoder Enable enhanced encoder
  --use-wandb            Log to Weights & Biases
```

## Output Files

Each training run creates:
- `best_model.pt` - Best checkpoint (by validation metric)
- `results.json` - Final metrics and hyperparameters
- `training_log.csv` - Epoch-by-epoch metrics (if not using wandb)
