# Complete Pipeline Guide

**Project**: G-code Fingerprinting with Machine Learning
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Setup](#setup)
2. [Stage 1: Data Preparation](#stage-1-data-preparation)
3. [Stage 2: Vocabulary Building](#stage-2-vocabulary-building)
4. [Stage 3: Preprocessing](#stage-3-preprocessing)
5. [Stage 4: Training](#stage-4-training)
6. [Stage 5: Evaluation](#stage-5-evaluation)
7. [Stage 6: Visualization](#stage-6-visualization)
8. [Stage 7: Hyperparameter Sweeps](#stage-7-hyperparameter-sweeps)
9. [Stage 8: Deployment](#stage-8-deployment)

---

## Setup

### Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"
```

### Project Structure

```
gcode_fingerprinting/
├── src/miracle/              # Source code
│   ├── dataset/             # Data loading, preprocessing, augmentation
│   ├── model/               # Model architectures (MM_DTAE_LSTM, MultiHeadGCodeLM)
│   ├── training/            # Training loops, losses, metrics
│   └── utilities/           # Tokenization, helpers
├── scripts/                  # Executable scripts
│   ├── train_multihead.py   # Main training script
│   ├── train_*.py           # Training variations
│   ├── generate_*.py        # Visualization generators
│   ├── api_server.py        # FastAPI inference server
│   └── utils/               # Shell utilities
├── configs/                  # Configuration files
│   ├── sweep_config.yaml    # W&B sweep configuration
│   └── *.json               # Training configs
├── data/                    # Raw data & vocabulary
│   ├── *.csv                # Raw sensor + G-code data (100 files)
│   └── gcode_vocab_v2.json  # Vocabulary (170 tokens)
├── docs/                    # Documentation
├── outputs/                 # Training outputs
│   ├── processed_v2/        # Preprocessed sequences
│   ├── wandb_sweeps/        # Sweep checkpoints
│   └── figures/             # Visualization outputs
└── tests/                   # Unit tests
```

---

## Stage 1: Data Preparation

### Analyze Raw Data

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_raw_data.py --data-dir data/
```

**Output**: Generates analysis of:
- File counts and sizes
- Sensor feature statistics (motor currents, temperatures, etc.)
- G-code token distribution
- Missing value detection
- Class imbalance metrics

### Expected Data Format

Each CSV file contains:
- **Sensor Features** (8 columns): Motor currents (I1-I4), temperatures (T1-T4)
- **G-code Commands**: Target tokens for each timestep
- **Timestamps**: Optional temporal markers

---

## Stage 2: Vocabulary Building

### Build Vocabulary with 2-Digit Bucketing

```bash
PYTHONPATH=src .venv/bin/python -m miracle.utilities.gcode_tokenizer build-vocab \
    --data-dir data/ \
    --output data/gcode_vocab_v2.json \
    --bucket-digits 2 \
    --vocab-size 200 \
    --min-freq 3
```

**What This Does**:
- Scans all CSV files in `data/`
- Buckets numeric parameters to 2-digit precision (e.g., `X123.45` → `X120`)
- Creates vocabulary of ~170 unique tokens
- Includes special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`

**Output**: `data/gcode_vocab_v2.json`

---

## Stage 3: Preprocessing

### Create Training Sequences

```bash
PYTHONPATH=src .venv/bin/python -m miracle.dataset.preprocessing \
    --data-dir data/ \
    --output-dir outputs/processed_v2/ \
    --vocab-path data/gcode_vocab_v2.json \
    --window-size 64 \
    --stride 16
```

**Parameters**:
- `window-size 64`: Each sequence contains 64 timesteps
- `stride 16`: Overlapping windows with 75% overlap
- Creates train/val/test splits (70/15/15)

**Output Files**:
```
outputs/processed_v2/
├── train_sequences.npz      # 2212 sequences
├── val_sequences.npz        # 474 sequences
├── test_sequences.npz       # 474 sequences
└── metadata.json            # Dataset statistics
```

**Each .npz contains**:
- `sensor_data`: Shape [N, 64, 8] - Sensor readings
- `gcode_tokens`: Shape [N, 64] - Token indices
- `decomposed_targets`: Shape [N, 64, 4] - Hierarchical targets

---

## Stage 4: Training

### Single Training Run

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_multihead.py \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/training \
    --max-epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --use-wandb
```

### With Data Augmentation

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_with_augmentation.py \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/training_aug \
    --augmentation-prob 0.3 \
    --use-wandb
```

**Augmentation Techniques**:
1. Gaussian noise (σ=0.01)
2. Time warping (±5%)
3. Magnitude scaling (0.95-1.05)
4. Time masking (10% window)
5. Feature dropout (10%)
6. Class-aware oversampling (3x for rare tokens)

### Monitor Training

```bash
# W&B dashboard
wandb whoami

# Check local logs
tail -f outputs/training/train_log.txt

# GPU/MPS usage
top -o cpu
```

**Key Metrics**:
- `train/gcode_type_acc`: Token type accuracy (99%+)
- `train/gcode_cmd_acc`: Command accuracy (100%)
- `train/gcode_param_type_acc`: Parameter type (85%+)
- `train/gcode_param_val_acc`: Parameter value (60%+)
- `train/unique_tokens`: Unique predictions (target: >100/170)

---

## Stage 5: Evaluation

### Test on Holdout Set

```bash
PYTHONPATH=src .venv/bin/python scripts/test_local_checkpoint.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz \
    --vocab-path data/gcode_vocab_v2.json
```

**Output**: Per-head accuracies, confusion matrices, error analysis

### Batch Evaluation (Multiple Checkpoints)

```bash
PYTHONPATH=src .venv/bin/python scripts/load_results.py \
    --checkpoints-dir outputs/wandb_sweeps/ \
    --test-data outputs/processed_v2/test_sequences.npz \
    --output results/evaluation.json
```

---

## Stage 6: Visualization

### Generate All Figures

```bash
.venv/bin/python scripts/generate_visualizations.py --all --output outputs/figures/
```

### Generate Specific Figures

```bash
# Confusion matrices
.venv/bin/python scripts/generate_visualizations.py --confusion-matrices

# Bootstrap confidence intervals
.venv/bin/python scripts/generate_visualizations.py --confidence-intervals

# Accuracy distributions
.venv/bin/python scripts/generate_visualizations.py --accuracy-distribution

# Token embedding space (t-SNE)
.venv/bin/python scripts/generate_visualizations.py --embedding-space
```

### Using Real Results

```bash
.venv/bin/python scripts/generate_visualizations.py \
    --confidence-intervals \
    --accuracy-distribution \
    --use-real-data \
    --checkpoint-path outputs/wandb_sweeps/RUN_ID/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz \
    --output outputs/figures/real_results/
```

**See [VISUALIZATION.md](VISUALIZATION.md) for complete guide.**

---

## Stage 7: Hyperparameter Sweeps

### Configure Sweep

Edit `configs/sweep_config.yaml`:

```yaml
method: bayes
metric:
  name: val/gcode_acc
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  d_model:
    values: [128, 256, 512]
  nhead:
    values: [4, 8]
  num_layers:
    values: [2, 4, 6]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
```

### Run Sweep

```bash
# Create sweep
.venv/bin/wandb sweep configs/sweep_config.yaml

# Run agents (multiple terminals for parallel search)
.venv/bin/wandb agent YOUR_SWEEP_ID
```

### Analyze Results

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep_results.py \
    --sweep-id YOUR_SWEEP_ID \
    --output reports/sweep_analysis.csv
```

**See [TRAINING.md](TRAINING.md) for detailed sweep guide.**

---

## Stage 8: Deployment

### Export Model to ONNX

```bash
PYTHONPATH=src .venv/bin/python scripts/export_onnx.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --vocab-path data/gcode_vocab_v2.json \
    --output models/gcode_model.onnx
```

### Quantize Model (INT8)

```bash
PYTHONPATH=src .venv/bin/python scripts/quantize_model.py \
    --onnx-path models/gcode_model.onnx \
    --output models/gcode_model_int8.onnx
```

### Start API Server

```bash
PYTHONPATH=src .venv/bin/python scripts/api_server.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --vocab-path data/gcode_vocab_v2.json \
    --host 0.0.0.0 \
    --port 8000
```

**Test Inference**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [[1.2, 0.8, 1.1, 0.9, 25.3, 26.1, 24.8, 25.5], ...]
  }'
```

**See [API.md](API.md) for complete API reference.**

---

## Common Issues

### ModuleNotFoundError

**Solution**: Always use `PYTHONPATH=src`:
```bash
PYTHONPATH=src .venv/bin/python -m miracle.dataset.preprocessing ...
```

### MPS Out of Memory (Mac M1/M2)

**Solution**: Reduce batch size or use CPU:
```bash
# Reduce batch size
--batch-size 8

# Force CPU
--device cpu
```

### W&B Authentication

```bash
wandb login YOUR_API_KEY
```

### Old Checkpoints Incompatible

**Cause**: Preprocessing version mismatch (vocab v1 vs v2)

**Solution**: Use checkpoints from same vocab version

---

## Performance Benchmarks

### Expected Results (Vocab v2, 64-window)

| Metric | Target | Typical |
|--------|--------|---------|
| Token Type Accuracy | >99% | 99.8% |
| Command Accuracy | 100% | 100% |
| Param Type Accuracy | >80% | 84.3% |
| Param Value Accuracy | >50% | 56-60% |
| Unique Tokens Predicted | >100/170 | 120-140 |
| Training Time (50 epochs) | <2 hours | 1.5 hours |

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB RAM, GPU with 4GB VRAM
- **Optimal**: Mac M1/M2 with 16GB+ unified memory

---

## Next Steps

1. **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
2. **Training Details**: See [TRAINING.md](TRAINING.md)
3. **Visualization**: See [VISUALIZATION.md](VISUALIZATION.md)
4. **API Reference**: See [API.md](API.md)

---

**Questions?** Check the documentation or open an issue.
