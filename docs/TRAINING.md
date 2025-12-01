# Training Guide

**Project**: G-code Fingerprinting with Machine Learning
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Training Scripts](#training-scripts)
3. [Hyperparameter Sweeps](#hyperparameter-sweeps)
4. [Data Augmentation](#data-augmentation)
5. [Monitoring & Debugging](#monitoring--debugging)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)

---

## Model Architecture

### Overview

The system uses a **two-stage hierarchical architecture**:

1. **Backbone (MM_DTAE_LSTM)**: Multi-modal encoder
   - Processes 8-channel sensor data (motor currents + temperatures)
   - LSTM-based sequence encoder
   - Outputs contextualized representations

2. **Multi-Head Language Model (MultiHeadGCodeLM)**: 4-head decoder
   - **Head 1**: Token Type (Command vs Parameter vs Special)
   - **Head 2**: G-code Command (G0, G1, M104, etc.)
   - **Head 3**: Parameter Type (X, Y, Z, E, F, etc.)
   - **Head 4**: Parameter Value (bucketed numeric values)

### Hierarchical Token Decomposition

Each G-code token is decomposed into 4 semantic components:

**Example**: `X120.5` →
- Type: `PARAM` (index 2)
- Command: `<PAD>` (index 0)
- Param Type: `X` (index for 'X')
- Param Value: `120` (bucketed to 2 digits)

**Example**: `G1` →
- Type: `CMD` (index 1)
- Command: `G1` (index for 'G1')
- Param Type: `<PAD>` (index 0)
- Param Value: `<PAD>` (index 0)

### Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `d_model` | 128 | 64-512 | Embedding dimension |
| `nhead` | 4 | 2-8 | Attention heads |
| `num_layers` | 4 | 2-6 | Transformer layers |
| `dim_feedforward` | 512 | 256-2048 | FFN dimension |
| `dropout` | 0.2 | 0.1-0.5 | Dropout rate |

---

## Training Scripts

### Basic Training

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

### Using Configuration Files

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_multihead.py \
    --config configs/baseline.json
```

**Example config** (`configs/baseline.json`):
```json
{
  "data_dir": "outputs/processed_v2",
  "vocab_path": "data/gcode_vocab_v2.json",
  "output_dir": "outputs/baseline",
  "max_epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "d_model": 128,
  "nhead": 4,
  "num_layers": 4,
  "dropout": 0.2,
  "use_wandb": true
}
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | required | Preprocessed data directory |
| `--vocab-path` | str | required | Vocabulary JSON file |
| `--output-dir` | str | required | Output directory for checkpoints |
| `--max-epochs` | int | 50 | Maximum training epochs |
| `--batch-size` | int | 32 | Training batch size |
| `--learning-rate` | float | 0.001 | Initial learning rate |
| `--d-model` | int | 128 | Embedding dimension |
| `--nhead` | int | 4 | Number of attention heads |
| `--num-layers` | int | 4 | Number of transformer layers |
| `--dropout` | float | 0.2 | Dropout probability |
| `--device` | str | auto | Device (cpu/cuda/mps) |
| `--use-wandb` | flag | False | Enable W&B logging |
| `--wandb-project` | str | gcode-fingerprinting | W&B project name |

---

## Hyperparameter Sweeps

### W&B Sweep Configuration

**File**: `configs/sweep_config.yaml`

```yaml
program: scripts/train_sweep.py
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
  weight_decay:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  augmentation_prob:
    values: [0.0, 0.2, 0.3, 0.5]
```

### Running Sweeps

```bash
# 1. Create sweep
.venv/bin/wandb sweep configs/sweep_config.yaml
# Output: Created sweep with ID: abc123xyz

# 2. Run agent (single process)
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz

# 3. Run multiple agents in parallel (3 terminals)
# Terminal 1:
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
# Terminal 2:
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
# Terminal 3:
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

### Analyze Sweep Results

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep_results.py \
    --sweep-id abc123xyz \
    --output reports/sweep_analysis.csv
```

**Output**: CSV with columns:
- Run ID
- Best validation accuracy
- Hyperparameters (lr, batch_size, d_model, etc.)
- Training time
- Final loss

### Sweep Strategies

#### 1. **Grid Search** (Exhaustive)
```yaml
method: grid
```
- Tests all combinations
- Good for small search spaces
- Expensive but thorough

#### 2. **Random Search** (Exploration)
```yaml
method: random
```
- Random sampling
- Good for large search spaces
- Fast exploration

#### 3. **Bayesian Optimization** (Recommended)
```yaml
method: bayes
```
- Guided search using Gaussian processes
- Balances exploration/exploitation
- Most efficient for expensive training

---

## Data Augmentation

### Augmentation Techniques

#### 1. **Gaussian Noise** (σ=0.01)
```python
sensor_data += torch.randn_like(sensor_data) * 0.01
```
- Adds robustness to sensor noise
- Applied to all features

#### 2. **Time Warping** (±5%)
```python
new_length = int(seq_len * random.uniform(0.95, 1.05))
warped = F.interpolate(sensor_data, size=new_length)
```
- Simulates variable execution speeds
- Resampled to original length

#### 3. **Magnitude Scaling** (0.95-1.05×)
```python
scale = random.uniform(0.95, 1.05)
sensor_data *= scale
```
- Simulates calibration drift
- Per-sequence scaling

#### 4. **Time Masking** (10% window)
```python
mask_len = int(seq_len * 0.1)
mask_start = random.randint(0, seq_len - mask_len)
sensor_data[mask_start:mask_start+mask_len] = 0
```
- Simulates data dropouts
- Random contiguous masking

#### 5. **Feature Dropout** (10%)
```python
drop_mask = torch.rand(num_features) > 0.1
sensor_data[:, ~drop_mask] = 0
```
- Drops entire feature channels
- Improves robustness

#### 6. **Class-Aware Oversampling** (3× for rare tokens)
```python
rare_threshold = 0.01  # 1% frequency
rare_tokens = [tok for tok, freq in token_freq.items() if freq < rare_threshold]
oversample_factor = 3
```
- Upsamples rare G-code tokens
- Addresses class imbalance

### Augmentation Configuration

```bash
# Enable augmentation with 30% probability
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_with_augmentation.py \
    --augmentation-prob 0.3 \
    --noise-std 0.01 \
    --time-warp-range 0.05 \
    --magnitude-scale-range 0.05 \
    --time-mask-ratio 0.1 \
    --feature-dropout 0.1 \
    --oversample-factor 3
```

---

## Monitoring & Debugging

### W&B Metrics

**Training Metrics**:
- `train/loss` - Combined multi-head loss
- `train/gcode_type_acc` - Token type accuracy (99%+)
- `train/gcode_cmd_acc` - Command accuracy (100%)
- `train/gcode_param_type_acc` - Parameter type (85%+)
- `train/gcode_param_val_acc` - Parameter value (60%+)
- `train/unique_tokens` - Unique predictions (target: >100/170)

**Validation Metrics**:
- `val/loss` - Validation loss
- `val/gcode_acc` - Overall validation accuracy
- `val/gcode_type_acc`, `val/gcode_cmd_acc`, etc.

**System Metrics**:
- `system/gpu_utilization` - GPU usage %
- `system/memory_allocated` - Memory usage
- `system/epoch_time` - Time per epoch

### Local Logging

```bash
# View training log
tail -f outputs/training/train_log.txt

# Example output:
# Epoch 10/50 | Train Loss: 0.234 | Val Loss: 0.189
# Val Accuracies: type=99.8%, cmd=100%, param_type=84.3%, param_val=56.2%
# Unique Tokens: 127/170 (74.7%)
```

### Debugging

#### Check Data Loading

```bash
PYTHONPATH=src .venv/bin/python -c "
from miracle.dataset.dataset import GCodeDataset
import numpy as np

train_data = np.load('outputs/processed_v2/train_sequences.npz')
print('Sensor data shape:', train_data['sensor_data'].shape)
print('G-code tokens shape:', train_data['gcode_tokens'].shape)
print('Decomposed targets shape:', train_data['decomposed_targets'].shape)
"
```

#### Test Model Forward Pass

```bash
PYTHONPATH=src .venv/bin/python -c "
import torch
from miracle.model.multihead_lm import MultiHeadGCodeLM

model = MultiHeadGCodeLM(vocab_size=170, d_model=128, nhead=4, num_layers=4)
sensor_emb = torch.randn(2, 64, 128)  # batch=2, seq=64, d_model=128

outputs = model(sensor_emb)
print('Type logits:', outputs['type'].shape)
print('Command logits:', outputs['command'].shape)
print('Param type logits:', outputs['param_type'].shape)
print('Param value logits:', outputs['param_val'].shape)
"
```

#### Monitor GPU/MPS

```bash
# Mac MPS
top -o cpu | grep python

# NVIDIA GPU
nvidia-smi -l 1
```

---

## Configuration

### Learning Rate Schedules

#### Cosine Annealing (Recommended)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
```

#### Reduce on Plateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

### Early Stopping

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Best Practices

### 1. **Start Small, Scale Up**
```bash
# Quick test run (10 epochs, small model)
--max-epochs 10 --d-model 64 --num-layers 2

# Full training (50 epochs, default model)
--max-epochs 50 --d-model 128 --num-layers 4

# Large model (100 epochs, large architecture)
--max-epochs 100 --d-model 512 --num-layers 6
```

### 2. **Use Validation Accuracy for Model Selection**
- Save checkpoints based on `val/gcode_acc`
- Avoid overfitting by monitoring val/train gap
- Target: val_acc within 2% of train_acc

### 3. **Class Imbalance Strategies**
- **Weighted Loss**: Scale loss by inverse class frequency
- **Focal Loss**: Focus on hard examples (γ=2.0-3.0)
- **Oversampling**: 3× for tokens with <1% frequency
- **Augmentation**: Augment rare token sequences more

### 4. **Reproducibility**
```python
# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Deterministic operations (slower)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 5. **Checkpoint Management**
```bash
# Save best checkpoint only (saves disk space)
--save-best-only

# Save every N epochs (for ablation studies)
--save-every 10
```

### 6. **Memory Optimization**
```bash
# Reduce batch size
--batch-size 16

# Use gradient accumulation
--accumulation-steps 2  # Effective batch size = 16 * 2 = 32

# Mixed precision training (FP16)
--use-amp
```

---

## Troubleshooting

### Training Loss Not Decreasing

**Possible Causes**:
1. Learning rate too high/low → Try lr=1e-4 to 1e-3
2. Gradient vanishing/exploding → Add gradient clipping
3. Bad initialization → Check model weights at epoch 0

### Validation Accuracy Plateaus

**Solutions**:
1. Add data augmentation
2. Increase model capacity (d_model, num_layers)
3. Reduce dropout (from 0.5 to 0.2)
4. Use learning rate warmup

### Out of Memory

**Solutions**:
1. Reduce `--batch-size` (32 → 16 → 8)
2. Reduce `--d-model` (256 → 128)
3. Use gradient checkpointing
4. Clear cache: `torch.cuda.empty_cache()`

### Poor Performance on Rare Tokens

**Solutions**:
1. Enable class-aware oversampling
2. Use focal loss with γ=3.0
3. Increase augmentation probability (0.5)
4. Train longer (100+ epochs)

---

## Performance Benchmarks

### Expected Results (50 epochs, default config)

| Metric | Epoch 10 | Epoch 30 | Epoch 50 |
|--------|----------|----------|----------|
| Train Loss | 0.45 | 0.18 | 0.12 |
| Val Loss | 0.38 | 0.21 | 0.19 |
| Token Type Acc | 95% | 99% | 99.8% |
| Command Acc | 98% | 100% | 100% |
| Param Type Acc | 70% | 82% | 84% |
| Param Value Acc | 40% | 54% | 58% |
| Unique Tokens | 80/170 | 115/170 | 130/170 |

### Training Time

| Hardware | Batch Size | Epoch Time | Total (50 epochs) |
|----------|------------|------------|-------------------|
| Mac M1 (8GB) | 16 | 2 min | 1.7 hours |
| Mac M2 (16GB) | 32 | 1.5 min | 1.25 hours |
| RTX 3090 | 64 | 45 sec | 38 minutes |
| CPU (8 cores) | 8 | 15 min | 12.5 hours |

---

## Next Steps

- **Pipeline Overview**: [PIPELINE.md](PIPELINE.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Visualization**: [VISUALIZATION.md](VISUALIZATION.md)
- **API Deployment**: [API.md](API.md)

---

**Questions?** Check the documentation or open an issue.
