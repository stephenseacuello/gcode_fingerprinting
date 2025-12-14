# Hyperparameter Sweep Guide

**Project**: G-code Fingerprinting with Machine Learning
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setting Up W&B](#setting-up-wb)
4. [Creating Sweep Configuration](#creating-sweep-configuration)
5. [Running Sweeps](#running-sweeps)
6. [Monitoring & Analysis](#monitoring--analysis)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Hyperparameter sweeps use **Weights & Biases (W&B)** to automatically search for optimal model configurations. This guide covers:

- **Bayesian optimization**: Intelligent search using Gaussian processes
- **Parallel execution**: Run multiple agents simultaneously
- **Automatic logging**: Track all experiments centrally
- **Visualization**: Compare runs with interactive dashboards

**Key Benefits:**
- Find optimal hyperparameters 10-100× faster than manual search
- Reproducible experiments with automatic versioning
- Visual comparison of 100+ runs simultaneously

---

## Prerequisites

### 1. Install Weights & Biases

```bash
.venv/bin/pip install wandb
```

### 2. Create W&B Account

1. Go to [https://wandb.ai/](https://wandb.ai/)
2. Sign up (free tier available)
3. Note your username (you'll need it for sweep commands)

### 3. Login to W&B

```bash
.venv/bin/wandb login
```

This will prompt you for an API key (found at [https://wandb.ai/authorize](https://wandb.ai/authorize)).

### 4. Verify Setup

```bash
.venv/bin/wandb whoami
```

You should see your username.

---

## Setting Up W&B

### Initialize W&B in Your Project

From the project root:

```bash
.venv/bin/wandb init
```

**Prompts:**
- **Project name**: `gcode-fingerprinting` (or your choice)
- **Entity** (username/team): Your W&B username
- **Enable anonymous logging**: No (recommended)

This creates a `wandb/` directory and `.wandb` settings file.

---

## Creating Sweep Configuration

### Sweep Configuration File

Create [configs/sweep_config.yaml](../configs/sweep_config.yaml):

```yaml
# Hyperparameter Sweep Configuration for G-code Fingerprinting
# Uses Bayesian optimization to find optimal model parameters

program: scripts/train_multihead.py
method: bayes

metric:
  name: val/gcode_acc
  goal: maximize

# Early termination to save compute
early_terminate:
  type: hyperband
  min_iter: 10
  s: 2

parameters:
  # Learning rate (log-uniform for wide search)
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01

  # Batch size (categorical - must fit in memory)
  batch_size:
    values: [8, 16, 32, 64]

  # Model architecture
  d_model:
    values: [64, 128, 256, 512]

  nhead:
    values: [2, 4, 8]

  num_layers:
    values: [2, 3, 4, 6]

  # Regularization
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5

  weight_decay:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001

  # Data augmentation
  augmentation_prob:
    values: [0.0, 0.2, 0.3, 0.5]

  # Training
  max_epochs:
    value: 50

  # Label smoothing
  label_smoothing:
    distribution: uniform
    min: 0.0
    max: 0.2

  # Gradient clipping
  grad_clip:
    values: [0.5, 1.0, 2.0, 5.0]

  # Optimizer
  optimizer:
    values: ['adam', 'adamw', 'sgd']

  # Command loss weight (important for hierarchical model)
  command_weight:
    distribution: uniform
    min: 1.0
    max: 5.0
```

### Sweep Configuration Explained

| Parameter | Range | Why |
|-----------|-------|-----|
| `learning_rate` | 0.0001 - 0.01 | Log-uniform for exponential search |
| `batch_size` | 8, 16, 32, 64 | Categorical - depends on memory |
| `d_model` | 64-512 | Model capacity (larger = more expressive) |
| `nhead` | 2-8 | Attention heads (must divide d_model) |
| `num_layers` | 2-6 | Transformer depth |
| `dropout` | 0.1-0.5 | Regularization strength |
| `weight_decay` | 1e-5 - 1e-3 | L2 regularization |
| `augmentation_prob` | 0-0.5 | Data augmentation probability |

---

## Running Sweeps

### Step 1: Create Sweep

```bash
.venv/bin/wandb sweep configs/sweep_config.yaml
```

**Output:**
```
wandb: Creating sweep from: configs/sweep_config.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

**Save the sweep ID** (`abc123xyz`) - you'll need it to run agents.

### Step 2: Run Single Agent

```bash
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

This starts a worker that:
1. Requests hyperparameters from W&B server
2. Trains model with those parameters
3. Logs results (loss, accuracy, etc.)
4. Repeats until sweep completes

### Step 3: Run Multiple Agents in Parallel

To speed up sweeps, run multiple agents simultaneously:

**Terminal 1:**
```bash
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

**Terminal 2:**
```bash
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

**Terminal 3:**
```bash
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

**Recommendations:**
- **Mac M1/M2**: 1-2 agents (memory-limited)
- **RTX 3090**: 2-4 agents (GPU-limited)
- **Multi-GPU cluster**: N agents (1 per GPU)

### Step 4: Stop Agents

Press `Ctrl+C` in each terminal to gracefully stop agents.

They will finish the current run before exiting.

---

## Monitoring & Analysis

### 1. Real-Time Dashboard

Open the sweep URL from Step 1:
```
https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/abc123xyz
```

**Dashboard Features:**
- **Parallel Coordinates Plot**: Visualize parameter relationships
- **Importance Plot**: Which hyperparameters matter most
- **Table View**: Sortable list of all runs
- **Metric Charts**: Training curves for all runs

### 2. View Best Run

Click "Best Run" in the dashboard to see:
- Final validation accuracy
- Hyperparameters used
- Training curves (loss, accuracy per head)
- System metrics (GPU utilization, memory)

### 3. Download Results

```bash
# Get best run's checkpoint
wandb artifact get YOUR_USERNAME/gcode-fingerprinting/run-abc123xyz-checkpoint:best

# Export sweep results to CSV
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep_results.py \
    --sweep-id abc123xyz \
    --output reports/sweep_results.csv
```

### 4. Compare Multiple Runs

In the W&B dashboard:
1. Select multiple runs (checkbox on left)
2. Click "Compare" button
3. View side-by-side comparison of:
   - Hyperparameters
   - Metrics (accuracy, loss)
   - Training curves
   - System metrics

---

## Best Practices

### 1. Start Small, Scale Up

**Phase 1: Quick Exploration (5-10 runs)**
```yaml
method: random  # Fast exploration
max_runs: 10    # Limit runs
parameters:
  learning_rate:
    values: [0.0001, 0.001, 0.01]  # Coarse grid
  d_model:
    values: [128, 256]  # 2 options only
```

**Phase 2: Focused Optimization (50-100 runs)**
```yaml
method: bayes  # Intelligent search
max_runs: 100
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0005  # Narrow range based on Phase 1
    max: 0.002
  d_model:
    values: [256, 512]  # Best from Phase 1
```

### 2. Use Early Termination

Save compute by stopping bad runs early:

```yaml
early_terminate:
  type: hyperband
  min_iter: 10  # Run at least 10 epochs
  s: 2          # Aggressiveness (higher = more aggressive)
```

**How it works:**
- Compares run's metric to top performers after 10 epochs
- Terminates if significantly worse
- Saves ~50% compute time on average

### 3. Prioritize Important Parameters

**High Impact** (optimize first):
- `learning_rate`: Biggest single factor
- `batch_size`: Affects training stability
- `d_model`: Model capacity
- `weight_decay`: Prevents overfitting

**Medium Impact**:
- `num_layers`: Depth vs speed tradeoff
- `dropout`: Regularization
- `command_weight`: Loss balancing

**Low Impact** (fix or omit):
- `optimizer`: Adam/AdamW usually best
- `grad_clip`: 1.0 is usually fine

### 4. Monitor Resource Usage

Track GPU/CPU utilization:
```bash
# Mac
top -o cpu | grep python

# NVIDIA GPU
nvidia-smi -l 1  # Update every 1 second
```

**Warning signs:**
- GPU utilization < 50%: Batch size too small
- Out of memory errors: Reduce batch size or d_model
- CPU bottleneck: Increase num_workers in DataLoader

### 5. Checkpoint Management

Large sweeps generate many checkpoints:

```bash
# Auto-cleanup in training script
--save-best-only  # Only save best checkpoint per run

# Manual cleanup
find outputs/wandb_sweeps -name "checkpoint_*.pt" -mtime +7 -delete
```

---

## Troubleshooting

### Issue 1: "wandb: ERROR Sweep agent failed"

**Cause**: Network error or invalid sweep ID

**Fix:**
```bash
# Check internet connection
ping wandb.ai

# Verify sweep ID
wandb sweep --list
```

### Issue 2: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Fix:**
Reduce batch size or model size:
```yaml
batch_size:
  values: [8, 16]  # Smaller batches

d_model:
  values: [64, 128]  # Smaller model
```

### Issue 3: No Improvement After Many Runs

**Possible causes:**
1. **Search space too wide**: Narrow ranges based on early runs
2. **Metric not improving**: Check for bugs in training loop
3. **Data issues**: Verify preprocessing is correct

**Fix:**
```bash
# Analyze completed runs
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep_results.py \
    --sweep-id abc123xyz \
    --plot-importance \
    --output reports/
```

### Issue 4: Agents Not Starting

**Symptoms**: `wandb agent` command hangs

**Fix:**
```bash
# Clear cache
rm -rf ~/.cache/wandb

# Reinstall
.venv/bin/pip uninstall wandb -y
.venv/bin/pip install wandb

# Login again
.venv/bin/wandb login
```

### Issue 5: Duplicate Runs

**Cause**: Multiple agents picked same hyperparameters

**Fix:**
Increase search space or use grid method for small spaces:
```yaml
method: grid  # Exhaustive search (no duplicates)
```

---

## Advanced Sweep Strategies

### 1. Grid Search (Exhaustive)

Tests all combinations - good for small spaces:

```yaml
method: grid
parameters:
  learning_rate: {values: [0.0001, 0.001, 0.01]}
  batch_size: {values: [16, 32]}
  # Total runs: 3 × 2 = 6
```

### 2. Random Search (Exploration)

Random sampling - good for large spaces:

```yaml
method: random
max_runs: 50
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
```

### 3. Bayesian Optimization (Recommended)

Intelligent search using past results:

```yaml
method: bayes
max_runs: 100  # More runs = better optimization
parameters:
  # Same as random, but smarter sampling
```

**When to use:**
- **Grid**: ≤10 total combinations
- **Random**: Initial exploration, very large spaces
- **Bayes**: Focused optimization (most cases)

### 4. Multi-Objective Optimization

Optimize multiple metrics simultaneously:

```yaml
metric:
  name: combined_score
  goal: maximize

# In training script, log:
# wandb.log({"combined_score": 0.7*val_acc + 0.3*(1-val_loss)})
```

---

## Example Workflow

### Complete Sweep Pipeline

```bash
# 1. Create sweep
.venv/bin/wandb sweep configs/sweep_config.yaml
# Output: Sweep ID = abc123xyz

# 2. Start 3 parallel agents (in separate terminals)
.venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz

# 3. Monitor progress (while agents run)
# Open: https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/abc123xyz

# 4. After completion, analyze results
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep_results.py \
    --sweep-id abc123xyz \
    --output reports/sweep_analysis.csv

# 5. Download best checkpoint
BEST_RUN_ID="run_abc123_best"  # From W&B dashboard
wandb artifact get YOUR_USERNAME/gcode-fingerprinting/$BEST_RUN_ID:best

# 6. Test best model
PYTHONPATH=src .venv/bin/python scripts/test_local_checkpoint.py \
    --checkpoint outputs/wandb_sweeps/$BEST_RUN_ID/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz

# 7. Generate visualizations with best model
.venv/bin/python scripts/generate_visualizations.py \
    --all \
    --use-real-data \
    --checkpoint-path outputs/wandb_sweeps/$BEST_RUN_ID/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz \
    --output outputs/figures/best_model/
```

---

## Cost Estimation

### Compute Time

| Hardware | Runs/Day | 100-Run Sweep |
|----------|----------|---------------|
| Mac M1 (1 agent) | ~12 | 8-9 days |
| Mac M2 (2 agents) | ~24 | 4-5 days |
| RTX 3090 (4 agents) | ~96 | ~24 hours |
| Cloud (10× 3090) | ~960 | ~2.5 hours |

**Assumptions**: 50 epochs, ~2 hours per run

### W&B Costs

- **Free tier**: Unlimited projects, 100 GB storage
- **Team tier** ($35/user/month): More storage, private projects
- **Enterprise**: Custom pricing

**For academic use**: Free tier is usually sufficient

---

## Next Steps

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Training Details**: [TRAINING.md](TRAINING.md)
- **Visualization**: [VISUALIZATION.md](VISUALIZATION.md)
- **API Deployment**: [API.md](API.md)

---

**Questions?** Check the [W&B documentation](https://docs.wandb.ai/) or open an issue.
