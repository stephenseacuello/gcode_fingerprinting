# Hyperparameter Sweep Quick Start Guide

Complete guide to running W&B hyperparameter sweeps and generating visualization figures.

---

## Prerequisites

Your W&B sweep is ready to run with:
- **Config**: `configs/sweep_config.yaml` (Bayesian optimization)
- **Analysis Script**: `scripts/analyze_sweep.py` (generates figures)
- **Sweep Parameters**: learning_rate, batch_size, hidden_dim, num_heads, num_layers, weight_decay, grad_clip, command_weight

---

## Step 1: Authenticate with W&B

First-time setup (one time only):

```bash
# Option 1: Login via CLI
.venv/bin/wandb login

# This will prompt you for an API key
# Get it from: https://wandb.ai/authorize
```

**Verify authentication:**
```bash
.venv/bin/wandb status
# Should show "api_key": "<your-key>" (not null)
```

---

## Step 2: Create the Sweep

```bash
# Create sweep and save the ID
.venv/bin/wandb sweep configs/sweep_config.yaml
```

**Output will look like:**
```
wandb: Creating sweep from: configs/sweep_config.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz
```

**IMPORTANT**: Save the sweep ID (`abc123xyz` in example above)!

---

## Step 3: Run Sweep Agents

### Option A: Single Agent (Recommended for Mac)

```bash
# Copy the exact command from Step 2 output:
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/SWEEP_ID
```

**Example:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent stepheneacuello/gcode-fingerprinting/abc123xyz
```

This will:
1. Request hyperparameters from W&B
2. Train a model with those parameters
3. Log results (accuracy, loss, etc.)
4. Repeat until sweep completes (or you press Ctrl+C)

### Option B: Multiple Agents (Faster, use multiple terminals)

**Terminal 1:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/SWEEP_ID
```

**Terminal 2:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/SWEEP_ID
```

**Terminal 3:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/SWEEP_ID
```

**Mac M1/M2**: Run 1-2 agents (memory limited)
**Mac M3**: Run 2-3 agents
**NVIDIA GPU**: Run 2-4 agents

---

## Step 4: Monitor Progress

**Open the sweep dashboard:**
```
https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/SWEEP_ID
```

**Dashboard shows:**
- Parallel coordinates plot (parameter relationships)
- Parameter importance ranking
- Best run so far
- All completed runs in a table

**Stop agents anytime with Ctrl+C** (they'll finish the current run gracefully)

---

## Step 5: Analyze Results & Generate Figures

After completing at least 10-20 runs:

```bash
# Analyze sweep and generate visualizations
.venv/bin/python scripts/analyze_sweep.py \
    --sweep-id SWEEP_ID \
    --entity YOUR_USERNAME \
    --output outputs/sweep_analysis/
```

**This generates:**
1. `sweep_results.csv` - All runs with hyperparameters and metrics
2. `parameter_importance.png` - Which hyperparameters matter most
3. `parallel_coordinates.png` - Top 20 runs visualization
4. `metric_distributions.png` - Histograms of all metrics
5. `best_vs_worst.png` - Comparison chart
6. `summary_report.txt` - Text summary with statistics

**Example output:**
```
✓ Fetching sweep results...
✓ Found 25 runs (20 completed, 5 failed)
✓ Saved: outputs/sweep_analysis/sweep_results.csv
✓ Saved: outputs/sweep_analysis/parameter_importance.png
✓ Saved: outputs/sweep_analysis/parallel_coordinates.png
✓ Saved: outputs/sweep_analysis/metric_distributions.png
✓ Saved: outputs/sweep_analysis/best_vs_worst.png
✓ Saved: outputs/sweep_analysis/summary_report.txt

================================================================================
SWEEP SUMMARY REPORT
================================================================================
Total Runs: 25
Completed Runs: 20
Failed Runs: 5

val/overall_acc Statistics:
  Mean: 0.4823
  Std:  0.0451
  Min:  0.3912
  Max:  0.5634

Best Run: noble-sweep-17
  val/overall_acc: 0.5634
  Config:
    learning_rate: 0.0008
    batch_size: 32
    hidden_dim: 256
    num_heads: 4
    num_layers: 3

================================================================================
```

---

## Complete Example Workflow

Here's the complete command sequence:

```bash
# 1. Authenticate (one time)
.venv/bin/wandb login
# Paste your API key from https://wandb.ai/authorize

# 2. Create sweep
.venv/bin/wandb sweep configs/sweep_config.yaml
# Output: sweep ID = abc123xyz

# 3. Run 2 agents in parallel (open 2 terminals)
# Terminal 1:
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz

# Terminal 2:
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_USERNAME/gcode-fingerprinting/abc123xyz

# 4. Monitor at: https://wandb.ai/YOUR_USERNAME/gcode-fingerprinting/sweeps/abc123xyz

# 5. After ~20 runs, press Ctrl+C in both terminals

# 6. Analyze results
.venv/bin/python scripts/analyze_sweep.py \
    --sweep-id abc123xyz \
    --entity YOUR_USERNAME \
    --output outputs/sweep_analysis/

# 7. View figures in outputs/sweep_analysis/
```

---

## Sweep Configuration Details

The sweep is configured to optimize `val/overall_acc` using:

**Search Method**: Bayesian optimization (smart parameter selection)

**Early Termination**: Hyperband (stops bad runs early to save time)
- Minimum 5 epochs before stopping
- Aggressiveness level: 2

**Parameters Being Tuned:**

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| `learning_rate` | 0.0001 - 0.01 | log-uniform |
| `batch_size` | [8, 16, 32] | categorical |
| `hidden_dim` | [64, 128, 256] | categorical |
| `num_heads` | [2, 4, 8] | categorical |
| `num_layers` | [2, 3, 4] | categorical |
| `weight_decay` | 0.00001 - 0.01 | log-uniform |
| `grad_clip` | [0.5, 1.0, 2.0] | categorical |
| `command_weight` | 1.0 - 5.0 | uniform |

**Fixed Parameters:**
- `max_epochs`: 50
- `data_dir`: outputs/processed_quick
- `vocab_path`: data/vocabulary.json

---

## Troubleshooting

### Issue: "api_key": null

**Fix:**
```bash
.venv/bin/wandb login
# Paste API key from https://wandb.ai/authorize
```

### Issue: Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory` or system freezes

**Fix:** Edit `configs/sweep_config.yaml`:
```yaml
batch_size:
  values: [8, 16]  # Remove 32, 64

hidden_dim:
  values: [64, 128]  # Remove 256, 512
```

### Issue: Agents not progressing

**Symptoms:** Agents hang or show no output

**Fix:**
```bash
# Stop all agents (Ctrl+C)
# Clear cache
rm -rf ~/.cache/wandb

# Try again
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/wandb agent YOUR_SWEEP_PATH
```

### Issue: Training script errors

**Common errors and fixes:**

1. **Module not found**: Add `PYTHONPATH=src` before command
2. **Data not found**: Make sure `outputs/processed_quick/` exists
3. **Vocab not found**: Check `data/vocabulary.json` exists

---

## Tips for Best Results

### 1. Start Small
Run 10-20 runs first to understand the search space:
```bash
# Let agents run until ~20 runs complete, then Ctrl+C
```

### 2. Analyze Early
Check parameter importance after 20 runs:
```bash
.venv/bin/python scripts/analyze_sweep.py --sweep-id SWEEP_ID --output outputs/sweep_analysis/
```

### 3. Adjust Search Space
If parameter_importance.png shows some parameters don't matter, fix them:
```yaml
# In configs/sweep_config.yaml
# Change this:
num_heads:
  values: [2, 4, 8]

# To this (if num_heads=4 always works best):
num_heads:
  value: 4  # Fixed
```

### 4. Run More Agents for Faster Results
**Expected time for 50 runs:**
- 1 agent: ~100 hours (4+ days)
- 2 agents: ~50 hours (2 days)
- 3 agents: ~33 hours (1.5 days)

### 5. Monitor GPU/CPU Usage
```bash
# Mac - check CPU
top -o cpu | grep python

# NVIDIA GPU
nvidia-smi -l 1  # Update every 1 second
```

---

## Next Steps After Sweep

### 1. Get Best Model Checkpoint

Best checkpoint is automatically saved in:
```
outputs/training_50epoch/checkpoint_best.pt
```

Or download from W&B:
```bash
# From sweep dashboard, click best run
# Copy run ID (e.g., "noble-sweep-17")

wandb artifact get YOUR_USERNAME/gcode-fingerprinting/run-BEST_RUN_ID-checkpoint:best
```

### 2. Test Best Model

```bash
PYTHONPATH=src .venv/bin/python scripts/test_local_checkpoint.py \
    --checkpoint outputs/training_50epoch/checkpoint_best.pt \
    --test-data outputs/processed_quick/test_sequences.npz
```

### 3. Generate Visualizations with Best Model

```bash
.venv/bin/python scripts/generate_visualizations.py \
    --all \
    --use-real-data \
    --checkpoint-path outputs/training_50epoch/checkpoint_best.pt \
    --test-data outputs/processed_quick/test_sequences.npz \
    --vocab-path data/vocabulary.json \
    --output outputs/figures/best_model/
```

### 4. Deploy Best Model to API

Update the API server to use the best checkpoint:
```bash
# Edit src/miracle/api/server.py line 63
# Change to best checkpoint path

# Start API
PYTHONPATH=src .venv/bin/python scripts/api_server.py
```

---

## Additional Resources

- **W&B Sweeps Documentation**: https://docs.wandb.ai/guides/sweeps
- **Bayesian Optimization Explained**: https://docs.wandb.ai/guides/sweeps/sweep-config-keys
- **Project SWEEPS_GUIDE.md**: [docs/SWEEPS_GUIDE.md](docs/SWEEPS_GUIDE.md)
- **Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)

---

**Questions?** Check the documentation or W&B dashboard for detailed metrics and logs.

**Ready to start?** Jump to [Step 1: Authenticate with W&B](#step-1-authenticate-with-wb)
