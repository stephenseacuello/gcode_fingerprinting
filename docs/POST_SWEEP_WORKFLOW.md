# Post-Sweep Workflow Guide

This guide explains how to analyze W&B sweep results, deploy the best checkpoint, and test it via the API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Step-by-Step Guide](#step-by-step-guide)
3. [API Checkpoint Loading](#api-checkpoint-loading)
4. [Testing Inference](#testing-inference)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Automated Workflow (Recommended)

```bash
# Run complete post-sweep workflow
./scripts/post_sweep_workflow.sh \
    --sweep-id njo48wle \
    --entity seacuello-university-of-rhode-island \
    --project uncategorized \
    --deploy
```

This single command will:
1. Analyze sweep results
2. Get best checkpoint from W&B
3. Evaluate on test set
4. Deploy to production (if `--deploy` flag is set)
5. Generate comparison report

---

## Step-by-Step Guide

### Step 1: Analyze Sweep Results

After your W&B sweep completes (20 runs), analyze the results:

```bash
.venv/bin/python scripts/analyze_sweep.py \
    --sweep-id njo48wle \
    --entity seacuello-university-of-rhode-island \
    --project uncategorized \
    --output outputs/sweep_analysis
```

**Output**:
- `outputs/sweep_analysis/sweep_results.csv` - All runs ranked by performance
- `outputs/sweep_analysis/parameter_importance.png` - Which hyperparameters matter most
- `outputs/sweep_analysis/parallel_coordinates.png` - Top 20 run configurations
- `outputs/sweep_analysis/metric_distributions.png` - Performance distributions
- `outputs/sweep_analysis/summary_report.txt` - Text summary

**Example Output**:
```
================================================================================
SWEEP SUMMARY REPORT
================================================================================

Total Runs: 20
Completed Runs: 18
Failed Runs: 2

val/overall_acc Statistics:
  Mean: 0.8452
  Std:  0.0342
  Min:  0.7811
  Max:  0.9123

Best Run: swept-rain-42
  val/overall_acc: 0.9123
  Config:
    learning_rate: 0.0008534
    batch_size: 16
    hidden_dim: 256
    num_heads: 8
    num_layers: 3
```

### Step 2: Get Best Checkpoint from W&B

Retrieve the best checkpoint:

```bash
.venv/bin/python scripts/get_best_checkpoint_from_sweep.py \
    --sweep-id njo48wle \
    --entity seacuello-university-of-rhode-island \
    --project uncategorized \
    --output-dir outputs/best_from_sweep
```

**Output**:
- `outputs/best_from_sweep/best_config.json` - Best hyperparameters
- `outputs/best_from_sweep/all_runs.csv` - All runs with metrics
- `outputs/best_from_sweep/checkpoint_best.pt` - Best checkpoint (if found locally)

**If Checkpoint Not Found Locally**:

The script will provide a W&B URL to download the checkpoint manually:

```
⚠️  Checkpoint not found locally for run abc123de
    To download from W&B, visit:
    https://wandb.ai/your-entity/your-project/runs/abc123de/files
```

1. Visit the URL
2. Go to the "Files" tab
3. Download the checkpoint file
4. Copy it to `outputs/best_from_sweep/checkpoint_best.pt`

### Step 3: Evaluate on Test Set

Run comprehensive evaluation:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
    --checkpoint outputs/best_from_sweep/checkpoint_best.pt \
    --test-data outputs/processed_quick/test_sequences.npz \
    --vocab-path data/vocabulary.json \
    --output outputs/evaluation_best
```

**Output**:
```
================================================================================
EVALUATION RESULTS
================================================================================
Test Samples: 500
Command Accuracy: 0.9234 (92.34%)
Parameter Type Accuracy: 0.8876 (88.76%)
Parameter Value Accuracy: 0.8543 (85.43%)
Overall Token Accuracy: 0.8123 (81.23%)
================================================================================
```

**Files Created**:
- `outputs/evaluation_best/evaluation_results.json` - Detailed metrics
- `outputs/evaluation_best/evaluation_results.csv` - CSV format for comparison

### Step 4: Deploy Checkpoint

Deploy the checkpoint to the API's production location:

```bash
.venv/bin/python scripts/deploy_checkpoint.py \
    --source outputs/best_from_sweep/checkpoint_best.pt \
    --target outputs/production/checkpoint_best.pt
```

**Output**:
```
================================================================================
CHECKPOINT DEPLOYMENT
================================================================================

Validating checkpoint: outputs/best_from_sweep/checkpoint_best.pt
✓ Checkpoint is valid
  Epoch: 18
  Val Accuracy: 0.9123
  Hidden Dim: 256
  Num Layers: 3

Creating backup: outputs/production/checkpoint_best_backup.pt
Copying checkpoint:
  From: outputs/best_from_sweep/checkpoint_best.pt
  To:   outputs/production/checkpoint_best.pt
✓ Checkpoint deployed successfully
```

### Step 5: Compare Checkpoints (Optional)

Compare multiple checkpoints side-by-side:

```bash
.venv/bin/python scripts/compare_checkpoints.py \
    --checkpoints \
        outputs/training_10epoch/checkpoint_best.pt \
        outputs/training_50epoch/checkpoint_best.pt \
        outputs/best_from_sweep/checkpoint_best.pt \
    --test-data outputs/processed_quick/test_sequences.npz \
    --vocab-path data/vocabulary.json \
    --output outputs/checkpoint_comparison
```

**Output**:
- `outputs/checkpoint_comparison/checkpoint_comparison.csv` - Metrics table
- `outputs/checkpoint_comparison/checkpoint_comparison.png` - Bar chart
- `outputs/checkpoint_comparison/comparison_report.md` - Markdown report

**Example Table**:
```
| Rank | Checkpoint | Command Acc | Param Type Acc | Param Value Acc | Overall Acc |
|------|-----------|-------------|----------------|-----------------|-------------|
| 1 | checkpoint_best.pt (sweep) | 0.9234 | 0.8876 | 0.8543 | 0.8123 |
| 2 | checkpoint_best.pt (50ep)  | 0.9102 | 0.8654 | 0.8321 | 0.7987 |
| 3 | checkpoint_best.pt (10ep)  | 0.8765 | 0.8432 | 0.8102 | 0.7654 |
```

---

## API Checkpoint Loading

### Method 1: Dynamic Loading via API Endpoint

Load a checkpoint without restarting the server:

```bash
curl -X POST http://localhost:8000/load_checkpoint \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "outputs/best_from_sweep/checkpoint_best.pt",
    "vocab_path": "data/vocabulary.json",
    "device": "cpu"
  }'
```

**Response**:
```json
{
  "status": "success",
  "checkpoint_path": "outputs/best_from_sweep/checkpoint_best.pt",
  "model_version": "multihead_sweep_v1",
  "vocab_size": 170,
  "d_model": 256,
  "load_time_ms": 542.3
}
```

### Method 2: Update Default Checkpoint

Edit [src/miracle/api/server.py](../src/miracle/api/server.py#L63-L64):

```python
# Change these lines
default_checkpoint = "outputs/best_from_sweep/checkpoint_best.pt"
default_vocab = "data/vocabulary.json"
```

Then restart the API server:

```bash
# If running in background
pkill -f api_server.py

# Start server
PYTHONPATH=src .venv/bin/python src/miracle/api/server.py
```

---

## Testing Inference

### Test via API

1. **Create test payload** (`test_payload.json`):

```json
{
  "sensor_data": {
    "continuous": [[0.1, 0.2, ...], ...],
    "categorical": [[0, 1, 2, 3], ...]
  },
  "inference_config": {
    "method": "greedy",
    "max_length": 64
  }
}
```

2. **Make prediction request**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

3. **Response**:

```json
{
  "gcode_sequence": ["G0", "X10.5", "Y20.3", "Z5.0", "G1", "F1500"],
  "inference_time_ms": 45.2,
  "model_version": "multihead_sweep_v1"
}
```

### Test via Python

```python
import requests
import numpy as np

# Generate random sensor data (for testing)
sensor_data = {
    "continuous": np.random.randn(64, 135).tolist(),
    "categorical": np.random.randint(0, 4, (64, 4)).tolist()
}

# Make request
response = requests.post(
    "http://localhost:8000/predict",
    json={"sensor_data": sensor_data}
)

result = response.json()
print("Predicted G-code:", result['gcode_sequence'])
print("Inference time:", result['inference_time_ms'], "ms")
```

### Check Model Info

```bash
curl http://localhost:8000/info
```

**Response**:
```json
{
  "model_name": "MM-DTAE-LSTM-MultiHead",
  "model_version": "multihead_sweep_v1",
  "vocab_size": 170,
  "d_model": 256,
  "num_parameters": 3245672,
  "supported_endpoints": ["/predict", "/batch_predict", "/fingerprint", "/health", "/info", "/load_checkpoint"],
  "supported_generation_methods": ["greedy", "beam_search", "temperature", "top_k", "nucleus"]
}
```

---

## Troubleshooting

### Issue: Sweep Analysis Shows No Completed Runs

**Symptoms**:
```
⚠️ Warning: No completed runs found
Run states: {'running': 10, 'failed': 3}
```

**Solution**: Wait for at least one run to complete. Check W&B dashboard to monitor progress.

### Issue: Checkpoint Not Found Locally

**Symptoms**:
```
⚠️ Checkpoint not found locally for run abc123de
```

**Solution**:
1. Visit the W&B run URL provided in the output
2. Navigate to "Files" tab
3. Download `checkpoint.pt` or similar file
4. Copy to the expected location: `outputs/best_from_sweep/checkpoint_best.pt`

### Issue: API Fails to Load Checkpoint

**Symptoms**:
```
Failed to load checkpoint: Checkpoint missing required keys: ['backbone_state_dict']
```

**Solution**: Ensure the checkpoint is from the multi-head training script (`train_multihead.py`), not the old single-head script.

### Issue: Import Error When Running Scripts

**Symptoms**:
```
ModuleNotFoundError: No module named 'miracle'
```

**Solution**: Make sure to set `PYTHONPATH`:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py ...
```

Or use the full path as scripts do internally.

### Issue: Evaluation Accuracy Much Lower Than Expected

**Possible Causes**:
1. **Wrong test set**: Make sure you're using the correct test data file
2. **Vocabulary mismatch**: Ensure vocab file matches the one used during training
3. **Device mismatch**: Some operations behave differently on different devices (CPU vs GPU)

**Solution**:
```bash
# Verify vocabulary
diff data/vocabulary.json data/vocabulary_backup.json

# Check test data shape
python -c "import numpy as np; data=np.load('outputs/processed_quick/test_sequences.npz'); print(data.files, len(data['sequences']))"
```

### Issue: API Server Won't Start

**Symptoms**:
```
Address already in use: ('0.0.0.0', 8000)
```

**Solution**: Kill existing server:

```bash
lsof -ti:8000 | xargs kill -9
```

Or use a different port:

```python
# In server.py
uvicorn.run("server:app", host="0.0.0.0", port=8001)
```

---

## Quick Reference Commands

```bash
# Complete automated workflow
./scripts/post_sweep_workflow.sh --sweep-id SWEEP_ID --entity ENTITY --deploy

# Analyze sweep
.venv/bin/python scripts/analyze_sweep.py --sweep-id SWEEP_ID --entity ENTITY

# Get best checkpoint
.venv/bin/python scripts/get_best_checkpoint_from_sweep.py --sweep-id SWEEP_ID --entity ENTITY

# Evaluate checkpoint
.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint PATH --test-data PATH --vocab-path PATH

# Deploy checkpoint
.venv/bin/python scripts/deploy_checkpoint.py --source PATH --target PATH

# Compare checkpoints
.venv/bin/python scripts/compare_checkpoints.py --checkpoints PATH1 PATH2 PATH3 --test-data PATH --vocab-path PATH

# Load checkpoint via API
curl -X POST http://localhost:8000/load_checkpoint -H "Content-Type: application/json" -d '{"checkpoint_path": "PATH"}'

# Test prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_payload.json
```

---

For more information, see:
- [Training Guide](TRAINING.md)
- [API Reference](API.md)
- [Hyperparameter Sweeps Guide](SWEEPS_GUIDE.md)
