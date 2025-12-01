># Hyperparameter Tuning Guide

This guide describes the comprehensive hyperparameter optimization strategy for the G-code fingerprinting project.

## Overview

We use **Weights & Biases (W&B) Sweeps** for systematic hyperparameter search with:
- **5 major sweeps** covering vocabulary, augmentation, warmup, architecture, and loss weighting
- **300+ total experiments** expected
- **Bayesian optimization** for efficient search
- **Early termination** with Hyperband to save compute

---

## Sweep Configurations

### 1. Vocabulary Bucketing Sweep
**File:** `sweeps/phase3/vocabulary_bucketing.yaml`
**Method:** Grid search
**Runs:** ~25

**Parameters:**
- `vocab_bucket_digits`: [null, 1, 2, 3]
  - `null`: No bucketing (full precision, ~668 tokens)
  - `1`: 1-digit (1575 → 1, very coarse)
  - `2`: 2-digit (1575 → 15, current v2, ~170 tokens)
  - `3`: 3-digit (1575 → 157, finer, ~350 tokens)

**Goal:** Find optimal vocabulary size/precision tradeoff for >70% accuracy

**Expected Outcome:**
- 3-digit bucketing likely best (balances precision vs size)
- Parameter value accuracy should improve 10-15%

---

### 2. Augmentation Optimization Sweep
**File:** `sweeps/phase3/augmentation_optimization.yaml`
**Method:** Bayesian optimization
**Runs:** ~50

**Parameters:**
- Oversampling factor: [1, 2, 3, 4, 5]
- Noise std: [0.005 - 0.05]
- Noise/shift/scale probabilities: [0.3, 0.5, 0.7, 1.0]
- Temporal shift max: [0, 1, 2, 3, 5]
- Magnitude scale range
- Mixup alpha: [0.0 - 0.8]

**Goal:** Maximize command accuracy (most critical metric)

**Expected Insights:**
- Optimal oversampling: 3-4x
- Noise level that helps generalization: ~0.02
- Best augmentation combinations

---

### 3. Warmup Scheduler Sweep
**File:** `sweeps/phase3/warmup_optimization.yaml`
**Method:** Grid search
**Runs:** ~30

**Parameters:**
- Warmup epochs: [0, 3, 5, 10, 15, 20]
- Warmup type: [linear, exponential, cosine]
- Base scheduler: [cosine, onecycle, plateau, step]
- Peak learning rate: [0.0001 - 0.01]

**Goal:** Find scheduler configuration for fastest convergence

**Hypothesis:**
- 5-10 epochs warmup optimal
- Linear warmup most effective
- Cosine annealing after warmup works well

---

### 4. Architecture Sweep
**File:** `sweeps/phase3/architecture_sweep.yaml`
**Method:** Bayesian optimization
**Runs:** ~60-100

**Parameters:**
- Model capacity: d_model [96-256], layers [1-4], heads [4-8]
- Regularization: dropout [0.0-0.3], weight_decay [0.0-0.1]
- Training: batch_size, learning_rate, optimizer
- Scheduler types

**Goal:**
- Best architecture for accuracy
- Pareto frontier (accuracy vs efficiency)
- Parameter sensitivity analysis

**Key Questions:**
- Does d_model=256 improve accuracy significantly?
- Optimal depth (encoder vs decoder layers)?
- Best optimizer (AdamW vs Adam vs SGD)?

---

### 5. Loss Weighting Sweep
**File:** `sweeps/phase3/loss_weighting.yaml`
**Method:** Bayesian optimization
**Runs:** ~40

**Parameters:**
- Multi-head weights: type_gate, command, param_type, param_value
- Auxiliary weights: reconstruction, fingerprint, contrastive
- Loss types: cross_entropy, focal, label_smoothing
- Focal gamma, label smoothing epsilon

**Goal:**
- Maintain 100% command accuracy
- Improve parameter value accuracy >70%
- Understand loss weight sensitivity

**Expected Finding:**
- Commands need 3-5x weight
- Reconstruction helps (weight ~0.5)
- Focal loss may help rare tokens

---

## Running Sweeps

### Using the Sweep Runner

```bash
# Single sweep
./scripts/run_sweeps.sh --sweep vocabulary

# With multiple parallel agents
./scripts/run_sweeps.sh --sweep architecture --agents 4

# All sweeps sequentially
./scripts/run_sweeps.sh --sweep all --agents 2

# Dry run (see what would execute)
./scripts/run_sweeps.sh --sweep vocabulary --dry-run
```

### Manual Sweep Execution

```bash
# 1. Create sweep
wandb sweep sweeps/phase3/vocabulary_bucketing.yaml

# 2. Run agent(s)
wandb agent <sweep_id>

# 3. Run multiple agents in parallel
wandb agent <sweep_id> &
wandb agent <sweep_id> &
wandb agent <sweep_id> &
wait
```

---

## Analyzing Sweep Results

### View Results in W&B Dashboard

1. Go to https://wandb.ai/your-entity/gcode-fingerprinting
2. Click "Sweeps" tab
3. Select your sweep
4. View:
   - Parallel coordinates plot
   - Parameter importance
   - Best runs
   - Training curves

### Extract Best Configurations

```python
import wandb

api = wandb.Api()
sweep = api.sweep("your-entity/gcode-fingerprinting/sweep_id")

# Get top 10 runs
best_runs = sorted(sweep.runs, key=lambda r: r.summary.get('val/overall_accuracy', 0), reverse=True)[:10]

for run in best_runs:
    print(f"{run.name}: {run.summary['val/overall_accuracy']:.2f}%")
    print(f"  Config: {run.config}")
```

### Use Analysis Script

```bash
# Analyze all sweeps
python scripts/analyze_all_sweeps.py --sweep-ids <id1>,<id2>,<id3>

# Generate report
python scripts/analyze_all_sweeps.py --output results/sweep_summary.pdf
```

---

## Sweep Timeline

### Week 1: Foundation Sweeps
- **Day 1-2:** Vocabulary bucketing (25 runs)
- **Day 3-4:** Augmentation optimization (50 runs)
- **Day 5:** Warmup scheduler (30 runs)

### Week 2: Architecture & Training
- **Day 1-3:** Architecture sweep (60-100 runs)
- **Day 4-5:** Loss weighting (40 runs)

### Week 3: Inference & Ensemble
- **Day 1-2:** Inference parameter sweep (50 runs)
- **Day 3-4:** Multi-head temperature tuning (40 runs)
- **Day 5:** Build ensemble from top-10 models

**Total Expected:** ~300 runs, ~120-160 GPU/MPS hours

---

## Best Practices

### Before Starting Sweeps
1. **Login to W&B:** `wandb login`
2. **Test single run:** Ensure training script works
3. **Check data:** Verify preprocessed data exists
4. **Set project name:** Update configs if needed

### During Sweeps
1. **Monitor progress:** Check W&B dashboard regularly
2. **Early stopping:** Let Hyperband terminate poor runs
3. **Resource management:** Don't run too many agents simultaneously
4. **Save checkpoints:** Best models saved automatically

### After Sweeps
1. **Analyze results:** Use parallel coordinates, parameter importance
2. **Validate best:** Re-train top configs to verify
3. **Document findings:** Update results summary
4. **Create ensemble:** Combine top models

---

## Expected Results

### Vocabulary Sweep
- **Baseline (2-digit):** 58.5% overall, 100% command
- **Expected (3-digit):** 65-70% overall, 100% command
- **Improvement:** +6-12% overall accuracy

### Augmentation Sweep
- **Insight:** Optimal augmentation strategy
- **Impact:** +2-5% command accuracy
- **Benefit:** Better generalization

### Architecture Sweep
- **Find:** Best model size (likely d_model=192 or 256)
- **Impact:** +3-7% overall accuracy
- **Tradeoff:** 2-3x more parameters

### Combined Optimizations
- **Target:** >70% overall accuracy
- **Stretch:** >75% overall accuracy
- **Maintain:** 100% command accuracy

---

## Troubleshooting

### Sweep Not Starting
```bash
# Check W&B login
wandb whoami

# Verify config file
cat sweeps/phase3/vocabulary_bucketing.yaml

# Test training script
python train_multihead.py --config configs/phase1_best.json --epochs 1
```

### Poor Results
- Check data preprocessing
- Verify vocabulary version (v2 vs v3)
- Ensure augmentation is enabled
- Review loss weights

### Out of Memory
- Reduce batch_size
- Use smaller d_model
- Enable gradient checkpointing
- Use CPU if MPS fails

---

## Advanced Sweeps (Optional)

### Inference Parameter Sweep
- Temperature: [0.1 - 2.0]
- Beam width: [1, 3, 5, 10, 20]
- Top-k: [10, 50, 100, 200]
- Top-p: [0.7 - 0.99]
- Per-head temperatures

### Ensemble Optimization
- Select top-K models (K=5, 10, 20)
- Optimize ensemble weights
- Compare: uniform vs learned weights

---

## Resources

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Hyperband Paper](https://arxiv.org/abs/1603.06560)
- [Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/)

---

**Last Updated:** November 19, 2025
**Status:** Ready for execution
