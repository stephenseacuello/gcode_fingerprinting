# Mode Collapse Prevention Guide

## Summary

This document explains the mode collapse prevention features added to prevent training failures like the one experienced with `full_featured_FIXED_3`, where the model only predicted "G0" for all inputs.

## What is Mode Collapse?

Mode collapse occurs when a neural network learns to always predict the most common class, achieving high accuracy on that class but failing to learn diverse patterns. Symptoms include:
- High command accuracy (100%) but low overall accuracy (0%)
- Dashboard showing identical predictions for all inputs
- Low prediction diversity and entropy
- Model only predicting class 0 (or most frequent class)

## New Features

### 1. Mode Collapse Detection Module

**Location**: [`src/miracle/training/mode_collapse_prevention.py`](../src/miracle/training/mode_collapse_prevention.py)

**Features**:
- **Class Frequency Analysis**: Compute distribution of commands, param types, and param values
- **Class Weight Computation**: Generate inverse frequency weights to balance rare classes
- **Prediction Diversity Metrics**: Track entropy, unique predictions, and per-class accuracy
- **Focal Loss**: Alternative loss function that focuses on hard examples
- **Mode Collapse Warnings**: Automatic detection and warning system

### 2. Class Frequency Analysis Script

**Location**: [`scripts/analyze_class_frequencies.py`](../scripts/analyze_class_frequencies.py)

**Usage**:
```bash
python scripts/analyze_class_frequencies.py \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output outputs/class_weights.json \
    --method inverse \
    --smooth 0.1
```

**What it does**:
1. Analyzes training data to compute class frequencies
2. Generates class weights using inverse frequency or sqrt inverse methods
3. Saves weights to JSON for use in training
4. Displays top classes and their distributions

**Methods**:
- `inverse`: weight = 1 / frequency (more aggressive)
- `sqrt_inverse`: weight = 1 / sqrt(frequency) (less aggressive)

### 3. Training Integration

The mode collapse prevention features integrate seamlessly into existing training:

**Already Available** (no code changes needed):
- Label smoothing: `--label_smoothing 0.0` (reduce from 0.1-0.2)
- Composite accuracy metric: Already computed automatically

**To Add** (requires integration into [train_multihead.py](../scripts/train_multihead.py)):
- Class-balanced loss with weights
- Focal loss option
- Diversity monitoring during training
- Mode collapse warnings

## Recommended Training Strategy

### Step 1: Analyze Your Data

Before training, analyze class frequencies:

```bash
python scripts/analyze_class_frequencies.py \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output outputs/class_weights_hybrid.json
```

This will show you:
- Top 10 most frequent commands
- Distribution across parameter types and values
- Suggested class weights

### Step 2: Train with Balanced Loss

**Option A: Class-Balanced Loss** (Recommended)

Train with class weights generated from Step 1:

```bash
python scripts/train_multihead.py \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output-dir outputs/balanced_training \
    --class-weights-path outputs/class_weights_hybrid.json \
    --label-smoothing 0.0 \
    --max-epochs 20 \
    --patience 8 \
    --use-wandb
```

**Option B: Focal Loss** (For severe imbalance)

Use focal loss which automatically focuses on hard examples:

```bash
python scripts/train_multihead.py \
    --data-dir outputs/processed_hybrid \
    --vocab-path data/vocabulary_1digit_hybrid.json \
    --output-dir outputs/focal_training \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --label-smoothing 0.0 \
    --max-epochs 20 \
    --use-wandb
```

### Step 3: Monitor for Mode Collapse

During training, watch for these warning signs:

**In Console Output**:
```
⚠️  WARNING: POSSIBLE MODE COLLAPSE DETECTED
  Prediction Diversity: 15% (threshold: 30%)
  Entropy Ratio: 35% (healthy: > 50%)
```

**In W&B Dashboard**:
- `pred_diversity_ratio`: Should be > 0.3 (30% of classes used)
- `pred_entropy_ratio`: Should be > 0.5 (50% of maximum entropy)
- `cmd_0_acc` through `cmd_9_acc`: Per-class accuracies (should vary)

### Step 4: Adjust if Needed

If mode collapse is detected:

1. **Increase class weight aggressiveness**: Use `inverse` instead of `sqrt_inverse`
2. **Try focal loss**: Set `--focal-gamma 2.0` or higher (up to 5.0)
3. **Remove label smoothing**: Set `--label-smoothing 0.0`
4. **Use composite metric for early stopping**: Already done automatically

## API Reference

### compute_class_frequencies

```python
from miracle.training.mode_collapse_prevention import compute_class_frequencies

frequencies = compute_class_frequencies(train_loader, decomposer)
# Returns: Dict with 'command', 'param_type', 'param_value' frequency tensors
```

### compute_class_weights

```python
from miracle.training.mode_collapse_prevention import compute_class_weights

weights = compute_class_weights(
    frequencies,
    method='inverse',  # or 'sqrt_inverse'
    smooth=0.1
)
# Returns: Dict with 'command', 'param_type', 'param_value' weight tensors
```

### FocalLoss

```python
from miracle.training.mode_collapse_prevention import FocalLoss

focal_loss = FocalLoss(
    num_classes=n_commands,
    alpha=class_weights,  # Optional: from compute_class_weights
    gamma=2.0,           # Higher = more focus on hard examples
    label_smoothing=0.0
)

loss = focal_loss(logits, targets)
```

### compute_diversity_metrics

```python
from miracle.training.mode_collapse_prevention import compute_diversity_metrics

metrics = compute_diversity_metrics(
    command_logits,
    command_targets,
    decomposer,
    top_k=10
)
# Returns metrics including entropy, diversity, per-class accuracy
```

## Troubleshooting

### Problem: Model still collapses even with class weights

**Solution**: Try focal loss with higher gamma (3.0-5.0) and remove label smoothing

### Problem: Training is slower with class weights

**Expected**: Class weighting adds minimal overhead (<5%). If slower, check batch size.

### Problem: Weights file not loading

**Check**:
1. File exists at specified path
2. JSON format is correct: `{"command": [...], "param_type": [...], "param_value": [...]}`
3. Weight tensor sizes match model head sizes

### Problem: Still getting mode collapse warnings despite fixes

**Investigate**:
1. Check if most frequent class dominates (>80% of data)
2. Consider data augmentation to oversample rare classes
3. Try reducing model capacity (fewer layers/hidden dims)
4. Check if dataset is inherently imbalanced beyond fixing

## Comparison: full_featured_FIXED_3 vs. Proposed Approach

### Original Approach (Failed)
- No class weighting
- Label smoothing: 0.1
- Early stopping on tolerance accuracy (masked the problem)
- Result: 100% command acc (always G0), 0% overall acc

### Proposed Approach
- Class-balanced loss or Focal loss
- Label smoothing: 0.0
- Early stopping on composite accuracy
- Diversity monitoring during training
- Expected: Balanced predictions across all classes

## Next Steps

To fully integrate these features, the training script needs updates:

**Required Changes to** [`train_multihead.py`](../scripts/train_multihead.py):
1. Add `--class-weights-path` and `--use-focal-loss` argparse options
2. Load class weights if provided
3. Use FocalLoss or pass weights to loss function
4. Compute and log diversity metrics during validation
5. Add mode collapse warning system

**Implementation Status**: Utility modules complete, training integration pending

---

**Files Added**:
- [`src/miracle/training/mode_collapse_prevention.py`](../src/miracle/training/mode_collapse_prevention.py)
- [`scripts/analyze_class_frequencies.py`](../scripts/analyze_class_frequencies.py)
- This guide

**Next Training Run**: Use analyze_class_frequencies.py first, then train with generated weights
