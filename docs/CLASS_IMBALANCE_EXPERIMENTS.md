# Class Imbalance Experiments - 2025-11-29

## 🚨 Problem Identified

**Degenerate Solution:** Model always predicts G0 (majority class) → gets 100% command accuracy but learns nothing

### Data Distribution:
```
Command 0 (G0):     7582 (79.02%) ← Dominates!
Command 1 (G1):     1880 (19.59%)
Command 4:            41 ( 0.43%)
Command 5:            51 ( 0.53%)
Command 10:           41 ( 0.43%)
Other commands:        0 ( 0.00%)
```

**Mathematical Proof of Degenerate Solution:**
- Always predict G0: Get 79% correct with ZERO effort
- With minimal G1 prediction: Reach 90-100% accuracy
- Rare commands (1.4% total): Can be ignored with minimal penalty

---

## Experiment 1: Standard Class Weights (FAILED)

**Configuration:**
- Class weights from [outputs/class_weights_hybrid.json](outputs/class_weights_hybrid.json)
- G0 weight: 0.114
- Rare command weight: 1.15
- Ratio: 10x

**Results (Epochs 1-30):**
```
Epoch 29:
  Command Distribution:
    G0: 100.0% ← Only predicting G0!
    G1: 0.0%
    G2: 0.0%
    G3: 0.0%
  Val Accuracy: 93.77%
```

**Verdict:** ❌ 10x ratio insufficient - model found degenerate solution

---

## Experiment 2: Extreme Class Weights (FAILED)

**Configuration:**
- Generated via [scripts/generate_extreme_class_weights.py](scripts/generate_extreme_class_weights.py)
- G0 weight: 0.001
- Rare command weight: 100.0
- Ratio: 100,000x (!!!)
- Strategy: sqrt dampening

**Results (Epochs 1-4):**
```
Epoch 1:
  Command Acc: 54.90% train, 90.91% val
  G1: 0.0% | G2: 0.0% | G3: 0.0%

Epoch 4:
  Command Acc: 100.00% train, 100.00% val ← Converged back!
  G1: 0.0% | G2: 0.0% | G3: 0.0%
```

**Verdict:** ❌ Even 100,000x ratio failed - model reverted to G0-only prediction by epoch 4

---

## Experiment 3: Focal Loss γ=5.0 + Extreme Weights (FAILED)

**Configuration:**
- Focal loss: γ=5.0 (very aggressive)
- Class weights: extreme (100,000x ratio)
- Combined approach

**Results (Epoch 1):**
```
Epoch 1:
  Command Acc: 9.80% train, 9.09% val ← TERRIBLE!
  Composite: 0.0335 (was 0.51 before)
  G1: 0.0% | G2: 0.0% | G3: 0.0%
```

**Verdict:** ❌ γ=5.0 TOO aggressive - model can't learn anything

---

## Experiment 4: Focal Loss γ=2.0 + Extreme Weights (RUNNING)

**Configuration:**
- Focal loss: γ=2.0 (standard value from original paper)
- Class weights: extreme (100,000x ratio)
- Output dir: [outputs/focal_loss_gamma2](outputs/focal_loss_gamma2)
- W&B run: [https://wandb.ai/seacuello-university-of-rhode-island/gcode-fingerprinting](https://wandb.ai/seacuello-university-of-rhode-island/gcode-fingerprinting)

**Status:** 🏃 Training in progress...

**Expected Behavior:**
- Focal loss should down-weight easy examples (G0)
- Extreme weights should up-weight rare classes
- γ=2.0 should be balanced (not too aggressive)

**Success Criteria:**
- G1 prediction > 0%
- Rare command (4, 5, 10) prediction > 0%
- Command accuracy > 80% (balanced across classes)

---

## Alternative Approaches (If Focal Loss Fails)

### Option 1: Balanced Sampling
```python
from torch.utils.data import WeightedRandomSampler

# Force each batch to have balanced class representation
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
```

**Pros:** Guarantees balanced training batches
**Cons:** May overtrain on rare examples (overfitting risk)

### Option 2: Curriculum Learning
1. Phase 1: Train on balanced subset (equal G0, G1, rare commands)
2. Phase 2: Fine-tune on full dataset

**Pros:** Model learns all classes first, then adapts to true distribution
**Cons:** More complex, requires careful subset selection

### Option 3: Two-Stage Training
1. Stage 1: Train classifier on balanced data (oversample rare classes)
2. Stage 2: Calibrate probabilities on true distribution

**Pros:** Separates learning from calibration
**Cons:** Requires careful implementation

### Option 4: Accept the Distribution?
**Hypothesis:** Maybe G0 dominance is the TRUE data distribution?

If G-code files genuinely contain 79% G0 commands, then:
- Model SHOULD predict G0 most of the time
- 100% command accuracy might be CORRECT if it predicts 79% G0, 20% G1, 1% rare
- The issue is NOT degenerate solution, but lack of rare class examples in validation set

**To Test:**
1. Check per-class recall (not just overall accuracy)
2. Examine confusion matrix (are rare classes being predicted at all?)
3. Analyze validation set composition (how many rare class examples exist?)

---

## Metrics to Track

### Primary Metrics:
- **Per-Class Recall:** G0, G1, G2, G3, etc.
- **Rare Class F1:** Average F1 for commands with <1% frequency
- **Composite Accuracy:** command_acc × param_type_acc × param_value_acc

### Diagnostic Metrics:
- **Command Distribution:** % predictions for each command
- **Confusion Matrix:** See which classes are being confused
- **Class-Specific Loss:** Monitor loss per command class

---

## Files Generated

1. [scripts/generate_extreme_class_weights.py](scripts/generate_extreme_class_weights.py) - Extreme weight generator
2. [outputs/class_weights_extreme.json](outputs/class_weights_extreme.json) - Extreme weights (100,000x ratio)
3. This summary document

---

## Next Steps

1. ⏳ **Wait for Focal Loss γ=2.0 results** (Experiment 4)
2. 📊 **Analyze per-class metrics** (not just overall accuracy)
3. 🔍 **Inspect validation set** (how many rare class examples?)
4. 🎯 **If γ=2.0 fails:** Try balanced sampling or curriculum learning
5. 🤔 **Re-evaluate problem:** Is G0 dominance actually correct?

---

## Key Learnings

1. **Class imbalance is SEVERE:** 79% G0, 1.4% rare classes combined
2. **Standard approaches fail:** 10x weights, 100,000x weights, focal loss γ=5.0
3. **Degenerate solutions are mathematically optimal:** Model correctly minimizes loss by predicting G0
4. **May need architectural changes:** Not just loss/weight tuning
5. **Data augmentation?** Could generate synthetic rare class examples

---

## W&B Tracking

View all experiments:
- Project: [gcode-fingerprinting](https://wandb.ai/seacuello-university-of-rhode-island/gcode-fingerprinting)
- Filter by tags: `class-imbalance`, `focal-loss`, `extreme-weights`

---

Last updated: 2025-11-29 22:00 UTC
