# Class Imbalance Crisis - Summary & Path Forward

## 🚨 The Problem

**All training runs converge to degenerate solution: Always predict G0**

Despite trying EVERY standard technique for class imbalance, the model consistently learns to predict only the majority class (G0) and ignores all rare classes.

---

## 📊 Data Distribution (The Root Cause)

### Command Class Distribution:
```
Command 0 (G0):     7,582 tokens (79.02%) ← DOMINATES
Command 1 (G1):     1,880 tokens (19.59%)
Command 4:             41 tokens (0.43%)
Command 5:             51 tokens (0.53%)
Command 10:            41 tokens (0.43%)
Other commands:         0 tokens (0.00%)
```

**Combined rare classes (4, 5, 10): Only 1.39% of data!**

---

## ❌ Failed Experiments

### Experiment 1: Standard Class Weights
- **Config:** 10x ratio (G0=0.114, rare=1.15)
- **Result:** Command=100%, G1/G2/G3=0% (epochs 1-30)
- **Verdict:** Ratio too small

### Experiment 2: Extreme Class Weights
- **Config:** 100,000x ratio (G0=0.001, rare=100.0)
- **Result:** Started at 54.90% acc, converged to 100% G0-only by epoch 4
- **Verdict:** Even 100,000x insufficient!

### Experiment 3: Focal Loss γ=5.0
- **Config:** γ=5.0 + extreme weights
- **Result:** Command=9.09% (too aggressive - can't learn)
- **Verdict:** γ=5.0 destroys training

### Experiment 4: Focal Loss γ=2.0
- **Config:** γ=2.0 + extreme weights
- **Result:** Epoch 1: 63.64%, Epoch 2: 100% G0-only
- **Verdict:** Still reverts to degenerate solution

---

## 🤔 Why ALL Techniques Fail

### The Mathematical Truth:

The model is **correctly optimizing the loss function** by predicting only G0!

Even with extreme penalties, the numbers work out:

**Strategy A: Predict only G0**
- Get 79% of examples right
- Loss from rare classes (1.39%): Even with 100x weight = 1.39 loss contribution

**Strategy B: Try to learn rare classes**
- Risk getting some of the 79% G0 examples wrong
- Only gain 1.39% accuracy on rare classes
- Net loss: Higher!

**The model is being SMART, not lazy!**

---

## 🎯 What We Actually Need

### Critical Question: Is this even a problem?

**Option A: The data distribution is correct**
- Maybe G-code really IS 79% G0 commands
- Maybe the model SHOULD predict G0 most of the time
- The "degenerate solution" might be the CORRECT solution!

**Option B: There's a fundamental data issue**
- Preprocessing removed rare classes?
- Vocabulary mapping collapsed different commands?
- Test/val splits don't have rare class examples?

---

## 🔍 Diagnostic Steps BEFORE More Training

### 1. Inspect Raw G-Code Files

Check if original files actually have rare commands:

```bash
# Sample a few raw G-code files
head -100 data/clean_output/file1.gcode
head -100 data/clean_output/file2.gcode

# Count command frequencies
grep -E "^G[0-9]+" data/clean_output/*.gcode | cut -d' ' -f1 | sort | uniq -c | sort -rn
```

**Question:** Do raw files have G2, G3, etc., or is it actually 79% G0?

### 2. Check Vocabulary Mapping

Examine [data/vocabulary_1digit_hybrid.json](data/vocabulary_1digit_hybrid.json):

```python
import json
with open('data/vocabulary_1digit_hybrid.json') as f:
    vocab = json.load(f)

# Print all command tokens
for token, idx in vocab['token2id'].items():
    if token.startswith('G'):
        print(f"{token} -> ID {idx}")
```

**Question:** Are rare G-codes being mapped to the same ID?

### 3. Analyze Validation Set Composition

```python
import numpy as np

# Load validation data
val_data = np.load('outputs/processed_hybrid/val_sequences.npz', allow_pickle=True)
tokens = val_data['tokens']

# Count how many examples have rare commands
from miracle.dataset.target_utils import TokenDecomposer
decomposer = TokenDecomposer('data/vocabulary_1digit_hybrid.json')

rare_commands = {4, 5, 10}
sequences_with_rare = 0

for seq in tokens:
    for token_id in seq:
        _, cmd_id, _, _ = decomposer.decompose_token(token_id)
        if cmd_id in rare_commands:
            sequences_with_rare += 1
            break

print(f"Validation sequences with rare commands: {sequences_with_rare} / {len(tokens)}")
print(f"Percentage: {sequences_with_rare / len(tokens) * 100:.2f}%")
```

**Question:** How many validation examples even HAVE rare commands?

### 4. Check if "Command Accuracy 100%" is Misleading

The composite_acc = command_acc × param_type_acc × param_value_acc

But what if:
- Command acc = 100% (predicting all G0)
- Param type acc = 94% (actually learning param types)
- Composite = 94%

This would LOOK good but only be learning param types, not commands!

**Check per-class F1 scores, not just overall accuracy.**

---

## 🚀 Next Steps (Ordered by Priority)

### Step 1: DATA INVESTIGATION (Do this FIRST!)

Run the 4 diagnostic checks above to understand:
1. What's actually in the raw G-code?
2. Is the vocabulary mapping correct?
3. Does val set even have rare classes?
4. Are we measuring the right metrics?

### Step 2: If rare commands exist in data → Try Balanced Sampling

```python
from torch.utils.data import WeightedRandomSampler

# Compute sample weights (inverse frequency)
sample_weights = []
for seq in dataset:
    # Weight sequence by rarest command it contains
    weight = compute_sequence_weight(seq)
    sample_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # Allow repeating rare examples
)

# Use sampler in DataLoader
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

This FORCES the model to see rare classes equally often as common ones.

### Step 3: If balanced sampling fails → Curriculum Learning

**Phase 1:** Train on artificially balanced dataset (oversample rare, undersample G0)
**Phase 2:** Fine-tune on true distribution

### Step 4: If nothing works → Re-examine the problem definition

Maybe the task is:
- **Not** "predict all commands equally well"
- **But** "predict the true G-code distribution (79% G0)"

In which case, the current "degenerate" behavior is CORRECT!

---

## 📁 Files for Investigation

1. **Raw data:** `data/clean_output/*.gcode`
2. **Vocabulary:** [data/vocabulary_1digit_hybrid.json](data/vocabulary_1digit_hybrid.json)
3. **Processed data:** `outputs/processed_hybrid/*.npz`
4. **Class weights:** [outputs/class_weights_extreme.json](outputs/class_weights_extreme.json)
5. **Training script:** [scripts/train_multihead.py](scripts/train_multihead.py)

---

## 💡 Key Insight

**The model is not broken - it's optimizing correctly!**

The issue is either:
1. The data truly is 79% G0 (in which case this is expected behavior)
2. The preprocessing corrupted the rare classes
3. We're measuring the wrong thing (overall acc instead of per-class recall)
4. We need a fundamentally different sampling strategy (not just loss weighting)

---

## ⏭️ Recommended Action

**STOP training more models right now.**

Instead:
1. Run the 4 diagnostic checks above
2. Understand what's actually in the data
3. Decide if "100% command accuracy" is even wrong
4. Only then resume training with the appropriate approach

**Do not throw more compute at this without understanding the data first!**

---

Last updated: 2025-11-29 22:03 UTC
Status: 🔴 All training attempts failed - needs data investigation
