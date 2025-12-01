# Training Comparison Guide - Phase 2

**Date**: 2025-11-19
**Version**: 2.0 (Vocabulary v2 + All Improvements)

---

## Overview

This guide compares **three training approaches** to solve the model collapse problem:

1. **Baseline (Vocab v2 only)** - ❌ Still collapses (11-14 unique tokens)
2. **With Data Augmentation** - ✅ Vocabulary v2 + 3x oversampling
3. **Multi-Head Architecture** - ✅ Vocabulary v2 + hierarchical prediction + optional augmentation

---

## Quick Comparison

| Approach | Vocab | Augmentation | Multi-Head | Expected Performance |
|----------|-------|--------------|------------|---------------------|
| **Baseline** | v2 (170) | ❌ | ❌ | ❌ Collapses (11-14 tokens) |
| **Augmented** | v2 (170) | ✅ 3x | ❌ | ✅ Good (>100 tokens, >60% acc) |
| **Multi-Head** | v2 (170) | Optional | ✅ | ✅ Best (>120 tokens, >70% acc) |

---

## Approach 1: Baseline (Vocab v2 Only)

### Status
❌ **Still collapses** - vocabulary reduction alone is insufficient

### Command
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_phase1_fixed.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --output-dir outputs/baseline_v2 \
    --max-epochs 50 \
    --use-wandb
```

### Results (From Testing)
- ❌ Unique tokens: 11-14 / 170 (6-8%)
- ❌ G-command accuracy: 0-7%
- ❌ Overall accuracy: 31-54%
- ⚠️ Early stopping: Model collapse detected

### Analysis
**Problem**: Even with 170-token vocabulary (reduced from 668), the ~10:1 class imbalance is still too severe without additional techniques.

---

## Approach 2: With Data Augmentation

### Features
✅ Vocabulary v2 (170 tokens)
✅ 3x oversampling for G/M commands
✅ Sensor noise injection
✅ Temporal shifting
✅ Magnitude scaling

### Command
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_with_augmentation.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/augmented_v2 \
    --oversample-factor 3 \
    --max-epochs 50 \
    --use-wandb \
    --run-name "augmented-vocab-v2-3x"
```

### Expected Results
- ✅ Unique tokens: >100 / 170 (>60%)
- ✅ G-command accuracy: >60%
- ✅ Overall accuracy: >60%
- ✅ Effective training data: 2212 → ~3500 for G-commands

### How It Works
1. **Base dataset**: 2212 train sequences
2. **Identify rare tokens**: G0, G1, G2, G3, M3, M5, etc.
3. **Oversample 3x**: Sequences with G/M commands appear 3 times
4. **Augment on-the-fly**: Each epoch sees different augmented versions
   - Sensor noise: ±2% Gaussian
   - Temporal shift: ±2 timesteps
   - Magnitude scale: 0.95-1.05x

---

## Approach 3: Multi-Head Architecture

### Features
✅ Vocabulary v2 (170 tokens)
✅ Hierarchical token prediction (4 heads)
✅ Eliminates gradient competition
✅ Optional data augmentation
✅ Separate loss weights per head

### Command (Without Augmentation)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_multihead.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/multihead_v2 \
    --max-epochs 50 \
    --use-wandb \
    --run-name "multihead-vocab-v2"
```

### Command (With Augmentation) - **RECOMMENDED**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_multihead.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/multihead_aug_v2 \
    --use-augmentation \
    --oversample-factor 3 \
    --max-epochs 50 \
    --use-wandb \
    --run-name "multihead-aug-vocab-v2"
```

### Expected Results
- ✅ Unique tokens: >120 / 170 (>70%)
- ✅ Command accuracy: >70%
- ✅ Overall accuracy: >70%
- ✅ Type accuracy: >85%
- ✅ Parameter type accuracy: >75%

### Architecture
```
Multi-Head G-code LM:
  ├─ Type Gate: SPECIAL/COMMAND/PARAMETER/NUMERIC (4 classes)
  ├─ Command Head: G0, G1, M3, etc. (~6-10 classes)
  ├─ Param Type Head: X, Y, Z, F, R, S (~5-10 classes)
  └─ Param Value Head: 00-99 (100 classes)
```

### How It Works
1. **Token decomposition**: Each token split into hierarchy
   - `G0` → type=COMMAND, command_id=0
   - `X` → type=PARAMETER, param_type=X
   - `NUM_X_15` → type=NUMERIC, param_type=X, param_value=15

2. **Separate prediction**: Each head trained independently
   - No gradient competition between token types
   - Commands get 3x loss weight (rarer tokens)

3. **Token composition**: Predictions combined back into full tokens

---

## Training All Three Approaches

### Run Sequential Comparison

```bash
# 1. Baseline (will collapse - for comparison)
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_phase1_fixed.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --output-dir outputs/baseline_v2 \
    --max-epochs 20 \
    --use-wandb \
    --run-name "baseline-v2"

# 2. With augmentation
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_with_augmentation.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/augmented_v2 \
    --oversample-factor 3 \
    --max-epochs 50 \
    --use-wandb \
    --run-name "augmented-v2"

# 3. Multi-head with augmentation (BEST)
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_multihead.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/multihead_aug_v2 \
    --use-augmentation \
    --oversample-factor 3 \
    --max-epochs 50 \
    --use-wandb \
    --run-name "multihead-aug-v2"
```

### Monitor on WandB

All runs will appear on your WandB dashboard:
```
https://wandb.ai/<your-username>/gcode-fingerprinting
```

Compare metrics:
- `val/overall_acc`
- `val/g_command_acc`
- `unique_tokens_predicted`
- Training time per epoch

---

## Recommended Approach

### For Best Results: Multi-Head + Augmentation

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_multihead.py \
    --config configs/phase1_best.json \
    --data-dir outputs/processed_v2 \
    --vocab-path data/gcode_vocab_v2.json \
    --output-dir outputs/final_model \
    --use-augmentation \
    --oversample-factor 3 \
    --max-epochs 50 \
    --patience 10 \
    --use-wandb \
    --run-name "final-multihead-aug"
```

**Why this works**:
1. **Vocabulary v2**: 668 → 170 tokens (74.5% reduction)
2. **Data augmentation**: Effective 3x more G-command training examples
3. **Multi-head**: Separates command vs parameter prediction
4. **Combined effect**: Addresses root cause (imbalance) + symptom (collapse)

---

## Expected Training Times (Mac M1/M2)

| Approach | Seconds/Epoch | Total (50 epochs) |
|----------|---------------|-------------------|
| Baseline | ~12s | ~10 min |
| Augmented | ~18s | ~15 min |
| Multi-Head | ~15s | ~12.5 min |
| Multi-Head + Aug | ~22s | ~18 min |

---

## Success Criteria

### Minimum (Augmented)
- ✅ Unique tokens predicted: >100 / 170
- ✅ G-command accuracy: >60%
- ✅ Overall accuracy: >60%
- ✅ No model collapse

### Target (Multi-Head + Aug)
- ✅ Unique tokens predicted: >120 / 170
- ✅ Command accuracy: >70%
- ✅ Overall accuracy: >70%
- ✅ Type accuracy: >85%

---

## Troubleshooting

### Issue: Still collapsing (<20 unique tokens)

**Solutions**:
1. Verify vocabulary v2 is being used: Check `vocab_size=170` in logs
2. Increase oversample factor: Try `--oversample-factor 5`
3. Increase command loss weight: Modify `command_weight=5.0` in `train_multihead.py`

### Issue: Low command accuracy (<50%)

**Solutions**:
1. Use multi-head architecture (stronger signal for commands)
2. Increase oversample factor to 5x
3. Reduce batch size to 16 (more gradient updates)

### Issue: Out of memory

**Solutions**:
1. Reduce batch size: `"batch_size": 16` in config
2. Reduce hidden_dim: `"hidden_dim": 256` in config
3. Use augmentation without multi-head (less memory)

---

## Files Created

| File | Purpose | Lines |
|------|---------|-------|
| [train_with_augmentation.py](train_with_augmentation.py) | Training with data augmentation | ~380 |
| [train_multihead.py](train_multihead.py) | Training with multi-head architecture | ~450 |
| [src/miracle/dataset/data_augmentation.py](src/miracle/dataset/data_augmentation.py) | Augmentation classes | ~297 |
| [src/miracle/model/multihead_lm.py](src/miracle/model/multihead_lm.py) | Multi-head LM | ~273 |
| [src/miracle/dataset/target_utils.py](src/miracle/dataset/target_utils.py) | Token decomposition | ~290 |
| [src/miracle/training/losses.py](src/miracle/training/losses.py) | Multi-head loss (added) | +150 |

---

## Next Steps

1. **Train all three approaches** to compare results
2. **Select best model** based on WandB metrics
3. **Evaluate on test set** using best checkpoint
4. **Generate visualizations** for publication

---

**Last Updated**: 2025-11-19
**Status**: Ready for comparative training
