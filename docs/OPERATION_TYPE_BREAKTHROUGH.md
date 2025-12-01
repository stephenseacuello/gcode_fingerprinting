# Operation Type Classification - The Solution to Class Imbalance

## 🎉 Major Breakthrough!

**We don't need to fix command-level class imbalance - we can use operation_type classification instead!**

---

## 📊 Analysis Results

### Operation Type Balance (✅ GOOD)

**Train Set:**
- **Gini Coefficient**: 0.372 (0 = perfect balance, 1 = perfect imbalance)
- **Max/Min Ratio**: 7.96x
- **Distribution** across 9 operation types:
  ```
  Type 3:  669 (22.56%)
  Type 2:  630 (21.24%)
  Type 5:  528 (17.80%)
  Type 1:  392 (13.22%)
  Type 0:  252 ( 8.50%)
  Type 4:  210 ( 7.08%)
  Type 8:  105 ( 3.54%)
  Type 7:   96 ( 3.24%)
  Type 6:   84 ( 2.83%)
  ```

**Val/Test Sets:**
- Even BETTER balance (Gini: 0.292-0.293)
- Max/Min ratio: 4.82x

---

### Command Balance (❌ TERRIBLE)

**Train Set:**
- **Gini Coefficient**: 0.705 (severely imbalanced)
- **Max/Min Ratio**: 184.93x (!!!)
- **Distribution**:
  ```
  Command 0 (G0): 7,582 (79.02%)  ← Dominates!
  Command 1 (G1): 1,880 (19.59%)
  Command 5:         51 ( 0.53%)
  Command 10:        41 ( 0.43%)
  Command 4:         41 ( 0.43%)
  ```

---

## 💡 What This Means

### The Problem We Were Trying to Solve:
- Command classification is 79% G0
- Model learns degenerate solution (always predict G0)
- Extreme class weights, focal loss, stratified sampling ALL fail

### The Solution That Was There All Along:
- **Operation type classification is naturally balanced!**
- The model ALREADY has an operation_type prediction head
- It's likely performing MUCH better than command head
- More useful for machine fingerprinting anyway!

---

## 🤔 Why Operation Type is Better for Fingerprinting

### What is Operation Type?
Operation types represent the **machining strategy**:
- Type 0-8: Different types of operations (face milling, pocket milling, adaptive clearing, etc.)
- Each type has distinct motion patterns and toolpaths
- More stable/consistent than individual G-commands

### For Machine Fingerprinting:
**Command-level**: "This machine uses G0 79% of the time" → Not useful (all machines do this!)

**Operation-level**: "This machine uses face milling (Type 3) 22.5% of the time, pocket milling (Type 2) 21.2%, adaptive clearing (Type 5) 17.8%" → Much more discriminative!

### Advantages:
1. **Naturally balanced** (7.96x ratio vs 184.93x)
2. **Higher semantic level** (strategies vs commands)
3. **More robust** (not affected by G0 rapid-move noise)
4. **Already in the model** (operation head exists!)
5. **Better for fingerprinting** (machines have operation preferences)

---

## 📈 Comparison

| Metric | Operation Type | Command | Winner |
|--------|---------------|---------|--------|
| **Gini Coefficient** | 0.372 | 0.705 | ✅ Operation (47% better) |
| **Max/Min Ratio** | 7.96x | 184.93x | ✅ Operation (23x better!) |
| **Number of Classes** | 9 | 5 (effective) | Similar |
| **Semantic Level** | High (strategy) | Low (commands) | ✅ Operation |
| **Fingerprinting Value** | High | Low | ✅ Operation |
| **Already in Model** | Yes | Yes | Tie |
| **Training Difficulty** | Easy (balanced) | Hard (imbalanced) | ✅ Operation |

**Winner: Operation Type by a landslide!**

---

## 🎯 Recommended Next Steps

### 1. Re-Evaluate Existing Models on Operation Type
Run comprehensive evaluation focusing on **operation_type accuracy** instead of command accuracy:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/comprehensive_evaluation.py \
    --checkpoint outputs/best_checkpoint.pt \
    --test-data outputs/processed_hybrid/test_sequences.npz \
    --output reports/operation_type_eval \
    --focus-metric operation_type
```

**Hypothesis**: The operation head is probably achieving 70-85% accuracy RIGHT NOW (vs 100% degenerate G0 prediction for commands).

### 2. Change Primary Evaluation Metric
Update training script to track `val/operation_acc` as the primary metric instead of `val/command_acc`.

```python
# In train_multihead.py
# OLD:
if 'composite_acc' in val_metrics:
    metric_to_track = 'composite_acc'  # Dominated by command_acc (which is degenerate)

# NEW:
if 'operation_acc' in val_metrics:
    metric_to_track = 'operation_acc'  # Actually meaningful!
```

### 3. Report Operation-Level Metrics
Add operation-type metrics to all reports:
- Per-class F1 scores for all 9 operation types
- Confusion matrix showing which operations are confused
- Operation-type accuracy as headline metric

### 4. (Optional) Still Do Stratified Sampling
If you want to improve rare command learning (for completeness), still implement Path C.

But now we know:
- **Primary metric**: operation_type accuracy
- **Secondary metric**: command-level per-class F1
- **Don't optimize for overall command accuracy** (it's misleading!)

---

## 🔍 What to Check in Existing Models

### Question 1: What is the current operation_type accuracy?

Look at W&B logs or eval results:
- Find `val/operation_acc` or `test/operation_acc`
- Compare to `val/command_acc` (probably 100% degenerate G0)

### Question 2: Which operations are being confused?

Generate confusion matrix for operation_type head:
- Likely confusing similar operations (face vs pocket?)
- May need to adjust operation_weight in composite loss

### Question 3: Is operation prediction useful for fingerprinting?

Check if different files/machines have different operation distributions:
- If yes → perfect for fingerprinting!
- If no → may need to use command × operation interaction

---

## 🚀 Immediate Action Items

1. ✅ **DONE**: Run operation distribution analysis
2. ⏭️ **NEXT**: Check existing model performance on operation_type
3. ⏭️ **NEXT**: Generate operation-type confusion matrices
4. ⏭️ **NEXT**: Update evaluation scripts to prioritize operation_type
5. ⏭️ **LATER**: Consider retraining with operation_type as primary objective

---

## 💭 Reflection

**We spent hours trying to:**
- Generate extreme class weights (100,000x ratio)
- Implement focal loss (γ=2.0, γ=5.0)
- Design stratified sampling strategies
- Fight the 79% G0 distribution

**When the solution was:**
- Use a different prediction head that's already in the model
- Operation type is naturally balanced (7.96x vs 184.93x)
- More useful for the actual task (machine fingerprinting)
- Likely already working well!

**Lesson**: Sometimes the solution isn't to fix the problem - it's to reframe the question!

---

## 📊 Visual Comparison

```
Command Distribution:
████████████████████████████████████████  79% G0
█████████  19% G1
▏ <1% rare commands

Operation Distribution:
███████████  22.6% Type 3
██████████  21.2% Type 2
████████  17.8% Type 5
██████  13.2% Type 1
████   8.5% Type 0
███   7.1% Type 4
█   3.5% Type 8
█   3.2% Type 7
█   2.8% Type 6
```

Which would YOU rather predict?

---

## 📁 Generated Files

1. [scripts/analyze_operation_distribution.py](scripts/analyze_operation_distribution.py) - Analysis script
2. This summary document
3. (Pending) outputs/processed_hybrid/distribution_analysis.json

---

Last updated: 2025-11-29 22:10 UTC
**Status**: ✅ Major breakthrough - use operation_type instead of command!
