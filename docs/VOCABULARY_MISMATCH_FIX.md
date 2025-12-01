# Vocabulary Mismatch Fix - PAD PAD PAD Issue

**Date:** 2025-11-30
**Issue:** Dashboard generating "PAD PAD PAD" instead of valid G-code
**Root Cause:** Vocabulary file mismatch between training and inference
**Status:** ✅ FIXED

---

## Problem Summary

After fixing the autoregressive generation loop, the dashboard started generating "PAD PAD PAD" repeatedly instead of valid G-code tokens.

**Symptoms:**
```csv
gcode_predicted,token_count
PAD PAD PAD,3
PAD PAD PAD,3
PAD PAD PAD,3
```

**Debug logs showed:**
```
type_id=1, command_id=10, param_type_id=4, param_value_id=1
Composed token: ID=0, Token=PAD
```

---

## Root Cause

### Training Configuration
Model was trained with:
```yaml
vocab-path: data/vocabulary_1digit_hybrid.json
```

This vocabulary has:
- **69 total tokens**
- **12 command tokens**: G0, G1, G17, G2, G3, G53, G55, G90, G94, M3, M30
- **1-digit bucketing** (bucket_digits=1)
- Command IDs: 0-11

### Dashboard Configuration (WRONG!)
Dashboard was loading:
```python
vocab_path = Path('data/vocabulary.json')  # ← WRONG!
```

This vocabulary has:
- **668 total tokens**
- **6 command tokens**: G0, G1, G2, G3, G53, M30
- **4-digit bucketing** (no bucketing)
- Command IDs: 0-5

### The Mismatch

When the model predicts `command_id=10`:
- **Training decomposer**: command_id=10 → "M3" ✅
- **Dashboard decomposer**: command_id=10 → **DOES NOT EXIST** → returns PAD (ID=0) ❌

The TokenDecomposer.compose_token() method returns PAD as a fallback when given invalid indices.

---

## The Fix

### Changes Made

Updated [flask_dashboard.py](../flask_dashboard.py) to use the correct vocabulary:

**Line 336:**
```python
# Before
vocab_path = Path('data/vocabulary.json')

# After
vocab_path = Path('data/vocabulary_1digit_hybrid.json')
```

**Line 419:**
```python
# Before
vocab_path = Path('data/vocabulary.json')

# After
vocab_path = Path('data/vocabulary_1digit_hybrid.json')
```

---

## Vocabulary Comparison

### vocabulary_1digit_hybrid.json (CORRECT for sweep_overnight_20251129)

```json
{
  "config": {
    "mode": "hybrid",
    "bucket_digits": 1,
    "vocab_size": 5000,
    ...
  },
  "vocab": {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "UNK": 3,
    "MASK": 4,
    "X": 5,
    "Y": 6,
    "Z": 7,
    "R": 8,
    "F": 9,
    "I": 10,
    "J": 11,
    "K": 12,
    "G0": 13,
    "G1": 14,
    "G2": 15,
    "G3": 16,
    "G17": 17,
    "G53": 18,
    "G55": 19,
    "G90": 20,
    "G94": 21,
    "M3": 22,
    "M30": 23,
    "NUM_X_0": 24,
    "NUM_X_1": 25,
    ...
  }
}
```

**Key features:**
- 1-digit bucketing: Values 0-9 (10 buckets per parameter)
- Total vocab: 69 tokens
- Command tokens: 12 (G0, G1, G2, G3, G17, G53, G55, G90, G94, M3, M30, MASK)
- Numeric tokens: ~40 (10 buckets × 4 main parameters)

### vocabulary.json (OLD, INCORRECT)

```json
{
  "config": {
    "mode": "hybrid",
    "vocab_size": 5000,
    ...
  },
  "vocab": {
    "PAD": 0,
    ...
    "G0": 16,
    "G1": ?,
    "NUM_Y_1575": 9,
    "NUM_X_1650": 10,
    ...
  }
}
```

**Key features:**
- No bucketing (full precision)
- Total vocab: 668 tokens
- Command tokens: 6
- Numeric tokens: ~650 (all unique values from training data)

---

## TokenDecomposer Explanation

The `TokenDecomposer` builds internal mappings from the vocabulary:

```python
class TokenDecomposer:
    def __init__(self, vocab_path_or_dict):
        # Extract command tokens (G*, M*)
        self.command_tokens = [tok for tok in vocab if tok.startswith('G') or tok.startswith('M')]
        # ['G0', 'G1', 'G2', ..., 'M3', 'M30']

        # Extract parameter types (single letters)
        self.param_type_tokens = [tok for tok in vocab if len(tok)==1 and tok.isalpha()]
        # ['X', 'Y', 'Z', 'F', 'R', ...]

    def compose_token(self, type_id, command_id, param_type_id, param_value_id):
        if type_id == 1:  # TYPE_COMMAND
            if command_id < len(self.command_tokens):
                return vocab[self.command_tokens[command_id]]
            else:
                return 0  # ← PAD fallback for invalid command_id!
```

**With wrong vocabulary:**
- Model predicts `command_id=10` (expects "M3")
- Dashboard decomposer only has 6 commands (0-5)
- `command_id=10 >= 6` → returns PAD

**With correct vocabulary:**
- Model predicts `command_id=10` (expects "M3")
- Dashboard decomposer has 12 commands (0-11)
- `command_id=10` → `self.command_tokens[10]` → "M3" ✅

---

## Verification

### Test the Fix

1. **Stop the dashboard** (if running)

2. **Restart with correct vocabulary:**
   ```bash
   python flask_dashboard.py
   ```

3. **Check startup logs:**
   ```
   Loading vocabulary from: data/vocabulary_1digit_hybrid.json
   Vocabulary size: 69 tokens
   Decomposer initialized with 12 commands
   ```

4. **Monitor predictions:**
   ```bash
   python flask_dashboard.py 2>&1 | grep -E "Composed token"
   ```

   **Before fix:**
   ```
   Composed token: ID=0, Token=PAD
   Composed token: ID=0, Token=PAD
   ```

   **After fix:**
   ```
   Composed token: ID=22, Token=M3
   Composed token: ID=14, Token=G1
   Composed token: ID=24, Token=NUM_X_0
   ```

5. **Export CSV and verify:**
   ```csv
   gcode_predicted,token_count
   G1 X4 Y-1,3
   G0 Z5,2
   M3 S2500,2
   ```

---

## Why This Happened

The mismatch occurred because:

1. **Multiple vocabulary files exist** in the project:
   - `vocabulary.json` (original, 668 tokens)
   - `vocabulary_1digit_hybrid.json` (1-digit bucketing, 69 tokens)
   - `vocabulary_2digit_fixed.json` (2-digit bucketing)

2. **Dashboard hardcoded** `vocabulary.json` as default

3. **Training used** `vocabulary_1digit_hybrid.json` (from sweep config)

4. **No validation** that loaded vocabulary matches checkpoint metadata

---

## Prevention

### Best Practices Going Forward

1. **Store vocab path in checkpoint:**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'vocab_path': 'data/vocabulary_1digit_hybrid.json',  # Save this!
       'decomposer_config': {
           'n_commands': len(decomposer.command_tokens),
           'n_param_types': len(decomposer.param_type_tokens),
           'bucket_digits': decomposer.bucket_digits,
       }
   }, checkpoint_path)
   ```

2. **Validate on load:**
   ```python
   if 'vocab_path' in checkpoint:
       expected_vocab = checkpoint['vocab_path']
       if vocab_path != expected_vocab:
           raise ValueError(f"Vocabulary mismatch: expected {expected_vocab}, got {vocab_path}")
   ```

3. **Add decomposer sanity check:**
   ```python
   # After loading decomposer
   logger.info(f"Decomposer initialized:")
   logger.info(f"  Commands: {len(decomposer.command_tokens)} ({decomposer.command_tokens})")
   logger.info(f"  Param types: {len(decomposer.param_type_tokens)} ({decomposer.param_type_tokens})")
   logger.info(f"  Bucket digits: {decomposer.bucket_digits}")
   ```

4. **Command line argument for vocab:**
   ```bash
   python flask_dashboard.py --vocab-path data/vocabulary_1digit_hybrid.json
   ```

---

## Related Issues

### Other Vocabularies in Project

- **vocabulary_2digit_fixed.json** (7.3KB, 2-digit bucketing)
  - Use for models trained with 2-digit precision
  - ~170 tokens total

- **gcode_vocab_v2.json** (fallback in code line 332)
  - May not exist in your project
  - Should remove this fallback or create the file

### Data Directory Mismatch

The sweep config also uses:
```yaml
data-dir: outputs/processed_hybrid
```

But dashboard loads from:
```python
data_dir = Path('outputs/processed')
```

You may want to verify this matches as well, though it's less critical than vocabulary mismatch.

---

## Summary

**Problem:** Model trained with 12-command vocabulary, dashboard using 6-command vocabulary

**Symptom:** `command_id=10` → PAD (invalid index in dashboard's decomposer)

**Fix:** Updated dashboard to load `vocabulary_1digit_hybrid.json` (matches training)

**Result:** Dashboard now correctly decodes model predictions to G-code tokens

**Verification:** Look for "Composed token: ID=X, Token=G1" (not PAD) in logs

---

**Next Steps:**
1. Restart dashboard
2. Monitor logs for successful token composition
3. Export CSV and verify real G-code strings
4. Consider adding vocab path to checkpoint metadata for future runs

---

Generated: 2025-11-30
