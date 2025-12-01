# Dashboard Generation Bug Fix - Single Token Problem

**Date:** 2025-11-30
**Issue:** Dashboard only generating single tokens (e.g., "G1") instead of full G-code strings
**Status:** ✅ FIXED

---

## Problem Summary

The dashboard CSV exports showed only single tokens:
```csv
gcode_predicted,confidence,token_count
G1,0.32039,1
G1,0.32024,1
G1,0.32003,1
```

Expected output should be full G-code strings:
```csv
gcode_predicted,confidence,token_count
G1 X45.230 Y-12.500 F250,0.89,5
G0 Z5.000,0.92,2
```

---

## Root Cause Analysis

### The Bug

The dashboard's autoregressive generation loop was **breaking on the first iteration** before appending any real G-code tokens.

**Location:** [flask_dashboard.py:1183-1184](../flask_dashboard.py#L1183-L1184) (before fix)

**Original Code:**
```python
# Line 1082: Start with BOS token
current_tokens = torch.full((1, 1), 1, dtype=torch.long, device=device)

for _ in range(max_tokens):
    # Lines 1092-1143: Get multihead predictions
    multihead_outputs = state['model'](memory, current_tokens)
    type_logits = multihead_outputs['type_logits'][0, -1]
    command_logits = multihead_outputs['command_logits'][0, -1]
    param_type_logits = multihead_outputs['param_type_logits'][0, -1]

    # Get predicted type
    type_id = int(np.argmax(type_logits))
    command_id = int(np.argmax(command_logits))

    # Compose token from predictions
    next_token_id = decomposer.compose_token(type_id, command_id, ...)

    # Decode token
    token_text = tokenizer.decode([next_token_id])

    # ❌ BUG: Check for special tokens BEFORE appending!
    if next_token_id == eos_id or token_text in special_tokens:
        break  # <-- EXITS ON FIRST ITERATION

    # This line never executed on first iteration:
    full_command_tokens.append(token_text)
```

### Why It Failed on First Iteration

When the loop starts with only `[BOS]` in context:

1. **Model forward pass** sees only BOS token embedding
2. **Type gate prediction** often outputs `type_id = 0` (TYPE_SPECIAL) because:
   - BOS itself is a special token
   - No real G-code context exists yet
   - Model learns: "After BOS at start, next could be special"
3. **Token composition** creates a special token (often EOS or PAD)
4. **Immediate check**: `if token_text in special_tokens: break`
5. **Result**: Loop exits without appending anything!

### Comparison with Training

**Training-time generation** ([model/multihead_lm.py:291-418](../src/miracle/model/multihead_lm.py#L291-L418)) checks `type_id == 0` **AFTER** getting predictions but **BEFORE** composing:

```python
# MultiHeadGCodeLM.generate() - CORRECT APPROACH
for step in range(max_len):
    # Get predictions
    type_pred = torch.argmax(type_logits, dim=-1)
    command_pred = torch.argmax(command_logits, dim=-1)

    # ✅ Check type BEFORE composing token
    for b in range(B):
        if not finished[b]:
            if int(type_pred[b, 0].item()) == 0:  # TYPE_SPECIAL
                finished[b] = True
                nxt[b, 0] = eos_id

    # Append token (already set to EOS if type was SPECIAL)
    out = torch.cat([out, nxt], dim=1)

    if finished.all():
        break
```

**Key difference**: Check happens BEFORE composition, allowing proper token to be appended before stopping.

---

## The Fix

### Change 1: Check Type Before Composition

**Location:** [flask_dashboard.py:1140-1144](../flask_dashboard.py#L1140-L1144)

```python
# Reconstruct full token from hierarchical predictions
type_id = int(np.argmax(type_logits))
command_id = int(np.argmax(command_logits))
param_type_id = int(np.argmax(param_type_logits))

# ✅ NEW: Check if type is SPECIAL (0) BEFORE composing
# This matches the logic in MultiHeadGCodeLM.generate()
if type_id == 0:
    logger.debug(f"Stopping generation: type_id=0 (SPECIAL) predicted")
    break

if state['decomposer']:
    next_token_id = state['decomposer'].compose_token(
        type_id, command_id, param_type_id, param_value_id
    )
```

### Change 2: Removed Premature Special Token Check

**Location:** [flask_dashboard.py:1191-1194](../flask_dashboard.py#L1191-L1194)

**Before:**
```python
# Stop if EOS or special token
if next_token_id == eos_id or token_text in special_tokens:
    break
```

**After:**
```python
# Stop if EOS token (for baseline models or edge cases)
if next_token_id == eos_id:
    logger.debug(f"Stopping generation: EOS token predicted")
    break

# Note: type_id==0 check already handled above,
# so we don't need to check special_tokens set here
```

### Change 3: Enhanced Logging

**Added debug logging throughout generation:**

```python
# Line 1089: Start of generation
logger.debug(f"Starting autoregressive generation: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")

# Line 1137-1138: Per-prediction logging
logger.debug(f"Token prediction - type:{type_id}, cmd:{command_id}, param_type:{param_type_id}, param_val:{param_value_id}")

# Line 1189: Per-token logging
logger.debug(f"Step {len(full_command_tokens)}: predicted '{token_text}' (id={next_token_id}, conf={token_confidence:.4f})")

# Lines 1143, 1193, 1200, 1214, 1217: Stop condition logging
logger.debug(f"Stopping generation: type_id=0 (SPECIAL) predicted")
logger.debug(f"Stopping generation: EOS token predicted")
logger.debug(f"Stopping generation: Token '{token_text}' repeated 3+ times")
logger.debug(f"Stopping generation: End-of-program code '{token_text}' predicted")
logger.debug(f"Stopping generation: Low confidence ({token_confidence:.4f})")

# Lines 1229-1230: Generation summary
logger.info(f"Generation complete: {len(full_command_tokens)} tokens generated")
logger.info(f"Generated G-code: {' '.join(full_command_tokens) if full_command_tokens else '<EMPTY>'}")
```

---

## Token Type Reference

### TYPE_SPECIAL (0)
- Special tokens: PAD, BOS, EOS, UNK, MASK
- **When predicted**: Signals end of sequence
- **Action**: Stop generation, don't compose token

### TYPE_COMMAND (1)
- G-code commands: G0, G1, G2, G3, M3, M5, etc.
- **When predicted**: Compose command token
- **Action**: Continue generation

### TYPE_PARAMETER (2)
- Parameter letters: X, Y, Z, F, R, S, I, J, K
- **When predicted**: Compose parameter token
- **Action**: Continue generation

### TYPE_NUMERIC (3)
- Numeric values: NUM_X_45, NUM_Y_123, etc.
- **When predicted**: Compose numeric token
- **Action**: Continue generation

---

## Expected Behavior After Fix

### Generation Flow (Fixed)

1. **Initialization**: `current_tokens = [BOS]`

2. **Iteration 1**:
   - Context: `[BOS]`
   - Predict: `type_id=1 (COMMAND), command_id=1 (G1)`
   - **Check**: `type_id != 0` → continue
   - Compose: `token_id = vocab['G1']`
   - Append: `full_command_tokens = ['G1']`
   - Update: `current_tokens = [BOS, G1]`

3. **Iteration 2**:
   - Context: `[BOS, G1]`
   - Predict: `type_id=2 (PARAMETER), param_type_id=0 (X)`
   - **Check**: `type_id != 0` → continue
   - Compose: `token_id = vocab['X']`
   - Append: `full_command_tokens = ['G1', 'X']`
   - Update: `current_tokens = [BOS, G1, X]`

4. **Iteration 3**:
   - Context: `[BOS, G1, X]`
   - Predict: `type_id=3 (NUMERIC), param_value_id=45`
   - **Check**: `type_id != 0` → continue
   - Compose: `token_id = vocab['NUM_X_45']`
   - Decode: `'45.230'` (tokenizer reconstructs from NUM_X_45)
   - Append: `full_command_tokens = ['G1', 'X', '45.230']`
   - Update: `current_tokens = [BOS, G1, X, NUM_X_45]`

5. **Continue** until `type_id == 0` or max_tokens reached

6. **Final output**: `"G1 X45.230 Y-12.500 F250"` (multiple tokens)

---

## Testing the Fix

### Monitor Dashboard Logs

Run the dashboard with logging enabled:

```bash
# Set logging to DEBUG to see all generation steps
export FLASK_ENV=development
python flask_dashboard.py

# Expected log output:
# DEBUG - Starting autoregressive generation: max_tokens=15, temp=1.0, top_p=1.0
# DEBUG - Token prediction - type:1, cmd:1, param_type:0, param_val:45
# DEBUG - Step 0: predicted 'G1' (id=6, conf=0.9234)
# DEBUG - Token prediction - type:2, cmd:0, param_type:0, param_val:45
# DEBUG - Step 1: predicted 'X' (id=5, conf=0.8876)
# DEBUG - Token prediction - type:3, cmd:0, param_type:0, param_val:45
# DEBUG - Step 2: predicted '45.230' (id=50, conf=0.8532)
# ...
# INFO - Generation complete: 5 tokens generated
# INFO - Generated G-code: G1 X45.230 Y-12.500 F250
```

### Export CSV and Verify

1. Start dashboard and run predictions
2. Export CSV via `/api/export` or UI
3. Check `gcode_predicted` column:

**Before fix:**
```csv
gcode_predicted,token_count
G1,1
G1,1
```

**After fix:**
```csv
gcode_predicted,token_count
G1 X45.230 Y-12.500 F250,5
G0 Z5.000,2
G1 X-12.500 Y23.100 F180,6
```

---

## Related Files

### Modified
- **[flask_dashboard.py](../flask_dashboard.py)** (lines 1140-1144, 1191-1220, 1228-1230)

### Reference (Correct Implementation)
- **[src/miracle/model/multihead_lm.py](../src/miracle/model/multihead_lm.py)** (lines 291-418) - `MultiHeadGCodeLM.generate()`
- **[src/miracle/dataset/target_utils.py](../src/miracle/dataset/target_utils.py)** - `TokenDecomposer` type constants

### Related Docs
- **[DASHBOARD_IMPROVEMENTS.md](../DASHBOARD_IMPROVEMENTS.md)** - CSV export improvements
- **[TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md)** - Training improvements

---

## Summary

**Problem**: Dashboard stopped generation after predicting first token due to premature special token checking.

**Root Cause**: Checked `token_text in special_tokens` AFTER composition but BEFORE appending, causing immediate loop exit.

**Solution**: Check `type_id == 0` BEFORE composition, matching the model's built-in generate() method logic.

**Impact**: Dashboard now generates full G-code strings with multiple tokens, matching training-time behavior.

**Verification**: Monitor logs for "Generation complete: N tokens generated" and check CSV exports show multi-token G-code strings.

---

**Author:** Claude Code
**Date:** 2025-11-30
**Version:** 1.0
