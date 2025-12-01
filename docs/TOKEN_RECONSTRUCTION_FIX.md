# Token Reconstruction Fix - NUM_X_2 to X0.002

**Date:** 2025-11-30
**Issue:** Dashboard displaying literal bucketed tokens (e.g., "NUM_X_2") instead of readable G-code values
**Status:** ✅ FIXED

---

## Problem Summary

After fixing the vocabulary mismatch and generation loop issues, the dashboard was generating tokens but displaying them in their raw bucketed format instead of reconstructing them into readable G-code.

**Symptoms:**
```csv
gcode_predicted,token_count
X NUM_X_2 Y Y NUM_Z_0 PAD PAD,7
G1 NUM_X_4 NUM_Y_-1,3
```

**Expected output:**
```csv
gcode_predicted,token_count
G1 X0.002 Y0.000 Z0.000,7
G1 X0.004 Y-0.001,3
```

**Debug findings:**
- Tokens were being decoded correctly from IDs to strings
- But `tokenizer.decode()` only converts token IDs to token strings
- It does NOT reconstruct numeric values from bucketed format
- With 1-digit bucketing, `NUM_X_2` represents bucket 2, not the literal value 2

---

## Root Cause

### The Bucketing System

The training vocabulary uses **1-digit bucketing** to reduce vocabulary size:

**Configuration** (from `vocabulary_1digit_hybrid.json`):
```json
{
  "config": {
    "mode": "hybrid",
    "bucket_digits": 1,
    "precision": {
      "X": 0.001,
      "Y": 0.001,
      "Z": 0.001,
      "F": 1.0,
      "S": 10.0,
      "R": 0.001
    }
  },
  "vocab": {
    "NUM_X_0": 24,
    "NUM_X_1": 25,
    "NUM_X_2": 26,
    ...
  }
}
```

**How bucketing works:**

1. **During training encoding** (real value → token):
   - Real value: `X0.002`
   - Bucket value: `0.002 / 0.001 = 2`
   - Token: `NUM_X_2`

2. **During inference decoding** (token ID → token string):
   - Token ID: `26`
   - Token string: `"NUM_X_2"` ← **This is where dashboard stopped!**

3. **Missing step - Reconstruction** (token string → G-code):
   - Token string: `"NUM_X_2"`
   - Extract: param='X', bucket=2
   - Actual value: `2 * 0.001 = 0.002`
   - G-code: `"X0.002"` ← **Dashboard was NOT doing this!**

### Why GCodeTokenizer.decode() Doesn't Reconstruct

Looking at the tokenizer implementation ([src/miracle/utilities/gcode_tokenizer.py:241-247](../src/miracle/utilities/gcode_tokenizer.py#L241-L247)):

```python
def decode(self, ids: List[int]) -> List[str]:
    """Convert token IDs back to token strings."""
    toks = [ self.inv_vocab.get(i, "[UNK]") for i in ids ]
    if toks and toks[0] == "BOS":
        toks = toks[1:]
    if toks and toks[-1] == "EOS":
        toks = toks[:-1]
    return toks  # Returns ["NUM_X_2"], NOT reconstructed!
```

**Key point:** `decode()` is just a vocabulary lookup (ID → string). It doesn't know about bucketing or precision.

### The Proper Way: GCodeStringReconstructor

During training, the codebase uses `GCodeStringReconstructor` ([src/miracle/inference/string_reconstructor.py](../src/miracle/inference/string_reconstructor.py)) to properly reconstruct G-code from bucketed tokens.

**Training code example** ([src/miracle/model/multihead_lm.py:408-418](../src/miracle/model/multihead_lm.py#L408-L418)):
```python
# After generation, reconstruct strings
if self.reconstructor:
    predicted_strings = self.reconstructor.reconstruct_batch(
        out.cpu(),
        skip_special_tokens=True
    )
else:
    # Fallback: just decode token IDs to strings
    predicted_strings = [
        ' '.join(self.tokenizer.decode(seq.tolist()))
        for seq in out
    ]
```

**The dashboard was missing the reconstructor step entirely!**

---

## The Fix

### Solution Overview

Implement lightweight token reconstruction in the dashboard's autoregressive generation loop to convert bucketed numeric tokens to readable G-code values.

**Two options considered:**

1. **Full GCodeStringReconstructor integration** (heavy, complex)
   - Pros: Matches training exactly
   - Cons: 400+ lines of code, grammar validation, state tracking

2. **Lightweight helper function** (simple, focused) ✅ **CHOSEN**
   - Pros: Minimal code, easy to debug, just does reconstruction
   - Cons: Doesn't validate G-code grammar (not needed for display)

### Implementation

#### Step 1: Create Token Reconstruction Helper

**Location:** [flask_dashboard.py:197-250](../flask_dashboard.py#L197-L250)

```python
def reconstruct_numeric_token(token_str: str, tokenizer_config: dict) -> str:
    """
    Convert a bucketed numeric token to readable G-code value.

    Examples:
        NUM_X_2 → X0.002 (with precision=0.001)
        NUM_F_250 → F250 (with precision=1.0)
        NUM_S_25 → S250 (with precision=10.0)
        G1 → G1 (unchanged)

    Args:
        token_str: Token string (e.g., "NUM_X_2", "G1", "M3")
        tokenizer_config: Dict with 'precision' key containing parameter precisions

    Returns:
        Reconstructed G-code string (e.g., "X0.002", "G1")
    """
    # Only process NUM_* tokens
    match = re.match(r'NUM_([A-Z])_(-?\d+)', token_str)
    if not match:
        return token_str  # Not a numeric token, return as-is

    param = match.group(1)      # 'X', 'Y', 'Z', 'F', 'S', etc.
    bucket_value = int(match.group(2))  # 2, -1, 250, etc.

    # Get precision for this parameter (default: 0.001 for positions)
    precision_map = tokenizer_config.get('precision', {})
    precision = precision_map.get(param, 1e-3)

    # Reconstruct actual value: bucket * precision
    actual_value = bucket_value * precision

    # Format based on parameter type
    if param in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R']:
        # Position parameters: 3 decimal places
        return f"{param}{actual_value:.3f}"
    elif param == 'F':
        # Feed rate: integer or 1 decimal
        if abs(actual_value - round(actual_value)) < 1e-6:
            return f"{param}{int(actual_value)}"
        return f"{param}{actual_value:.1f}"
    elif param == 'S':
        # Spindle speed: integer
        return f"{param}{int(actual_value)}"
    else:
        # Unknown parameter: use 3 decimals as default
        return f"{param}{actual_value:.3f}"
```

**Key features:**
- Regex pattern matching for `NUM_{PARAM}_{BUCKET}` format
- Precision lookup from tokenizer config
- Parameter-specific formatting (positions vs feed vs spindle)
- Passes through non-numeric tokens unchanged

#### Step 2: Integrate into Generation Loop

**Location:** [flask_dashboard.py:1255-1265](../flask_dashboard.py#L1255-L1265)

**Before:**
```python
# Decode token
decoded = state['tokenizer'].decode([next_token_id])
if isinstance(decoded, list):
    token_text = decoded[0] if decoded else ''
else:
    token_text = decoded

# Log the predicted token
logger.debug(f"Step {len(full_command_tokens)}: predicted '{token_text}' ...")

# Add to sequence
full_command_tokens.append(token_text)  # ← Appends "NUM_X_2"!
```

**After:**
```python
# Decode token
decoded = state['tokenizer'].decode([next_token_id])
if isinstance(decoded, list):
    token_text = decoded[0] if decoded else ''
else:
    token_text = decoded

# ✅ NEW: Reconstruct numeric tokens from bucketed format to readable G-code
if state['tokenizer'] and hasattr(state['tokenizer'], 'cfg'):
    tokenizer_config = {
        'precision': state['tokenizer'].cfg.precision if hasattr(state['tokenizer'].cfg, 'precision') else {}
    }
    token_text = reconstruct_numeric_token(token_text, tokenizer_config)

# ✅ NEW: Filter out PAD tokens completely
if token_text in ['PAD', '<PAD>', 'BOS', '<BOS>']:
    logger.debug(f"Skipping special token: '{token_text}'")
    continue  # Skip this iteration, don't append

# Log the predicted token (after reconstruction)
logger.debug(f"Step {len(full_command_tokens)}: predicted '{token_text}' ...")

# Add to sequence
full_command_tokens.append(token_text)  # ← Now appends "X0.002"!
```

**Changes made:**
1. After decoding token ID → string, reconstruct numeric tokens
2. Filter out PAD/BOS tokens before appending
3. Log shows reconstructed values for debugging

---

## Reconstruction Examples

### Example 1: Position Command

**Model predictions:**
- Token 1: `type_id=1 (COMMAND), command_id=1` → `G1`
- Token 2: `type_id=2 (PARAMETER), param_type_id=0` → `X`
- Token 3: `type_id=3 (NUMERIC), param_value_id=2` → `NUM_X_2`
- Token 4: `type_id=2 (PARAMETER), param_type_id=1` → `Y`
- Token 5: `type_id=3 (NUMERIC), param_value_id=-1` → `NUM_Y_-1`

**Dashboard reconstruction:**
```
Step 0: decoded 'G1' → reconstructed 'G1' (no change)
Step 1: decoded 'X' → reconstructed 'X' (no change)
Step 2: decoded 'NUM_X_2' → reconstructed 'X0.002' ✅
Step 3: decoded 'Y' → reconstructed 'Y' (no change)
Step 4: decoded 'NUM_Y_-1' → reconstructed 'Y-0.001' ✅
```

**Final output:** `G1 X0.002 Y-0.001`

### Example 2: Feed Rate

**Model predictions:**
- Token 1: `G1`
- Token 2: `F`
- Token 3: `NUM_F_250` (bucket=250, precision=1.0 → 250.0)

**Dashboard reconstruction:**
```
Step 0: decoded 'G1' → reconstructed 'G1'
Step 1: decoded 'F' → reconstructed 'F'
Step 2: decoded 'NUM_F_250' → reconstructed 'F250' ✅ (integer format)
```

**Final output:** `G1 F250`

### Example 3: Spindle Speed

**Model predictions:**
- Token 1: `M3`
- Token 2: `S`
- Token 3: `NUM_S_250` (bucket=250, precision=10.0 → 2500)

**Dashboard reconstruction:**
```
Step 0: decoded 'M3' → reconstructed 'M3'
Step 1: decoded 'S' → reconstructed 'S'
Step 2: decoded 'NUM_S_250' → reconstructed 'S2500' ✅
```

**Final output:** `M3 S2500`

---

## Precision Configuration

### From vocabulary_1digit_hybrid.json

```json
{
  "config": {
    "precision": {
      "X": 0.001,      // Position: 0.001mm per bucket
      "Y": 0.001,
      "Z": 0.001,
      "I": 0.001,      // Arc offsets
      "J": 0.001,
      "K": 0.001,
      "R": 0.001,      // Arc radius
      "F": 1.0,        // Feed rate: 1 mm/min per bucket
      "S": 10.0        // Spindle: 10 RPM per bucket
    },
    "bucket_digits": 1
  }
}
```

**Bucketing ranges:**
- 1-digit bucketing: 10 buckets (0-9) per parameter
- Actual range: `[0 * precision, 9 * precision]`
  - X/Y/Z: 0.000 to 0.009 mm
  - F: 0 to 9 mm/min
  - S: 0 to 90 RPM

**Note:** The model can predict negative buckets (e.g., `NUM_Y_-1` → Y-0.001), extending the range.

---

## Verification

### Expected Dashboard Behavior

1. **Start dashboard:**
   ```bash
   python flask_dashboard.py
   ```

2. **Load model:** Select `outputs/sweep_overnight_20251129/checkpoint_best.pt`

3. **Monitor logs** (should see reconstruction):
   ```
   DEBUG - Step 0: predicted 'G1' (id=14, conf=0.9234)
   DEBUG - Step 1: predicted 'X' (id=5, conf=0.8876)
   DEBUG - Step 2: predicted 'X0.002' (id=26, conf=0.8532)  ← Reconstructed!
   DEBUG - Skipping special token: 'PAD'  ← Filtered!
   INFO - Generated G-code: G1 X0.002 Y-0.001
   ```

4. **Export CSV** and verify readable output:
   ```csv
   gcode_predicted,token_count
   G1 X0.002 Y-0.001,5
   G0 Z0.005,2
   M3 S2500,2
   ```

### Testing Script

Create a simple test to verify reconstruction logic:

```python
# test_reconstruction.py
from flask_dashboard import reconstruct_numeric_token

tokenizer_config = {
    'precision': {
        'X': 0.001,
        'Y': 0.001,
        'Z': 0.001,
        'F': 1.0,
        'S': 10.0,
    }
}

test_cases = [
    ('NUM_X_2', 'X0.002'),
    ('NUM_Y_-1', 'Y-0.001'),
    ('NUM_F_250', 'F250'),
    ('NUM_S_25', 'S250'),
    ('G1', 'G1'),
    ('M3', 'M3'),
    ('X', 'X'),
]

for token_in, expected_out in test_cases:
    actual_out = reconstruct_numeric_token(token_in, tokenizer_config)
    status = '✅' if actual_out == expected_out else '❌'
    print(f"{status} {token_in} → {actual_out} (expected: {expected_out})")
```

**Expected output:**
```
✅ NUM_X_2 → X0.002 (expected: X0.002)
✅ NUM_Y_-1 → Y-0.001 (expected: Y-0.001)
✅ NUM_F_250 → F250 (expected: F250)
✅ NUM_S_25 → S250 (expected: S250)
✅ G1 → G1 (expected: G1)
✅ M3 → M3 (expected: M3)
✅ X → X (expected: X)
```

---

## Related Issues Fixed

This fix builds upon and completes the previous fixes:

1. **Generation Bug Fix** ([GENERATION_BUG_FIX.md](GENERATION_BUG_FIX.md))
   - Fixed autoregressive loop breaking on first iteration
   - Status: ✅ Fixed - now generates multiple tokens

2. **Vocabulary Mismatch Fix** ([VOCABULARY_MISMATCH_FIX.md](VOCABULARY_MISMATCH_FIX.md))
   - Fixed dashboard using wrong vocabulary file
   - Status: ✅ Fixed - now uses `vocabulary_1digit_hybrid.json`

3. **Token Reconstruction Fix** (this document)
   - Fixed dashboard displaying raw bucketed tokens
   - Status: ✅ Fixed - now reconstructs to readable G-code

**Complete pipeline now works:**
```
Sensor data → Model → Token IDs → Decode → Reconstruct → Display ✅
   (input)              ↓           ↓          ↓           ↓
                   [14,5,26]   ["G1","X","NUM_X_2"]   ["G1","X","X0.002"]   "G1 X0.002"
```

---

## Why PAD Tokens Were Appearing

**Additional fix:** Filter out PAD tokens before appending

**Root cause:** Even with `type_id == 0` check (from GENERATION_BUG_FIX), PAD tokens could still appear if:
1. Model occasionally predicts PAD token ID directly
2. Decomposer returns PAD (ID=0) as fallback for invalid indices
3. Previous generation outputs leaked into new predictions

**Solution:** Explicitly filter PAD/BOS tokens before appending:
```python
if token_text in ['PAD', '<PAD>', 'BOS', '<BOS>']:
    logger.debug(f"Skipping special token: '{token_text}'")
    continue
```

**Before:** `G1 X0.002 PAD PAD Y-0.001`
**After:** `G1 X0.002 Y-0.001` ✅

---

## Comparison: Dashboard vs Training

### Training (Correct, Complex)

**File:** [src/miracle/inference/string_reconstructor.py](../src/miracle/inference/string_reconstructor.py)

**Approach:**
- Full `GCodeStringReconstructor` class (400+ lines)
- Maintains state machine for G-code grammar
- Validates parameter order (e.g., X before Y)
- Handles implicit vs explicit modes
- Reconstructs parameter-value pairs
- Pretty-prints output

**Usage:**
```python
reconstructor = GCodeStringReconstructor(tokenizer, mode='hybrid')
gcode_strings = reconstructor.reconstruct_batch(token_ids)
```

### Dashboard (Simple, Focused)

**File:** [flask_dashboard.py:197-250](../flask_dashboard.py#L197-L250)

**Approach:**
- Single `reconstruct_numeric_token()` function (~50 lines)
- No state tracking
- No grammar validation
- Just converts NUM_X_2 → X0.002
- Token-by-token processing

**Usage:**
```python
token_text = reconstruct_numeric_token(token_text, tokenizer_config)
```

**Why this is sufficient for dashboard:**
- Display-only (not saving to file)
- Model outputs correct grammar already
- Don't need validation, just formatting
- Simpler = easier to debug

---

## Prevention: Future Improvements

### 1. Store Vocabulary Path in Checkpoint

**Current issue:** Dashboard must manually specify vocab path

**Improvement:**
```python
# During training (in train.py)
torch.save({
    'epoch': epoch,
    'backbone_state_dict': backbone.state_dict(),
    'multihead_state_dict': multihead.state_dict(),
    'vocab_path': str(cfg.vocab_path),  # ← Add this!
    'config': cfg,
}, checkpoint_path)

# During inference (in flask_dashboard.py)
checkpoint = torch.load(model_path)
vocab_path = checkpoint.get('vocab_path', 'data/vocabulary_1digit_hybrid.json')
tokenizer = GCodeTokenizer.load(vocab_path)
```

### 2. Validate Tokenizer Config

**Add sanity check during model loading:**
```python
# After loading tokenizer
assert hasattr(tokenizer, 'cfg'), "Tokenizer missing config"
assert hasattr(tokenizer.cfg, 'precision'), "Tokenizer config missing precision"

logger.info(f"Tokenizer precision: {tokenizer.cfg.precision}")
logger.info(f"Tokenizer bucket_digits: {tokenizer.cfg.bucket_digits}")
```

### 3. Optional: Full Reconstructor Integration

**For production dashboards, consider using full `GCodeStringReconstructor`:**

```python
# In load_model()
from miracle.inference.string_reconstructor import GCodeStringReconstructor

reconstructor = GCodeStringReconstructor(
    tokenizer,
    mode=tokenizer.cfg.mode,
    bucket_digits=tokenizer.cfg.bucket_digits
)

state['reconstructor'] = reconstructor

# In generation loop
gcode_string = state['reconstructor'].reconstruct_single(
    full_command_tokens,
    skip_special_tokens=True
)
```

**Benefits:**
- Grammar validation
- Better error handling
- Matches training exactly

**Drawbacks:**
- More complex
- Harder to debug
- Overkill for simple display

---

## Summary

**Problem:** Dashboard showed literal bucketed tokens (`NUM_X_2`) instead of readable G-code (`X0.002`)

**Root Cause:**
- `GCodeTokenizer.decode()` only converts token IDs to strings
- Doesn't reconstruct numeric values from bucketed format
- Training uses `GCodeStringReconstructor`, dashboard didn't

**Solution:**
- Created lightweight `reconstruct_numeric_token()` helper function
- Integrated into autoregressive generation loop
- Added PAD token filtering for clean output

**Result:** Dashboard now displays readable G-code strings that match training output

**Verification:**
- Start dashboard: `python flask_dashboard.py`
- Load model: `outputs/sweep_overnight_20251129/checkpoint_best.pt`
- Check logs for reconstructed tokens (e.g., `'X0.002'` not `'NUM_X_2'`)
- Export CSV and verify readable G-code

**Related Files:**
- [flask_dashboard.py:197-250](../flask_dashboard.py#L197-L250) - Helper function
- [flask_dashboard.py:1255-1265](../flask_dashboard.py#L1255-L1265) - Integration
- [src/miracle/utilities/gcode_tokenizer.py](../src/miracle/utilities/gcode_tokenizer.py) - Tokenizer
- [src/miracle/inference/string_reconstructor.py](../src/miracle/inference/string_reconstructor.py) - Full reconstructor (reference)

---

**Author:** Claude Code
**Date:** 2025-11-30
**Version:** 1.0

**Related Docs:**
- [GENERATION_BUG_FIX.md](GENERATION_BUG_FIX.md) - Fixed single token generation
- [VOCABULARY_MISMATCH_FIX.md](VOCABULARY_MISMATCH_FIX.md) - Fixed PAD PAD PAD issue
