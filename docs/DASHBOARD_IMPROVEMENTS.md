# Dashboard G-Code Reconstruction Improvements

## Summary

Fixed the dashboard CSV export to show full reconstructed G-code strings (e.g., `G1 X45.230 Y-12.500 F250`) instead of single tokens (e.g., `Z`).

## Problem Identified

The dashboard CSV export was using `gcode_text` (single token prediction) instead of `full_command` (full autoregressive generation), causing exports to show only partial predictions like "Z" instead of complete G-code strings.

## Changes Made

### 1. **Fixed Standard CSV Export** ([flask_dashboard.py:2376-2389](flask_dashboard.py#L2376-L2389))

**Before:**
```python
'gcode_predicted': pred['gcode_text'],  # Single token only
'confidence': pred['gcode_confidence'],
```

**After:**
```python
# Use full_command (autoregressive generation) if available
gcode_output = pred.get('full_command', pred.get('gcode_text', '<EMPTY>'))
full_conf = pred.get('full_command_confidence', pred.get('gcode_confidence', 0.0))

export_data.append({
    'gcode_predicted': gcode_output,  # Full G-code string
    'confidence': full_conf,
    'operation_type': pred.get('operation_type', 'unknown'),
    'operation_confidence': pred.get('operation_confidence', 0.0),
    'token_count': len(gcode_output.split()),
})
```

**Benefits:**
- Exports now show complete G-code commands
- Includes operation type and confidence
- Counts tokens in generated sequence

---

### 2. **Added Detailed CSV Export Endpoint** ([flask_dashboard.py:2407-2466](flask_dashboard.py#L2407-L2466))

New API endpoint: `/api/export/detailed`

**Features:**
- **Full G-code reconstruction** with autoregressive generation
- **Token-by-token breakdown** with individual confidence scores
- **Multi-head prediction details** (type, command, param_type, param_value)
- **Operation classification** (adaptive, face, pocket, etc.)
- **Per-token analysis** - up to 10 tokens with individual confidences

**CSV Structure:**
```csv
index,timestamp,operation_type,operation_confidence,gcode_predicted,full_confidence,token_count,avg_token_confidence,token_1,token_1_conf,token_2,token_2_conf,...
0,2025-11-30T12:34:56,face,0.95,G1 X45.230 Y-12.500 F250,0.89,5,0.91,G1,0.95,X45.230,0.88,...
```

**Usage:**
```bash
# From the dashboard UI
# Navigate to Export → Download Detailed CSV

# Or via API
curl http://localhost:5000/api/export/detailed --output detailed_export.csv
```

---

### 3. **Added Tokenizer Configuration Endpoint** ([flask_dashboard.py:2625-2688](flask_dashboard.py#L2625-L2688))

New API endpoint: `/api/tokenizer_info`

**Returns:**
- **Vocabulary size** and tokenization mode (literal/split/hybrid)
- **Precision settings** for X, Y, Z, F, S parameters
- **Bucketing configuration** (bucket_digits, max_bucket_value)
- **Vocabulary breakdown** by category:
  - Special tokens (PAD, BOS, EOS, UNK, MASK)
  - Command tokens (G0, G1, G2, M3, M5, etc.)
  - Parameter tokens (X, Y, Z, F, R, S, etc.)
  - Numeric tokens (NUM_X_015, NUM_Y_123, etc.)
- **Decomposer details** (for multi-head models):
  - Number of commands, parameter types, parameter values
  - Bucketing strategy
  - Example tokens from each category

**Example Response:**
```json
{
  "vocab_size": 170,
  "mode": "hybrid",
  "precision": {
    "X": 0.001,
    "Y": 0.001,
    "Z": 0.001,
    "F": 1.0,
    "S": 10.0
  },
  "bucket_digits": 2,
  "max_bucket_value": 100,
  "vocab_breakdown": {
    "special_count": 5,
    "command_count": 15,
    "parameter_count": 10,
    "numeric_count": 140,
    "examples": {
      "commands": ["G0", "G1", "G2", "G3", "M3", "M5"],
      "parameters": ["X", "Y", "Z", "F", "R", "S"],
      "numeric_samples": ["NUM_X_015", "NUM_Y_023", "NUM_Z_100"]
    }
  },
  "decomposer": {
    "n_commands": 15,
    "n_param_types": 10,
    "n_param_values": 100,
    "bucket_digits": 2,
    "command_examples": ["G0", "G1", "G2", "G3"],
    "param_type_examples": ["X", "Y", "Z", "F"]
  }
}
```

**Usage:**
```bash
# Query tokenizer config
curl http://localhost:5000/api/tokenizer_info | jq
```

---

## Understanding the Output

### Single Token vs Full Command

The dashboard now distinguishes between:

1. **Token-Level Prediction** (`gcode_text`)
   - Single-step prediction from the model
   - Shows next most likely token
   - Useful for debugging individual predictions
   - Example: `Z` or `G1` or `X45.230`

2. **Full Command Generation** (`full_command`)
   - Autoregressive sequence generation
   - Reconstructs complete G-code lines
   - Uses beam search or greedy/sampling strategies
   - Example: `G1 X45.230 Y-12.500 F250`

### Bucketing Strategy

The tokenizer uses **bucketing** to reduce vocabulary size for numeric values:

- **2-digit bucketing** (bucket_digits=2): Values grouped by first 2 digits
  - Example: `45.230` → `NUM_X_45`, `45.789` → `NUM_X_45`
  - Vocabulary size: ~100 values per parameter

- **3-digit bucketing** (bucket_digits=3): More precise grouping
  - Example: `45.230` → `NUM_X_452`, `45.789` → `NUM_X_457`
  - Vocabulary size: ~1000 values per parameter

This trades off precision for vocabulary efficiency, reducing the total token count from ~10,000 to ~170.

### Multi-Head Architecture

For multi-head models, predictions are decomposed into hierarchical components:

1. **Type Gate** (4 classes): Special, Command, Parameter, Numeric
2. **Command Head** (~15 classes): G0, G1, G2, G3, M3, M5, etc.
3. **Parameter Type Head** (~10 classes): X, Y, Z, F, R, S, I, J, K
4. **Parameter Value Head** (regression or bucketed): Actual numeric value

These are **composed** back into full tokens during generation using the `TokenDecomposer`.

---

## Verification

### Check Current Exports

1. **Standard Export** - Now shows full commands:
```csv
index,timestamp,gcode_predicted,confidence,anomaly_score,operation_type,operation_confidence,token_count
0,2025-11-30T06:56:35,G1 X45.230 Y-12.500 F250,0.89,0.0,face,0.95,5
```

2. **Detailed Export** - Includes token breakdown:
```csv
index,gcode_predicted,token_count,token_1,token_1_conf,token_2,token_2_conf,...
0,G1 X45.230 Y-12.500 F250,5,G1,0.95,X45.230,0.88,Y-12.500,0.91,...
```

### Compare with Training Output

During training, you should see similar string reconstruction using:
- [src/miracle/inference/string_reconstructor.py](src/miracle/inference/string_reconstructor.py) - `GCodeStringReconstructor` class
- [src/miracle/training/train.py](src/miracle/training/train.py) - String metrics computation

The dashboard now uses the **same autoregressive generation** as training, ensuring consistency.

---

## Next Steps

### Recommended Actions

1. **Test the improved export:**
   ```bash
   python flask_dashboard.py
   # Navigate to http://localhost:5000
   # Generate some predictions
   # Click Export → Download CSV
   # Verify full G-code strings appear
   ```

2. **Compare training vs dashboard output:**
   - Run a training session and capture G-code string outputs
   - Run the dashboard and export predictions
   - Verify both show similar reconstruction quality

3. **Adjust generation settings if needed:**
   ```python
   # In the dashboard UI or via API
   POST /api/settings
   {
     "enable_autoregressive": true,
     "max_tokens": 15,
     "temperature": 1.0,
     "top_p": 1.0,
     "use_beam_search": false,
     "beam_size": 3
   }
   ```

4. **Monitor bucketing accuracy:**
   - Check `/api/tokenizer_info` to see bucketing configuration
   - If numeric accuracy is poor, consider:
     - Increasing `bucket_digits` (e.g., 2 → 3)
     - Using regression head instead of classification
     - Adjusting precision settings in tokenizer config

---

## Technical Details

### File Modifications

1. **[flask_dashboard.py](flask_dashboard.py)**
   - Line 2376-2389: Updated standard CSV export
   - Line 2407-2466: Added detailed CSV export endpoint
   - Line 2625-2688: Added tokenizer info endpoint

### API Endpoints Added

- `GET /api/export/detailed` - Detailed CSV with token breakdown
- `GET /api/tokenizer_info` - Tokenizer configuration and vocab stats

### Backward Compatibility

- Original `/api/export` endpoint still works
- Fallback to `gcode_text` if `full_command` not available
- No changes required to existing dashboard UI

---

## Troubleshooting

### Issue: Still seeing single tokens in export

**Check:**
1. Is `enable_autoregressive` set to `true`?
   ```bash
   curl http://localhost:5000/api/settings
   ```
2. Are predictions in history actually running autoregressive generation?
   - Check dashboard console for "FULL COMMAND GENERATION" logs

**Fix:**
```bash
# Enable autoregressive generation
curl -X POST http://localhost:5000/api/settings \
  -H "Content-Type: application/json" \
  -d '{"enable_autoregressive": true}'
```

### Issue: Bucketing seems inaccurate

**Check:**
```bash
curl http://localhost:5000/api/tokenizer_info | jq '.bucket_digits'
```

**Understanding:**
- `bucket_digits=2`: Groups to first 2 digits (45.230 → 45)
- `bucket_digits=3`: Groups to first 3 digits (45.230 → 452)

Bucketing intentionally reduces precision to shrink vocabulary size. If you need higher precision, consider:
1. Using regression head for param_value (continuous output)
2. Increasing bucket_digits in tokenizer config
3. Using "split" mode instead of "hybrid" mode

### Issue: Generation produces repetitive tokens

**Check:**
- The dashboard has repetition detection (breaks after 3 identical tokens)
- Adjust temperature to increase diversity:
  ```bash
  curl -X POST http://localhost:5000/api/settings \
    -H "Content-Type: application/json" \
    -d '{"temperature": 1.2}'  # Increase from 1.0
  ```

---

## Related Code

- **Tokenization**: [src/miracle/utilities/gcode_tokenizer.py](src/miracle/utilities/gcode_tokenizer.py)
- **Token Decomposition**: [src/miracle/dataset/target_utils.py](src/miracle/dataset/target_utils.py)
- **Multi-Head Model**: [src/miracle/model/multihead_lm.py](src/miracle/model/multihead_lm.py)
- **String Reconstruction**: [src/miracle/inference/string_reconstructor.py](src/miracle/inference/string_reconstructor.py)
- **Training**: [src/miracle/training/train.py](src/miracle/training/train.py)

---

Generated: 2025-11-30
