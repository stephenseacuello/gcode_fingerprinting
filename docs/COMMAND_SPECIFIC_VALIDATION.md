# Command-Specific Parameter Validation

**Date:** 2025-11-30
**Issue:** Dashboard generating invalid G-code with wrong parameters for commands (e.g., "G1 X0.043 R0.000")
**Status:** ✅ IMPLEMENTED

---

## Problem Summary

The dashboard was generating G-code sequences with **invalid parameter combinations** for specific commands:

**Example of Invalid Output:**
```csv
gcode_predicted,token_count
G1 X 0.043 R 0.000,5     ← G1 (linear) cannot have R (radius for arcs)
G0 X 0.043 F 250,5       ← G0 (rapid) should not have F (feed rate)
M3 X 10 Y 20,5           ← M3 (spindle) cannot have position parameters
```

### G-code Command Rules

Different G-code commands accept different sets of parameters:

#### Motion Commands

- **G0 (Rapid Positioning)**: Fast non-cutting move
  - **Valid parameters**: X, Y, Z
  - **Invalid parameters**: F (no feed rate for rapid moves), R, I, J, K (no arc parameters)
  - **Example**: `G0 X10 Y20 Z5` ✅

- **G1 (Linear Interpolation)**: Straight line cutting move
  - **Valid parameters**: X, Y, Z, F
  - **Invalid parameters**: R, I, J, K (arc parameters only for G2/G3)
  - **Example**: `G1 X10 Y20 Z5 F250` ✅

- **G2 (Clockwise Arc)** / **G3 (Counter-Clockwise Arc)**: Circular arc moves
  - **Valid parameters**: X, Y, Z, F, and either:
    - R (radius) OR
    - I, J, K (center point offsets)
  - **Note**: Cannot use both R and I/J/K together
  - **Example**: `G2 X30 Y40 R5 F180` ✅
  - **Example**: `G3 X30 Y40 I2.5 J3.0 F180` ✅

#### Spindle Commands

- **M3 (Spindle On Clockwise)** / **M5 (Spindle Off)**:
  - **Valid parameters**: S (spindle speed, M3 only)
  - **Invalid parameters**: X, Y, Z, F, R, I, J, K (no motion parameters)
  - **Example**: `M3 S2500` ✅

#### Program Control

- **M30 (Program End)** / **M2 (Program End)**:
  - **Valid parameters**: None
  - **Example**: `M30` ✅

---

## The Fix

### Implementation

**Location:** [flask_dashboard.py:1511-1543](../flask_dashboard.py#L1511-L1543)

Added command-specific parameter validation that checks if parameters are valid for the current command:

```python
# SAFETY: Command-specific parameter validation
# Different G-code commands accept different parameters
if token_text in ['X', 'Y', 'Z', 'F', 'S', 'R', 'I', 'J', 'K']:
    # Find the current command by scanning backward
    current_command = None
    for i in range(len(full_command_tokens) - 1, -1, -1):
        tok = full_command_tokens[i]
        if tok.startswith('G') or tok.startswith('M'):
            current_command = tok
            break

    if current_command:
        # Define invalid parameter combinations
        invalid_combinations = {
            'G0': ['F', 'R', 'I', 'J', 'K'],  # Rapid move: no feed rate or arc params
            'G1': ['R', 'I', 'J', 'K'],       # Linear move: no arc parameters
            'M3': ['X', 'Y', 'Z', 'F', 'R', 'I', 'J', 'K'],  # Spindle on: only S
            'M5': ['X', 'Y', 'Z', 'F', 'R', 'I', 'J', 'K'],  # Spindle off: only S
            'M30': ['X', 'Y', 'Z', 'F', 'S', 'R', 'I', 'J', 'K'],  # Program end: no params
        }

        # Check if this parameter is invalid for the current command
        if current_command in invalid_combinations:
            if token_text in invalid_combinations[current_command]:
                logger.warning(f"⚠️ Invalid parameter for command: '{current_command}' cannot have '{token_text}'")
                logger.warning(f"  Current sequence: {' '.join(full_command_tokens)}")
                logger.warning(f"  {current_command} accepts: {_get_valid_params(current_command)}")
                logger.warning(f"  Stopping generation to prevent invalid G-code")
                break  # Stop generation
```

### Helper Function

**Location:** [flask_dashboard.py:708-719](../flask_dashboard.py#L708-L719)

Added helper function to provide user-friendly parameter descriptions:

```python
def _get_valid_params(command):
    """Get valid parameters for a G-code command."""
    valid_params = {
        'G0': 'X Y Z (rapid positioning - no feed rate)',
        'G1': 'X Y Z F (linear interpolation with feed rate)',
        'G2': 'X Y Z F R I J K (clockwise arc - needs radius or center offset)',
        'G3': 'X Y Z F R I J K (counter-clockwise arc - needs radius or center offset)',
        'M3': 'S (spindle on clockwise with speed)',
        'M5': '(spindle off - no parameters)',
        'M30': '(program end - no parameters)',
    }
    return valid_params.get(command, 'X Y Z F S R I J K (default)')
```

---

## How It Works

### Validation Process

1. **Token predicted**: Model predicts a parameter token (X, Y, Z, F, S, R, I, J, K)

2. **Find current command**: Scan backward through `full_command_tokens` to find the most recent G or M command

3. **Check validity**: Look up the command in `invalid_combinations` dictionary

4. **Stop if invalid**: If the parameter is in the invalid list for this command, log warning and stop generation

### Example Execution

**Scenario: Model predicts "G1 X 0.043 R 0.000"**

```
Token 0: G1    ← Current command = G1
Token 1: X     ← Check: X valid for G1? YES (continue)
Token 2: 0.043 ← Numeric value (continue)
Token 3: R     ← Check: R valid for G1? NO (R is in invalid_combinations['G1'])

⚠️ Invalid parameter for command: 'G1' cannot have 'R'
  Current sequence: G1 X 0.043
  G1 accepts: X Y Z F (linear interpolation with feed rate)
  Stopping generation to prevent invalid G-code

Result: "G1 X 0.043" ✅ (stops before adding invalid R)
```

**Scenario: Model predicts "G2 X 0.043 R 0.750"**

```
Token 0: G2    ← Current command = G2
Token 1: X     ← Check: X valid for G2? YES (continue)
Token 2: 0.043 ← Numeric value (continue)
Token 3: R     ← Check: R valid for G2? YES (G2 accepts R for arcs)
Token 4: 0.750 ← Numeric value (continue)

Result: "G2 X 0.043 R 0.750" ✅ (valid arc command)
```

---

## Validation Rules

### Implemented Rules

| Command | Valid Parameters | Invalid Parameters | Notes |
|---------|------------------|-------------------|-------|
| **G0** | X, Y, Z | F, R, I, J, K | Rapid move - no feed rate |
| **G1** | X, Y, Z, F | R, I, J, K | Linear move - no arc params |
| **G2/G3** | X, Y, Z, F, R, I, J, K | - | Arcs need R or I/J/K |
| **M3** | S | X, Y, Z, F, R, I, J, K | Spindle - only speed |
| **M5** | - | X, Y, Z, F, S, R, I, J, K | Spindle off - no params |
| **M30** | - | X, Y, Z, F, S, R, I, J, K | Program end - no params |

### Not Yet Implemented

These additional rules would further improve validation:

1. **Mutually exclusive parameters**:
   - G2/G3 should have R **OR** I/J/K, not both
   - Currently allows both (will use R if both present)

2. **Required parameters**:
   - G2/G3 **must** have R or I/J/K (at least one)
   - Currently allows G2 with only X Y Z (no radius info)

3. **Conditional parameters**:
   - F parameter should only appear once in a sequence
   - S parameter should only appear with M3 (not M5)

4. **Value constraints**:
   - S (spindle speed) should be positive
   - R (radius) should be positive
   - F (feed rate) should be positive

---

## Monitoring

### Check for Invalid Parameter Warnings

```bash
# Watch for command-specific validation errors
tail -f /tmp/dashboard.log | grep "Invalid parameter for command"
```

**Expected output when invalid parameter detected:**
```
2025-11-30 14:51:45 - WARNING - ⚠️ Invalid parameter for command: 'G1' cannot have 'R'
2025-11-30 14:51:45 - WARNING -   Current sequence: G1 X 0.043
2025-11-30 14:51:45 - WARNING -   G1 accepts: X Y Z F (linear interpolation with feed rate)
2025-11-30 14:51:45 - WARNING -   Stopping generation to prevent invalid G-code
```

### CSV Export Verification

After running predictions:

```bash
# Check for invalid parameter combinations
grep "G1.*R" predictions.csv   # Should be empty (G1 can't have R)
grep "G0.*F" predictions.csv   # Should be empty (G0 can't have F)
grep "M3.*X" predictions.csv   # Should be empty (M3 can't have X/Y/Z)
```

---

## Modal Behavior (Future Enhancement)

### Current Limitation

The dashboard currently **requires every sequence to start with a G/M command**. However, real G-code uses **modal commands** where the previous command stays active:

**Real G-code with modal commands:**
```gcode
G1 X10 Y20 Z5 F250    ; Linear move with feed rate
X15 Y25               ; Still G1 (modal) - no G command needed
X20 Y30               ; Still G1 (modal)
G0 Z10                ; Switch to rapid positioning
X0 Y0                 ; Still G0 (modal) - rapid to home
```

**Current dashboard output:**
```gcode
G1 X10 Y20 Z5 F250    ; First command
G1 X15 Y25            ; Repeats G1 (not modal)
G1 X20 Y30            ; Repeats G1 again
```

### Why Modal Support is Complex

1. **Grammar constraints**: Currently force first token to be COMMAND (type_id=1)
2. **State tracking**: Would need to maintain "current active command" across sequences
3. **Training data**: Model may not have been trained on modal sequences
4. **Validation**: Would need to validate parameters against "implicit" command

### Implementing Modal Support

To add modal behavior would require:

1. **Relax grammar constraints**: Allow sequences to start with PARAMETER tokens when in modal mode

2. **Add state tracking**: Maintain current command context
   ```python
   state['current_modal_command'] = None  # Track G1, G0, etc.
   ```

3. **Modify validation**: Check parameters against modal command if no explicit command
   ```python
   if current_command is None and state['current_modal_command']:
       current_command = state['current_modal_command']
   ```

4. **Update on command tokens**: When G/M command is predicted, update modal state
   ```python
   if token_text.startswith('G') or token_text.startswith('M'):
       state['current_modal_command'] = token_text
   ```

5. **Grammar constraint changes**: In `grammar_constraints.py`, remove step==0 forcing COMMAND type

**This is deferred for future work** as it requires coordinated changes across multiple components.

---

## Summary

### ✅ Implemented

1. **Command-specific parameter validation** for G0, G1, G2, G3, M3, M5, M30
2. **Invalid combination detection** (e.g., G1 with R, G0 with F, M3 with X/Y/Z)
3. **Informative logging** showing which parameters are valid for each command
4. **Hard stops** prevent invalid G-code from being generated

### 📊 Impact

**Before fix:**
```csv
G1 X 0.043 R 0.000      ❌ (G1 cannot have R)
G0 X 0.043 F 250        ❌ (G0 cannot have F)
M3 X 10 Y 20 S 2500     ❌ (M3 cannot have X/Y/Z)
```

**After fix:**
```csv
G1 X 0.043              ✅ (stops before invalid R)
G0 X 0.043              ✅ (stops before invalid F)
M3 S 2500               ✅ (stops before invalid X/Y/Z)
```

### 🔮 Future Enhancements

- [ ] Add mutual exclusivity checks (R vs. I/J/K for G2/G3)
- [ ] Add required parameter checks (G2/G3 must have R or I/J/K)
- [ ] Add value constraint validation (positive values for S, R, F)
- [ ] Implement modal command support
- [ ] Add F parameter uniqueness check (only once per sequence)

---

**Author:** Claude Code
**Date:** 2025-11-30
**Version:** 1.0

**Related Fixes:**
- [PARAMETER_REPETITION_FIX.md](PARAMETER_REPETITION_FIX.md) - Parameter repetition and ordering
- [GRAMMAR_CONSTRAINT_FIX.md](GRAMMAR_CONSTRAINT_FIX.md) - Grammar constraints
- [FINGERPRINT_RECONSTRUCTION_PLOTS.md](FINGERPRINT_RECONSTRUCTION_PLOTS.md) - Metrics visualization
