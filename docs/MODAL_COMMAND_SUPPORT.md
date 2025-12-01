# Modal Command Support

**Date:** 2025-11-30
**Feature:** Support for modal G-code commands in dashboard generation
**Status:** ✅ IMPLEMENTED

---

## Overview

Added support for **modal G-code commands** - a standard feature in real G-code where commands remain active across multiple sequences until changed.

### What are Modal Commands?

In G-code programming, most motion commands are **modal**, meaning they stay active until explicitly changed:

**Real G-code with modal commands:**
```gcode
G1 X10 Y20 Z5 F250    ; Linear move with feed rate
X15 Y25               ; Still G1 (modal) - no G command needed
X20 Y30               ; Still G1 (modal)
G0 Z10                ; Switch to rapid positioning
X0 Y0                 ; Still G0 (modal) - rapid to home
```

**Without modal support (previous behavior):**
```gcode
G1 X10 Y20 Z5 F250    ; First command
G1 X15 Y25            ; Repeats G1 (not modal)
G1 X20 Y30            ; Repeats G1 again
```

### Modal Commands in G-code

The following commands are modal (stay active):
- **G0**: Rapid positioning (modal)
- **G1**: Linear interpolation (modal)
- **G2**: Clockwise arc (modal)
- **G3**: Counter-clockwise arc (modal)
- **G17/G18/G19**: Plane selection (modal)

The following are **non-modal** (must be repeated):
- **M3/M5**: Spindle control (non-modal)
- **M30/M2**: Program end (non-modal)

---

## Implementation

### 1. State Tracking

**Location:** [flask_dashboard.py:120](../flask_dashboard.py#L120)

Added `current_modal_command` to global state to track the active modal command:

```python
state = {
    # ... other state fields ...
    'current_modal_command': None,  # Track current modal G-code command (e.g., 'G1', 'G0')
}
```

### 2. Grammar Constraints Initialization

**Location:** [flask_dashboard.py:460](../flask_dashboard.py#L460)

Enabled modal command support in grammar constraints:

```python
grammar_constraints = GCodeGrammarConstraints(
    decomposer.vocab,
    device=device,
    decomposer=decomposer,
    allow_modal_commands=True  # Enable modal G-code behavior
)
logger.info(f"✅ Grammar constraints initialized on {device} (modal commands enabled)")
```

### 3. Passing Modal Context to Grammar Constraints

**Location:** [flask_dashboard.py:1234-1242](../flask_dashboard.py#L1234-L1242)

During generation, pass the current modal command to grammar constraints:

```python
# Pass modal command context to grammar constraints
# This allows sequences to start with parameters if we have a modal command active
modal_cmd = state.get('current_modal_command')
if modal_cmd and len(full_command_tokens) == 0:
    logger.info(f"[Modal Mode] Using modal command context: {modal_cmd}")

multihead_outputs = state['grammar_constraints'].apply_inference_constraints(
    multihead_outputs,
    current_tokens,
    step=len(full_command_tokens),
    modal_command=modal_cmd  # Pass modal command for context
)
```

### 4. Tracking Modal Commands

**Location:** [flask_dashboard.py:1573-1578](../flask_dashboard.py#L1573-L1578)

Update modal command state when a command token is predicted:

```python
# Track modal command: Update when a G or M command is predicted
# This allows future sequences to use this command as context
# Example: After "G1 X10 Y20", next sequence "X15 Y25" implicitly uses G1
if token_text.startswith('G') or token_text.startswith('M'):
    state['current_modal_command'] = token_text
    logger.debug(f"[Modal] Command '{token_text}' is now active")
```

### 5. Modal Command Validation

**Location:** [flask_dashboard.py:1549-1552](../flask_dashboard.py#L1549-L1552)

Use modal command for parameter validation when no explicit command in sequence:

```python
# Modal mode: If no explicit command in sequence, use modal command context
if current_command is None and state.get('current_modal_command'):
    current_command = state['current_modal_command']
    logger.debug(f"[Modal] Using modal command for validation: {current_command}")
```

---

## Grammar Constraints Implementation

**Location:** [src/miracle/training/grammar_constraints.py:41-53](../src/miracle/training/grammar_constraints.py#L41-L53)

Added `allow_modal_commands` parameter:

```python
def __init__(self, vocab, device='cpu', decomposer=None, allow_modal_commands=False):
    """
    Args:
        vocab: Vocabulary dictionary mapping tokens to IDs
        device: torch device
        decomposer: TokenDecomposer instance for type checking (optional)
        allow_modal_commands: If True, allows sequences to start with parameters
                             instead of commands (modal G-code behavior)
    """
    self.vocab = vocab
    self.device = device
    self.decomposer = decomposer
    self.allow_modal_commands = allow_modal_commands
```

**Location:** [src/miracle/training/grammar_constraints.py:727-776](../src/miracle/training/grammar_constraints.py#L727-L776)

Modified constraint logic to allow PARAMETER tokens at step 0 when modal mode enabled:

```python
def apply_inference_constraints(
    self,
    logits: Dict[str, torch.Tensor],
    current_sequence: torch.Tensor,
    step: int,
    modal_command: str = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply hard constraints during inference by masking invalid predictions.

    Args:
        logits: Model output logits
        current_sequence: Current generated sequence [B, step]
        step: Current generation step
        modal_command: Optional modal command context (e.g., 'G1' if previous command was G1)
                      Used when allow_modal_commands=True to allow parameter-only sequences
    """
    # ...

    # First token handling: Allow COMMAND or PARAMETER (if modal mode enabled)
    if step == 0:
        if 'type_logits' in constrained_logits:
            type_logits = constrained_logits['type_logits']  # [B, 1, 4]

            # Modal mode: Allow PARAMETER tokens if we have a modal command context
            if self.allow_modal_commands and modal_command is not None:
                # Allow both COMMAND and PARAMETER types
                type_logits[:, :, 0] -= 100.0  # Suppress TYPE_SPECIAL (PAD, BOS, EOS)
                type_logits[:, :, 3] -= 100.0  # Suppress TYPE_NUMERIC (must start with param letter)
                # Boost TYPE_COMMAND and TYPE_PARAMETER equally
                type_logits[:, :, 1] += 5.0   # TYPE_COMMAND
                type_logits[:, :, 2] += 5.0   # TYPE_PARAMETER
            else:
                # Standard mode: First token MUST be a command (type_id=1)
                type_logits[:, :, 0] -= 100.0  # Suppress TYPE_SPECIAL (PAD, BOS, EOS)
                type_logits[:, :, 2] -= 100.0  # Suppress TYPE_PARAMETER (X, Y, Z)
                type_logits[:, :, 3] -= 100.0  # Suppress TYPE_NUMERIC (NUM_X_2)
                # Boost TYPE_COMMAND (G0, G1, M3, etc.)
                type_logits[:, :, 1] += 10.0

            constrained_logits['type_logits'] = type_logits
        return constrained_logits
```

---

## How It Works

### Normal Mode (No Modal Context)

**Sequence 1:**
```
Step 0: Grammar forces COMMAND → Predicts G1
Step 1: Grammar allows PARAMETER → Predicts X
Step 2: Grammar forces NUMERIC → Predicts 1200
Step 3: Grammar allows PARAMETER → Predicts Y
Step 4: Grammar forces NUMERIC → Predicts 0043
Step 5: Grammar allows SPECIAL (EOS) → Generation stops

Result: "G1 X1200 Y0043"
Modal state updated: current_modal_command = 'G1'
```

### Modal Mode (With Modal Context)

**Sequence 2 (after G1 was predicted in Sequence 1):**
```
Step 0: Grammar allows PARAMETER (modal_command='G1') → Predicts X
Step 1: Grammar forces NUMERIC → Predicts 1500
Step 2: Grammar allows PARAMETER → Predicts Y
Step 3: Grammar forces NUMERIC → Predicts 0025
Step 4: Grammar allows SPECIAL (EOS) → Generation stops

Result: "X1500 Y0025" (implicitly uses G1)
Modal state unchanged: current_modal_command = 'G1'
```

**Sequence 3 (switching to rapid positioning):**
```
Step 0: Grammar allows COMMAND or PARAMETER → Predicts G0 (model chooses to switch)
Step 1: Grammar allows PARAMETER → Predicts Z
Step 2: Grammar forces NUMERIC → Predicts 0010
Step 3: Grammar allows SPECIAL (EOS) → Generation stops

Result: "G0 Z0010"
Modal state updated: current_modal_command = 'G0'
```

---

## Benefits

### 1. More Natural G-code
Generated sequences match real G-code behavior where commands don't need to be repeated.

**Before:**
```gcode
G1 X10 Y20 F250
G1 X15 Y25 F250
G1 X20 Y30 F250
```

**After:**
```gcode
G1 X10 Y20 F250
X15 Y25
X20 Y30
```

### 2. Shorter Sequences
Parameter-only sequences are more compact, reducing token count.

### 3. Context Awareness
Model can learn that certain operations naturally follow each other without explicit commands.

### 4. Validation Still Works
Parameter validation uses modal command context:
- `X Y Z` sequence after `G1` → validates against G1 rules
- `X Y Z` sequence after `G0` → validates against G0 rules (no F allowed)

---

## Monitoring

### Check Modal Mode Activation

```bash
# Watch for modal mode messages in logs
tail -f /tmp/dashboard.log | grep "\[Modal"
```

**Expected output:**
```
2025-11-30 15:30:12 - INFO - [Modal Mode] Using modal command context: G1
2025-11-30 15:30:12 - DEBUG - [Modal] Command 'G1' is now active
2025-11-30 15:30:15 - DEBUG - [Modal] Using modal command for validation: G1
2025-11-30 15:30:20 - DEBUG - [Modal] Command 'G0' is now active
```

### Check Grammar Constraints Initialization

```bash
# Verify modal mode enabled
tail -f /tmp/dashboard.log | grep "Grammar constraints initialized"
```

**Expected output:**
```
2025-11-30 15:25:30 - INFO - ✅ Grammar constraints initialized on cpu (modal commands enabled)
```

---

## Example Scenarios

### Scenario 1: Linear Moves with Modal G1

**Prediction 1:**
```
Output: G1 X10 Y20 Z5 F250
Modal state: G1
```

**Prediction 2:**
```
Output: X15 Y25        (implicitly G1)
Modal state: G1        (unchanged)
Validation: Parameters validated against G1 rules
```

**Prediction 3:**
```
Output: X20 Y30        (implicitly G1)
Modal state: G1        (unchanged)
```

### Scenario 2: Switching from G1 to G0

**Prediction 1:**
```
Output: G1 X10 Y20 F250
Modal state: G1
```

**Prediction 2:**
```
Output: G0 Z10         (explicit switch to rapid)
Modal state: G0        (updated)
```

**Prediction 3:**
```
Output: X0 Y0          (implicitly G0)
Modal state: G0        (unchanged)
Validation: F parameter rejected (G0 doesn't allow F)
```

### Scenario 3: Arc Moves with Modal G2

**Prediction 1:**
```
Output: G2 X30 Y40 R5 F180
Modal state: G2
```

**Prediction 2:**
```
Output: X35 Y45 R3     (implicitly G2)
Modal state: G2        (unchanged)
Validation: R parameter allowed (G2 accepts arc parameters)
```

---

## Current Limitations

### 1. Non-Modal Commands Not Implemented

Currently, **all** commands are treated as modal. In real G-code:
- Modal commands: G0, G1, G2, G3, G17, G18, G19
- Non-modal commands: M3, M5, M30, M2

**Enhancement needed:**
```python
NON_MODAL_COMMANDS = {'M3', 'M5', 'M30', 'M2'}

if token_text.startswith('G') or token_text.startswith('M'):
    # Only update modal state for modal commands
    if token_text not in NON_MODAL_COMMANDS:
        state['current_modal_command'] = token_text
```

### 2. Modal State Not Reset

Modal command persists across all predictions. In real machining:
- Modal state resets on program restart
- Some controllers reset modal state after M30 (program end)

**Enhancement needed:**
```python
# Reset modal state on program end
if token_text in ['M30', 'M2']:
    state['current_modal_command'] = None
    logger.debug(f"[Modal] Reset modal command on program end")
```

### 3. Plane Selection Not Tracked

G17/G18/G19 (plane selection) are also modal but not currently tracked:
- G17: XY plane (most common)
- G18: XZ plane
- G19: YZ plane

**Enhancement needed:**
```python
state['current_plane'] = 'G17'  # Default XY plane

if token_text in ['G17', 'G18', 'G19']:
    state['current_plane'] = token_text
```

### 4. Feed Rate and Spindle Speed Modality

F (feed rate) and S (spindle speed) are also modal but not tracked:
- Last specified F value stays active
- Last specified S value stays active

**Enhancement needed:**
```python
state['current_feed_rate'] = None
state['current_spindle_speed'] = None

# Track when F or S is predicted with numeric value
```

---

## Future Enhancements

### 1. Modal State Display in Dashboard

Add visual indicator showing current modal state:
```
Current Modal State:
  Command: G1 (Linear Interpolation)
  Plane: G17 (XY)
  Feed Rate: 250 mm/min
  Spindle: 2500 RPM (ON)
```

### 2. Modal State Reset API

```python
@app.route('/api/reset_modal_state', methods=['POST'])
def reset_modal_state():
    """Reset modal command state (simulates program restart)."""
    state['current_modal_command'] = None
    state['current_plane'] = 'G17'
    state['current_feed_rate'] = None
    state['current_spindle_speed'] = None
    return jsonify({'success': True, 'message': 'Modal state reset'})
```

### 3. Modal History Tracking

Track modal command changes over time:
```python
state['modal_history'] = deque(maxlen=50)

# When modal command changes
state['modal_history'].append({
    'timestamp': datetime.now().isoformat(),
    'from': old_command,
    'to': new_command,
    'sequence': full_command_tokens
})
```

### 4. Training Data Integration

Currently model may not be trained on modal sequences. To fully leverage modal behavior:
1. Preprocess training data to include modal sequences
2. Augment data: Convert "G1 X Y" followed by "G1 X Y" → "G1 X Y" followed by "X Y"
3. Add modal context to model inputs during training

---

## Testing

### Test Cases

#### Test 1: Basic Modal Behavior
```python
# Generate first sequence
prediction1 = predict(sensor_data)
assert prediction1['gcode_text'].startswith('G1')

# Generate second sequence (should allow parameter-only)
prediction2 = predict(sensor_data)
# May or may not start with command (model decides)
```

#### Test 2: Modal Command Persistence
```python
# After predicting G1
assert state['current_modal_command'] == 'G1'

# After predicting G0
assert state['current_modal_command'] == 'G0'
```

#### Test 3: Parameter Validation with Modal Context
```python
# Set modal command to G1
state['current_modal_command'] = 'G1'

# Try to predict R parameter (invalid for G1)
# Should be rejected even if sequence has no explicit command
```

---

## Summary

**Implemented:**
- ✅ Modal command state tracking
- ✅ Grammar constraints support for modal mode
- ✅ Modal context passed to constraints during generation
- ✅ Modal command updated when G/M commands predicted
- ✅ Parameter validation uses modal command context

**Current Behavior:**
- Sequences can start with COMMAND or PARAMETER tokens (model decides)
- When modal command is active, parameter-only sequences are allowed
- Parameter validation works with modal context
- Modal command persists across predictions until changed

**Future Work:**
- [ ] Distinguish modal vs. non-modal commands (M3, M5, M30 should not be modal)
- [ ] Reset modal state on program end (M30, M2)
- [ ] Track plane selection (G17, G18, G19)
- [ ] Track feed rate (F) and spindle speed (S) modality
- [ ] Add modal state visualization to dashboard
- [ ] Add modal state reset API endpoint
- [ ] Augment training data with modal sequences

---

**Author:** Claude Code
**Date:** 2025-11-30
**Version:** 1.0

**Related Documentation:**
- [COMMAND_SPECIFIC_VALIDATION.md](COMMAND_SPECIFIC_VALIDATION.md) - Parameter validation rules
- [PARAMETER_REPETITION_FIX.md](PARAMETER_REPETITION_FIX.md) - Parameter repetition prevention
- [GRAMMAR_CONSTRAINT_FIX.md](GRAMMAR_CONSTRAINT_FIX.md) - Grammar constraint fixes
