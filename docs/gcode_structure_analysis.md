# G-Code Structure Analysis for Better Prediction

## Executive Summary
Understanding G-code grammar and constraints is critical for building effective prediction models. This document analyzes patterns from the dataset to inform model design.

## Key G-Code Patterns Observed

### 1. **Modal Commands** (State Persistence)
```gcode
G1 X3.291 Z0.0003      # Set linear interpolation mode
X3.275 Y0.3746 R0.083  # Still in G1 mode
X3.275 Y0.7064 R0.083  # Still in G1 mode
```

**Implication for Model**:
- The model must track state across sequences
- Predicting bare coordinates (X/Y/Z without command) is only valid after a motion command (G0/G1/G2/G3)
- **Current weakness**: Our model treats each token independently, missing this context

### 2. **Arc Commands (G2/G3) Require Radius or Center**
```gcode
G2 X-0.2084 Y0.1257 R0.083  # Clockwise arc MUST have R or I/J/K
G3 X3.358 Y0.2916 R0.083    # Counter-clockwise arc MUST have R or I/J/K
```

**Implication for Model**:
- G2/G3 commands have **mandatory** R (radius) OR I/J/K (arc center offset) parameters
- Predicting "G2" should strongly bias toward predicting R or I/J/K next
- **Current weakness**: No constraint enforcement - model can generate invalid "G2 X Y" without R

### 3. **Alternating Arc Patterns** (Repetitive Structures)
```gcode
G2 X-0.2084 Y0.4575 R0.083
X-0.1254 Y0.5405 R0.083
G3 X3.358 Y0.6235 R0.083
G2 X-0.2084 Y0.7894 R0.083
X-0.1254 Y0.8724 R0.083
G3 X3.358 Y0.9553 R0.083
```

**Pattern**: G2 → (modal G2) → G3 → G2 → (modal G2) → G3 (repeats)

**Implication for Model**:
- Manufacturing operations often have **periodic patterns**
- Pockets, faces, and adaptive toolpaths create repeating sequences
- **Opportunity**: Use attention mechanism to capture long-range periodicities
- **Current strength**: Transformer architecture can learn this if trained properly

### 4. **Feed Rate (F) Appears with Linear/Arc Moves**
```gcode
G1 Z0.025 F7.3  # Feed rate specified for linear motion
G2 X Y R F      # Arc motions can also have feed rate
```

**Implication for Model**:
- F parameter is **optional** but common with G1/G2/G3
- F is **sticky** (modal) - once set, applies to subsequent moves
- **Current weakness**: No differentiation between sticky vs one-time parameters

### 5. **Rapid Positioning (G0) vs Cutting Moves (G1/G2/G3)**
```gcode
G0 Z0.6         # Rapid move to safe height (NO feed rate)
G1 Z0.025 F7.3  # Cutting move (HAS feed rate)
```

**Implication for Model**:
- G0 should **never** have F parameter (rapids ignore feed rate)
- G1/G2/G3 should **often** have F parameter for first occurrence after G0
- **Current weakness**: No constraint preventing "G0 F10.0" (invalid)

### 6. **Z-Axis Retracts Between Features**
```gcode
G0 Z0.6         # Retract to safe height
G0 X... Y...    # Move to new position at safe height
G1 Z0.025       # Plunge to cutting depth
```

**Pattern**: Z-retract → XY-rapid → Z-plunge

**Implication for Model**:
- Strong sequence dependency: Z-retract (G0 Z+) often followed by XY-rapid then Z-plunge
- **Opportunity**: Multi-head architecture can specialize in predicting this pattern
- **Current design**: Multi-head architecture is well-suited for this!

### 7. **Parameter Value Correlations**
From dataset analysis:
```
R parameter: Often 0.083 (arc radius from tool diameter)
F parameter: 7.3, 22.0, 40.0 (discrete feedrate settings)
Z depth: Often -0.025, -0.05, etc (depth of cut increments)
```

**Implication for Model**:
- Numeric values are **NOT uniformly distributed**
- Values cluster around tool settings and manufacturing constraints
- **Current failure**: Regression model predicted mean value (1.4) for everything
- **Better approach**: Discrete bucketing captures these clusters better than regression

## How Our Multi-Head Architecture Addresses These Patterns

### ✅ Strengths
1. **Type Gate**: Separates command/parameter prediction → reduces confusion
2. **Separate Heads**: Command head can learn modal patterns independently
3. **Sequence Modeling**: Transformer can capture long-range dependencies (arc patterns)
4. **Operation Type**: Classifies overall operation (face/pocket/etc) to condition predictions

### ❌ Weaknesses
1. **No Modal State Tracking**: Model doesn't explicitly track current active command
2. **No Constraint Enforcement**: Can generate invalid sequences (G2 without R)
3. **Independent Token Prediction**: Doesn't enforce parameter dependencies
4. **Regression for Clustered Values**: Fails because values are discrete, not continuous

## Recommendations for Improvement

### 1. **Use Discrete Bucketing, Not Regression**
```python
# GOOD: Captures R=0.083 cluster
param_value_logits = model.param_value_head(hidden)  # → [B, T, 100] (buckets 00-99)

# BAD: Predicts mean=1.4 for everything
param_value_regression = model.param_value_regression_head(hidden)  # → [B, T, 1]
```

### 2. **Add Constraint Losses**
```python
# Penalize G2/G3 not followed by R parameter
arc_command_mask = (commands == G2_ID) | (commands == G3_ID)
next_token_is_R = (param_types[:, 1:] == R_ID)
constraint_loss = F.binary_cross_entropy(
    next_token_is_R.float(),
    arc_command_mask[:, :-1].float()
)
```

### 3. **Add Modal State Embeddings**
```python
# Track active command state
modal_state = torch.zeros(B, 1, D)  # Last command
for t in range(seq_len):
    if is_command[t]:
        modal_state = command_embedding[t]
    hidden[t] = hidden[t] + modal_state  # Condition on modal state
```

### 4. **Use Curriculum Learning**
```
Phase 1: Learn individual commands (G0, G1, G2, G3)
Phase 2: Learn command + immediate parameters (G1 X Y)
Phase 3: Learn modal patterns (G1; X Y; X Y)
Phase 4: Learn full sequences with arcs and retracts
```

### 5. **Analyze Value Distributions Per Parameter Type**
```python
# Don't bucket all parameters uniformly!
R_values → cluster at tool_diameter/2 (0.083, 0.125, etc)
F_values → cluster at machine settings (7.3, 22.0, 40.0)
Z_values → cluster at depth_of_cut increments (0.025, 0.05, etc)
XY_values → wide range, need finer bucketing

# Use per-parameter bucketing strategies
if param_type == 'R':
    buckets = [0.05, 0.083, 0.125, 0.25, 0.5]  # Tool radii
elif param_type == 'F':
    buckets = [5, 7.3, 10, 15, 22, 40, 60]  # Common feedrates
elif param_type == 'Z':
    buckets = [-10.0, -5.0, -1.0, ..., 10.0]  # Depth range
```

## Testing Model Understanding

### Validation Checks
1. **Grammar Check**: Does generated sequence follow modal command rules?
2. **Constraint Check**: Do G2/G3 commands have R parameter?
3. **Physical Validity**: Are Z-retracts followed by XY-rapids?
4. **Value Clustering**: Do predicted values match dataset distributions?

### Metrics Beyond Accuracy
```python
# Check diversity (not just predicting mean/mode)
unique_predictions = len(set(predictions))
entropy = -sum(p * log(p) for p in value_distribution)

# Check grammar validity
grammar_violations = count_invalid_sequences(predictions)

# Check physical plausibility
rapid_feed_violations = count(predictions matching "G0 F...")
arc_radius_violations = count(predictions matching "G2/G3" without "R")
```

## Conclusion

**Why Regression Failed**: G-code numeric values are **discrete, clustered, and context-dependent**, not continuous and smooth. The regression model predicted the mean value (1.4) because it minimized MAE, but this produces nonsensical G-code.

**Why Bucketing Works Better**: Discrete buckets capture the actual value distributions (R=0.083, F=7.3, etc) as distinct classes, allowing the model to learn which values appear in which contexts.

**Next Steps**:
1. Test hybrid_1digit_improved model (uses 1-digit coarse bucketing + residuals)
2. Compare against 2-digit and 3-digit bucketing approaches
3. Add constraint losses to enforce G-code grammar
4. Implement per-parameter value distribution analysis
