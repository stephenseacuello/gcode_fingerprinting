# Research Goals vs. Current Experimental Setup

## Your Research Objectives

**Primary Goal:** Determine which sensor locations and modalities provide the most discriminative information for classifying CNC machining operations.

**Specific Questions:**
1. Which physical locations (spindle, frame, bed, gantry) are most informative?
2. Which sensor modalities (accelerometer, gyroscope, magnetometer, RMS, etc.) matter most?
3. Does multimodal fusion improve classification over single modalities?
4. Is sensor redundancy necessary, or can fewer sensors achieve similar performance?

**These are excellent research questions!** However, the current data collection and preprocessing approach creates confounding that prevents answering them.

---

## The Confounding Problem

### What Actually Happened During Data Collection

All Arduino units were physically placed at their locations for all operations. However:

**Different operations engaged different machine components**, causing:
- Some sensors to capture strong signals ("fired")
- Other sensors to remain relatively silent (minimal activity)

### Sensor Activity Patterns

| Sensor Location | adaptive | pocket | face | Pattern |
|-----------------|----------|--------|------|---------|
| spindle1 | ✅ 100% active | ⚠️ 65% active | ⚠️ 75% active | Strongly correlated with adaptive |
| xa_motor | ⚠️ 75% active | ✅ 100% active | ⚠️ 55% active | Strongly correlated with pocket |
| z_gant_2 | ⚠️ 50% active | ✅ 100% active | ⚠️ 70% active | Strongly correlated with pocket |
| y_bed__1 | ✅ 100% active | ⚠️ 60% active | ⚠️ 90% active | Strongly correlated with adaptive |
| frame_r2 | ✅ 100% active | ✅ 100% active | ✅ 100% active | No correlation (good!) |
| y_bed__3 | ✅ 100% active | ✅ 100% active | ✅ 100% active | No correlation (good!) |

**"Active" = sensor produced meaningful signal (not constant zeros)**

### Why This Creates Confounding

**Confounding means:** Two variables are correlated, making it impossible to determine causation.

In your case:
- **Variable A:** Which sensors are active (produced signal)
- **Variable B:** Operation type
- **Correlation:** A predicts B with 62% accuracy

**The problem:** You cannot distinguish:
1. **Sensor data informativeness:** "When xa_motor captures vibrations, those vibrations are discriminative"
2. **Sensor activity correlation:** "xa_motor being active (vs silent) correlates with operation type"

**What you want to measure:** #1
**What models learn:** #2

---

## What Your Current Results Actually Tell You

### What You CAN Validly Conclude:

✅ **"Different operations engage different machine components"**
- Adaptive operations primarily engage spindle and Y-bed
- Pocket operations heavily engage X-axis motor and Z-gantry
- This is valuable domain knowledge about machining physics!

✅ **"Sensor activity patterns correlate with operation types"**
- If xa_motor is active → likely pocket operation
- If spindle1 is active → likely adaptive operation

✅ **"Multimodal sensors successfully capture machining activity"**
- Accelerometer, gyroscope, RMS all captured signals
- Environmental sensors (temperature, pressure) also recorded activity

### What You CANNOT Validly Conclude:

❌ **"xa_motor is the most important sensor location"**
- Current results: xa_motor shows high feature importance
- Reality: Could be because (a) its vibration patterns are discriminative, OR (b) its presence/absence correlates with operation type
- **Cannot distinguish these!**

❌ **"Accelerometer is more informative than gyroscope"**
- Current results: Accelerometer features rank highly
- Reality: Could be because accelerometer data is genuinely more discriminative, OR because accelerometer channels had more zero-padding artifacts
- **Cannot distinguish these!**

❌ **"Multimodal fusion improves over single modalities"**
- Current results: Multimodal model gets 100% accuracy
- Reality: 100% is due to data leakage (file-level split + zero-padding shortcuts), not genuine learning
- **Cannot evaluate fusion effectiveness with leaky data!**

❌ **"We can deploy with fewer sensors and maintain accuracy"**
- Current results: Removing certain sensors barely hurts accuracy
- Reality: Model learned sensor presence patterns, not sensor data characteristics
- **Deployment with different sensor configuration will fail!**

---

## The Mathematical Problem

### What Ablation Studies Require

To determine sensor importance via ablation:

**Assumption required:** Sensors must be **conditionally independent** given operation type.

That is: `P(sensor_i active | operation) ≈ P(sensor_i active)`

**Your data:** Sensors are **conditionally dependent**:
- `P(xa_motor active | pocket) = 1.0`
- `P(xa_motor active | face150025) = 0.15`
- **Strong dependence!**

### Why This Breaks Ablation Analysis

**Ablation experiment:** "Remove xa_motor, measure accuracy drop"

**What you want to measure:**
```
Importance = Accuracy(all_sensors) - Accuracy(all_except_xa_motor)
           = Information in xa_motor vibration patterns
```

**What you actually measure:**
```
Importance = (Ability to detect xa_motor presence)
           + (Information in xa_motor vibration patterns)
           + (Correlation between xa_motor and operation type)
```

**Cannot separate these components!**

### Example Calculation

```
Baseline accuracy: 100%

Remove xa_motor: 92% accuracy
  → Conclusion: "xa_motor is important (8% drop)"

But break down the 8% drop:
  - 5% from losing "xa_motor present → pocket" rule
  - 2% from losing xa_motor vibration patterns
  - 1% from correlated information in z_gant_2

True discriminative value of xa_motor vibrations: ~2%
Confounded "importance": 8%
```

**You've overestimated xa_motor importance by 4×!**

---

## Why Zero-Padding Makes This Worse

Even if sensor activity naturally correlates with operations (physics-based), zero-padding creates **artificial binary signals** that are trivially exploitable.

### Real Low Activity (Physics)

```python
# Sensor present but minimal motion
xa_motor.Ax = [0.002, -0.001, 0.003, -0.002, 0.001, ...]

Characteristics:
- Mean ≈ 0.0
- Std ≈ 0.002  (small but non-zero)
- Has noise, drift, electronic artifacts
- Frequency content: mostly noise, some harmonics
```

**Model must learn:** "Low amplitude, noise-dominated signal → minimal X-axis activity"

### Zero-Padded (Preprocessing Artifact)

```python
# Preprocessing filled with zeros
xa_motor.Ax = [0.000000, 0.000000, 0.000000, ...]

Characteristics:
- Mean = 0.0 exactly
- Std = 0.0 exactly  (impossible for real sensor!)
- No noise, no drift, no artifacts
- Frequency content: DC only (0 Hz)
```

**Model learns:** `if std(xa_motor.Ax) == 0: return "not pocket"`

**This is a trivial decision rule that requires no understanding of vibration dynamics!**

---

## Solutions for Your Research Goals

### Solution 1: Sensor Activity Descriptive Analysis (Immediate)

**What to do:**
Analyze and report which sensors were active for which operations, but **do NOT claim this measures sensor informativeness**.

**Report structure:**
```markdown
## Sensor Activity Patterns

Different machining operations engaged different machine components,
resulting in characteristic sensor activity patterns:

### Adaptive Operations
- Spindle vibrations: Active in 100% of files
- Y-bed vibrations: Active in 100% of files
- X-axis motor: Active in 75% of files
- Z-gantry: Active in 50% of files

**Interpretation:** Adaptive toolpaths primarily engage the spindle
and Y-axis motion, with variable engagement of X and Z axes.

### Pocket Operations
- X-axis motor: Active in 100% of files
- Z-gantry: Active in 100% of files
- Spindle: Active in 65% of files

**Interpretation:** Pocket milling consistently engages X-axis and
Z-axis motion, with variable spindle engagement.

## Implications
These activity patterns reflect the kinematic requirements of each
operation type and suggest which sensor locations would be most
relevant for condition monitoring of specific operations.
```

**This is valid and valuable domain knowledge!** Just don't claim it proves sensor importance for classification.

### Solution 2: Matched-Configuration Classification (Short-term)

**What to do:**
Use only sensors that fired consistently (>95%) across ALL operation types.

**Implementation:**
```python
# Find sensors active in >95% of all files
consistently_active = [
    'frame_r2',      # 100% of files
    'y_bed__3',      # 100% of files
    'frame_l2',      # 99.3% of files
    'spindle2',      # 97.8% of files
    'y_bed__4',      # 95.6% of files
    'frame_l3',      # 94.8% of files
]

# Use ONLY these 6 sensors (6 × 17 = 102 channels + 8 electrical = 110 features)
# No zero-padding needed
# All operations have signal from these sensors
```

**Benefits:**
- No sensor presence/absence confounding
- Model must learn from signal characteristics
- Ablation study is valid (all sensors active for all operations)
- Can answer: "Among reliably-active sensors, which are most informative?"

**Limitations:**
- Cannot evaluate sensors like xa_motor that don't fire consistently
- Reduced set of sensors to analyze

**Valid conclusions:**
- "Among sensors capturing consistent activity, frame_l2 is most discriminative"
- "Accelerometer modality outperforms gyroscope when sensors are active"
- "We can achieve X% accuracy using only 3 of the 6 consistently-active sensors"

### Solution 3: Two-Stage Analysis (Recommended)

**Stage 1: Sensor Engagement Analysis (Descriptive)**

Report which sensors engaged for which operations (as in Solution 1).

**Stage 2: Signal Discriminability Analysis (Predictive)**

For sensors that engaged, analyze discriminative power:

```python
# For each sensor that sometimes fires:
for sensor in all_sensors:
    # Take only files where THIS sensor was active
    active_files = [f for f in all_files if sensor_fired(f, sensor)]

    # Among these files (mixed operations), can sensor discriminate?
    # Example: 40 files where xa_motor fired
    #   - 20 pocket files
    #   - 12 adaptive files
    #   - 8 face files

    # Train classifier on xa_motor data only, for these 40 files
    accuracy = classify_using_only(sensor, active_files)

    # If accuracy >> random: sensor data is discriminative
    # If accuracy ≈ random: sensor just correlated, not discriminative
```

**Example results:**

| Sensor | Files Where Active | Operations (when active) | Accuracy | Discriminative? |
|--------|-------------------|-------------------------|----------|-----------------|
| xa_motor | 97 files | 55 pocket, 30 adaptive, 12 face | 78% | ✅ Yes |
| spindle1 | 97 files | 40 adaptive, 35 face, 22 pocket | 65% | ✅ Yes |
| frame_r2 | 135 files | Mixed (all operations) | 72% | ✅ Yes |
| y_bed__2 | 51 files | Mixed (all operations) | 15% | ❌ No |

**Interpretation:**
- xa_motor: Active mostly for pocket, AND when active, its vibrations discriminate → **Important sensor**
- spindle1: Active mostly for adaptive, AND when active, its vibrations discriminate → **Important sensor**
- frame_r2: Active for all operations, AND its vibrations discriminate → **Very important sensor** (no confounding!)
- y_bed__2: Inconsistent activity, AND vibrations don't discriminate → **Not important**

**This separates sensor engagement from sensor informativeness!**

### Solution 4: Explicit Activity Modeling (Advanced)

**What to do:**
Build a model that explicitly separates sensor activity from sensor signal:

```python
# Features
sensor_activity = [is_active(sensor) for sensor in all_sensors]  # 16 binary
sensor_signals = [extract_features(sensor) for sensor in all_sensors]  # 280 continuous

# Model architecture
activity_embedding = MLP(sensor_activity)  # Learn from activity pattern
signal_embedding = CNN_LSTM(sensor_signals)  # Learn from signal characteristics

prediction = Classifier(activity_embedding + signal_embedding)

# Ablation experiments
acc_activity_only = evaluate(activity_embedding only)
acc_signal_only = evaluate(signal_embedding only)
acc_both = evaluate(both)

# Analysis
if acc_activity_only >> acc_signal_only:
    conclusion = "Sensor presence matters more than sensor data" ❌

elif acc_signal_only >> acc_activity_only:
    conclusion = "Sensor data is discriminative" ✅

elif acc_both >> max(acc_activity, acc_signal):
    conclusion = "Both matter, synergistic effect" ⚠️
```

**This quantifies the contribution of each!**

---

## Critical First Step: Fix File-Level Splitting

**BEFORE doing anything else, you MUST fix the window-level splitting issue!**

**Current (broken):**
```python
# Mix all windows from all files
all_windows = []
for file in all_135_files:
    windows = create_windows(file)
    all_windows.extend(windows)

# Randomly split windows
train, val, test = random_split(all_windows)
# ⚠️ Same file's windows in train AND test!
```

**Correct:**
```python
# Split files FIRST
train_files, val_files, test_files = stratified_split(
    all_135_files,
    by='operation_type',
    train=0.7, val=0.15, test=0.15
)

# Create windows per split
train_windows = [create_windows(f) for f in train_files]
val_windows = [create_windows(f) for f in val_files]
test_windows = [create_windows(f) for f in test_files]

# No file appears in multiple splits ✅
```

**Why this is critical:**
- Without this fix, test accuracy is meaningless (file memorization)
- All other fixes are pointless if this isn't fixed first
- This is the primary cause of 100% accuracy

**Expected result after fix:**
- Accuracy drops from 100% to 65-85%
- This is GOOD - reveals true generalization performance

---

## Revised Research Workflow

### Phase 1: Fix Data Leakage (Week 1)

1. ✅ Implement file-level splitting
2. ✅ Re-run all models
3. ✅ Verify accuracy drops to realistic levels (70-85%)

**Deliverable:** "Models achieve 75% accuracy with proper train/test separation, demonstrating genuine learning from sensor dynamics."

### Phase 2: Sensor Activity Analysis (Week 1-2)

1. ✅ Create sensor activity heatmap (operation × sensor × activity %)
2. ✅ Report kinematic interpretations
3. ✅ Identify consistently-active sensors

**Deliverable:** "Different operations engage different machine components. We identify 6 sensors with >95% activity across all operations."

### Phase 3: Matched-Configuration Classification (Week 2-3)

1. ✅ Use only consistently-active sensors
2. ✅ Train models on clean data (no zero-padding confounding)
3. ✅ Perform ablation study

**Deliverable:** "Among consistently-active sensors, frame_l2 and spindle2 are most discriminative. We achieve 82% accuracy using only 3 of 6 sensors."

### Phase 4: Signal Discriminability Analysis (Week 3-4)

1. ✅ For each sensor, evaluate discriminative power when active
2. ✅ Separate sensor engagement from sensor informativeness
3. ✅ Build two-component model (activity + signal)

**Deliverable:** "We quantify that sensor activity patterns contribute 35% of classification performance, while sensor signal characteristics contribute 65%."

### Phase 5: Modality Analysis (Week 4-5)

1. ✅ Compare accelerometer vs gyroscope vs magnetometer vs RMS
2. ✅ Test multimodal fusion vs single modalities
3. ✅ Quantify modality importance

**Deliverable:** "Accelerometer and RMS are most informative. Multimodal fusion improves accuracy from 78% (best single modality) to 82%."

---

## Key Takeaways

### Your Research Goals Are Excellent ✅

Determining sensor importance for CNC monitoring is valuable research!

### Your Data Has Confounding ⚠️

Sensor activity correlates with operation type, preventing clean analysis.

### Current Results Are Invalid ❌

100% accuracy from data leakage, not genuine learning.

### Solutions Exist ✅

- Fix file-level splitting (critical, immediate)
- Use consistently-active sensors (reduces confounding)
- Explicit activity modeling (separates confounding from informativeness)
- Two-stage analysis (descriptive + predictive)

### Revised Expectations 📊

- **Accuracy after fixes:** 70-85% (down from 100%, but VALID)
- **Sensor importance:** Can be measured for consistently-active sensors
- **Modality comparison:** Valid once confounding is addressed
- **Deployment:** Will generalize to new files (unlike current approach)

---

## Recommendations for Your PhD Student

### Immediate Actions (This Week):

1. **Fix file-level splitting** in preprocessing script
2. **Re-run all experiments** with proper split
3. **Document accuracy drop** (100% → ~75%) in results

### Short-term (Next 2 Weeks):

4. **Create sensor activity analysis** (descriptive, no modeling)
5. **Identify consistently-active sensors** (>95% activity)
6. **Re-run experiments with matched sensors** only

### Medium-term (Next Month):

7. **Implement two-stage analysis** (engagement + discriminability)
8. **Quantify confounding contribution** (activity patterns vs signal)
9. **Compare modalities** on clean data

### Paper Revision:

10. **Add "Limitations" section** explaining sensor activity confounding
11. **Reframe contributions** as:
    - Characterization of sensor engagement patterns
    - Classification performance on reliably-active sensors
    - Modality comparison for condition monitoring
12. **Report realistic accuracy** (75-85%) with proper validation

---

## Final Note

The confounding between sensor activity and operation type is a **natural consequence** of physics—different operations engage different machine components. This is **not a flaw in your experimental design**, but it does limit which research questions you can answer with this dataset.

**You can still publish valuable research** by:
- Being transparent about the confounding
- Separating descriptive analysis (sensor engagement) from predictive analysis (sensor informativeness)
- Using consistently-active sensors for fair comparison
- Fixing data leakage to demonstrate genuine learning

**The key is intellectual honesty:** Report what the data can and cannot tell you, rather than claiming conclusions the confounding prevents.
