# Preprocessing Pipeline: Step-by-Step Analysis

## Dataset Overview

**Raw Data Location:** `data/`
- **9 operation types:** adaptive, adaptive150025, face, face150025, pocket, pocket150025, damageadaptive, damageface, damagepocket
- **Number of files per type:**
  - adaptive: 20 files
  - adaptive150025: 20 files
  - face: 20 files
  - face150025: 20 files
  - pocket: 20 files
  - pocket150025: 20 files
  - damageadaptive: 5 files ⚠️
  - damageface: 5 files ⚠️
  - damagepocket: 5 files ⚠️
- **Total:** 135 CSV files

**Raw CSV Structure:** 214 columns per file

| Column Range | Type | Count | Features |
|--------------|------|-------|----------|
| 1-17 | Metadata/Machine Controller | 17 | t_console, stat, line, posx, posy, posz, mpox, mpoy, mpoz, vel, feed, unit, dist, plane, coor, momo, raw_json |
| 18-25 | Electrical | 8 | spindle, x_motor, y_motor, z_motor, spindle_A, x_motor_A, y_motor_A, z_motor_A |
| 26-212 | Sensors (11 × 17) | 187 | frame_l2, frame_r2, frame_r1, y_bed__3, frame_b2, frame_l3, y_bed__1, spindle2, y_bed__4, z_gant_2, spindle1 |
| 213-214 | G-code | 2 | gcode_line, gcode_string |

**Sensors (11 total):**
1. frame_l2
2. frame_r2
3. frame_r3
4. y_bed__3
5. frame_b2
6. frame_l3
7. y_bed__1
8. spindle2
9. y_bed__4
10. z_gant_2
11. spindle1

**Sensor Channels (17 per sensor):**
- IMU: Ax, Ay, Az (accelerometer), Gx, Gy, Gz (gyroscope), Mx, My, Mz (magnetometer)
- Environmental: Pressure, Temperature, Proximity
- Color: ColorR, ColorG, ColorB, ColorA
- Audio: RMS

---

## Preprocessing Pipeline (preprocessing.py)

### Command
```bash
python -m miracle.dataset.preprocessing \
    --data-dir data/ \
    --output-dir outputs/processed/ \
    --vocab-path data/vocabulary.json \
    --window-size 64 \
    --stride 16
```

### Configuration (preprocessing_config.py)

```python
scaler_type = 'robust'  # RobustScaler (median + IQR)
nan_strategy = 'forward_fill'  # Propagate last valid value
outlier_method = 'clip'  # Clip to Q1-3*IQR, Q3+3*IQR
remove_zero_variance = True
correlation_threshold = 0.95  # Remove highly correlated
window_size = 64  # timesteps
stride = 16  # sliding window stride
```

---

## Step-by-Step Transformation

### STEP 1: Feature Exclusion

**Excluded from continuous features** (line 105-110):
```python
exclude_features = [
    # Metadata
    'time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',  # (not in CSV)
    't_console', 'gcode_line', 'gcode_string', 'raw_json',   # (in CSV)

    # NaN columns
    'vel', 'plane',

    # "Data leakage" (marked as such in config)
    'line', 'posx', 'posy', 'posz', 'feed', 'momo'
]
```

**Total excluded: 12 features** (from raw CSV)
- t_console
- raw_json
- gcode_line
- gcode_string
- vel
- plane
- line
- posx, posy, posz (commanded positions)
- feed (feed rate)
- momo (motion mode)

**Moved to CATEGORICAL** (line 101):
```python
categorical_features = ['stat', 'unit', 'dist', 'coor']
```

**Total: 4 features** moved to separate categorical array

### After STEP 1:
- **Continuous:** 214 - 12 (excluded) - 4 (categorical) - 2 (gcode) = **196 features**
  - Electrical: 8
  - Machine positions: mpox, mpoy, mpoz (3)
  - Sensors: 11 × 17 = 187
  - Total: 8 + 3 + 187 = 198 ✓ (matches)
- **Categorical:** 4 features (stat, unit, dist, coor)

---

### STEP 2: Load All Files & Build Master Column List (lines 453-468)

Scans ALL 135 CSV files to build a master column list for consistent dimensions across files.

**Why?** Some sensors might be missing in some files.

---

### STEP 3: Feature Selection (lines 470-508)

#### 3a. Remove High-Missing Features
```python
if max_missing_pct < 100:  # default: 50%
    # Remove columns with >50% missing values
```

#### 3b. Remove Zero-Variance Features
```python
if remove_zero_variance:  # default: True
    # Remove columns with zero or near-zero variance
```

#### 3c. Remove Highly Correlated Features
```python
if correlation_threshold < 1.0:  # default: 0.95
    # Remove one feature from pairs with |corr| > 0.95
```

**Impact:** Unknown number of features removed (depends on data characteristics)

---

### STEP 4: DATA LEAKAGE - Fit Scaler on ALL FILES (lines 513-524)

```python
# CRITICAL ISSUE: Fits scaler on ALL data BEFORE train/val/test split
all_continuous_data = []
for csv_path in input_files:  # ALL 135 files!
    df = preprocessor.load_csv(csv_path)
    continuous, _, _ = preprocessor.extract_features(df)
    all_continuous_data.append(continuous)

combined_data = np.vstack(all_continuous_data)  # Combine ALL data
preprocessor.fit_scaler(combined_data)  # FIT ON ALL DATA ⚠️
```

**Uses RobustScaler** (median + IQR, not StandardScaler):
```python
# From preprocessing_config.py line 27
scaler_type = 'robust'  # RobustScaler, NOT StandardScaler!
```

**Outlier Clipping** (lines 298-300):
```python
if outlier_method == 'clip':  # default: 'clip'
    # Clip to [Q1 - 3*IQR, Q3 + 3*IQR] before fitting scaler
    continuous_data = clip_outliers(continuous_data, threshold=3.0)
```

---

### STEP 5: Normalize All Data (line 340)

```python
continuous = preprocessor.transform(continuous)
# Applies RobustScaler fitted on ALL data
```

---

### STEP 6: Create Sliding Windows (lines 260-288)

```python
window_size = 64
stride = 16

for start_idx in range(0, T - window_size + 1, stride):
    end_idx = start_idx + window_size
    cont_window = continuous[start_idx:end_idx]  # [64, n_features]
    cat_window = categorical[start_idx:end_idx]   # [64, 4]
    # ...
```

**Number of windows per file:**
- If file has T timesteps: `(T - 64) // 16 + 1` windows

**Total windows:** ~thousands (depends on file lengths)

---

### STEP 7: RANDOM SPLIT (lines 535-548) - AFTER NORMALIZATION!

```python
# WRONG ORDER! Should split BEFORE normalizing!
indices = np.random.permutation(n_total)
train_indices = indices[:n_train]  # 70%
val_indices = indices[n_train:n_train + n_val]  # 15%
test_indices = indices[n_train + n_val:]  # 15%
```

**Data leakage:** Test set statistics (median, IQR) leaked into normalization!

---

### STEP 8: Save to .npz Files (lines 390-399)

```python
np.savez(output_path,
    continuous=continuous_data,  # [N, 64, n_features] - RobustScaler normalized
    categorical=categorical_data,  # [N, 64, 4]
    tokens=token_data,  # [N, max_len] - tokenized gcode
    lengths=lengths,  # [N]
    gcode_texts=gcode_texts,  # [N] object array
    operation_type=operation_type_ids,  # [N] - 0-8 for 9 classes
    operation_type_names=operation_types,  # [N] object array
)
```

**Saved:**
- `train_sequences.npz`
- `val_sequences.npz`
- `test_sequences.npz`
- `metadata.json`

---

## What Evaluation Scripts Do (run_9class_direct.py, run_baseline_models.py)

### STEP 1: Load Preprocessed Data (lines 189-190)

```python
data = np.load(npz_path, allow_pickle=True)
scaled = data['continuous'][:, :, keep_indices]
# ⚠️ Only loads 'continuous', IGNORES 'categorical'!
```

**PROBLEM:** The 4 categorical features (stat, unit, dist, coor) are completely ignored!

### STEP 2: Remove 16 More Features (line 122)

```python
MACHINE_CONTROLLER_FEATURES = {
    'coor', 'dist', 'feed', 'gcode_line', 'line', 'momo',
    'mpox', 'mpoy', 'mpoz', 'plane', 'posx', 'posy', 'posz',
    'stat', 'unit', 'vel',
}

keep_indices = [i for i, c in enumerate(orig_columns)
                if c not in MACHINE_CONTROLLER_FEATURES]
```

**But only 3 are actually in continuous:**
- mpox, mpoy, mpoz (machine positions)

**The other 13 were already excluded or moved to categorical:**
- stat, unit, dist, coor → categorical (ignored anyway)
- feed, line, posx, posy, posz, momo, plane, vel → excluded
- gcode_line → excluded

### STEP 3: Fix Data Leakage (lines 192-214)

```python
# Inverse transform (undo RobustScaler)
raw = scaled * keep_std + keep_mean

# Re-fit StandardScaler on TRAINING ONLY
scaler = StandardScaler()  # ⚠️ Different scaler type!
scaler.fit(train_flat)  # ✅ Only training data

# Apply to all splits
scaled = scaler.transform(flat)
```

**CHANGES:**
1. ✅ Fixes data leakage (train-only fitting)
2. ⚠️ Changes scaler type: RobustScaler → StandardScaler
   - RobustScaler: median + IQR (robust to outliers)
   - StandardScaler: mean + std (assumes normal distribution)

---

## Final Feature Count

### What SHOULD be kept (my calculation):
- Electrical: 8 (spindle, motors, currents)
- Machine positions: 3 (mpox, mpoy, mpoz) → **REMOVED by eval script**
- Sensors: 11 × 17 = 187
- **Total if mpox/mpoy/mpoz kept:** 8 + 3 + 187 = 198
- **Total after eval script removes mpox/mpoy/mpoz:** 8 + 187 = **195**

### What student says:
- **280 features**

### Discrepancy:
**280 - 195 = 85 extra features!**

**Possible explanations:**
1. **Feature engineering** (but add_derivatives and add_rolling_stats are disabled by default)
2. **Different preprocessing config** was actually used
3. **Multiple processed datasets** exist with different settings
4. **Incorrect count** by student

---

## Critical Issues Identified

### 1. ⚠️ DATA LEAKAGE (CONFIRMED)
- **Problem:** RobustScaler fitted on ALL data (train + val + test) BEFORE splitting
- **Impact:** Test set statistics leak into normalization
- **Fix:** Evaluation scripts inverse transform and re-fit on train only ✅
- **Remaining issue:** Changes scaler type (Robust → Standard)

### 2. ⚠️ CATEGORICAL FEATURES IGNORED
- **Problem:** 4 features (stat, unit, dist, coor) moved to categorical array, then ignored by eval scripts
- **Impact:** Potentially informative features lost
- **Recommendation:** Include categorical features or add them back to continuous

### 3. ⚠️ MACHINE POSITIONS REMOVED
- **Problem:** mpox, mpoy, mpoz (actual machine positions) removed by eval scripts
- **Impact:** Important positional information lost
- **Note:** posx, posy, posz (commanded positions) already excluded as "data leakage"
- **Question:** Should machine positions be used?

### 4. ⚠️ NORMALIZATION MISMATCH
- **Original:** RobustScaler (median + IQR)
- **Evaluation:** StandardScaler (mean + std)
- **Impact:** Different feature scaling, could affect model performance

### 5. ⚠️ FEATURE COUNT MISMATCH
- **Expected:** 195 features
- **Student claims:** 280 features
- **Need to check:** Actual metadata from .npz files

### 6. ⚠️ IMBALANCED CLASSES
- **Damage classes:** Only 5 files each (damageadaptive, damageface, damagepocket)
- **Other classes:** 20 files each
- **Impact:** 4× fewer samples for damage classes
- **Consequence:** Model might overfit to majority classes

---

## Information Loss Summary

| Step | Features Lost | Type | Reason |
|------|---------------|------|--------|
| Preprocessing exclusions | 12 | Metadata, NaN, "leakage" | Config setting |
| Moved to categorical | 4 | Machine state | Config setting |
| Categorical ignored | 4 | Machine state | Eval script bug |
| Machine positions removed | 3 | mpox, mpoy, mpoz | Eval script |
| Feature selection | ??? | High-missing, zero-var, high-corr | Data-dependent |
| **TOTAL LOST** | **19+** | — | — |
| **KEPT** | **195** | Electrical (8) + Sensors (187) | — |
| **Original** | **214** | All raw CSV columns | — |
| **Retention Rate** | **91.1%** | — | — |

---

## Recommendations

### 1. Verify Actual Feature Count
```bash
# Check what's actually in the .npz files
python << EOF
import numpy as np
import json

# Load preprocessed data
data_path = "outputs/7class_cascade_to_9class/9class_moddropout_final/data"
d = np.load(f"{data_path}/train_sequences.npz", allow_pickle=True)

print(f"continuous shape: {d['continuous'].shape}")
print(f"categorical shape: {d['categorical'].shape}")

# Load metadata
with open(f"{data_path}/metadata.json") as f:
    meta = json.load(f)
print(f"\\nmetadata n_continuous_features: {meta.get('n_continuous_features')}")
print(f"continuous_columns: {len(meta.get('continuous_columns', []))}")
print(f"\\nFirst 20 columns:")
for i, col in enumerate(meta.get('continuous_columns', [])[:20]):
    print(f"  {i+1}. {col}")
EOF
```

### 2. Fix Categorical Features
**Option A:** Include categorical features in continuous
```python
# In prepare_data_9class(), concatenate categorical to continuous
```

**Option B:** Use categorical properly
```python
# Add categorical embedding in model
```

### 3. Reconsider Machine Positions
**Question:** Should mpox, mpoy, mpoz be included?
- **PRO:** Contains actual machine state
- **CON:** Might correlate with operation type (data leakage)
- **Test:** Check correlation with operation_type labels

### 4. Use Consistent Scaler
**Option A:** Stick with RobustScaler (better for non-normal data)
**Option B:** Use StandardScaler everywhere
**Recommendation:** RobustScaler is better for sensor data with outliers

### 5. Investigate 100% Accuracy
Even after fixing leakage, all models get 100%. Possible causes:
- **File-level information leaking:** All windows from same file in same split
- **Operation name in features:** Check if any feature encodes operation type
- **Trivially separable:** Check sensor patterns manually

### 6. Re-run Preprocessing Properly
```python
# 1. Split FILES into train/val/test FIRST (not windows!)
# 2. Create windows from each split separately
# 3. Fit scaler on TRAINING windows only
# 4. Transform val/test with training scaler
# 5. Use consistent scaler type
# 6. Include categorical features properly
```

---

## Next Steps

1. **Inspect actual .npz files** to verify feature count and column names
2. **Check if file-level splitting** was done (same file's windows shouldn't be in train AND test)
3. **Visualize sensor patterns** for each operation type to see if trivially separable
4. **Re-run preprocessing** with proper train/val/test splitting at FILE level
5. **Include categorical features** (stat, unit, dist, coor)
6. **Decide on machine positions** (mpox, mpoy, mpoz) - keep or remove?
