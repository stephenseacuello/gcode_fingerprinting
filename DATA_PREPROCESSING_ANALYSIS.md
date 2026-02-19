# Data Preprocessing Analysis

## Executive Summary

**Your suspicion is CORRECT!** There are multiple issues with the normalized data pipeline that could explain the suspicious 100% accuracy:

1. ✅ **Data Leakage** - Scaler fitted on all data before train/test split (your student fixes this)
2. ⚠️ **Normalization Mismatch** - RobustScaler vs StandardScaler
3. ⚠️ **Missing Features** - Categorical features ignored, machine positions excluded
4. ⚠️ **Feature Count Discrepancy** - Actual features may differ from expected 280

---

## Data Flow Pipeline

### 1. Original Preprocessing (`src/miracle/dataset/preprocessing.py`)

**Input:** Raw CSV files (214 columns)
- Time/metadata columns: 4
- Machine controller: 16 features
- Electrical: 8 features (spindle, motors, currents)
- Sensors: 11 sensors × 17 channels = 187 features

**Excluded during preprocessing:**
```python
# From preprocessing_config.py line 105-110
exclude_features = [
    'time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',  # metadata
    't_console', 'gcode_line', 'gcode_string', 'raw_json',   # metadata
    'vel', 'plane',                                           # NaN columns
    'line', 'posx', 'posy', 'posz', 'feed', 'momo'           # marked as "data leakage"
]
# Total excluded: 14 features
```

**Categorical features (NOT in continuous):**
```python
# From preprocessing_config.py line 101
categorical_features = ['stat', 'unit', 'dist', 'coor']
# Total: 4 features
```

**Normalization:**
```python
# From preprocessing_config.py line 27
scaler_type = 'robust'  # RobustScaler (uses median + IQR)
```

**⚠️ DATA LEAKAGE ISSUE:**
```python
# From preprocessing.py line 513-524
# FITS SCALER ON ALL FILES (train + val + test) BEFORE SPLITTING!
all_continuous_data = []
for csv_path in input_files:
    continuous, _, _ = preprocessor.extract_features(df)
    all_continuous_data.append(continuous)

combined_data = np.vstack(all_continuous_data)  # All data combined
preprocessor.fit_scaler(combined_data)           # ⚠️ LEAKAGE!

# THEN splits into train/val/test (line 535-548)
indices = np.random.permutation(n_total)
train_indices = indices[:n_train]  # Split AFTER normalization
```

**Output:** `.npz` files with:
- `continuous`: Features normalized with RobustScaler (fitted on ALL data)
- `categorical`: ['stat', 'unit', 'dist', 'coor'] as integers
- `operation_type`: Labels (0-8 for 9 classes)

---

### 2. Evaluation Scripts (`run_9class_direct.py`, `run_baseline_models.py`)

**Load preprocessed data:**
```python
# Line 189-190 in run_9class_direct.py
data = np.load(npz_path, allow_pickle=True)
scaled = data['continuous'][:, :, keep_indices]  # ⚠️ Only loads 'continuous'
# 'categorical' features are IGNORED!
```

**Additional features removed:**
```python
# Line 44-48 in both files
MACHINE_CONTROLLER_FEATURES = {
    'coor', 'dist', 'feed', 'gcode_line', 'line', 'momo',
    'mpox', 'mpoy', 'mpoz', 'plane', 'posx', 'posy', 'posz',
    'stat', 'unit', 'vel',
}
# Total: 16 features

# Line 122: Remove these from loaded data
keep_indices = [i for i, c in enumerate(orig_columns) if c not in MACHINE_CONTROLLER_FEATURES]
```

**⚠️ PROBLEMS:**
- `mpox, mpoy, mpoz` were NOT excluded originally, but removed here
- `stat, unit, dist, coor` are in categorical, NOT continuous (but code tries to remove them anyway)
- This creates a mismatch!

**Fix data leakage:**
```python
# Line 193: Inverse transform (undo RobustScaler)
raw = scaled * keep_std + keep_mean

# Line 205-206: Re-fit StandardScaler on TRAINING data only
scaler = StandardScaler()
scaler.fit(train_flat)  # ✅ Only training data

# Line 213: Apply to all splits
scaled = scaler.transform(flat)  # ✅ Fixes leakage
```

**⚠️ NEW ISSUE:**
- Original preprocessing: **RobustScaler** (median + IQR)
- Evaluation: **StandardScaler** (mean + std)
- Different normalization methods!

---

## Feature Count Analysis

Let's trace where 280 comes from:

**Starting point (raw CSV):** 214 columns

**After original preprocessing excludes 14 features:**
214 - 14 = 200 columns

**Minus 4 categorical features (separate array):**
200 - 4 = 196 continuous features

**But evaluation scripts expect 280?**

This suggests the original preprocessed data has MORE features than raw CSV, possibly from:
- Feature engineering (derivatives, rolling stats, etc.)
- Different data source
- Multiple time-aligned sensor streams

**Need to check:** What does `orig_meta['continuous_columns']` actually contain?

---

## Critical Questions to Investigate

1. **What's actually in the .npz files?**
   ```bash
   # Load and inspect
   python -c "import numpy as np; d = np.load('path/to/train_sequences.npz', allow_pickle=True); print(d['continuous'].shape)"
   ```

2. **What are the column names?**
   ```bash
   # Check metadata
   cat path/to/metadata.json
   ```

3. **Are categorical features being used?**
   - Current code ignores them
   - Should they be included?

4. **Why 100% accuracy?**
   - Even after fixing data leakage, all models get 100%
   - Suggests remaining issues:
     - Feature that directly encodes class (e.g., operation name in data)
     - Test set too small or not representative
     - Classes are trivially separable even with proper preprocessing

---

## Recommendations

### Immediate Actions:

1. **Inspect the actual preprocessed data:**
   ```python
   import numpy as np
   import json

   # Load data
   data = np.load('path/to/train_sequences.npz', allow_pickle=True)
   print(f"continuous shape: {data['continuous'].shape}")
   print(f"categorical shape: {data['categorical'].shape}")
   print(f"operation_type unique: {np.unique(data['operation_type'])}")

   # Load metadata
   with open('path/to/metadata.json') as f:
       meta = json.load(f)
   print(f"n_continuous_features: {meta['n_continuous_features']}")
   print(f"continuous_columns: {meta['continuous_columns'][:10]}...")  # First 10
   ```

2. **Check if categorical features should be included:**
   - `stat, unit, dist, coor` might be important!
   - Currently ignored by evaluation scripts

3. **Verify the 16 features being removed actually exist:**
   - Some might be in categorical, not continuous
   - This could cause index errors or remove wrong features

4. **Re-run preprocessing from scratch with correct settings:**
   - Split data FIRST (train/val/test)
   - Fit scaler on training only
   - Use consistent scaler type (StandardScaler everywhere)

5. **Investigate why 100% accuracy persists:**
   - Even with leakage fixed, still getting 100%
   - Check for:
     - Operation type encoded in feature names
     - File-level information leaking into features
     - Temporal patterns that trivially separate classes

### Long-term Fix:

Create a new preprocessing script:
```python
# 1. Load raw CSV files
# 2. Extract operation type from filename
# 3. Split files into train/val/test FIRST (stratified by operation type)
# 4. Process each split separately:
#    - Extract features (continuous + categorical)
#    - Fit scaler on TRAINING set only
#    - Transform each split with training scaler
# 5. Save with proper metadata
```

---

## Bottom Line

Your intuition was right! The normalized data has multiple issues:

1. ✅ **Data leakage** (scaler fitted on all data) - Your student's code fixes this
2. ⚠️ **Normalization mismatch** (RobustScaler → StandardScaler) - Could affect results
3. ⚠️ **Feature mismatch** (categorical ignored, wrong features removed) - Could lose information
4. ⚠️ **100% accuracy** - Even with fixes, suggests deeper issue with data or task

**Next step:** Inspect the actual .npz files to see what's really in there, then decide whether to:
- Fix the evaluation scripts to match preprocessing
- Re-run preprocessing with correct settings
- Investigate why the task is so easy (100% accuracy)
