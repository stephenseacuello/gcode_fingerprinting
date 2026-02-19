# Experiment: 95% Consistent Sensors (No Zero-Padding)

## Overview

This experiment addresses **Issue #1 (Zero-Padding)** from the comprehensive preprocessing analysis by using only sensors that are present and active in ≥95% of files.

**Expected Impact:**
- ✅ Eliminates zero-padding shortcuts
- ✅ Forces models to learn from actual sensor dynamics (not sensor presence)
- ⚠️ May reduce accuracy from 100% to realistic 70-85%
- ✅ Validates whether zero-padding was inflating accuracy

---

## Methodology

### 1. Sensor Selection

From the sensor consistency analysis, we identified **5 sensors with ≥95% activity**:

| Sensor Location | Activity % | Channels | Total Features |
|----------------|------------|----------|----------------|
| frame_r2 | 100.0% | 17 | 17 |
| y_bed__3 | 100.0% | 17 | 17 |
| frame_l2 | 98.5% | 17 | 17 |
| spindle2 | 97.8% | 17 | 17 |
| y_bed__4 | 95.6% | 17 | 17 |

**Total input features:**
- 5 sensors × 17 channels = **85 sensor features**
- \+ 8 electrical features = **93 total features**

**Key advantage:** NO ZERO-PADDING needed - all files have these sensors!

### 2. Comparison Setup

**Current (with zero-padding):**
```
16 sensor locations × 17 channels = 272 features
+ 8 electrical = 280 total features
Missing sensors filled with zeros → creates shortcuts!
Accuracy: 100% (all models)
```

**New (95% consistent sensors):**
```
5 sensor locations × 17 channels = 85 features
+ 8 electrical = 93 total features
NO zero-padding - all sensors present in all files
Accuracy: ??? (to be determined)
```

### 3. Models to Compare

**Encoder:**
- MM-LSTM-DAE (multimodal LSTM denoising autoencoder)

**Baselines (ML):**
- XGBoost
- Random Forest
- Logistic Regression

**Baselines (NN):**
- MLP (3-layer)
- Simple LSTM

---

## Running the Experiment

### Prerequisites

1. **Data directory:** `data/` with aligned CSV files
2. **Vocabulary file:** `outputs/vocabulary/gcode_vocabulary_v2.json`
   - If missing, generate with: `python scripts/utilities/build_vocabulary.py --data-dir data/ --output outputs/vocabulary/gcode_vocabulary_v2.json`

### Quick Start

Run the complete experiment pipeline:

```bash
./scripts/experiments/run_95pct_sensors_experiment.sh
```

This will:
1. ✅ Analyze sensor consistency (if not already done)
2. ✅ Preprocess data with only 95% consistent sensors
3. ✅ Train MM-LSTM-DAE encoder
4. ✅ Train all baseline models
5. ✅ Generate comparison report

**Estimated time:** 30-60 minutes (depending on hardware)

### Step-by-Step (Manual Execution)

If you prefer to run steps individually:

#### Step 1: Sensor Consistency Analysis

```bash
python scripts/analysis/identify_consistent_sensors.py \
    --data-dir data/ \
    --threshold 95.0 \
    --output outputs/sensor_consistency_report.json
```

**Output:** `outputs/sensor_consistency_report.json`

#### Step 2: Preprocessing

```bash
python scripts/preprocessing/run_preprocessing_consistent_sensors.py \
    --data-dir data/ \
    --output-dir outputs/experiments/95pct_sensors/preprocessed \
    --vocab-path outputs/vocabulary/gcode_vocabulary_v2.json \
    --sensor-report outputs/sensor_consistency_report.json \
    --threshold 95.0 \
    --seed 42
```

**Output:** `outputs/experiments/95pct_sensors/preprocessed/`
- `train_sequences.npz`
- `val_sequences.npz`
- `test_sequences.npz`
- `metadata.json`

#### Step 3: Train MM-LSTM-DAE Encoder

```bash
python scripts/evaluation/run_9class_direct.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/encoder \
    --iteration 95pct_sensors \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

**Output:** `outputs/experiments/95pct_sensors/encoder/`
- `model_best.pt`
- `metrics.json`
- `confusion_matrix.png`

#### Step 4: Train Baseline Models

XGBoost:
```bash
python scripts/evaluation/run_baseline_models.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/baselines/xgboost \
    --model xgboost \
    --seed 42
```

Random Forest:
```bash
python scripts/evaluation/run_baseline_models.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/baselines/random_forest \
    --model random_forest \
    --seed 42
```

MLP:
```bash
python scripts/evaluation/run_baseline_models.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/baselines/mlp \
    --model mlp \
    --seed 42
```

Simple LSTM:
```bash
python scripts/evaluation/run_baseline_models.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/baselines/lstm_simple \
    --model lstm_simple \
    --seed 42
```

---

## Expected Results

### Scenario A: Zero-Padding Was the Main Issue

**Before (with zero-padding):**
- All models: 100% test accuracy
- No train-test gap

**After (95% sensors, no padding):**
- Test accuracy: 70-85%
- Train accuracy: 80-95%
- Clear train-test gap (normal overfitting)

**Conclusion:** Zero-padding was creating trivial shortcuts. Models now learn real dynamics.

### Scenario B: Other Issues Dominate

**Before (with zero-padding):**
- All models: 100% test accuracy

**After (95% sensors, no padding):**
- Test accuracy: Still ~100%
- Still no train-test gap

**Conclusion:** Issue #2 (window-level splitting) and Issue #3 (sensor confounding) are stronger. Zero-padding alone wasn't the main cause.

### Scenario C: Partial Improvement

**Before (with zero-padding):**
- All models: 100% test accuracy

**After (95% sensors, no padding):**
- Test accuracy: 90-95%
- Some train-test gap appears

**Conclusion:** Zero-padding contributed to inflated accuracy, but other issues remain.

---

## Analysis Questions

After running the experiment, analyze:

1. **Did accuracy drop from 100%?**
   - If YES → Zero-padding was inflating accuracy
   - If NO → Other issues (window-level split, sensor confounding) dominate

2. **Is there a train-test gap?**
   - If YES → Normal learning, model is overfitting (good sign!)
   - If NO → Data leakage still present from other issues

3. **Are results similar across all models?**
   - If YES → Suggests dataset issues, not model-specific
   - If NO → Some models better at avoiding shortcuts

4. **Do simpler models still get high accuracy?**
   - Logistic regression > 90% → Still trivial shortcuts available
   - Logistic regression < 70% → Genuinely learning complex patterns

---

## Output Structure

```
outputs/experiments/95pct_sensors/
├── preprocessed/
│   ├── train_sequences.npz       # Training data (93 features)
│   ├── val_sequences.npz          # Validation data
│   ├── test_sequences.npz         # Test data
│   ├── metadata.json              # Preprocessing metadata
│   └── consistent_sensors.json    # Sensor configuration
├── encoder/
│   ├── model_best.pt              # Best MM-LSTM-DAE model
│   ├── metrics.json               # Training/test metrics
│   ├── confusion_matrix.png       # Confusion matrix plot
│   └── training_log.txt           # Full training log
└── baselines/
    ├── xgboost/
    │   ├── model.pkl
    │   ├── metrics.json
    │   └── confusion_matrix.png
    ├── random_forest/
    ├── logistic_regression/
    ├── mlp/
    └── lstm_simple/
```

---

## Next Steps

### If Accuracy Drops (Good!)

1. ✅ Confirms zero-padding was inflating accuracy
2. ⏭️ Next: Address Issue #2 (file-level splitting)
3. ⏭️ Then: Address Issue #3 (sensor placement confounding)
4. ⏭️ Eventually: Restore critical features (Issue #5)

### If Accuracy Stays 100% (Need More Fixes)

1. ⚠️ Zero-padding alone wasn't the main issue
2. ⏭️ **Priority:** Fix Issue #2 (window-level splitting) - this is CRITICAL
3. ⏭️ Combine with sensor filtering for maximum effect
4. ⏭️ Check if Issue #3 (sensor confounding) dominates

### Recommended Combined Fix

If accuracy remains very high (>95%), combine fixes:

```bash
# 1. Use 95% sensors (this experiment)
# 2. Add file-level splitting (instead of window-level)
# 3. This should drop accuracy to realistic 70-85%
```

See `PREPROCESSING_ISSUES_COMPREHENSIVE.md` for full fix recommendations.

---

## Comparison with Original Results

To compare with original preprocessing results:

```bash
# Original (all sensors, zero-padding)
# → outputs/original_preprocessing/

# New (95% sensors, no padding)
# → outputs/experiments/95pct_sensors/

# Compare metrics
python -c "
import json
from pathlib import Path

original = Path('outputs/original_preprocessing/encoder/metrics.json')
new = Path('outputs/experiments/95pct_sensors/encoder/metrics.json')

if original.exists() and new.exists():
    with open(original) as f: orig_data = json.load(f)
    with open(new) as f: new_data = json.load(f)

    print('Accuracy Comparison:')
    print(f\"Original (with zero-padding):  {orig_data['test_accuracy']:.2f}%\")
    print(f\"New (95% sensors, no padding): {new_data['test_accuracy']:.2f}%\")
    print(f\"Difference: {orig_data['test_accuracy'] - new_data['test_accuracy']:.2f}%\")
"
```

---

## Troubleshooting

### Issue: Vocabulary file not found

```bash
# Generate vocabulary from your G-code data
python scripts/utilities/build_vocabulary.py \
    --data-dir data/ \
    --output outputs/vocabulary/gcode_vocabulary_v2.json
```

### Issue: Preprocessing fails with dimension mismatch

Check that all CSV files have the 5 consistent sensors:
```bash
python -c "
import pandas as pd
from pathlib import Path

consistent = ['frame_l2', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

for csv in Path('data').glob('*.csv'):
    df = pd.read_csv(csv, nrows=1)
    cols = df.columns
    sensors = set()
    for col in cols:
        if '.' in col:
            sensors.add(col.split('.')[0])

    missing = [s for s in consistent if s not in sensors]
    if missing:
        print(f'{csv.name}: MISSING {missing}')
"
```

### Issue: CUDA out of memory

Reduce batch size:
```bash
# In run_9class_direct.py, use --batch-size 64 instead of 128
python scripts/evaluation/run_9class_direct.py ... --batch-size 64
```

---

## References

- **Issue Analysis:** `PREPROCESSING_ISSUES_COMPREHENSIVE.md`
- **Sensor Report:** `outputs/sensor_consistency_report.json`
- **Original Papers:** Multimodal sensor fusion for manufacturing
