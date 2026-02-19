# Romesh's Changes - 95% Sensor Experiment

This directory contains Python scripts created to address **Issue #1 (Zero-Padding)** from the preprocessing analysis.

## Files in This Directory

### 1. `identify_consistent_sensors.py`
**Purpose:** Analyze sensor consistency across all CSV files

**Original Location:** `scripts/analysis/identify_consistent_sensors.py`

**What it does:**
- Scans all CSV files in the data directory
- For each sensor location, checks:
  - **Presence**: Is the sensor in the file (columns exist)?
  - **Activity**: Does it have actual signal (std > 1e-6, not constant zeros)?
- Calculates coverage percentages
- Identifies sensors with ≥95% activity (no zero-padding needed)

**Usage:**
```bash
python romesh_changes/identify_consistent_sensors.py \
    --data-dir data/ \
    --threshold 95.0 \
    --output outputs/sensor_consistency_report.json
```

**Output:** `outputs/sensor_consistency_report.json` with sensor statistics

---

### 2. `run_preprocessing_consistent_sensors.py`
**Purpose:** Preprocess data using only sensors with ≥95% activity

**Original Location:** `scripts/preprocessing/run_preprocessing_consistent_sensors.py`

**What it does:**
- Loads sensor consistency report
- Filters to only use consistent sensors (eliminates zero-padding)
- Runs preprocessing with filtered column list
- Creates train/val/test splits with 93 features (85 sensors + 8 electrical)

**Usage:**
```bash
python romesh_changes/run_preprocessing_consistent_sensors.py \
    --data-dir data/ \
    --output-dir outputs/experiments/95pct_sensors/preprocessed \
    --vocab-path outputs/vocabulary/gcode_vocabulary_v2.json \
    --sensor-report outputs/sensor_consistency_report.json \
    --threshold 95.0 \
    --seed 42
```

**Output:** Preprocessed NPZ files in specified output directory

---

## Related Files (Not in This Directory)

### Bash Orchestration Script
**Location:** `scripts/experiments/run_95pct_sensors_experiment.sh`

**Purpose:** Runs the complete experiment pipeline:
1. Sensor consistency analysis
2. Preprocessing with filtered sensors
3. Train MM-LSTM-DAE encoder
4. Train baseline models
5. Generate comparison report

### Documentation
**Location:** `EXPERIMENT_95PCT_SENSORS.md` (project root)

**Purpose:** Complete documentation of the 95% sensor experiment including:
- Methodology
- Expected results
- Troubleshooting
- Analysis guide

---

## Quick Start

### Step 1: Analyze Sensors
```bash
python romesh_changes/identify_consistent_sensors.py \
    --data-dir data/ \
    --threshold 95.0
```

### Step 2: Preprocess with Consistent Sensors
```bash
python romesh_changes/run_preprocessing_consistent_sensors.py \
    --data-dir data/ \
    --output-dir outputs/experiments/95pct_sensors/preprocessed \
    --vocab-path outputs/vocabulary/gcode_vocabulary_v2.json
```

### Step 3: Train and Evaluate (use existing scripts)
```bash
# Train MM-LSTM-DAE
python scripts/evaluation/run_9class_direct.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/encoder

# Train baselines
python scripts/evaluation/run_baseline_models.py \
    --data-dir outputs/experiments/95pct_sensors/preprocessed \
    --output-dir outputs/experiments/95pct_sensors/baselines/xgboost \
    --model xgboost
```

---

## Expected Results

**Current (with zero-padding):**
- 280 features (16 sensors × 17 + 8 electrical)
- Missing sensors filled with zeros
- All models: 100% accuracy (shortcut learning!)

**New (95% sensors, no padding):**
- 93 features (5 sensors × 17 + 8 electrical)
- NO zero-padding - all sensors present
- Expected accuracy: 70-85% (realistic learning)

---

## Sensor Details

The 5 sensors with ≥95% activity:

| Sensor | Activity % | Channels | Notes |
|--------|-----------|----------|-------|
| frame_r2 | 100.0% | 17 | Frame right position 2 |
| y_bed__3 | 100.0% | 17 | Y-bed position 3 |
| frame_l2 | 98.5% | 17 | Frame left position 2 |
| spindle2 | 97.8% | 17 | Spindle sensor 2 |
| y_bed__4 | 95.6% | 17 | Y-bed position 4 |

Each sensor has 17 channels:
- Accelerometer: Ax, Ay, Az
- Gyroscope: Gx, Gy, Gz
- Magnetometer: Mx, My, Mz
- Environmental: Pressure, Temperature, Proximity
- Color: ColorR, ColorG, ColorB, ColorA
- Audio: RMS

---

## Changes Made to Original Code

These scripts are **new additions** - they don't modify existing code. They provide:

1. **New analysis capability**: Identify which sensors are consistently present
2. **New preprocessing option**: Filter to consistent sensors only
3. **Backward compatible**: Original preprocessing still works

---

## Testing the Impact

To measure the impact of eliminating zero-padding:

1. **Run original preprocessing** (with zero-padding)
2. **Run this experiment** (without zero-padding)
3. **Compare accuracies**:
   - If accuracy drops: Zero-padding was inflating results ✅
   - If accuracy stays 100%: Other issues dominate ⚠️

---

## Next Steps After This Experiment

If this experiment shows accuracy is still 100%:
- **Issue #2** (window-level splitting) is likely the dominant problem
- Need to implement file-level splitting next
- See `PREPROCESSING_ISSUES_COMPREHENSIVE.md` for details

If accuracy drops to 70-85%:
- ✅ Confirms zero-padding was a major issue
- Continue fixing remaining issues (file-level split, feature restoration, etc.)

---

## Contact

For questions about these scripts, refer to:
- `PREPROCESSING_ISSUES_COMPREHENSIVE.md` - Full issue analysis
- `EXPERIMENT_95PCT_SENSORS.md` - Experiment documentation
