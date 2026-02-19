#!/bin/bash
################################################################################
# Experiment: 95% Consistent Sensors (No Zero-Padding)
#
# This experiment addresses Issue #1 (Zero-Padding) by using only sensors
# that are present and active in ≥95% of files.
#
# Pipeline:
# 1. Run sensor consistency analysis (if not already done)
# 2. Preprocess data with only consistent sensors
# 3. Train MM-LSTM-DAE encoder
# 4. Train baseline models (ML + NN)
# 5. Compare results
#
# Expected outcome:
# - Eliminates zero-padding shortcuts
# - Forces model to learn from actual sensor dynamics
# - May reduce accuracy from 100% to more realistic 70-85%
################################################################################

set -e  # Exit on error

# Configuration
DATA_DIR="data"
VOCAB_PATH="outputs/vocabulary/gcode_vocabulary_v2.json"
OUTPUT_BASE="outputs/experiments/95pct_sensors"
SEED=42

# Setup logging
mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/experiment_log.txt"
echo "Logging output to: $LOG_FILE"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "Experiment: 95% Consistent Sensors (No Zero-Padding)"
echo "================================================================================"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "This experiment eliminates Issue #1 by using only sensors with ≥95% activity"
echo ""

# Step 1: Sensor consistency analysis
echo "--------------------------------------------------------------------------------"
echo "STEP 1: Analyzing sensor consistency..."
echo "--------------------------------------------------------------------------------"
echo ""

if [ -f "outputs/sensor_consistency_report.json" ]; then
    echo "✓ Sensor consistency report already exists"
    echo "  Using: outputs/sensor_consistency_report.json"
else
    echo "Running sensor consistency analysis..."
    python scripts/analysis/identify_consistent_sensors.py \
        --data-dir "$DATA_DIR" \
        --threshold 95.0 \
        --output outputs/sensor_consistency_report.json
fi

echo ""
echo "Sensors with ≥95% activity:"
python -c "
import json
with open('outputs/sensor_consistency_report.json', 'r') as f:
    data = json.load(f)
sensors = [name for name, info in data['sensors'].items() if info['activity_percentage'] >= 95.0]
for s in sorted(sensors):
    info = data['sensors'][s]
    print(f\"  • {s:<15} - {info['activity_percentage']:>5.1f}% active ({info['num_channels']} channels)\")
total_channels = sum(data['sensors'][s]['num_channels'] for s in sensors)
print(f\"\\nTotal: {len(sensors)} sensors × 17 channels = {total_channels} features + 8 electrical = {total_channels + 8} total\")
"

# Step 2: Preprocessing
echo ""
echo "--------------------------------------------------------------------------------"
echo "STEP 2: Preprocessing with consistent sensors only..."
echo "--------------------------------------------------------------------------------"
echo ""

PREPROCESS_DIR="$OUTPUT_BASE/preprocessed"

if [ -d "$PREPROCESS_DIR" ] && [ -f "$PREPROCESS_DIR/metadata.json" ]; then
    echo "⚠️  Preprocessed data already exists at: $PREPROCESS_DIR"
    read -p "Re-run preprocessing? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping preprocessing step"
    else
        echo "Re-running preprocessing..."
        python scripts/preprocessing/run_preprocessing_consistent_sensors.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$PREPROCESS_DIR" \
            --vocab-path "$VOCAB_PATH" \
            --sensor-report outputs/sensor_consistency_report.json \
            --threshold 95.0 \
            --window-size 64 \
            --stride 16 \
            --seed $SEED
    fi
else
    echo "Running preprocessing..."
    mkdir -p "$PREPROCESS_DIR"
    python scripts/preprocessing/run_preprocessing_consistent_sensors.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PREPROCESS_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --sensor-report outputs/sensor_consistency_report.json \
        --threshold 95.0 \
        --window-size 64 \
        --stride 16 \
        --seed $SEED
fi

# Step 3: Train MM-LSTM-DAE encoder
echo ""
echo "--------------------------------------------------------------------------------"
echo "STEP 3: Training MM-LSTM-DAE encoder..."
echo "--------------------------------------------------------------------------------"
echo ""

ENCODER_DIR="$OUTPUT_BASE/encoder"
mkdir -p "$ENCODER_DIR"

echo "Training with consistent sensors (93 features, no zero-padding)..."
python scripts/evaluation/run_9class_direct.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$ENCODER_DIR" \
    --iteration 95pct_sensors \
    --max_epochs 100 \
    --seed $SEED

# Step 4: Train baseline models
echo ""
echo "--------------------------------------------------------------------------------"
echo "STEP 4: Training baseline models..."
echo "--------------------------------------------------------------------------------"
echo ""

BASELINE_DIR="$OUTPUT_BASE/baselines"
mkdir -p "$BASELINE_DIR"

# ML baselines
echo ""
echo "4a. XGBoost baseline..."
python scripts/evaluation/run_baseline_models.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$BASELINE_DIR/xgboost" \
    --model xgboost \
    --seed $SEED

echo ""
echo "4b. Random Forest baseline..."
python scripts/evaluation/run_baseline_models.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$BASELINE_DIR/random_forest" \
    --model random_forest \
    --seed $SEED

echo ""
echo "4c. Logistic Regression baseline..."
python scripts/evaluation/run_baseline_models.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$BASELINE_DIR/logistic_regression" \
    --model logistic_regression \
    --seed $SEED

# NN baselines
echo ""
echo "4d. MLP baseline..."
python scripts/evaluation/run_baseline_models.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$BASELINE_DIR/mlp" \
    --model mlp \
    --seed $SEED

echo ""
echo "4e. Simple LSTM baseline..."
python scripts/evaluation/run_baseline_models.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$BASELINE_DIR/lstm_simple" \
    --model lstm_simple \
    --seed $SEED

# Step 5: Compare results
echo ""
echo "================================================================================"
echo "STEP 5: Results Comparison"
echo "================================================================================"
echo ""

python -c "
import json
from pathlib import Path

output_base = Path('$OUTPUT_BASE')

# Collect results
results = {}

# Encoder
encoder_metrics = output_base / 'encoder' / 'metrics.json'
if encoder_metrics.exists():
    with open(encoder_metrics) as f:
        data = json.load(f)
        results['MM-LSTM-DAE'] = {
            'test_acc': data.get('test_accuracy', 'N/A'),
            'train_acc': data.get('train_accuracy', 'N/A')
        }

# Baselines
baseline_models = ['xgboost', 'random_forest', 'logistic_regression', 'mlp', 'lstm_simple']
for model in baseline_models:
    metrics_path = output_base / 'baselines' / model / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            results[model.upper().replace('_', ' ')] = {
                'test_acc': data.get('test_accuracy', 'N/A'),
                'train_acc': data.get('train_accuracy', 'N/A')
            }

# Print comparison table
print('Model Comparison (95% Consistent Sensors - No Zero-Padding)')
print('-' * 80)
print(f\"{'Model':<25} {'Train Acc':>15} {'Test Acc':>15} {'Gap':>10}\")
print('-' * 80)

for model_name, metrics in sorted(results.items()):
    train_acc = metrics['train_acc']
    test_acc = metrics['test_acc']

    if isinstance(train_acc, (int, float)) and isinstance(test_acc, (int, float)):
        gap = train_acc - test_acc
        print(f\"{model_name:<25} {train_acc:>14.2f}% {test_acc:>14.2f}% {gap:>9.2f}%\")
    else:
        print(f\"{model_name:<25} {str(train_acc):>15} {str(test_acc):>15} {'N/A':>10}\")

print('-' * 80)
print()
print('Expected Results:')
print('  • Accuracy should DROP from previous 100% (this is GOOD!)')
print('  • Realistic accuracy: 70-85% (depends on remaining issues)')
print('  • Train-test gap should exist (overfitting indication)')
print('  • If still 100%, other issues (window-level split, etc.) dominate')
print()
"

echo "================================================================================"
echo "✅ Experiment Complete!"
echo "================================================================================"
echo "Finished at: $(date)"
echo ""
echo "Output directory: $OUTPUT_BASE"
echo "Log file: $LOG_FILE"
echo ""
echo "Key files:"
echo "  • Preprocessed data: $PREPROCESS_DIR"
echo "  • Encoder results:   $ENCODER_DIR"
echo "  • Baseline results:  $BASELINE_DIR"
echo ""
echo "Next steps:"
echo "  1. Review results to see if accuracy dropped from 100%"
echo "  2. If still 100%, Issue #2 (window-level split) likely dominates"
echo "  3. Compare with original results to quantify impact of zero-padding"
echo ""
echo "================================================================================"
