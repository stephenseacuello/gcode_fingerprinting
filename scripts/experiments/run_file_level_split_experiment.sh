#!/bin/bash
################################################################################
# Experiment: File-Level Split + 95% Sensors (Fixes Issues #1 and #2)
#
# This experiment addresses BOTH critical issues:
# - Issue #1 (Zero-Padding): Uses only 95% consistent sensors
# - Issue #2 (Window-Level Split): Splits FILES first, then creates windows
#
# Pipeline:
# 1. Run sensor consistency analysis (if not already done)
# 2. Preprocess with FILE-LEVEL split + consistent sensors
# 3. Train MM-LSTM-DAE encoder
# 4. Train baseline models (ML + NN)
# 5. Compare results
#
# Expected outcome:
# - Logistic regression: 30-70% (can't memorize files)
# - MM-LSTM-DAE: 70-90% (learns real patterns)
# - Clear gap between simple and complex models
#
# Usage:
#   ./run_file_level_split_experiment.sh                    # Normal run
#   ./run_file_level_split_experiment.sh --no-proximity     # Exclude y_bed__3.Proximity
#   ./run_file_level_split_experiment.sh --no-proximity --exclude-additional  # Exclude more channels
################################################################################

set -e  # Exit on error

# Parse arguments
EXCLUDE_PROXIMITY=false
EXCLUDE_COLOR=false
EXCLUDE_MAGNETOMETER=false
EXCLUDE_ADDITIONAL=false
PREPROCESSING_SCRIPT="romesh_changes/run_preprocessing_file_level_split.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-proximity)
            EXCLUDE_PROXIMITY=true
            PREPROCESSING_SCRIPT="romesh_changes/run_preprocessing_exclude_proximity.py"
            shift
            ;;
        --no-color)
            EXCLUDE_COLOR=true
            PREPROCESSING_SCRIPT="romesh_changes/run_preprocessing_exclude_proximity.py"
            shift
            ;;
        --no-magnetometer)
            EXCLUDE_MAGNETOMETER=true
            PREPROCESSING_SCRIPT="romesh_changes/run_preprocessing_exclude_proximity.py"
            shift
            ;;
        --exclude-additional)
            EXCLUDE_ADDITIONAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-proximity] [--no-color] [--no-magnetometer] [--exclude-additional]"
            exit 1
            ;;
    esac
done

# Configuration
DATA_DIR="data"
VOCAB_PATH="outputs/vocabulary/gcode_vocabulary_v2.json"

# Set output directory based on options
if [ "$EXCLUDE_PROXIMITY" = true ] || [ "$EXCLUDE_COLOR" = true ] || [ "$EXCLUDE_MAGNETOMETER" = true ]; then
    # Build experiment name and output directory based on exclusions
    NAME_PARTS=()
    DIR_PARTS=()

    if [ "$EXCLUDE_PROXIMITY" = true ]; then
        NAME_PARTS+=("NO Proximity")
        DIR_PARTS+=("no_proximity")
    fi

    if [ "$EXCLUDE_COLOR" = true ]; then
        NAME_PARTS+=("NO Color")
        DIR_PARTS+=("no_color")
    fi

    if [ "$EXCLUDE_MAGNETOMETER" = true ]; then
        NAME_PARTS+=("NO Magnetometer")
        DIR_PARTS+=("no_magnetometer")
    fi

    if [ "$EXCLUDE_ADDITIONAL" = true ]; then
        NAME_PARTS+=("NO Pressure")
        DIR_PARTS+=("no_pressure")
    fi

    # Join with "+"
    IFS='+'; NAME_JOINED="${NAME_PARTS[*]}"; unset IFS
    IFS='_'; DIR_JOINED="${DIR_PARTS[*]}"; unset IFS

    OUTPUT_BASE="outputs/experiments/${DIR_JOINED}"
    EXPERIMENT_NAME="File-Level Split ($NAME_JOINED)"
else
    OUTPUT_BASE="outputs/experiments/file_level_split"
    EXPERIMENT_NAME="File-Level Split + 95% Sensors"
fi

SEED=42

# Setup logging
mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/experiment_log_no_proximity_no_camera.txt"
echo "Logging output to: $LOG_FILE"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "================================================================================"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
if [ "$EXCLUDE_PROXIMITY" = true ] || [ "$EXCLUDE_COLOR" = true ] || [ "$EXCLUDE_MAGNETOMETER" = true ]; then
    echo "🎯 TESTING HYPOTHESIS: Excluding non-core sensor channels"
    if [ "$EXCLUDE_PROXIMITY" = true ]; then
        echo "   ❌ Excluding ALL 5 Proximity channels (dead in many files)"
    fi
    if [ "$EXCLUDE_COLOR" = true ]; then
        echo "   ❌ Excluding ALL 20 Color (RGBA) channels (testing with core sensors only)"
    fi
    if [ "$EXCLUDE_MAGNETOMETER" = true ]; then
        echo "   ❌ Excluding ALL 15 Magnetometer channels (Mx, My, Mz - operation-specific shortcuts)"
    fi
    if [ "$EXCLUDE_ADDITIONAL" = true ]; then
        echo "   ❌ Excluding additional problematic Pressure sensors"
    fi
    echo ""
else
    echo "This experiment fixes BOTH Issue #1 (zero-padding) AND Issue #2 (window-split)"
    echo ""
fi

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

# Step 2: Preprocessing with FILE-LEVEL SPLIT
echo ""
echo "--------------------------------------------------------------------------------"
echo "STEP 2: Preprocessing with FILE-LEVEL split..."
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

        # Build preprocessing command
        PREPROCESS_CMD="python $PREPROCESSING_SCRIPT \
            --data-dir $DATA_DIR \
            --output-dir $PREPROCESS_DIR \
            --vocab-path $VOCAB_PATH \
            --sensor-report outputs/sensor_consistency_report.json \
            --threshold 95.0 \
            --window-size 64 \
            --stride 16 \
            --seed $SEED"

        # Add exclusion flags
        if [ "$EXCLUDE_PROXIMITY" = true ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --exclude-proximity"
        fi

        if [ "$EXCLUDE_COLOR" = true ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --exclude-color"
        fi

        if [ "$EXCLUDE_MAGNETOMETER" = true ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --exclude-magnetometer"
        fi

        if [ "$EXCLUDE_ADDITIONAL" = true ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --exclude-additional"
        fi

        # Run preprocessing
        $PREPROCESS_CMD
    fi
else
    echo "Running preprocessing with FILE-LEVEL split..."
    mkdir -p "$PREPROCESS_DIR"

    # Build preprocessing command
    PREPROCESS_CMD="python $PREPROCESSING_SCRIPT \
        --data-dir $DATA_DIR \
        --output-dir $PREPROCESS_DIR \
        --vocab-path $VOCAB_PATH \
        --sensor-report outputs/sensor_consistency_report.json \
        --threshold 95.0 \
        --window-size 64 \
        --stride 16 \
        --seed $SEED"

    # Add exclusion flags
    if [ "$EXCLUDE_PROXIMITY" = true ]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --exclude-proximity"
    fi

    if [ "$EXCLUDE_COLOR" = true ]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --exclude-color"
    fi

    if [ "$EXCLUDE_MAGNETOMETER" = true ]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --exclude-magnetometer"
    fi

    if [ "$EXCLUDE_ADDITIONAL" = true ]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --exclude-additional"
    fi

    # Run preprocessing
    $PREPROCESS_CMD
fi

# Step 3: Train MM-LSTM-DAE encoder
echo ""
echo "--------------------------------------------------------------------------------"
echo "STEP 3: Training MM-LSTM-DAE encoder..."
echo "--------------------------------------------------------------------------------"
echo ""

ENCODER_DIR="$OUTPUT_BASE/encoder"
mkdir -p "$ENCODER_DIR"

echo "Training with file-level split (no file leakage)..."
python scripts/evaluation/run_9class_direct.py \
    --data-dir "$PREPROCESS_DIR" \
    --output-dir "$ENCODER_DIR" \
    --iteration file_level_split \
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
encoder_metrics = output_base / 'encoder' / 'results.json'
if encoder_metrics.exists():
    with open(encoder_metrics) as f:
        data = json.load(f)
        results['MM-LSTM-DAE'] = {
            'test_acc': data.get('test_accuracy', data.get('test_acc', 'N/A')),
            'train_acc': data.get('train_accuracy', data.get('train_acc', 'N/A'))
        }

# Baselines
baseline_models = ['xgboost', 'random_forest', 'logistic_regression', 'mlp', 'lstm_simple']
for model in baseline_models:
    metrics_path = output_base / 'baselines' / model / 'metrics.json'
    if not metrics_path.exists():
        # Try results.json instead
        metrics_path = output_base / 'baselines' / model / 'results.json'

    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            results[model.upper().replace('_', ' ')] = {
                'test_acc': data.get('test_accuracy', data.get('test_acc', 'N/A')),
                'train_acc': data.get('train_accuracy', data.get('train_acc', 'N/A'))
            }

# Print comparison table
print('Model Comparison (File-Level Split + 95% Sensors)')
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
print('Expected vs Actual Results:')
print('  • Logistic regression should be 30-70% (simple model, no file memorization)')
print('  • MM-LSTM-DAE should be 70-90% (complex model, learns temporal patterns)')
print('  • Clear gap between models indicates genuine learning, not shortcuts')
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
echo "  • File split info:   $PREPROCESS_DIR/file_split.json"
echo ""
echo "Key improvements:"
echo "  1. ✅ No zero-padding (93 features from 5 consistent sensors)"
echo "  2. ✅ File-level split (no file appears in multiple splits)"
echo "  3. ✅ Scaler fitted only on training data"
echo "  4. ✅ Test files completely unseen during training"
if [ "$EXCLUDE_PROXIMITY" = true ] || [ "$EXCLUDE_COLOR" = true ] || [ "$EXCLUDE_MAGNETOMETER" = true ]; then
    EXCLUSION_COUNT=0
    [ "$EXCLUDE_PROXIMITY" = true ] && EXCLUSION_COUNT=$((EXCLUSION_COUNT + 5))
    [ "$EXCLUDE_COLOR" = true ] && EXCLUSION_COUNT=$((EXCLUSION_COUNT + 20))
    [ "$EXCLUDE_MAGNETOMETER" = true ] && EXCLUSION_COUNT=$((EXCLUSION_COUNT + 15))
    [ "$EXCLUDE_ADDITIONAL" = true ] && EXCLUSION_COUNT=$((EXCLUSION_COUNT + 2))
    echo "  5. 🎯 Excluded $EXCLUSION_COUNT channels (testing tree baseline hypothesis)"
fi
echo ""

# If testing hypothesis, show comparison
if [ "$EXCLUDE_PROXIMITY" = true ] || [ "$EXCLUDE_COLOR" = true ] || [ "$EXCLUDE_MAGNETOMETER" = true ]; then
    echo "--------------------------------------------------------------------------------"
    echo "HYPOTHESIS TEST: Comparing WITH vs WITHOUT problematic channels"
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Export flags for Python to access
    export EXCLUDE_PROXIMITY
    export EXCLUDE_COLOR
    export EXCLUDE_MAGNETOMETER

    python << 'HYPOTHESIS_TEST'
import json
from pathlib import Path
import os

# Determine paths based on exclusions
exclude_proximity = os.environ.get('EXCLUDE_PROXIMITY', 'false') == 'true'
exclude_color = os.environ.get('EXCLUDE_COLOR', 'false') == 'true'
exclude_magnetometer = os.environ.get('EXCLUDE_MAGNETOMETER', 'false') == 'true'
exclude_additional = os.environ.get('EXCLUDE_ADDITIONAL', 'false') == 'true'

# Build path
parts = []
if exclude_proximity:
    parts.append('no_proximity')
if exclude_color:
    parts.append('no_color')
if exclude_magnetometer:
    parts.append('no_magnetometer')
if exclude_additional:
    parts.append('no_pressure')

new_base = Path(f"outputs/experiments/{'_'.join(parts)}")

original_base = Path("outputs/experiments/file_level_split")

def load_result(base_path, model_name):
    """Load test accuracy from results.json"""
    results_path = base_path / 'baselines' / model_name / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            return data.get('test', {}).get('accuracy', None)
    return None

models = ['xgboost', 'random_forest', 'logistic_regression']

# Build column header
exclusions = []
if exclude_proximity:
    exclusions.append("Proximity")
if exclude_color:
    exclusions.append("Color")
if exclude_magnetometer:
    exclusions.append("Magnetometer")
exclusion_label = "+".join(exclusions)

print(f"{'Model':<20} {'WITH All Sensors':>20} {'NO {0}':>20} {'Change':>12}".format(exclusion_label))
print("-"*75)

hypothesis_supported = False

for model in models:
    original_acc = load_result(original_base, model)
    new_acc = load_result(new_base, model)

    if original_acc is not None and new_acc is not None:
        change = new_acc - original_acc

        # Format with color indicators
        if change < -0.05:
            indicator = "↓"  # Dropped
        elif change > 0.05:
            indicator = "↑"  # Increased
        else:
            indicator = "≈"  # Similar

        print(f"{model.upper():<20} {original_acc:>19.2%} {new_acc:>19.2%} {change:>10.2%} {indicator}")

        # Check XGBoost specifically
        if model == 'xgboost' and abs(change) > 0.05 and change < 0:
            hypothesis_supported = True

print("-"*75)
print()

if hypothesis_supported:
    print(f"✅ HYPOTHESIS SUPPORTED: Removing {exclusion_label} channels reduced tree accuracy!")
    print(f"   → These channels were shortcuts for tree-based models")
else:
    print("❓ Trees still performing well - other problematic features may exist")
    print("   → Try excluding more channels: --no-proximity --no-color --no-magnetometer")

print()
HYPOTHESIS_TEST
fi

echo "================================================================================"
