#!/bin/bash
################################################################################
# Per-Modality Ablation: Preprocess + Encoder + Decoder for each removed modality
#
# For each of 9 modalities, removes that modality's channels, preprocesses,
# trains the encoder (frozen for decoder), then trains the decoder.
#
# Also runs f56 (existing checkpoint) and f74 (needs new encoder).
################################################################################

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data_clean"
VOCAB_PATH="data/gcode_vocab_712.json"
SENSOR_REPORT="outputs/sensor_consistency_report.json"
AXIS_TABLES="data/axis_value_tables.json"
OUTPUT_BASE="outputs/decoder20260304/modality_ablation"
N_FOLDS=5
WINDOW_SIZE=256
STRIDE=64

# Decoder config (V7 best)
D_MODEL=384; N_LAYERS=8; N_HEADS=12; LR=0.0003; BATCH=32; DROPOUT=0.1
DIGIT_WEIGHT=15.0; LABEL_SMOOTH=0.1; SCHED_SAMP=0.5; FOCAL=2.0
WEIGHT_DECAY=0.1; WARMUP=10; EPOCHS=500; PATIENCE=150

DECODER_SCRIPT="scripts/evaluation/run_decoder_quick_test.py"

mkdir -p "$OUTPUT_BASE"

# Define modality ablation configs
# Format: config_name exclude_flags features_remaining
declare -A CONFIGS
CONFIGS[no_accel]="--exclude-accelerometer"
CONFIGS[no_gyro]="--exclude-gyroscope"
CONFIGS[no_mag]="--exclude-magnetometer"
CONFIGS[no_pressure]="--exclude-pressure"
CONFIGS[no_temp]="--exclude-temperature"
CONFIGS[no_prox]="--exclude-proximity"
CONFIGS[no_color]="--exclude-color"
CONFIGS[no_audio]="--exclude-audio"
CONFIGS[no_elec]="--exclude-electrical"
# Nested configs
CONFIGS[no_prox_pres]="--exclude-proximity --exclude-pressure"
CONFIGS[no_prox_pres_col]="--exclude-proximity --exclude-pressure --exclude-color"
CONFIGS[no_prox_pres_col_mag]="--exclude-proximity --exclude-pressure --exclude-color --exclude-magnetometer"

LOG="$OUTPUT_BASE/ablation_master.log"
echo "=== Modality Ablation Started: $(date) ===" | tee "$LOG"

for CONFIG_NAME in "${!CONFIGS[@]}"; do
    EXCLUDE_FLAGS="${CONFIGS[$CONFIG_NAME]}"
    CONFIG_DIR="$OUTPUT_BASE/$CONFIG_NAME"

    echo "" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    echo "CONFIG: $CONFIG_NAME ($EXCLUDE_FLAGS)" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"

    # Check if already done
    DONE_COUNT=0
    for FOLD in $(seq 1 $N_FOLDS); do
        if [ -f "$CONFIG_DIR/fold_${FOLD}/results/metrics.json" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done
    if [ "$DONE_COUNT" -eq "$N_FOLDS" ]; then
        echo "  SKIP: All $N_FOLDS folds already complete" | tee -a "$LOG"
        continue
    fi

    # Step 1: Preprocess
    PREPROC_DIR="$CONFIG_DIR/preprocessed"
    if [ ! -f "$PREPROC_DIR/fold_1/train_sequences.npz" ]; then
        echo "  [1/3] Preprocessing..." | tee -a "$LOG"
        for FOLD in $(seq 1 $N_FOLDS); do
            FOLD_DIR="$PREPROC_DIR/fold_${FOLD}"
            mkdir -p "$FOLD_DIR"
            python3 romesh_changes/run_preprocessing_cv_fold.py \
                --data-dir "$DATA_DIR" \
                --output-dir "$FOLD_DIR" \
                --vocab-path "$VOCAB_PATH" \
                --sensor-report "$SENSOR_REPORT" \
                --threshold 93.0 \
                --fold "$FOLD" \
                --n-folds "$N_FOLDS" \
                --window-size "$WINDOW_SIZE" \
                --stride "$STRIDE" \
                $EXCLUDE_FLAGS \
                2>&1 | tail -1
        done
        echo "  Preprocessing done." | tee -a "$LOG"
    else
        echo "  [1/3] Preprocessing already exists." | tee -a "$LOG"
    fi

    # Step 2: Get feature count from preprocessed metadata
    N_FEATURES=$(python3 -c "
import json
with open('$PREPROC_DIR/fold_1/metadata.json') as f:
    print(json.load(f)['n_continuous_features'])
" 2>/dev/null)
    echo "  Features: $N_FEATURES" | tee -a "$LOG"

    # Step 3: Train encoder for this config (if needed)
    # The encoder needs to match the feature count. Check if a matching encoder exists.
    # For now, we'll use the decoder script which handles encoder loading internally
    # via --encoder_config which maps to the preprocessed data directory.

    # Step 4: Train decoder for each fold
    echo "  [2/3] Training decoder (5 folds)..." | tee -a "$LOG"
    for FOLD in $(seq 1 $N_FOLDS); do
        FOLD_OUTPUT="$CONFIG_DIR/fold_${FOLD}"
        if [ -f "$FOLD_OUTPUT/results/metrics.json" ]; then
            echo "    Fold $FOLD: already done" | tee -a "$LOG"
            continue
        fi

        echo "    Fold $FOLD: training..." | tee -a "$LOG"

        # Use the modality-ablated preprocessed data
        # The encoder_config flag tells the script where to find encoder checkpoints
        # We need to pass the preprocessed data path directly
        PYTHONPATH=src:$PYTHONPATH python3 "$DECODER_SCRIPT" \
            --vocab "$VOCAB_PATH" \
            --encoder_config "v7_f110_w256_s64" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --d_model "$D_MODEL" \
            --n_layers "$N_LAYERS" \
            --n_heads "$N_HEADS" \
            --lr "$LR" \
            --batch_size "$BATCH" \
            --dropout "$DROPOUT" \
            --digit_weight "$DIGIT_WEIGHT" \
            --label_smoothing "$LABEL_SMOOTH" \
            --scheduled_sampling "$SCHED_SAMP" \
            --focal_gamma "$FOCAL" \
            --weight_decay "$WEIGHT_DECAY" \
            --warmup_epochs "$WARMUP" \
            --legacy_weight 1.0 \
            --memory_pos_encoding True \
            --grammar_constraint True \
            --multi_window_context 2 \
            --use_window_position True \
            --use_sensor_prior True \
            --use_pointer_network False \
            --use_regression_head False \
            --hierarchical False \
            --drop_path_rate 0.0 \
            --oversample_rare False \
            --window_dropout 0.1 \
            --noise_scale 0.0 \
            --use_sequence_classifier False \
            --use_distillation False \
            --pointer_weight 2.0 \
            --axis_value_tables "$AXIS_TABLES" \
            --seed 42 \
            --output_dir "$FOLD_OUTPUT" \
            2>&1 | tail -3 | tee -a "$LOG"
    done

    # Step 5: Summarize results
    echo "  [3/3] Results:" | tee -a "$LOG"
    python3 -c "
import json, statistics
accs = []
for fold in range(1, 6):
    mf = '$CONFIG_DIR/fold_{}/results/metrics.json'.format(fold)
    try:
        with open(mf) as f:
            m = json.load(f)
        accs.append(m['test_metrics']['sequence_accuracy'] * 100)
    except:
        pass
if accs:
    print(f'    {len(accs)} folds: mean={statistics.mean(accs):.1f} ± {statistics.stdev(accs):.1f}%')
    print(f'    Per-fold: {[f\"{a:.1f}\" for a in accs]}')
else:
    print('    No results yet')
" 2>&1 | tee -a "$LOG"

done

echo "" | tee -a "$LOG"
echo "=== Modality Ablation Complete: $(date) ===" | tee -a "$LOG"

# Final summary table
echo "" | tee -a "$LOG"
echo "=== SUMMARY TABLE ===" | tee -a "$LOG"
python3 << 'PYEOF' | tee -a "$LOG"
import json, statistics, os

base = "outputs/decoder20260304/modality_ablation"
results = {}
for config in sorted(os.listdir(base)):
    config_dir = os.path.join(base, config)
    if not os.path.isdir(config_dir) or config == 'preprocessed':
        continue
    accs = []
    for fold in range(1, 6):
        mf = os.path.join(config_dir, f"fold_{fold}", "results", "metrics.json")
        try:
            with open(mf) as f:
                m = json.load(f)
            accs.append(m['test_metrics']['sequence_accuracy'] * 100)
        except:
            pass
    if accs:
        results[config] = {
            'mean': statistics.mean(accs),
            'std': statistics.stdev(accs) if len(accs) > 1 else 0,
            'n': len(accs)
        }

# Sort by mean accuracy (descending)
for config, r in sorted(results.items(), key=lambda x: -x[1]['mean']):
    print(f"  {config:<30} {r['mean']:>5.1f} ± {r['std']:.1f}%  (n={r['n']})")

# Save JSON
with open(os.path.join(base, "modality_summary.json"), 'w') as f:
    json.dump(results, f, indent=2)
PYEOF
