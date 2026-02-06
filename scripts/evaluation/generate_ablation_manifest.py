#!/usr/bin/env python3
"""Generate Ablation Study Manifest.

Creates a JSON manifest of all ablation experiments to run.
Supports full (3×3) or reduced (3 model seeds only) configurations.

Usage:
    # Full study (1,143 runs)
    python scripts/evaluation/generate_ablation_manifest.py \
        --output outputs/ablation_study_2026_02_06/manifest.json \
        --mode full

    # Reduced study (381 runs)
    python scripts/evaluation/generate_ablation_manifest.py \
        --output outputs/ablation_study_2026_02_06/manifest.json \
        --mode reduced

    # Specific studies only
    python scripts/evaluation/generate_ablation_manifest.py \
        --output outputs/ablation_study_2026_02_06/manifest.json \
        --studies modality_grouped,architecture
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_SPLIT_SEEDS = [42, 123, 456]
MODEL_SEEDS = [42, 123, 456]

# Modality groups
ALL_MODALITIES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental', 'color', 'rms', 'electrical']

# Individual components
ALL_COMPONENTS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz',
                  'Pressure', 'Temperature', 'Proximity',
                  'ColorR', 'ColorG', 'ColorB', 'ColorA', 'RMS']

# Sensors
ALL_SENSORS = ['frame_b1', 'frame_b2', 'frame_l1', 'frame_l2', 'frame_l3', 'frame_r1', 'frame_r2',
               'spindle1', 'spindle2', 'xa_motor',
               'y_bed__1', 'y_bed__2', 'y_bed__3', 'y_bed__4',
               'z_gant_1', 'z_gant_2']

# Sensor groups
SENSOR_GROUPS = ['frame', 'spindle', 'motor', 'bed', 'gantry']

# Architecture configs
ARCHITECTURE_CONFIGS = {
    # Size ablation
    'size_tiny': {'d_model': 64, 'lstm_layers': 1, 'n_heads': 2},
    'size_small': {'d_model': 128, 'lstm_layers': 1, 'n_heads': 4},
    'size_base': {'d_model': 256, 'lstm_layers': 2, 'n_heads': 4},  # baseline
    'size_large': {'d_model': 512, 'lstm_layers': 2, 'n_heads': 8},
    'size_xlarge': {'d_model': 512, 'lstm_layers': 3, 'n_heads': 8},

    # Component ablation
    'no_lstm': {'use_lstm': False, 'use_attention': True},
    'no_attention': {'use_lstm': True, 'use_attention': False},
    'single_lstm_layer': {'lstm_layers': 1},
    'triple_lstm_layer': {'lstm_layers': 3},
    'heads_2': {'n_heads': 2},
    'heads_8': {'n_heads': 8},

    # Regularization ablation
    'no_dropout': {'dropout': 0.0},
    'high_dropout': {'dropout': 0.4},
    'no_modality_dropout': {'modality_dropout': 0.0},
    'high_modality_dropout': {'modality_dropout': 0.3},
    'no_label_smoothing': {'label_smoothing': 0.0},
    'high_label_smoothing': {'label_smoothing': 0.2},

    # Pooling ablation
    'pooling_last': {'cls_pooling': 'last'},  # baseline
    'pooling_mean': {'cls_pooling': 'mean'},
    'pooling_max': {'cls_pooling': 'max'},
    'pooling_attention': {'cls_pooling': 'attention_pool'},

    # Training ablation
    'no_class_weights': {'use_class_weights': False},
    'lr_low': {'lr': 0.0001},
    'lr_high': {'lr': 0.01},
}

# Baseline models
BASELINE_ML_MODELS = ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression', 'knn']
BASELINE_NN_MODELS = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']
ALL_BASELINE_MODELS = BASELINE_ML_MODELS + BASELINE_NN_MODELS


def generate_modality_grouped_experiments():
    """Generate modality group ablation experiments."""
    experiments = []

    # Baseline (all modalities)
    experiments.append({
        'study': 'modality_grouped',
        'config_name': 'baseline',
        'ablation_type': 'baseline',
        'ablation_config': {},
    })

    # LOO (remove one modality)
    for mod in ALL_MODALITIES:
        experiments.append({
            'study': 'modality_grouped',
            'config_name': f'loo_{mod}',
            'ablation_type': 'loo',
            'ablation_config': {'exclude_modalities': [mod]},
        })

    # LOI (keep only one modality)
    for mod in ALL_MODALITIES:
        experiments.append({
            'study': 'modality_grouped',
            'config_name': f'loi_{mod}',
            'ablation_type': 'loi',
            'ablation_config': {'include_only_modalities': [mod]},
        })

    return experiments


def generate_modality_individual_experiments():
    """Generate individual modality component ablation experiments."""
    experiments = []

    # Baseline (all components)
    experiments.append({
        'study': 'modality_individual',
        'config_name': 'baseline',
        'ablation_type': 'baseline',
        'ablation_config': {},
    })

    # LOO (remove one component)
    for comp in ALL_COMPONENTS:
        experiments.append({
            'study': 'modality_individual',
            'config_name': f'loo_{comp}',
            'ablation_type': 'loo',
            'ablation_config': {'exclude_components': [comp]},
        })

    # LOI (keep only one component)
    for comp in ALL_COMPONENTS:
        experiments.append({
            'study': 'modality_individual',
            'config_name': f'loi_{comp}',
            'ablation_type': 'loi',
            'ablation_config': {'include_only_components': [comp]},
        })

    return experiments


def generate_sensor_individual_experiments():
    """Generate individual sensor ablation experiments."""
    experiments = []

    # Baseline (all sensors)
    experiments.append({
        'study': 'sensor_individual',
        'config_name': 'baseline',
        'ablation_type': 'baseline',
        'ablation_config': {},
    })

    # LOO (remove one sensor)
    for sensor in ALL_SENSORS:
        experiments.append({
            'study': 'sensor_individual',
            'config_name': f'loo_{sensor}',
            'ablation_type': 'loo',
            'ablation_config': {'exclude_sensors': [sensor]},
        })

    # LOI (keep only one sensor)
    for sensor in ALL_SENSORS:
        experiments.append({
            'study': 'sensor_individual',
            'config_name': f'loi_{sensor}',
            'ablation_type': 'loi',
            'ablation_config': {'include_only_sensors': [sensor]},
        })

    return experiments


def generate_sensor_group_experiments():
    """Generate sensor group ablation experiments."""
    experiments = []

    # Baseline (all groups)
    experiments.append({
        'study': 'sensor_group',
        'config_name': 'baseline',
        'ablation_type': 'baseline',
        'ablation_config': {},
    })

    # LOO (remove one group)
    for group in SENSOR_GROUPS:
        experiments.append({
            'study': 'sensor_group',
            'config_name': f'loo_{group}',
            'ablation_type': 'loo',
            'ablation_config': {'exclude_sensor_groups': [group]},
        })

    # LOI (keep only one group)
    for group in SENSOR_GROUPS:
        experiments.append({
            'study': 'sensor_group',
            'config_name': f'loi_{group}',
            'ablation_type': 'loi',
            'ablation_config': {'include_only_sensor_groups': [group]},
        })

    return experiments


def generate_architecture_experiments():
    """Generate architecture ablation experiments."""
    experiments = []

    for config_name, params in ARCHITECTURE_CONFIGS.items():
        experiments.append({
            'study': 'architecture',
            'config_name': config_name,
            'ablation_type': 'architecture',
            'architecture_config': params,
        })

    return experiments


def generate_baseline_experiments():
    """Generate baseline model comparison experiments."""
    experiments = []

    for model in ALL_BASELINE_MODELS:
        model_type = 'ml' if model in BASELINE_ML_MODELS else 'nn'
        experiments.append({
            'study': 'baseline',
            'config_name': model,
            'ablation_type': 'baseline_comparison',
            'model': model,
            'model_type': model_type,
        })

    return experiments


def generate_manifest(mode='full', studies=None):
    """Generate complete manifest of all experiments.

    Args:
        mode: 'full' (3×3 seeds) or 'reduced' (3 model seeds only)
        studies: List of study names to include, or None for all

    Returns:
        dict: Manifest with metadata and experiment list
    """
    all_studies = {
        'modality_grouped': generate_modality_grouped_experiments,
        'modality_individual': generate_modality_individual_experiments,
        'sensor_individual': generate_sensor_individual_experiments,
        'sensor_group': generate_sensor_group_experiments,
        'architecture': generate_architecture_experiments,
        'baseline': generate_baseline_experiments,
    }

    if studies is None:
        studies = list(all_studies.keys())

    # Generate base configs
    base_configs = []
    for study_name in studies:
        if study_name in all_studies:
            base_configs.extend(all_studies[study_name]())

    # Determine seeds
    if mode == 'full':
        data_split_seeds = DATA_SPLIT_SEEDS
        model_seeds = MODEL_SEEDS
    else:
        data_split_seeds = [42]  # Single data split
        model_seeds = MODEL_SEEDS

    # Expand with seeds
    experiments = []
    for config in base_configs:
        for ds_seed in data_split_seeds:
            for ms_seed in model_seeds:
                exp = dict(config)
                exp['data_split_seed'] = ds_seed
                exp['model_seed'] = ms_seed
                exp['experiment_id'] = f"{config['study']}_{config['config_name']}_ds{ds_seed}_ms{ms_seed}"

                # Generate output directory
                if config['study'] in ['modality_grouped', 'modality_individual']:
                    subdir = 'modality/' + ('grouped' if config['study'] == 'modality_grouped' else 'individual')
                elif config['study'] in ['sensor_individual', 'sensor_group']:
                    subdir = 'sensor/' + ('individual' if config['study'] == 'sensor_individual' else 'group')
                elif config['study'] == 'architecture':
                    subdir = 'architecture'
                elif config['study'] == 'baseline':
                    model_type = config.get('model_type', 'ml')
                    subdir = f'baseline/{model_type}'
                else:
                    subdir = config['study']

                exp['output_subdir'] = f"{subdir}/{config['config_name']}_ds{ds_seed}_ms{ms_seed}"

                # Determine script to use
                if config['study'] in ['modality_grouped', 'modality_individual', 'sensor_individual', 'sensor_group']:
                    exp['script'] = 'scripts/evaluation/run_comprehensive_ablation.py'
                elif config['study'] == 'architecture':
                    exp['script'] = 'scripts/evaluation/run_architecture_ablation.py'
                elif config['study'] == 'baseline':
                    exp['script'] = 'scripts/evaluation/run_baseline_models.py'

                experiments.append(exp)

    # Build manifest
    manifest = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'mode': mode,
            'total_configs': len(base_configs),
            'total_experiments': len(experiments),
            'data_split_seeds': data_split_seeds,
            'model_seeds': model_seeds,
            'studies': studies,
        },
        'study_counts': {},
        'experiments': experiments,
    }

    # Count by study
    for study in studies:
        study_exps = [e for e in experiments if e['study'] == study]
        n_configs = len(set(e['config_name'] for e in study_exps))
        manifest['study_counts'][study] = {
            'configs': n_configs,
            'experiments': len(study_exps),
        }

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Generate Ablation Study Manifest')
    parser.add_argument('--output', type=str, required=True,
                        help='Output manifest JSON path')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'reduced'],
                        help='full (3×3 seeds = 1,143 runs) or reduced (3 seeds = 381 runs)')
    parser.add_argument('--studies', type=str, default=None,
                        help='Comma-separated list of studies (default: all)')

    args = parser.parse_args()

    studies = None
    if args.studies:
        studies = [s.strip() for s in args.studies.split(',')]

    print(f"Generating manifest (mode={args.mode})")
    manifest = generate_manifest(mode=args.mode, studies=studies)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest Statistics:")
    print(f"  Mode: {manifest['metadata']['mode']}")
    print(f"  Total configs: {manifest['metadata']['total_configs']}")
    print(f"  Total experiments: {manifest['metadata']['total_experiments']}")
    print(f"\nPer-study breakdown:")
    for study, counts in manifest['study_counts'].items():
        print(f"  {study}: {counts['configs']} configs × {len(manifest['metadata']['data_split_seeds']) * len(manifest['metadata']['model_seeds'])} seeds = {counts['experiments']} experiments")

    print(f"\nSaved to: {output_path}")

    # Time estimate
    mins_per_run = 4
    total_mins = manifest['metadata']['total_experiments'] * mins_per_run
    lambda_time = total_mins / 8  # 8 GPUs
    bizon_time = total_mins / 2   # 2 GPUs

    print(f"\nEstimated time (~{mins_per_run} min/run):")
    print(f"  Lambda (8× L40): {lambda_time/60:.1f} hours")
    print(f"  Bizon (2× A6000): {bizon_time/60:.1f} hours")


if __name__ == '__main__':
    main()
