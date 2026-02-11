#!/usr/bin/env python3
"""
Comprehensive Crossed Ablation Study for CNC Operation Classification.

This script generates and manages 4 levels of crossed ablation experiments:

Level 1: Sensor × Modality (96 configs × 9 seeds = 864 experiments)
  - Each experiment: ONE sensor + ONE modality
  - Answers: "Best modality per sensor?"

Level 2: Sensor Group × Modality (30 configs × 9 seeds = 270 experiments)
  - Each experiment: ONE sensor group (all sensors at location) + ONE modality
  - Answers: "Best modality per location?"

Level 3: Pairwise Sensor Combinations (120 configs × 9 seeds = 1,080 experiments)
  - Each experiment: TWO sensors (all modalities)
  - Answers: "Best 2-sensor deployment?"

Level 4: Modality Combinations (35 configs × 9 seeds = 315 experiments)
  - Each experiment: 2 or 3 modalities (all sensors)
  - Answers: "Minimal modality set for >99%?"

TOTAL: 2,529 experiments (~10.5 hours with 16 parallel workers)

Usage:
    # Generate manifest for all levels
    python scripts/experiments/run_crossed_ablation.py --phase manifest \
        --output-dir outputs/ablation_study_2026_02_06

    # Generate manifest for specific levels only
    python scripts/experiments/run_crossed_ablation.py --phase manifest \
        --levels 1,2 --output-dir outputs/ablation_study_2026_02_06

    # Run with Ray parallel (2 GPUs, 8 per GPU = 16 parallel)
    python scripts/evaluation/run_ablation_parallel.py \
        --manifest outputs/ablation_study_2026_02_06/manifest_crossed.json \
        --data-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \
        --output-dir outputs/ablation_study_2026_02_06 \
        --num-gpus 2

    # Analyze results
    python scripts/experiments/run_crossed_ablation.py --phase analyze \
        --output-dir outputs/ablation_study_2026_02_06

Author: Claude Code
Date: February 2026
"""

import os
import sys
import json
import argparse
import time
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

# All 16 sensors (verified against metadata.json continuous_columns)
ALL_SENSORS = [
    # Frame (7)
    'frame_r1', 'frame_r2', 'frame_l1', 'frame_l2', 'frame_l3', 'frame_b1', 'frame_b2',
    # Spindle (2)
    'spindle1', 'spindle2',
    # Bed (4)
    'y_bed_1', 'y_bed_2', 'y_bed_3', 'y_bed_4',
    # Gantry (2)
    'z_gant_1', 'z_gant_2',
    # Motor (1)
    'xa_motor',
]

# Sensor groups by location
SENSOR_GROUPS = {
    'frame': ['frame_r1', 'frame_r2', 'frame_l1', 'frame_l2', 'frame_l3', 'frame_b1', 'frame_b2'],
    'spindle': ['spindle1', 'spindle2'],
    'bed': ['y_bed_1', 'y_bed_2', 'y_bed_3', 'y_bed_4'],
    'gantry': ['z_gant_1', 'z_gant_2'],
    'motor': ['xa_motor'],
}

# Sensor name normalization (metadata may have different naming)
SENSOR_ALIASES = {
    'y_bed_1': 'y_bed__1',
    'y_bed_2': 'y_bed__2',
    'y_bed_3': 'y_bed__3',
    'y_bed_4': 'y_bed__4',
}

# 6 sensor modalities (grouped from individual channels)
MODALITY_GROUPS = {
    'accelerometer': ['Ax', 'Ay', 'Az'],
    'gyroscope': ['Gx', 'Gy', 'Gz'],
    'magnetometer': ['Mx', 'My', 'Mz'],
    'environmental': ['Pressure', 'Temperature', 'Proximity'],
    'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
    'rms': ['RMS'],
}

MODALITIES = list(MODALITY_GROUPS.keys())

# Seeds for statistical testing (3×3 = 9 for two-way ANOVA)
DATA_SEEDS = [42, 123, 456]
MODEL_SEEDS = [42, 123, 456]

# Training script
TRAIN_SCRIPT = "scripts/evaluation/run_9class_direct.py"


def normalize_sensors(sensors: List[str]) -> List[str]:
    """Normalize sensor names to match metadata format."""
    return [SENSOR_ALIASES.get(s, s) for s in sensors]


def get_modality_channels(modalities: List[str]) -> List[str]:
    """Get all individual channels for a list of modality groups."""
    channels = []
    for mod in modalities:
        channels.extend(MODALITY_GROUPS[mod])
    return channels


# =============================================================================
# MANIFEST GENERATION
# =============================================================================

def generate_level1_sensor_modality(output_dir: Path) -> List[Dict]:
    """Level 1: Sensor × Modality crossed ablation.

    16 sensors × 6 modalities × 9 seeds = 864 experiments
    """
    manifest = []

    for sensor in ALL_SENSORS:
        sensor_norm = SENSOR_ALIASES.get(sensor, sensor)
        for modality in MODALITIES:
            for ds in DATA_SEEDS:
                for ms in MODEL_SEEDS:
                    exp_id = f"L1_sensor_mod_{sensor}_{modality}_ds{ds}_ms{ms}"
                    manifest.append({
                        'experiment_id': exp_id,
                        'study': 'crossed_L1_sensor_modality',
                        'level': 1,
                        'script': TRAIN_SCRIPT,
                        'config_name': f"{sensor}_{modality}",
                        'sensor': sensor,
                        'modality': modality,
                        'data_split_seed': ds,
                        'model_seed': ms,
                        'output_subdir': f"crossed/L1_sensor_modality/{sensor}_{modality}_ds{ds}_ms{ms}",
                        'ablation_config': {
                            'include_only_sensors': [sensor_norm],
                            'include_only_modalities': MODALITY_GROUPS[modality],
                        },
                    })

    return manifest


def generate_level2_group_modality(output_dir: Path) -> List[Dict]:
    """Level 2: Sensor Group × Modality crossed ablation.

    5 groups × 6 modalities × 9 seeds = 270 experiments
    """
    manifest = []

    for group_name, group_sensors in SENSOR_GROUPS.items():
        sensors_norm = normalize_sensors(group_sensors)
        for modality in MODALITIES:
            for ds in DATA_SEEDS:
                for ms in MODEL_SEEDS:
                    exp_id = f"L2_group_mod_{group_name}_{modality}_ds{ds}_ms{ms}"
                    manifest.append({
                        'experiment_id': exp_id,
                        'study': 'crossed_L2_group_modality',
                        'level': 2,
                        'script': TRAIN_SCRIPT,
                        'config_name': f"{group_name}_{modality}",
                        'group': group_name,
                        'modality': modality,
                        'data_split_seed': ds,
                        'model_seed': ms,
                        'output_subdir': f"crossed/L2_group_modality/{group_name}_{modality}_ds{ds}_ms{ms}",
                        'ablation_config': {
                            'include_only_sensors': sensors_norm,
                            'include_only_modalities': MODALITY_GROUPS[modality],
                        },
                    })

    return manifest


def generate_level3_sensor_pairs(output_dir: Path) -> List[Dict]:
    """Level 3: Pairwise sensor combinations.

    C(16,2) = 120 pairs × 9 seeds = 1,080 experiments
    """
    manifest = []

    for s1, s2 in combinations(ALL_SENSORS, 2):
        sensors_norm = normalize_sensors([s1, s2])
        for ds in DATA_SEEDS:
            for ms in MODEL_SEEDS:
                exp_id = f"L3_pair_{s1}_{s2}_ds{ds}_ms{ms}"
                manifest.append({
                    'experiment_id': exp_id,
                    'study': 'crossed_L3_sensor_pairs',
                    'level': 3,
                    'script': TRAIN_SCRIPT,
                    'config_name': f"{s1}_{s2}",
                    'sensors': [s1, s2],
                    'data_split_seed': ds,
                    'model_seed': ms,
                    'output_subdir': f"crossed/L3_sensor_pairs/{s1}_{s2}_ds{ds}_ms{ms}",
                    'ablation_config': {
                        'include_only_sensors': sensors_norm,
                        # All modalities (no filtering)
                    },
                })

    return manifest


def generate_level4_modality_combos(output_dir: Path) -> List[Dict]:
    """Level 4: Modality combinations (keep 2 or 3 modalities).

    C(6,2) + C(6,3) = 15 + 20 = 35 combos × 9 seeds = 315 experiments
    """
    manifest = []

    # Keep 2 modalities
    for mods in combinations(MODALITIES, 2):
        channels = get_modality_channels(mods)
        mod_str = '_'.join(mods)
        for ds in DATA_SEEDS:
            for ms in MODEL_SEEDS:
                exp_id = f"L4_mod2_{mod_str}_ds{ds}_ms{ms}"
                manifest.append({
                    'experiment_id': exp_id,
                    'study': 'crossed_L4_modality_combos',
                    'level': 4,
                    'script': TRAIN_SCRIPT,
                    'config_name': mod_str,
                    'modalities': list(mods),
                    'n_modalities': 2,
                    'data_split_seed': ds,
                    'model_seed': ms,
                    'output_subdir': f"crossed/L4_modality_combos/keep2_{mod_str}_ds{ds}_ms{ms}",
                    'ablation_config': {
                        'include_only_modalities': channels,
                        # All sensors (no filtering)
                    },
                })

    # Keep 3 modalities
    for mods in combinations(MODALITIES, 3):
        channels = get_modality_channels(mods)
        mod_str = '_'.join(mods)
        for ds in DATA_SEEDS:
            for ms in MODEL_SEEDS:
                exp_id = f"L4_mod3_{mod_str}_ds{ds}_ms{ms}"
                manifest.append({
                    'experiment_id': exp_id,
                    'study': 'crossed_L4_modality_combos',
                    'level': 4,
                    'script': TRAIN_SCRIPT,
                    'config_name': mod_str,
                    'modalities': list(mods),
                    'n_modalities': 3,
                    'data_split_seed': ds,
                    'model_seed': ms,
                    'output_subdir': f"crossed/L4_modality_combos/keep3_{mod_str}_ds{ds}_ms{ms}",
                    'ablation_config': {
                        'include_only_modalities': channels,
                    },
                })

    return manifest


def generate_manifest(output_dir: Path, data_dir: Path, levels: List[int] = None) -> List[Dict]:
    """Generate comprehensive crossed ablation manifest.

    Args:
        output_dir: Output directory
        data_dir: Data directory
        levels: Which levels to include (1-4). Default: all.

    Returns:
        Combined manifest for all requested levels.
    """
    if levels is None:
        levels = [1, 2, 3, 4]

    manifest = []
    level_counts = {}

    if 1 in levels:
        l1 = generate_level1_sensor_modality(output_dir)
        manifest.extend(l1)
        level_counts['L1_sensor_modality'] = len(l1)

    if 2 in levels:
        l2 = generate_level2_group_modality(output_dir)
        manifest.extend(l2)
        level_counts['L2_group_modality'] = len(l2)

    if 3 in levels:
        l3 = generate_level3_sensor_pairs(output_dir)
        manifest.extend(l3)
        level_counts['L3_sensor_pairs'] = len(l3)

    if 4 in levels:
        l4 = generate_level4_modality_combos(output_dir)
        manifest.extend(l4)
        level_counts['L4_modality_combos'] = len(l4)

    # Save manifest
    manifest_path = output_dir / 'manifest_crossed.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("=" * 70)
    print("COMPREHENSIVE CROSSED ABLATION MANIFEST")
    print("=" * 70)
    print(f"\nLevels included: {levels}")
    print(f"\nExperiment breakdown:")
    for level, count in level_counts.items():
        print(f"  {level}: {count}")
    print(f"\nTOTAL: {len(manifest)} experiments")
    print(f"\nEstimated runtime (16 parallel, ~4 min/exp):")
    print(f"  {len(manifest)} ÷ 16 × 4 min = {len(manifest) / 16 * 4 / 60:.1f} hours")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"\nTo run:")
    print(f"  python scripts/evaluation/run_ablation_parallel.py \\")
    print(f"    --manifest {manifest_path} \\")
    print(f"    --data-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \\")
    print(f"    --output-dir {output_dir} \\")
    print(f"    --num-gpus 2")

    return manifest


# =============================================================================
# TRAINING
# =============================================================================

def run_single_experiment(
    exp: Dict,
    data_dir: Path,
    output_base: Path,
    gpu_id: int = 0,
    dry_run: bool = False,
) -> Dict:
    """Run a single crossed ablation experiment.

    Note: This is a fallback for direct execution. Prefer using
    run_ablation_parallel.py with the manifest for better parallelization.
    """

    output_dir = output_base / exp['output_subdir']
    results_path = output_dir / 'results.json'

    # Skip if already completed
    if results_path.exists():
        return {
            'experiment_id': exp['experiment_id'],
            'status': 'skipped',
            'message': 'Already completed',
        }

    # Build command using ablation_config
    ablation_config = exp.get('ablation_config', {})
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--data-dir', str(data_dir),
        '--output-dir', str(output_dir),
        '--seed', str(exp['model_seed']),
    ]

    # Add sensor filtering if specified
    if 'include_only_sensors' in ablation_config:
        cmd.extend(['--include-only-sensors', ','.join(ablation_config['include_only_sensors'])])
    if 'exclude_sensors' in ablation_config:
        cmd.extend(['--exclude-sensors', ','.join(ablation_config['exclude_sensors'])])

    # Add modality filtering if specified
    if 'include_only_modalities' in ablation_config:
        cmd.extend(['--include-only-modalities', ','.join(ablation_config['include_only_modalities'])])
    if 'exclude_modalities' in ablation_config:
        cmd.extend(['--exclude-modalities', ','.join(ablation_config['exclude_modalities'])])

    if dry_run:
        print(f"  [DRY RUN] CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")
        return {'experiment_id': exp['experiment_id'], 'status': 'dry_run'}

    # Set environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"  Running {exp['experiment_id']} (GPU {gpu_id})...")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  ✓ {exp['experiment_id']} ({elapsed:.1f}s)")
            return {
                'experiment_id': exp['experiment_id'],
                'status': 'success',
                'elapsed': elapsed,
            }
        else:
            print(f"  ✗ {exp['experiment_id']} failed")
            # Save error log
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'error.log', 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return {
                'experiment_id': exp['experiment_id'],
                'status': 'failed',
                'error': result.stderr[-500:] if result.stderr else 'Unknown',
            }
    except subprocess.TimeoutExpired:
        print(f"  ✗ {exp['experiment_id']} timeout")
        return {'experiment_id': exp['experiment_id'], 'status': 'timeout'}
    except Exception as e:
        print(f"  ✗ {exp['experiment_id']} error: {e}")
        return {'experiment_id': exp['experiment_id'], 'status': 'error', 'error': str(e)}


def run_parallel(
    experiments: List[Dict],
    data_dir: Path,
    output_base: Path,
    num_gpus: int = 2,
    dry_run: bool = False,
) -> List[Dict]:
    """Run experiments in parallel across GPUs."""
    results = []

    def worker(args):
        exp, gpu_id = args
        return run_single_experiment(exp, data_dir, output_base, gpu_id, dry_run)

    # Create work items with round-robin GPU assignment
    work_items = [(exp, i % num_gpus) for i, exp in enumerate(experiments)]

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            results.append(future.result())

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(output_dir: Path) -> Dict:
    """Analyze crossed ablation results across all 4 levels."""
    import pandas as pd

    analysis = {
        'L1_sensor_modality': {},
        'L2_group_modality': {},
        'L3_sensor_pairs': {},
        'L4_modality_combos': {},
        'summary': {},
    }

    # Level 1: Sensor × Modality
    l1_results = []
    for results_file in output_dir.rglob('crossed/L1_sensor_modality/*/results.json'):
        with open(results_file) as f:
            data = json.load(f)
        # Parse: {sensor}_{modality}_ds{ds}_ms{ms}
        dir_name = results_file.parent.name
        parts = dir_name.rsplit('_ds', 1)[0].rsplit('_', 1)
        sensor = parts[0] if len(parts) == 2 else dir_name
        modality = parts[1] if len(parts) == 2 else 'unknown'
        l1_results.append({
            'sensor': sensor, 'modality': modality,
            'test_accuracy': data.get('test_accuracy', 0),
        })

    if l1_results:
        df = pd.DataFrame(l1_results)
        # Best modality per sensor
        best_per_sensor = {}
        for sensor in df['sensor'].unique():
            sensor_df = df[df['sensor'] == sensor]
            best_mod = sensor_df.groupby('modality')['test_accuracy'].mean().idxmax()
            best_acc = sensor_df.groupby('modality')['test_accuracy'].mean().max()
            best_per_sensor[sensor] = {'best_modality': best_mod, 'accuracy': float(best_acc)}

        # Best sensor per modality
        best_per_modality = {}
        for modality in df['modality'].unique():
            mod_df = df[df['modality'] == modality]
            best_sens = mod_df.groupby('sensor')['test_accuracy'].mean().idxmax()
            best_acc = mod_df.groupby('sensor')['test_accuracy'].mean().max()
            best_per_modality[modality] = {'best_sensor': best_sens, 'accuracy': float(best_acc)}

        analysis['L1_sensor_modality'] = {
            'count': len(l1_results),
            'best_per_sensor': best_per_sensor,
            'best_per_modality': best_per_modality,
            'overall_mean': float(df['test_accuracy'].mean()),
            'overall_std': float(df['test_accuracy'].std()),
        }

    # Level 2: Group × Modality
    l2_results = []
    for results_file in output_dir.rglob('crossed/L2_group_modality/*/results.json'):
        with open(results_file) as f:
            data = json.load(f)
        dir_name = results_file.parent.name
        parts = dir_name.rsplit('_ds', 1)[0].rsplit('_', 1)
        group = parts[0] if len(parts) == 2 else dir_name
        modality = parts[1] if len(parts) == 2 else 'unknown'
        l2_results.append({
            'group': group, 'modality': modality,
            'test_accuracy': data.get('test_accuracy', 0),
        })

    if l2_results:
        df = pd.DataFrame(l2_results)
        analysis['L2_group_modality'] = {
            'count': len(l2_results),
            'group_means': df.groupby('group')['test_accuracy'].mean().to_dict(),
            'modality_means': df.groupby('modality')['test_accuracy'].mean().to_dict(),
            'overall_mean': float(df['test_accuracy'].mean()),
        }

    # Level 3: Sensor Pairs
    l3_results = []
    for results_file in output_dir.rglob('crossed/L3_sensor_pairs/*/results.json'):
        with open(results_file) as f:
            data = json.load(f)
        dir_name = results_file.parent.name
        pair_str = dir_name.rsplit('_ds', 1)[0]
        l3_results.append({
            'pair': pair_str,
            'test_accuracy': data.get('test_accuracy', 0),
        })

    if l3_results:
        df = pd.DataFrame(l3_results)
        pair_means = df.groupby('pair')['test_accuracy'].mean().sort_values(ascending=False)
        analysis['L3_sensor_pairs'] = {
            'count': len(l3_results),
            'best_pairs': pair_means.head(10).to_dict(),
            'worst_pairs': pair_means.tail(10).to_dict(),
            'overall_mean': float(df['test_accuracy'].mean()),
        }

    # Level 4: Modality Combos
    l4_results = []
    for results_file in output_dir.rglob('crossed/L4_modality_combos/*/results.json'):
        with open(results_file) as f:
            data = json.load(f)
        dir_name = results_file.parent.name
        # keep2_{mods} or keep3_{mods}
        combo_str = dir_name.rsplit('_ds', 1)[0]
        n_mods = 2 if combo_str.startswith('keep2_') else 3
        l4_results.append({
            'combo': combo_str, 'n_modalities': n_mods,
            'test_accuracy': data.get('test_accuracy', 0),
        })

    if l4_results:
        df = pd.DataFrame(l4_results)
        combo_means = df.groupby('combo')['test_accuracy'].mean().sort_values(ascending=False)
        analysis['L4_modality_combos'] = {
            'count': len(l4_results),
            'best_combos': combo_means.head(10).to_dict(),
            'by_n_modalities': {
                2: float(df[df['n_modalities'] == 2]['test_accuracy'].mean()),
                3: float(df[df['n_modalities'] == 3]['test_accuracy'].mean()),
            },
            'overall_mean': float(df['test_accuracy'].mean()),
        }

    # Summary across all levels
    total_count = sum(analysis[k].get('count', 0) for k in ['L1_sensor_modality', 'L2_group_modality', 'L3_sensor_pairs', 'L4_modality_combos'])
    analysis['summary'] = {
        'total_experiments': total_count,
        'L1_count': analysis['L1_sensor_modality'].get('count', 0),
        'L2_count': analysis['L2_group_modality'].get('count', 0),
        'L3_count': analysis['L3_sensor_pairs'].get('count', 0),
        'L4_count': analysis['L4_modality_combos'].get('count', 0),
    }

    # Save analysis
    analysis_path = output_dir / 'crossed_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=float)

    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CROSSED ABLATION ANALYSIS")
    print("=" * 70)

    print(f"\nTotal experiments: {total_count}")
    print(f"  L1 (sensor × modality): {analysis['summary']['L1_count']}")
    print(f"  L2 (group × modality): {analysis['summary']['L2_count']}")
    print(f"  L3 (sensor pairs): {analysis['summary']['L3_count']}")
    print(f"  L4 (modality combos): {analysis['summary']['L4_count']}")

    if analysis['L1_sensor_modality']:
        print(f"\n--- Level 1: Sensor × Modality ---")
        print(f"Overall mean: {analysis['L1_sensor_modality']['overall_mean']*100:.2f}%")
        print("Best modality per sensor (top 5):")
        for sensor, info in list(sorted(
            analysis['L1_sensor_modality']['best_per_sensor'].items(),
            key=lambda x: -x[1]['accuracy']
        ))[:5]:
            print(f"  {sensor:15s}: {info['best_modality']:15s} ({info['accuracy']*100:.2f}%)")

    if analysis['L3_sensor_pairs']:
        print(f"\n--- Level 3: Best Sensor Pairs ---")
        for pair, acc in list(analysis['L3_sensor_pairs']['best_pairs'].items())[:5]:
            print(f"  {pair}: {acc*100:.2f}%")

    if analysis['L4_modality_combos']:
        print(f"\n--- Level 4: Best Modality Combos ---")
        for combo, acc in list(analysis['L4_modality_combos']['best_combos'].items())[:5]:
            print(f"  {combo}: {acc*100:.2f}%")

    print(f"\nAnalysis saved to: {analysis_path}")
    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Crossed Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Levels:
  1: Sensor × Modality (864 experiments)
  2: Sensor Group × Modality (270 experiments)
  3: Pairwise Sensor Combinations (1,080 experiments)
  4: Modality Combinations (315 experiments)

Examples:
  # Generate full manifest (2,529 experiments)
  python scripts/experiments/run_crossed_ablation.py --phase manifest

  # Generate only levels 1 and 2
  python scripts/experiments/run_crossed_ablation.py --phase manifest --levels 1,2

  # Analyze results
  python scripts/experiments/run_crossed_ablation.py --phase analyze
        """
    )
    parser.add_argument('--phase', required=True, choices=['manifest', 'train', 'analyze'],
                        help='Phase to run')
    parser.add_argument('--levels', type=str, default='1,2,3,4',
                        help='Comma-separated list of levels to include (default: all)')
    parser.add_argument('--data-dir', type=str, default='outputs/7class_cascade_to_9class/9class_moddropout_final/data',
                        help='Data directory with splits')
    parser.add_argument('--output-dir', type=str, default='outputs/ablation_study_2026_02_06',
                        help='Output directory')
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Parse levels
    levels = [int(l.strip()) for l in args.levels.split(',')]

    if args.phase == 'manifest':
        generate_manifest(output_dir, data_dir, levels=levels)

    elif args.phase == 'train':
        # Load manifest
        manifest_path = output_dir / 'manifest_crossed.json'
        if not manifest_path.exists():
            print("Manifest not found. Run --phase manifest first.")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"Running {len(manifest)} crossed ablation experiments")
        print(f"GPUs: {args.num_gpus}")
        print(f"Dry run: {args.dry_run}")
        print()

        t0 = time.time()
        results = run_parallel(manifest, data_dir, output_dir, args.num_gpus, args.dry_run)
        elapsed = time.time() - t0

        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total: {len(results)}")
        print(f"Success: {success}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Elapsed: {elapsed/60:.1f} minutes")

        # Save run log
        log_path = output_dir / 'crossed_run_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed,
                'summary': {'total': len(results), 'success': success, 'skipped': skipped, 'failed': failed},
                'results': results,
            }, f, indent=2)
        print(f"Log saved to: {log_path}")

    elif args.phase == 'analyze':
        try:
            analyze_results(output_dir)
        except ImportError:
            print("pandas required for analysis. Install with: pip install pandas")
            sys.exit(1)


if __name__ == '__main__':
    main()
