#!/usr/bin/env python3
"""Run Ablation Study in Parallel with Ray.

Executes experiments from a manifest file across multiple GPUs.

Usage:
    # Start Ray cluster (if not already running)
    ray start --head --num-gpus=8

    # Run all experiments
    python scripts/evaluation/run_ablation_parallel.py \
        --manifest outputs/ablation_study_2026_02_06/manifest.json \
        --data-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \
        --output-dir outputs/ablation_study_2026_02_06 \
        --num-gpus 8

    # Run specific studies only
    python scripts/evaluation/run_ablation_parallel.py \
        --manifest outputs/ablation_study_2026_02_06/manifest.json \
        --data-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \
        --output-dir outputs/ablation_study_2026_02_06 \
        --num-gpus 8 \
        --studies modality_grouped,architecture

    # Dry run (show commands without executing)
    python scripts/evaluation/run_ablation_parallel.py \
        --manifest outputs/ablation_study_2026_02_06/manifest.json \
        --dry-run

    # Multi-split mode: route each data_split_seed to its own split directory
    python scripts/evaluation/run_ablation_parallel.py \
        --manifest outputs/ablation_study_2026_02_06/manifest.json \
        --split-dir outputs/ablation_study_2026_02_06/data_splits \
        --output-dir outputs/ablation_study_2026_02_06 \
        --num-gpus 8
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def build_command(experiment, data_dir, output_dir, split_dir=None):
    """Build command line for an experiment.

    Args:
        experiment: Experiment dict from manifest.
        data_dir: Default data directory (used when split_dir is None).
        output_dir: Base output directory.
        split_dir: Base directory containing split_42/, split_123/, split_456/ subdirs.
                   When provided, routes each experiment to the correct split based on
                   its data_split_seed.
    """
    script = experiment['script']
    study = experiment['study']
    config_name = experiment['config_name']
    ds_seed = experiment['data_split_seed']
    ms_seed = experiment['model_seed']
    output_subdir = experiment['output_subdir']

    exp_output_dir = Path(output_dir) / output_subdir

    if split_dir:
        exp_data_dir = Path(split_dir) / f'split_{ds_seed}'
        if not exp_data_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {exp_data_dir}\n"
                f"Create it with: python scripts/data/create_data_splits.py "
                f"--source-dir <original_data> --output-base {split_dir} --seeds {ds_seed}"
            )
    else:
        exp_data_dir = Path(data_dir)

    cmd = [sys.executable, script]
    cmd.extend(['--data-dir', str(exp_data_dir)])
    cmd.extend(['--output-dir', str(exp_output_dir)])
    cmd.extend(['--seed', str(ms_seed)])

    # --iteration is only supported by run_9class_direct.py, not baseline models
    if study != 'baseline':
        cmd.extend(['--iteration', f"{config_name}_ds{ds_seed}_ms{ms_seed}"])

    # Add ablation-specific arguments
    # Support original studies + crossed ablation studies
    ablation_studies = [
        'modality_grouped', 'modality_individual', 'sensor_individual', 'sensor_group',
        'crossed_L1_sensor_modality', 'crossed_L2_group_modality',
        'crossed_L3_sensor_pairs', 'crossed_L4_modality_combos',
    ]
    if study in ablation_studies or study.startswith('crossed_'):
        ablation_config = experiment.get('ablation_config', {})
        needs_modality_dropout_off = False

        if 'exclude_modalities' in ablation_config:
            cmd.extend(['--exclude-modalities', ','.join(ablation_config['exclude_modalities'])])
        if 'include_only_modalities' in ablation_config:
            cmd.extend(['--include-only-modalities', ','.join(ablation_config['include_only_modalities'])])
            needs_modality_dropout_off = True

        if 'exclude_components' in ablation_config:
            cmd.extend(['--exclude-components', ','.join(ablation_config['exclude_components'])])
        if 'include_only_components' in ablation_config:
            cmd.extend(['--include-only-components', ','.join(ablation_config['include_only_components'])])
            needs_modality_dropout_off = True

        if 'exclude_sensors' in ablation_config:
            cmd.extend(['--exclude-sensors', ','.join(ablation_config['exclude_sensors'])])
        if 'include_only_sensors' in ablation_config:
            cmd.extend(['--include-only-sensors', ','.join(ablation_config['include_only_sensors'])])
            needs_modality_dropout_off = True

        if 'exclude_sensor_groups' in ablation_config:
            cmd.extend(['--exclude-sensor-groups', ','.join(ablation_config['exclude_sensor_groups'])])
        if 'include_only_sensor_groups' in ablation_config:
            cmd.extend(['--include-only-sensor-groups', ','.join(ablation_config['include_only_sensor_groups'])])
            needs_modality_dropout_off = True

        # Disable modality dropout for "leave only in" experiments (added once)
        if needs_modality_dropout_off:
            cmd.extend(['--modality_dropout', '0.0'])

        # L5 single-channel experiments converge by epoch ~100 (p95=94 across 2430 runs)
        if study == 'crossed_L5_sensor_channel':
            cmd.extend(['--max_epochs', '100'])

    elif study == 'architecture':
        arch_config = experiment.get('architecture_config', {})
        for key, value in arch_config.items():
            if isinstance(value, bool):
                cmd.extend([f'--{key}', 'true' if value else 'false'])
            else:
                cmd.extend([f'--{key}', str(value)])

    elif study == 'baseline':
        cmd.extend(['--model', experiment['model']])

    return cmd


def run_experiment_subprocess(experiment, data_dir, output_dir, gpu_id=None, split_dir=None):
    """Run a single experiment as subprocess."""
    cmd = build_command(experiment, data_dir, output_dir, split_dir=split_dir)
    exp_id = experiment['experiment_id']

    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=7200  # 2 hour timeout per experiment
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            'experiment_id': exp_id,
            'success': success,
            'elapsed': elapsed,
            'returncode': result.returncode,
            'stdout': result.stdout[-5000:] if result.stdout else '',
            'stderr': result.stderr[-5000:] if result.stderr else '',
        }
    except subprocess.TimeoutExpired:
        return {
            'experiment_id': exp_id,
            'success': False,
            'elapsed': time.time() - start_time,
            'error': 'Timeout',
        }
    except Exception as e:
        return {
            'experiment_id': exp_id,
            'success': False,
            'elapsed': time.time() - start_time,
            'error': str(e),
        }


if HAS_RAY:
    @ray.remote(num_gpus=1)
    def run_experiment_ray(experiment, data_dir, output_dir, split_dir=None):
        """Run a single experiment with Ray (GPU allocated)."""
        # Ray sets CUDA_VISIBLE_DEVICES to a single GPU (e.g., "0")
        # Use cuda:0 since that's what's visible to this worker
        cmd = build_command(experiment, data_dir, output_dir, split_dir=split_dir)
        exp_id = experiment['experiment_id']

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0

            return {
                'experiment_id': exp_id,
                'success': success,
                'elapsed': elapsed,
                'returncode': result.returncode,
                'stdout': result.stdout[-2000:] if result.stdout else '',
                'stderr': result.stderr[-2000:] if result.stderr else '',
            }
        except subprocess.TimeoutExpired:
            return {
                'experiment_id': exp_id,
                'success': False,
                'elapsed': time.time() - start_time,
                'error': 'Timeout',
            }
        except Exception as e:
            return {
                'experiment_id': exp_id,
                'success': False,
                'elapsed': time.time() - start_time,
                'error': str(e),
            }


def run_with_ray(experiments, data_dir, output_dir, num_gpus, split_dir=None):
    """Run experiments in parallel using Ray."""
    if not HAS_RAY:
        raise ImportError("Ray not installed. Install with: pip install ray")

    # Initialize Ray
    if not ray.is_initialized():
        # Try to connect to existing cluster first
        try:
            ray.init(address='auto')
            print("Connected to existing Ray cluster")
        except ConnectionError:
            # No existing cluster, start a new one
            ray.init(num_gpus=num_gpus)
            print(f"Started new Ray cluster with {num_gpus} GPUs")

    print(f"Running {len(experiments)} experiments with Ray ({num_gpus} GPUs)")
    if split_dir:
        print(f"Using multi-split data from: {split_dir}")

    # Submit all tasks
    futures = []
    for exp in experiments:
        future = run_experiment_ray.remote(exp, data_dir, output_dir, split_dir=split_dir)
        futures.append(future)

    # Collect results with progress
    results = []
    completed = 0
    start_time = time.time()

    for future in futures:
        result = ray.get(future)
        results.append(result)
        completed += 1

        status = '✓' if result['success'] else '✗'
        elapsed = result['elapsed']
        total_elapsed = time.time() - start_time
        eta = (total_elapsed / completed) * (len(experiments) - completed) if completed > 0 else 0

        print(f"[{completed}/{len(experiments)}] {status} {result['experiment_id']} "
              f"({elapsed:.1f}s) | ETA: {eta/60:.1f}min")

    return results


def run_with_threads(experiments, data_dir, output_dir, num_workers, num_physical_gpus=2, split_dir=None):
    """Run experiments in parallel using ThreadPoolExecutor."""
    print(f"Running {len(experiments)} experiments with {num_workers} threads ({num_physical_gpus} physical GPUs)")
    if split_dir:
        print(f"Using multi-split data from: {split_dir}")

    results = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_experiment_subprocess, exp, data_dir, output_dir, i % num_physical_gpus, split_dir=split_dir): exp
            for i, exp in enumerate(experiments)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            status = '✓' if result['success'] else '✗'
            elapsed = result['elapsed']
            total_elapsed = time.time() - start_time
            eta = (total_elapsed / completed) * (len(experiments) - completed) if completed > 0 else 0

            print(f"[{completed}/{len(experiments)}] {status} {result['experiment_id']} "
                  f"({elapsed:.1f}s) | ETA: {eta/60:.1f}min")

    return results


def check_completed(experiments, output_dir):
    """Check which experiments are already completed."""
    completed = []
    pending = []

    for exp in experiments:
        results_path = Path(output_dir) / exp['output_subdir'] / 'results.json'
        if results_path.exists():
            completed.append(exp)
        else:
            pending.append(exp)

    return completed, pending


def main():
    parser = argparse.ArgumentParser(description='Run Ablation Study in Parallel')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest JSON')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--num-physical-gpus', type=int, default=2,
                        help='Number of physical GPUs on the system (for thread backend)')
    parser.add_argument('--studies', type=str, default=None,
                        help='Comma-separated list of studies to run (default: all)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already completed experiments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--backend', type=str, default='ray', choices=['ray', 'threads'],
                        help='Parallelization backend')
    parser.add_argument('--split-dir', type=str, default=None,
                        help='Base directory with split_<seed>/ subdirs for multi-split ANOVA. '
                             'When set, each experiment routes to split_<data_split_seed>/ '
                             'instead of using --data-dir for all experiments.')

    args = parser.parse_args()

    # Load manifest (supports both dict with 'experiments' key and flat list)
    with open(args.manifest) as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        experiments = manifest
    else:
        experiments = manifest.get('experiments', [])
    print(f"Loaded manifest: {len(experiments)} experiments")

    # Filter by studies if specified
    if args.studies:
        study_filter = set(s.strip() for s in args.studies.split(','))
        experiments = [e for e in experiments if e['study'] in study_filter]
        print(f"Filtered to {len(experiments)} experiments (studies: {study_filter})")

    # Skip completed if resuming
    if args.resume and args.output_dir:
        completed, pending = check_completed(experiments, args.output_dir)
        print(f"Resume mode: {len(completed)} completed, {len(pending)} pending")
        experiments = pending

    if not experiments:
        print("No experiments to run!")
        return

    # Dry run: just print commands
    if args.dry_run:
        print("\nDry run - commands that would be executed:\n")
        if args.split_dir:
            print(f"Using multi-split data from: {args.split_dir}\n")
        for exp in experiments[:10]:  # Show first 10
            cmd = build_command(exp, args.data_dir or '/data', args.output_dir or '/output',
                                split_dir=args.split_dir)
            print(f"# {exp['experiment_id']}")
            print(' '.join(cmd))
            print()
        if len(experiments) > 10:
            print(f"... and {len(experiments) - 10} more experiments")
        return

    # Validate required args
    if not args.data_dir and not args.split_dir:
        print("ERROR: --data-dir or --split-dir required")
        return
    if not args.output_dir:
        print("ERROR: --output-dir required")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    start_time = time.time()

    if args.backend == 'ray':
        results = run_with_ray(experiments, args.data_dir, args.output_dir, args.num_gpus,
                               split_dir=args.split_dir)
    else:
        results = run_with_threads(experiments, args.data_dir, args.output_dir, args.num_gpus,
                                   args.num_physical_gpus, split_dir=args.split_dir)

    total_time = time.time() - start_time

    # Summary
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Avg time per experiment: {total_time/len(results)/60:.1f} minutes")

    # Save run summary
    summary = {
        'completed_at': datetime.now().isoformat(),
        'total_experiments': len(results),
        'successes': successes,
        'failures': failures,
        'total_time_seconds': total_time,
        'results': results,
    }

    summary_path = output_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # List failures
    if failures > 0:
        print(f"\nFailed experiments:")
        for r in results:
            if not r['success']:
                error = r.get('error', r.get('stderr', 'Unknown error')[:200])
                print(f"  - {r['experiment_id']}: {error}")


if __name__ == '__main__':
    main()
