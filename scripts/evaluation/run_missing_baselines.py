#!/usr/bin/env python3
"""Run missing baseline experiments for ablation study.

Runs the 30 baseline model comparisons that were skipped in the original run.

Usage:
    # Run all missing baselines (2 GPUs)
    python scripts/evaluation/run_missing_baselines.py

    # Dry run (show commands)
    python scripts/evaluation/run_missing_baselines.py --dry-run

    # Run only ML models (no GPU needed)
    python scripts/evaluation/run_missing_baselines.py --ml-only

    # Run only NN models
    python scripts/evaluation/run_missing_baselines.py --nn-only
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

# Configuration
DATA_DIR = "outputs/7class_cascade_to_9class/9class_moddropout_final/data"
OUTPUT_BASE = "outputs/ablation_study_2026_02_06/baseline"
SCRIPT = "scripts/evaluation/run_baseline_models.py"

# Note: svm_rbf removed due to timeout issues (>30 min per run)
# Use svm_linear instead or document as computationally intractable
ML_MODELS = ['xgboost', 'random_forest', 'logistic_regression', 'knn']
# ML_MODELS_WITH_SVM = ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression', 'knn']
NN_MODELS = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']
SEEDS = [42, 123, 456]


def run_experiment(model, seed, model_type, gpu_id=None, dry_run=False):
    """Run a single baseline experiment."""
    output_dir = Path(OUTPUT_BASE) / model_type / f"{model}_ds42_ms{seed}"
    results_path = output_dir / "results.json"

    # Skip if already completed
    if results_path.exists():
        return {
            'model': model,
            'seed': seed,
            'status': 'skipped',
            'message': 'Already completed',
        }

    cmd = [
        sys.executable, SCRIPT,
        '--data-dir', DATA_DIR,
        '--output-dir', str(output_dir),
        '--model', model,
        '--seed', str(seed),
    ]

    if dry_run:
        gpu_str = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        print(f"  {gpu_str}{' '.join(cmd)}")
        return {
            'model': model,
            'seed': seed,
            'status': 'dry_run',
        }

    # Set GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"  Running {model} seed={seed} (GPU={gpu_id})...")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  ✓ {model} seed={seed} completed in {elapsed:.1f}s")
            return {
                'model': model,
                'seed': seed,
                'status': 'success',
                'elapsed': elapsed,
            }
        else:
            print(f"  ✗ {model} seed={seed} failed (returncode={result.returncode})")
            # Save error log
            error_log = output_dir / "error.log"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(error_log, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return {
                'model': model,
                'seed': seed,
                'status': 'failed',
                'error': result.stderr[-500:] if result.stderr else 'Unknown error',
            }
    except subprocess.TimeoutExpired:
        print(f"  ✗ {model} seed={seed} timed out")
        return {
            'model': model,
            'seed': seed,
            'status': 'timeout',
        }
    except Exception as e:
        print(f"  ✗ {model} seed={seed} error: {e}")
        return {
            'model': model,
            'seed': seed,
            'status': 'error',
            'error': str(e),
        }


def run_sequential(experiments, gpu_id=None, dry_run=False):
    """Run experiments sequentially."""
    results = []
    for model, seed, model_type in experiments:
        result = run_experiment(model, seed, model_type, gpu_id, dry_run)
        results.append(result)
    return results


def run_parallel_gpus(experiments, num_gpus=2, dry_run=False):
    """Run experiments in parallel across GPUs."""
    results = []

    def worker(args):
        model, seed, model_type, gpu_id = args
        return run_experiment(model, seed, model_type, gpu_id, dry_run)

    # Create work items with GPU assignment (round-robin)
    work_items = []
    for i, (model, seed, model_type) in enumerate(experiments):
        gpu_id = i % num_gpus
        work_items.append((model, seed, model_type, gpu_id))

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(worker, item): item for item in work_items}
        for future in as_completed(futures):
            results.append(future.result())

    return results


def main():
    parser = argparse.ArgumentParser(description='Run missing baseline experiments')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without running')
    parser.add_argument('--ml-only', action='store_true', help='Only run ML models')
    parser.add_argument('--nn-only', action='store_true', help='Only run NN models')
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs for NN models')
    args = parser.parse_args()

    print("=" * 70)
    print("Running Missing Baseline Experiments")
    print("=" * 70)
    print(f"Data dir: {DATA_DIR}")
    print(f"Output base: {OUTPUT_BASE}")
    print(f"Dry run: {args.dry_run}")
    print(f"Num GPUs: {args.num_gpus}")
    print()

    # Check data directory
    if not Path(DATA_DIR).exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)

    all_results = []
    t0 = time.time()

    # Phase 1: ML models (sequential, CPU-bound)
    if not args.nn_only:
        print("=" * 70)
        print("PHASE 1: ML Models (sequential, CPU-bound)")
        print("=" * 70)

        ml_experiments = [(m, s, 'ml') for m in ML_MODELS for s in SEEDS]
        print(f"Experiments: {len(ml_experiments)}")
        print()

        ml_results = run_sequential(ml_experiments, gpu_id=0, dry_run=args.dry_run)
        all_results.extend(ml_results)

    # Phase 2: NN models (parallel across GPUs)
    if not args.ml_only:
        print()
        print("=" * 70)
        print(f"PHASE 2: NN Models (parallel, {args.num_gpus} GPUs)")
        print("=" * 70)

        nn_experiments = [(m, s, 'nn') for m in NN_MODELS for s in SEEDS]
        print(f"Experiments: {len(nn_experiments)}")
        print()

        nn_results = run_parallel_gpus(nn_experiments, num_gpus=args.num_gpus, dry_run=args.dry_run)
        all_results.extend(nn_results)

    # Summary
    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success = sum(1 for r in all_results if r['status'] == 'success')
    skipped = sum(1 for r in all_results if r['status'] == 'skipped')
    failed = sum(1 for r in all_results if r['status'] in ['failed', 'error', 'timeout'])

    print(f"Total: {len(all_results)}")
    print(f"Success: {success}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")

    if failed > 0:
        print()
        print("Failed experiments:")
        for r in all_results:
            if r['status'] in ['failed', 'error', 'timeout']:
                print(f"  - {r['model']} seed={r['seed']}: {r.get('error', r['status'])}")

    # Save results log
    log_path = Path(OUTPUT_BASE) / "run_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'summary': {
                'total': len(all_results),
                'success': success,
                'skipped': skipped,
                'failed': failed,
            },
            'results': all_results,
        }, f, indent=2)
    print(f"\nLog saved to: {log_path}")

    # Count total completed
    total_completed = len(list(Path(OUTPUT_BASE).rglob("results.json")))
    print(f"\nTotal baseline results: {total_completed}/30")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
