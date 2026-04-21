#!/usr/bin/env python3
"""Parallel Cross-Validation Experiment Runner.

Orchestrates the full experiment matrix from the email thread:
  Phase 1: 4 feature configs × 5 folds × 6 models = 120 experiments
  Phase 2: 3 window/stride configs × 5 folds × 6 models = 90 experiments
  Total: 210 experiments (20 GPU, 190 CPU-only)

Uses Ray for resource-aware scheduling:
  - Encoder (MM-DTAE-LSTM) tasks get 1 GPU each → 2 run in parallel on 2× A6000
  - Baseline tasks (XGBoost, RF, LR, MLP, SimpleLSTM) get CPU only → many in parallel

Preprocessing is done per (feature_config, window, stride, fold) before any
training tasks for that combination are submitted.

Usage:
    # Full run (all 210 experiments)
    python scripts/experiments/run_cv_parallel.py

    # Phase 1 only (feature ablation, default window)
    python scripts/experiments/run_cv_parallel.py --phase 1

    # Phase 2 only (window/stride ablation, specify feature config)
    python scripts/experiments/run_cv_parallel.py --phase 2 --phase2-features full

    # Dry run (show what would execute)
    python scripts/experiments/run_cv_parallel.py --dry-run

    # Resume (skip completed experiments)
    python scripts/experiments/run_cv_parallel.py --resume

    # Custom parallelism
    python scripts/experiments/run_cv_parallel.py --num-gpus 2 --baseline-workers 20
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

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREPROCESS_SCRIPT = PROJECT_ROOT / 'romesh_changes' / 'run_preprocessing_cv_fold.py'
ENCODER_SCRIPT = PROJECT_ROOT / 'scripts' / 'evaluation' / 'run_9class_direct.py'
BASELINE_SCRIPT = PROJECT_ROOT / 'scripts' / 'evaluation' / 'run_baseline_models.py'
AGGREGATE_SCRIPT = PROJECT_ROOT / 'romesh_changes' / 'aggregate_cv_results.py'
SENSOR_ANALYSIS_SCRIPT = PROJECT_ROOT / 'scripts' / 'analysis' / 'identify_consistent_sensors.py'

DATA_DIR = PROJECT_ROOT / 'data_clean'
VOCAB_PATH = PROJECT_ROOT / 'data' / 'gcode_vocab_v2.json'
SENSOR_REPORT = PROJECT_ROOT / 'outputs' / 'sensor_consistency_report.json'

SEED = 42
N_FOLDS = 5
BASELINE_MODELS = ['xgboost', 'random_forest', 'logistic_regression', 'mlp', 'lstm_simple']

# ── Experiment configurations ────────────────────────────────────────────────

FEATURE_CONFIGS = {
    'full': {
        'flags': [],
        'n_features': 110,
    },
    'no_proximity': {
        'flags': ['--exclude-proximity'],
        'n_features': 104,
    },
    'no_proximity_no_pressure': {
        'flags': ['--exclude-proximity', '--exclude-pressure'],
        'n_features': 98,
    },
    'no_proximity_no_pressure_no_color': {
        'flags': ['--exclude-proximity', '--exclude-pressure', '--exclude-color'],
        'n_features': 74,
    },
    'no_proximity_no_pressure_no_color_no_magnetometer': {
        'flags': ['--exclude-proximity', '--exclude-pressure', '--exclude-color', '--exclude-magnetometer'],
        'n_features': 56,
    },
}

PHASE1_WINDOW_STRIDE = [(64, 16)]  # default

PHASE2_WINDOW_STRIDES = [
    (16, 4),        # tiny window, 4:1 overlap
    (16, 8),        # tiny window, 2:1 overlap
    (32, 8),        # small window, 4:1 overlap
    # (64, 16)      # default — already covered in Phase 1
    (64, 32),       # default window, 2:1 overlap
    (64, 64),       # default window, no overlap
    (128, 32),      # large window, 4:1 overlap
    (128, 64),      # large window, 2:1 overlap
    (256, 64),      # very large window, 4:1 overlap
    (256, 128),     # very large window, 2:1 overlap
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def exp_name(feat_config, window, stride):
    return f"{feat_config}_w{window}_s{stride}_cv"


def is_experiment_complete(output_dir, fold, model_key):
    """Check if a specific model training has already produced results."""
    if model_key == 'encoder':
        results_path = output_dir / f'fold_{fold}' / 'encoder' / 'results.json'
    else:
        results_path = output_dir / f'fold_{fold}' / 'baselines' / model_key / 'results.json'
    return results_path.exists()


def is_preprocessing_complete(output_dir, fold):
    """Check if preprocessing for a fold has completed."""
    preprocess_dir = output_dir / f'fold_{fold}' / 'preprocessed'
    return (preprocess_dir / 'train_sequences.npz').exists()


def run_subprocess(cmd, env=None, timeout=14400):
    """Run a command as subprocess, return result dict."""
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env or os.environ.copy(), timeout=timeout,
        )
        return {
            'success': result.returncode == 0,
            'elapsed': time.time() - start,
            'returncode': result.returncode,
            'stdout': result.stdout[-3000:] if result.stdout else '',
            'stderr': result.stderr[-3000:] if result.stderr else '',
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'elapsed': time.time() - start, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'elapsed': time.time() - start, 'error': str(e)}


# ── Preprocessing ────────────────────────────────────────────────────────────

def run_preprocessing(feat_config, window, stride, fold, output_base):
    """Run preprocessing for one (config, fold) combination."""
    preprocess_dir = output_base / f'fold_{fold}' / 'preprocessed'

    cmd = [
        sys.executable, str(PREPROCESS_SCRIPT),
        '--data-dir', str(DATA_DIR),
        '--output-dir', str(preprocess_dir),
        '--vocab-path', str(VOCAB_PATH),
        '--sensor-report', str(SENSOR_REPORT),
        '--threshold', '93.0',
        '--fold', str(fold),
        '--n-folds', str(N_FOLDS),
        '--window-size', str(window),
        '--stride', str(stride),
    ]
    cmd.extend(FEATURE_CONFIGS[feat_config]['flags'])

    return run_subprocess(cmd, timeout=600)


# ── Training tasks ───────────────────────────────────────────────────────────

def build_encoder_cmd(preprocess_dir, encoder_dir, fold):
    return [
        sys.executable, str(ENCODER_SCRIPT),
        '--data-dir', str(preprocess_dir),
        '--output-dir', str(encoder_dir),
        '--iteration', f'cv_fold_{fold}',
        '--max_epochs', '200',
        '--patience', '40',
        '--seed', str(SEED),
    ]


def build_baseline_cmd(preprocess_dir, baseline_dir, model, fold):
    return [
        sys.executable, str(BASELINE_SCRIPT),
        '--data-dir', str(preprocess_dir),
        '--output-dir', str(baseline_dir),
        '--model', model,
        '--seed', str(SEED),
    ]


# ── Ray remote tasks ────────────────────────────────────────────────────────

if HAS_RAY:
    @ray.remote(num_gpus=1, num_cpus=2)
    def train_encoder_ray(cmd_args):
        """Train encoder with 1 GPU allocated by Ray."""
        result = run_subprocess(cmd_args)
        return result

    @ray.remote(num_gpus=0, num_cpus=4)
    def train_baseline_ray(cmd_args):
        """Train baseline with CPU only.

        Ray sets CUDA_VISIBLE_DEVICES="" for num_gpus=0 workers, which crashes
        XGBoost 3.x (CUDA-compiled) during CUDA driver init. We restore
        visibility to GPU 0 so the CUDA runtime initializes, but XGBoost still
        runs on CPU via device='cpu' / tree_method='hist'.
        """
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        result = run_subprocess(cmd_args, env=env)
        return result


# ── Manifest generation ──────────────────────────────────────────────────────

def generate_manifest(phase, phase2_features, output_base_root):
    """Generate the full list of experiments to run."""
    experiments = []

    if phase in (1, 'all'):
        for feat_config in FEATURE_CONFIGS:
            for window, stride in PHASE1_WINDOW_STRIDE:
                ename = exp_name(feat_config, window, stride)
                output_base = output_base_root / ename
                for fold in range(1, N_FOLDS + 1):
                    preprocess_dir = output_base / f'fold_{fold}' / 'preprocessed'
                    encoder_dir = output_base / f'fold_{fold}' / 'encoder'

                    experiments.append({
                        'experiment_id': f'{ename}_fold{fold}_encoder',
                        'feat_config': feat_config,
                        'window': window,
                        'stride': stride,
                        'fold': fold,
                        'model': 'encoder',
                        'needs_gpu': True,
                        'output_base': str(output_base),
                        'preprocess_dir': str(preprocess_dir),
                        'cmd': build_encoder_cmd(preprocess_dir, encoder_dir, fold),
                    })

                    for bmodel in BASELINE_MODELS:
                        baseline_dir = output_base / f'fold_{fold}' / 'baselines' / bmodel
                        experiments.append({
                            'experiment_id': f'{ename}_fold{fold}_{bmodel}',
                            'feat_config': feat_config,
                            'window': window,
                            'stride': stride,
                            'fold': fold,
                            'model': bmodel,
                            'needs_gpu': False,
                            'output_base': str(output_base),
                            'preprocess_dir': str(preprocess_dir),
                            'cmd': build_baseline_cmd(preprocess_dir, baseline_dir, bmodel, fold),
                        })

    if phase in (2, 'all'):
        feat_config = phase2_features
        for window, stride in PHASE2_WINDOW_STRIDES:
            ename = exp_name(feat_config, window, stride)
            output_base = output_base_root / ename
            for fold in range(1, N_FOLDS + 1):
                preprocess_dir = output_base / f'fold_{fold}' / 'preprocessed'
                encoder_dir = output_base / f'fold_{fold}' / 'encoder'

                experiments.append({
                    'experiment_id': f'{ename}_fold{fold}_encoder',
                    'feat_config': feat_config,
                    'window': window,
                    'stride': stride,
                    'fold': fold,
                    'model': 'encoder',
                    'needs_gpu': True,
                    'output_base': str(output_base),
                    'preprocess_dir': str(preprocess_dir),
                    'cmd': build_encoder_cmd(preprocess_dir, encoder_dir, fold),
                })

                for bmodel in BASELINE_MODELS:
                    baseline_dir = output_base / f'fold_{fold}' / 'baselines' / bmodel
                    experiments.append({
                        'experiment_id': f'{ename}_fold{fold}_{bmodel}',
                        'feat_config': feat_config,
                        'window': window,
                        'stride': stride,
                        'fold': fold,
                        'model': bmodel,
                        'needs_gpu': False,
                        'output_base': str(output_base),
                        'preprocess_dir': str(preprocess_dir),
                        'cmd': build_baseline_cmd(preprocess_dir, baseline_dir, bmodel, fold),
                    })

    return experiments


# ── Execution backends ───────────────────────────────────────────────────────

def run_with_ray(experiments, num_gpus=2, baseline_workers=20):
    """Run experiments with Ray, respecting GPU/CPU resource needs."""
    if not HAS_RAY:
        raise ImportError("Ray not installed. Install with: pip install ray")

    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus, num_cpus=os.cpu_count())
        print(f"Ray initialized: {num_gpus} GPUs, {os.cpu_count()} CPUs")

    gpu_exps = [e for e in experiments if e['needs_gpu']]
    cpu_exps = [e for e in experiments if not e['needs_gpu']]

    print(f"Submitting {len(gpu_exps)} GPU tasks + {len(cpu_exps)} CPU tasks to Ray")

    futures = {}
    for exp in gpu_exps:
        future = train_encoder_ray.remote(exp['cmd'])
        futures[future] = exp
    for exp in cpu_exps:
        future = train_baseline_ray.remote(exp['cmd'])
        futures[future] = exp

    results = []
    completed = 0
    total = len(futures)
    start_time = time.time()

    ready, remaining = ray.wait(list(futures.keys()), num_returns=1, timeout=None)
    while ready:
        for future in ready:
            result = ray.get(future)
            exp = futures[future]
            result['experiment_id'] = exp['experiment_id']
            result['model'] = exp['model']
            result['feat_config'] = exp['feat_config']
            result['fold'] = exp['fold']
            results.append(result)
            completed += 1

            status = 'OK' if result['success'] else 'FAIL'
            elapsed = result['elapsed']
            total_elapsed = time.time() - start_time
            eta = (total_elapsed / completed) * (total - completed) / 60 if completed else 0

            print(f"  [{completed}/{total}] {status} {exp['experiment_id']} "
                  f"({elapsed:.0f}s) | ETA: {eta:.1f}min")

            if not result['success']:
                stderr = result.get('stderr', result.get('error', ''))
                if stderr:
                    print(f"    ERROR: {stderr[:200]}")

        if remaining:
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        else:
            break

    return results


def run_with_threads(experiments, num_gpus=2, baseline_workers=20):
    """Fallback: run with ThreadPoolExecutor if Ray unavailable."""
    gpu_exps = [e for e in experiments if e['needs_gpu']]
    cpu_exps = [e for e in experiments if not e['needs_gpu']]

    results = []
    completed = 0
    total = len(experiments)
    start_time = time.time()

    def run_one(exp, gpu_id=None):
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        cmd = exp['cmd']
        result = run_subprocess(cmd, env=env)
        result['experiment_id'] = exp['experiment_id']
        return result

    # GPU tasks: limited to num_gpus concurrent
    print(f"Running {len(gpu_exps)} encoder tasks ({num_gpus} GPUs)...")
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        gpu_futures = {
            executor.submit(run_one, exp, i % num_gpus): exp
            for i, exp in enumerate(gpu_exps)
        }
        for future in as_completed(gpu_futures):
            result = future.result()
            results.append(result)
            completed += 1
            status = 'OK' if result['success'] else 'FAIL'
            print(f"  [{completed}/{total}] {status} {result['experiment_id']} ({result['elapsed']:.0f}s)")

    # CPU tasks: many concurrent
    print(f"Running {len(cpu_exps)} baseline tasks ({baseline_workers} workers)...")
    with ThreadPoolExecutor(max_workers=baseline_workers) as executor:
        cpu_futures = {
            executor.submit(run_one, exp): exp
            for exp in cpu_exps
        }
        for future in as_completed(cpu_futures):
            result = future.result()
            results.append(result)
            completed += 1
            status = 'OK' if result['success'] else 'FAIL'
            print(f"  [{completed}/{total}] {status} {result['experiment_id']} ({result['elapsed']:.0f}s)")

    return results


# ── Aggregation ──────────────────────────────────────────────────────────────

def run_aggregation(output_base):
    """Run the aggregation script for a single experiment config."""
    summary_path = output_base / 'cv_summary.json'
    cmd = [
        sys.executable, str(AGGREGATE_SCRIPT),
        '--cv-dir', str(output_base),
        '--n-folds', str(N_FOLDS),
        '--output', str(summary_path),
    ]
    result = run_subprocess(cmd, timeout=120)
    if result['success']:
        print(f"  Aggregated: {summary_path}")
    else:
        print(f"  Aggregation FAILED for {output_base}: {result.get('stderr', '')[:200]}")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Parallel Cross-Validation Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (all 210 experiments)
  python scripts/experiments/run_cv_parallel.py

  # Phase 1 only
  python scripts/experiments/run_cv_parallel.py --phase 1

  # Phase 2 only (must specify feature config)
  python scripts/experiments/run_cv_parallel.py --phase 2 --phase2-features full

  # Dry run
  python scripts/experiments/run_cv_parallel.py --dry-run

  # Resume incomplete run
  python scripts/experiments/run_cv_parallel.py --resume
        """,
    )
    parser.add_argument('--phase', type=str, default='all', choices=['1', '2', 'all'],
                        help='Which phase to run: 1 (feature ablation), 2 (window/stride), all (default)')
    parser.add_argument('--phase2-features', type=str, default='full',
                        choices=list(FEATURE_CONFIGS.keys()),
                        help='Feature config for Phase 2 window/stride ablation (default: full)')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Output root directory (default: outputs/experiments_YYYY_MM_DD)')
    parser.add_argument('--num-gpus', type=int, default=2,
                        help='Number of GPUs available (default: 2)')
    parser.add_argument('--baseline-workers', type=int, default=20,
                        help='Max concurrent baseline tasks (default: 20)')
    parser.add_argument('--backend', type=str, default='ray', choices=['ray', 'threads'],
                        help='Parallelization backend (default: ray)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments that already have results.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiment plan without executing')
    parser.add_argument('--save-manifest', type=str, default=None,
                        help='Save manifest JSON to this path (useful for inspection)')
    args = parser.parse_args()

    # Parse phase
    phase = args.phase if args.phase == 'all' else int(args.phase)

    # Output root
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        date_tag = datetime.now().strftime('%Y_%m_%d')
        output_root = PROJECT_ROOT / 'outputs' / f'experiments_{date_tag}'
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Generate manifest ────────────────────────────────────────────────────
    experiments = generate_manifest(phase, args.phase2_features, output_root)

    # Count summaries
    gpu_count = sum(1 for e in experiments if e['needs_gpu'])
    cpu_count = sum(1 for e in experiments if not e['needs_gpu'])
    configs = set((e['feat_config'], e['window'], e['stride']) for e in experiments)

    print("=" * 80)
    print("PARALLEL CV EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"  Phase:           {phase}")
    print(f"  Output root:     {output_root}")
    print(f"  Backend:         {args.backend}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Baseline workers:{args.baseline_workers}")
    print(f"  Total experiments: {len(experiments)} ({gpu_count} GPU, {cpu_count} CPU)")
    print(f"  Unique configs:  {len(configs)}")
    for fc, w, s in sorted(configs):
        n = sum(1 for e in experiments if e['feat_config'] == fc and e['window'] == w and e['stride'] == s)
        print(f"    {exp_name(fc, w, s)}: {n} experiments")
    print()

    # Save manifest if requested
    if args.save_manifest:
        manifest_path = Path(args.save_manifest)
        # Remove non-serializable cmd (has Path objects)
        serializable = []
        for e in experiments:
            se = dict(e)
            se['cmd'] = [str(c) for c in se['cmd']]
            serializable.append(se)
        with open(manifest_path, 'w') as f:
            json.dump({'experiments': serializable, 'total': len(serializable)}, f, indent=2)
        print(f"Manifest saved to: {manifest_path}")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDRY RUN — commands that would execute:\n")
        # Show preprocessing
        for fc, w, s in sorted(configs):
            ename = exp_name(fc, w, s)
            print(f"# Preprocessing for {ename}")
            for fold in range(1, N_FOLDS + 1):
                print(f"  python run_preprocessing_cv_fold.py ... --fold {fold} "
                      f"--window-size {w} --stride {s} {' '.join(FEATURE_CONFIGS[fc]['flags'])}")
            print()

        # Show training (first 10)
        print("# Training tasks (showing first 10):")
        for exp in experiments[:10]:
            gpu_tag = "[GPU]" if exp['needs_gpu'] else "[CPU]"
            print(f"  {gpu_tag} {exp['experiment_id']}")
        if len(experiments) > 10:
            print(f"  ... and {len(experiments) - 10} more")
        return

    # ── Sensor report ────────────────────────────────────────────────────────
    if not SENSOR_REPORT.exists():
        print("Running sensor consistency analysis...")
        run_subprocess([
            sys.executable, str(SENSOR_ANALYSIS_SCRIPT),
            '--data-dir', str(DATA_DIR),
            '--threshold', '93.0',
            '--output', str(SENSOR_REPORT),
        ])

    # ── Preprocessing ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING")
    print("=" * 80)

    preprocess_tasks = set()
    for exp in experiments:
        key = (exp['feat_config'], exp['window'], exp['stride'], exp['fold'])
        preprocess_tasks.add(key)

    preprocess_tasks = sorted(preprocess_tasks)
    print(f"  {len(preprocess_tasks)} preprocessing tasks")

    failed_preprocess = set()
    for i, (fc, w, s, fold) in enumerate(preprocess_tasks):
        ename = exp_name(fc, w, s)
        output_base = output_root / ename

        if args.resume and is_preprocessing_complete(output_base, fold):
            print(f"  [{i+1}/{len(preprocess_tasks)}] SKIP {ename} fold {fold} (already done)")
            continue

        print(f"  [{i+1}/{len(preprocess_tasks)}] Preprocessing {ename} fold {fold}...")
        result = run_preprocessing(fc, w, s, fold, output_base)
        if result['success']:
            print(f"    OK ({result['elapsed']:.0f}s)")
        else:
            print(f"    FAILED: {result.get('stderr', result.get('error', ''))[:200]}")
            failed_preprocess.add((fc, w, s, fold))

    if failed_preprocess:
        print(f"\n  WARNING: {len(failed_preprocess)} preprocessing tasks failed!")
        # Remove training experiments that depend on failed preprocessing
        before = len(experiments)
        experiments = [
            e for e in experiments
            if (e['feat_config'], e['window'], e['stride'], e['fold']) not in failed_preprocess
        ]
        print(f"  Removed {before - len(experiments)} dependent training tasks")

    # ── Resume: filter completed ─────────────────────────────────────────────
    if args.resume:
        before = len(experiments)
        experiments = [
            e for e in experiments
            if not is_experiment_complete(
                Path(e['output_base']), e['fold'], e['model']
            )
        ]
        skipped = before - len(experiments)
        if skipped:
            print(f"\n  Resume: skipping {skipped} completed experiments, {len(experiments)} remaining")

    if not experiments:
        print("\nAll experiments already complete!")
    else:
        # ── Training ─────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING")
        print("=" * 80)

        gpu_count = sum(1 for e in experiments if e['needs_gpu'])
        cpu_count = sum(1 for e in experiments if not e['needs_gpu'])
        print(f"  {len(experiments)} tasks: {gpu_count} GPU (encoder), {cpu_count} CPU (baselines)")

        start_time = time.time()

        if args.backend == 'ray':
            results = run_with_ray(experiments, args.num_gpus, args.baseline_workers)
        else:
            results = run_with_threads(experiments, args.num_gpus, args.baseline_workers)

        total_time = time.time() - start_time

        # ── Training summary ─────────────────────────────────────────────────
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes

        print(f"\n{'=' * 80}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 80}")
        print(f"  Successful: {successes}/{len(results)}")
        print(f"  Failed:     {failures}")
        print(f"  Total time: {total_time / 3600:.2f} hours")
        if results:
            print(f"  Avg time:   {total_time / len(results) / 60:.1f} min/experiment")

        if failures:
            print(f"\nFailed experiments:")
            for r in results:
                if not r['success']:
                    err = r.get('error', r.get('stderr', 'Unknown')[:200])
                    print(f"  - {r['experiment_id']}: {err}")

    # ── Aggregation ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("STEP 3: AGGREGATING RESULTS")
    print(f"{'=' * 80}")

    for fc, w, s in sorted(configs):
        ename = exp_name(fc, w, s)
        output_base = output_root / ename
        if output_base.exists():
            run_aggregation(output_base)

    # ── Save run summary ─────────────────────────────────────────────────────
    run_summary = {
        'completed_at': datetime.now().isoformat(),
        'phase': str(phase),
        'output_root': str(output_root),
        'total_configs': len(configs),
        'configs': [{'feat_config': fc, 'window': w, 'stride': s} for fc, w, s in sorted(configs)],
    }
    summary_path = output_root / 'parallel_run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"ALL DONE: {datetime.now()}")
    print(f"Output root: {output_root}")
    print(f"Summary:     {summary_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
