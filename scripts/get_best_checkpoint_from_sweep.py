#!/usr/bin/env python3
"""
Get the best checkpoint from a W&B sweep and optionally copy it locally.

Usage:
    python scripts/get_best_checkpoint_from_sweep.py \
        --sweep-id njo48wle \
        --entity seacuello-university-of-rhode-island \
        --project uncategorized \
        --output-dir outputs/best_from_sweep
"""

import argparse
import wandb
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def fetch_sweep_runs(sweep_id: str, entity: str = None, project: str = 'gcode-fingerprinting'):
    """
    Fetch all runs from a W&B sweep.

    Returns:
        List of run objects sorted by target metric (descending)
    """
    try:
        api = wandb.Api()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize W&B API: {e}\nMake sure you're logged in: wandb login")

    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    try:
        sweep = api.sweep(sweep_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Sweep not found: {sweep_path}\n"
                        f"Please check:\n"
                        f"  - Sweep ID is correct: {sweep_id}\n"
                        f"  - Entity is correct: {entity}\n"
                        f"  - Project is correct: {project}\n"
                        f"Original error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch sweep: {e}")

    print(f"✓ Found sweep: {sweep.name}")
    print(f"  Metric: {sweep.config.get('metric', {}).get('name', 'N/A')}")
    print(f"  Goal: {sweep.config.get('metric', {}).get('goal', 'N/A')}")
    print()

    runs = []
    for run in sweep.runs:
        # Only include finished runs
        if run.state == 'finished':
            runs.append(run)

    if len(runs) == 0:
        raise ValueError(f"No finished runs found for sweep: {sweep_id}")

    return runs, sweep


def display_top_runs(runs: List, metric: str = 'val/overall_acc', top_n: int = 5):
    """Display top N runs with their configurations and metrics."""

    # Extract data
    run_data = []
    for run in runs:
        try:
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary = run.summary._json_dict

            # Get metric value
            metric_value = summary.get(metric, None)
            if metric_value is None:
                continue

            run_data.append({
                'run_id': run.id,
                'name': run.name,
                'metric_value': metric_value,
                'config': config,
                'summary': summary
            })
        except Exception as e:
            print(f"Warning: Failed to process run {run.id}: {e}")
            continue

    if len(run_data) == 0:
        raise ValueError(f"No runs with metric '{metric}' found")

    # Sort by metric (descending)
    run_data.sort(key=lambda x: x['metric_value'], reverse=True)

    # Display top N
    print(f"=" * 80)
    print(f"TOP {top_n} RUNS (sorted by {metric})")
    print(f"=" * 80)
    print()

    for idx, run_info in enumerate(run_data[:top_n], 1):
        print(f"Rank {idx}: {run_info['name']} (ID: {run_info['run_id']})")
        print(f"  {metric}: {run_info['metric_value']:.4f}")

        # Display key hyperparameters
        config = run_info['config']
        print(f"  Hyperparameters:")
        for key in ['learning_rate', 'batch_size', 'hidden_dim', 'num_heads',
                    'num_layers', 'weight_decay', 'grad_clip', 'command_weight']:
            if key in config:
                print(f"    {key}: {config[key]}")
        print()

    return run_data


def save_best_config(run_data: List, output_dir: Path):
    """Save the best run's configuration to JSON."""
    best_run = run_data[0]

    config_file = output_dir / 'best_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'run_id': best_run['run_id'],
            'run_name': best_run['name'],
            'metric_value': best_run['metric_value'],
            'config': best_run['config']
        }, f, indent=2)

    print(f"✓ Saved best config to: {config_file}")
    return config_file


def save_all_runs_csv(runs: List, output_dir: Path, metric: str):
    """Save all run data to CSV for analysis."""

    run_records = []
    for run in runs:
        try:
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary = run.summary._json_dict

            record = {
                'run_id': run.id,
                'name': run.name,
                'state': run.state,
                'metric': summary.get(metric, None),
                **config
            }
            run_records.append(record)
        except Exception:
            continue

    df = pd.DataFrame(run_records)
    df = df.sort_values('metric', ascending=False)

    csv_file = output_dir / 'all_runs.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved all runs to: {csv_file}")
    return csv_file


def get_checkpoint_path_from_run(run, output_dir: Path):
    """
    Get the checkpoint file for a run.

    Note: W&B checkpoints need to be downloaded using wandb.restore() or manually from UI.
    This function looks for local checkpoint files matching the run ID pattern.
    """

    # Check for local checkpoint file with run ID
    project_root = Path(__file__).parent.parent

    # Common checkpoint locations
    possible_paths = [
        project_root / 'outputs' / f'sweep_best_{run.id}.pt',
        project_root / 'outputs' / 'training' / f'checkpoint_{run.id}_best.pt',
        project_root / 'outputs' / 'multihead_v2' / 'checkpoint_best.pt',
    ]

    for checkpoint_path in possible_paths:
        if checkpoint_path.exists():
            print(f"✓ Found local checkpoint: {checkpoint_path}")

            # Copy to output directory
            dest_path = output_dir / 'checkpoint_best.pt'
            shutil.copy2(checkpoint_path, dest_path)
            print(f"✓ Copied checkpoint to: {dest_path}")
            return dest_path

    # If not found locally, provide instructions to download from W&B
    print(f"⚠️  Checkpoint not found locally for run {run.id}")
    print(f"    To download from W&B, visit:")
    print(f"    https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}/files")
    print(f"    Look for checkpoint files in the 'files' tab")
    print()

    return None


def main():
    parser = argparse.ArgumentParser(description='Get best checkpoint from W&B sweep')
    parser.add_argument('--sweep-id', type=str, required=True, help='W&B sweep ID')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity/username')
    parser.add_argument('--project', type=str, default='gcode-fingerprinting', help='W&B project name')
    parser.add_argument('--metric', type=str, default='val/overall_acc', help='Metric to optimize')
    parser.add_argument('--output-dir', type=str, default='outputs/best_from_sweep',
                       help='Output directory for best checkpoint and config')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top runs to display')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch sweep runs
        print(f"Fetching sweep runs for: {args.sweep_id}")
        runs, sweep = fetch_sweep_runs(args.sweep_id, args.entity, args.project)
        print(f"Found {len(runs)} finished runs\n")

        # Display top runs
        run_data = display_top_runs(runs, args.metric, args.top_n)

        # Save best config
        save_best_config(run_data, output_dir)

        # Save all runs to CSV
        save_all_runs_csv(runs, output_dir, args.metric)

        # Try to get checkpoint
        best_run_obj = [r for r in runs if r.id == run_data[0]['run_id']][0]
        checkpoint_path = get_checkpoint_path_from_run(best_run_obj, output_dir)

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Best run: {run_data[0]['name']}")
        print(f"Run ID: {run_data[0]['run_id']}")
        print(f"{args.metric}: {run_data[0]['metric_value']:.4f}")
        print()
        print(f"Config saved to: {output_dir / 'best_config.json'}")
        print(f"All runs saved to: {output_dir / 'all_runs.csv'}")

        if checkpoint_path:
            print(f"Checkpoint saved to: {checkpoint_path}")
        else:
            print(f"Checkpoint needs to be downloaded manually from W&B")

        print()
        print("Next steps:")
        print(f"  1. Evaluate checkpoint: python scripts/evaluate_checkpoint.py --checkpoint {output_dir}/checkpoint_best.pt")
        print(f"  2. Deploy checkpoint: python scripts/deploy_checkpoint.py --source {output_dir}/checkpoint_best.pt")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
