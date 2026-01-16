#!/usr/bin/env python3
"""
Upload Ray Tune local results to WandB.

This script reads completed Ray Tune trials from disk and uploads them to WandB.
Useful when WandB logging didn't work during the sweep.

Usage:
    python scripts/utils/upload_ray_results_to_wandb.py \
        --results-dir outputs/push_to_90_sweep_bizon/gcode-final-sweep \
        --wandb-project gcode-push-to-90-2k-vocab \
        --wandb-group bizon-sweep
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import wandb


def find_trial_dirs(results_dir: Path) -> list:
    """Find all trial directories in the Ray Tune results."""
    trial_dirs = []

    # Ray Tune stores trials in subdirectories starting with 'train_model_'
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith('train_model_'):
            trial_dirs.append(item)

    return sorted(trial_dirs)


def load_trial_results(trial_dir: Path) -> dict:
    """Load results from a trial directory."""
    results = {
        'config': None,
        'metrics': [],
        'trial_id': trial_dir.name,
    }

    # Load params.json for config
    params_file = trial_dir / 'params.json'
    if params_file.exists():
        with open(params_file) as f:
            results['config'] = json.load(f)

    # Load result.json for final metrics
    result_file = trial_dir / 'result.json'
    if result_file.exists():
        with open(result_file) as f:
            for line in f:
                if line.strip():
                    try:
                        results['metrics'].append(json.load(line))
                    except:
                        pass

    # Try progress.csv if result.json doesn't have data
    progress_file = trial_dir / 'progress.csv'
    if progress_file.exists() and not results['metrics']:
        import csv
        with open(progress_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                metric = {}
                for k, v in row.items():
                    try:
                        metric[k] = float(v)
                    except:
                        metric[k] = v
                results['metrics'].append(metric)

    return results


def upload_trial_to_wandb(
    trial_results: dict,
    project: str,
    group: str,
    entity: str = None,
) -> None:
    """Upload a single trial's results to WandB."""

    if not trial_results['config'] or not trial_results['metrics']:
        print(f"  Skipping {trial_results['trial_id']} - no config or metrics")
        return

    config = trial_results['config']
    metrics = trial_results['metrics']

    # Initialize WandB run
    run = wandb.init(
        project=project,
        group=group,
        entity=entity,
        name=trial_results['trial_id'],
        config=config,
        reinit=True,
    )

    # Log all metrics
    for i, metric in enumerate(metrics):
        # Extract relevant metrics
        log_dict = {}

        # Common metric names from Ray Tune
        metric_keys = [
            'train_loss', 'val_loss', 'train_token_acc', 'val_token_acc',
            'epoch', 'training_iteration', 'time_total_s',
            'train_cmd_acc', 'val_cmd_acc', 'train_param_acc', 'val_param_acc',
            'learning_rate', 'lr',
        ]

        for key in metric_keys:
            if key in metric and metric[key] is not None:
                try:
                    log_dict[key] = float(metric[key])
                except (ValueError, TypeError):
                    pass

        if log_dict:
            wandb.log(log_dict, step=i)

    # Log final summary
    if metrics:
        final = metrics[-1]
        summary = {}
        for key in ['val_token_acc', 'val_loss', 'train_token_acc', 'train_loss', 'epoch']:
            if key in final:
                try:
                    summary[f'final_{key}'] = float(final[key])
                except:
                    pass

        # Find best val_token_acc
        best_val_acc = max(
            (float(m.get('val_token_acc', 0)) for m in metrics if m.get('val_token_acc')),
            default=0
        )
        summary['best_val_token_acc'] = best_val_acc

        for k, v in summary.items():
            wandb.run.summary[k] = v

    run.finish()
    print(f"  Uploaded {trial_results['trial_id']} ({len(metrics)} epochs)")


def main():
    parser = argparse.ArgumentParser(description='Upload Ray Tune results to WandB')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to Ray Tune results directory')
    parser.add_argument('--wandb-project', type=str, default='gcode-push-to-90-2k-vocab',
                        help='WandB project name')
    parser.add_argument('--wandb-group', type=str, default='bizon-sweep-upload',
                        help='WandB group name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (username or team)')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Maximum number of trials to upload')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Scanning for trials in: {results_dir}")
    trial_dirs = find_trial_dirs(results_dir)
    print(f"Found {len(trial_dirs)} trials")

    if args.max_trials:
        trial_dirs = trial_dirs[:args.max_trials]
        print(f"Uploading first {args.max_trials} trials")

    # Login to wandb
    wandb.login()

    uploaded = 0
    for i, trial_dir in enumerate(trial_dirs):
        print(f"\n[{i+1}/{len(trial_dirs)}] Processing {trial_dir.name}...")

        try:
            trial_results = load_trial_results(trial_dir)
            upload_trial_to_wandb(
                trial_results,
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
            )
            uploaded += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*50}")
    print(f"Uploaded {uploaded}/{len(trial_dirs)} trials to WandB")
    print(f"Project: {args.wandb_project}")
    print(f"Group: {args.wandb_group}")


if __name__ == '__main__':
    main()
