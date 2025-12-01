#!/usr/bin/env python3
"""
Extract Training Curves from LOCAL W&B Files

Reads LOCAL wandb run files (.wandb) to extract training and validation curves.
Computes aggregate statistics (avg, min, max, std) across all runs.
Generates visualizations of training dynamics.

Usage:
    python scripts/extract_local_curves.py \
        --wandb-dir wandb \
        --sweep-id 27v7pl9i \
        --output reports/training_curves

Output:
    - loss_curves.png (train/val loss over epochs)
    - accuracy_curves.png (train/val accuracies over epochs)
    - aggregate_statistics.json (avg±std, min, max)
    - per_run_summary.csv (final metrics for each run)
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import wandb
from tqdm import tqdm

sns.set_style("whitegrid")


def find_sweep_runs(wandb_dir: Path, sweep_id: str = None) -> List[Path]:
    """
    Find all wandb run directories, optionally filtered by sweep ID.

    Args:
        wandb_dir: Path to wandb directory
        sweep_id: Optional sweep ID to filter runs

    Returns:
        List of run directory paths
    """
    run_dirs = []

    for run_dir in wandb_dir.glob("run-*"):
        if not run_dir.is_dir():
            continue

        # Check if this run belongs to the sweep
        config_file = run_dir / "files" / "config.yaml"

        if sweep_id is not None:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_text = f.read()
                    # Check if sweep ID appears in config
                    if sweep_id not in config_text:
                        continue
            else:
                # Skip runs without config
                continue

        # Find .wandb file
        wandb_files = list(run_dir.glob("*.wandb"))
        if wandb_files:
            run_dirs.append(run_dir)

    return run_dirs


def extract_run_history(run_dir: Path) -> Dict:
    """
    Extract training history from a local wandb run directory.

    Args:
        run_dir: Path to wandb run directory

    Returns:
        Dict with run_id, config, and history dataframe
    """
    # Extract run ID from directory name
    run_id = run_dir.name.split('-')[-1]

    # Load config
    config_file = run_dir / "files" / "config.yaml"
    summary_file = run_dir / "files" / "wandb-summary.json"

    config = {}
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            # Extract actual hyperparameters (ignoring _wandb metadata)
            for key, value in config_data.items():
                if not key.startswith('_'):
                    if isinstance(value, dict) and 'value' in value:
                        config[key] = value['value']
                    else:
                        config[key] = value

    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

    # Extract history from .wandb file using wandb API
    wandb_file = list(run_dir.glob("*.wandb"))[0]

    # Use wandb.restore() to read local run file
    # Note: This requires wandb to be properly initialized
    history_data = []

    try:
        # Read the wandb file directly
        # The .wandb file is a binary protobuf format
        # We'll need to use wandb's internal API
        import wandb.sdk.internal.internal_api as internal_api

        # Alternative: Parse using wandb's History class
        # For now, we'll extract from summary which has final values
        # and estimate curves if possible from output.log

        # Try to parse output.log for epoch-by-epoch metrics
        output_log = run_dir / "files" / "output.log"
        if output_log.exists():
            history_data = parse_output_log(output_log)

    except Exception as e:
        print(f"  Warning: Could not extract full history for {run_id}: {e}")

    return {
        'run_id': run_id,
        'run_dir': str(run_dir),
        'config': config,
        'summary': summary,
        'history': history_data,
    }


def parse_output_log(log_path: Path) -> List[Dict]:
    """
    Parse output.log to extract epoch-by-epoch metrics.

    Looks for patterns like:
    Epoch 1/250 - Train Loss: 2.345, Val Loss: 1.234, Val Acc: 0.567
    """
    history = []

    with open(log_path, 'r') as f:
        for line in f:
            # Look for epoch summary lines
            # Pattern varies by training script
            # Try to extract common metrics

            # Example pattern: "Epoch 5/250"
            if 'Epoch ' in line and '/' in line:
                # Try to extract metrics from this line or surrounding lines
                # This is a simplified parser - may need adjustment
                parts = {}

                # Extract epoch number
                if 'Epoch ' in line:
                    try:
                        epoch_str = line.split('Epoch ')[1].split('/')[0].strip()
                        parts['epoch'] = int(epoch_str)
                    except:
                        continue

                # Look for loss values
                if 'train_loss' in line.lower() or 'train/loss' in line.lower():
                    try:
                        # Extract train loss
                        if 'train_loss:' in line:
                            parts['train_loss'] = float(line.split('train_loss:')[1].split()[0].strip(','))
                        elif 'train/loss:' in line:
                            parts['train_loss'] = float(line.split('train/loss:')[1].split()[0].strip(','))
                    except:
                        pass

                if 'val_loss' in line.lower() or 'val/loss' in line.lower():
                    try:
                        if 'val_loss:' in line:
                            parts['val_loss'] = float(line.split('val_loss:')[1].split()[0].strip(','))
                        elif 'val/loss:' in line:
                            parts['val_loss'] = float(line.split('val/loss:')[1].split()[0].strip(','))
                    except:
                        pass

                if 'val/param_type_acc' in line.lower():
                    try:
                        parts['val_param_type_acc'] = float(line.split('val/param_type_acc:')[1].split()[0].strip(','))
                    except:
                        pass

                if parts and 'epoch' in parts:
                    history.append(parts)

    return history


def aggregate_curves(all_runs: List[Dict]) -> Dict:
    """
    Aggregate training curves across multiple runs.

    Returns:
        Dict with aggregate statistics
    """
    # Collect all histories with valid data
    valid_histories = []

    for run in all_runs:
        if run['history'] and len(run['history']) > 0:
            df = pd.DataFrame(run['history'])
            if 'epoch' in df.columns:
                valid_histories.append(df)

    if not valid_histories:
        print("  Warning: No valid training histories found in logs")
        print("  Attempting to use summary statistics only...")
        return aggregate_from_summaries(all_runs)

    # Find common metrics across all runs
    common_metrics = set(valid_histories[0].columns)
    for df in valid_histories[1:]:
        common_metrics &= set(df.columns)

    common_metrics.discard('epoch')

    if not common_metrics:
        print("  Warning: No common metrics found across runs")
        return {}

    # Determine max epoch
    max_epochs = max(df['epoch'].max() for df in valid_histories)

    # Create aggregate dataframe
    epochs = np.arange(1, max_epochs + 1)
    aggregates = {'epoch': epochs}

    for metric in common_metrics:
        # Collect metric values across runs for each epoch
        metric_by_epoch = {epoch: [] for epoch in epochs}

        for df in valid_histories:
            for _, row in df.iterrows():
                epoch = int(row['epoch'])
                if metric in row and not pd.isna(row[metric]):
                    metric_by_epoch[epoch].append(row[metric])

        # Compute statistics
        means = []
        stds = []
        mins = []
        maxs = []

        for epoch in epochs:
            values = metric_by_epoch[epoch]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
                mins.append(np.min(values))
                maxs.append(np.max(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)

        aggregates[f'{metric}_mean'] = means
        aggregates[f'{metric}_std'] = stds
        aggregates[f'{metric}_min'] = mins
        aggregates[f'{metric}_max'] = maxs

    return pd.DataFrame(aggregates)


def aggregate_from_summaries(all_runs: List[Dict]) -> Dict:
    """
    Aggregate final summary statistics when full histories unavailable.
    """
    summaries = []

    for run in all_runs:
        if run['summary']:
            summary_data = {
                'run_id': run['run_id'],
            }

            # Extract key metrics
            for key in ['val/param_type_acc', 'val/loss', 'train/loss', 'epoch']:
                if key in run['summary']:
                    summary_data[key] = run['summary'][key]

            summaries.append(summary_data)

    if not summaries:
        return {}

    df = pd.DataFrame(summaries)

    # Compute aggregate statistics
    stats = {}
    for col in df.columns:
        if col != 'run_id' and pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
            }

    return {'summary_statistics': stats, 'per_run': df}


def plot_curves(aggregate_df: pd.DataFrame, output_dir: Path):
    """
    Generate training curve visualizations.
    """
    print("\n" + "="*80)
    print("GENERATING TRAINING CURVE PLOTS")
    print("="*80)

    # Check if we have epoch-wise data
    if 'epoch' not in aggregate_df.columns:
        print("  No epoch-wise data available, skipping curve plots")
        return

    epochs = aggregate_df['epoch'].values

    # 1. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train loss
    if 'train_loss_mean' in aggregate_df.columns:
        ax = axes[0]
        ax.plot(epochs, aggregate_df['train_loss_mean'], label='Mean', linewidth=2, color='#2E86AB')

        if 'train_loss_std' in aggregate_df.columns:
            ax.fill_between(
                epochs,
                aggregate_df['train_loss_mean'] - aggregate_df['train_loss_std'],
                aggregate_df['train_loss_mean'] + aggregate_df['train_loss_std'],
                alpha=0.3, color='#2E86AB', label='±1 std'
            )

        if 'train_loss_min' in aggregate_df.columns:
            ax.plot(epochs, aggregate_df['train_loss_min'], '--', alpha=0.5, color='#2E86AB', label='Min')
            ax.plot(epochs, aggregate_df['train_loss_max'], '--', alpha=0.5, color='#2E86AB', label='Max')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Val loss
    if 'val_loss_mean' in aggregate_df.columns:
        ax = axes[1]
        ax.plot(epochs, aggregate_df['val_loss_mean'], label='Mean', linewidth=2, color='#A23B72')

        if 'val_loss_std' in aggregate_df.columns:
            ax.fill_between(
                epochs,
                aggregate_df['val_loss_mean'] - aggregate_df['val_loss_std'],
                aggregate_df['val_loss_mean'] + aggregate_df['val_loss_std'],
                alpha=0.3, color='#A23B72', label='±1 std'
            )

        if 'val_loss_min' in aggregate_df.columns:
            ax.plot(epochs, aggregate_df['val_loss_min'], '--', alpha=0.5, color='#A23B72', label='Min')
            ax.plot(epochs, aggregate_df['val_loss_max'], '--', alpha=0.5, color='#A23B72', label='Max')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curve_path = output_dir / 'loss_curves.png'
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {loss_curve_path.name}")

    # 2. Accuracy curves
    if 'val_param_type_acc_mean' in aggregate_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, aggregate_df['val_param_type_acc_mean'] * 100, label='Mean', linewidth=2, color='#06A77D')

        if 'val_param_type_acc_std' in aggregate_df.columns:
            ax.fill_between(
                epochs,
                (aggregate_df['val_param_type_acc_mean'] - aggregate_df['val_param_type_acc_std']) * 100,
                (aggregate_df['val_param_type_acc_mean'] + aggregate_df['val_param_type_acc_std']) * 100,
                alpha=0.3, color='#06A77D', label='±1 std'
            )

        if 'val_param_type_acc_min' in aggregate_df.columns:
            ax.plot(epochs, aggregate_df['val_param_type_acc_min'] * 100, '--', alpha=0.5, color='#06A77D', label='Min')
            ax.plot(epochs, aggregate_df['val_param_type_acc_max'] * 100, '--', alpha=0.5, color='#06A77D', label='Max')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Validation Parameter Type Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        acc_curve_path = output_dir / 'accuracy_curves.png'
        plt.savefig(acc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {acc_curve_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract training curves from LOCAL W&B run files'
    )
    parser.add_argument(
        '--wandb-dir',
        type=str,
        default='wandb',
        help='Path to wandb directory (default: wandb)'
    )
    parser.add_argument(
        '--sweep-id',
        type=str,
        default=None,
        help='Optional sweep ID to filter runs (e.g., 27v7pl9i)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for curves and statistics'
    )

    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXTRACTING TRAINING CURVES FROM LOCAL W&B FILES")
    print("="*80)
    print(f"W&B Directory: {wandb_dir}")
    if args.sweep_id:
        print(f"Sweep ID: {args.sweep_id}")
    print(f"Output: {output_dir}")

    # Find runs
    print("\n" + "="*80)
    print("FINDING RUNS")
    print("="*80)

    run_dirs = find_sweep_runs(wandb_dir, args.sweep_id)
    print(f"Found {len(run_dirs)} runs")

    if not run_dirs:
        print("No runs found!")
        return

    # Extract histories
    print("\n" + "="*80)
    print("EXTRACTING RUN HISTORIES")
    print("="*80)

    all_runs = []
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        run_data = extract_run_history(run_dir)
        all_runs.append(run_data)

    # Create per-run summary
    print("\n" + "="*80)
    print("CREATING PER-RUN SUMMARY")
    print("="*80)

    summary_rows = []
    for run in all_runs:
        row = {'run_id': run['run_id']}

        # Add config
        for key in ['batch_size', 'hidden_dim', 'num_heads', 'num_layers', 'learning_rate', 'dropout', 'weight_decay']:
            row[key] = run['config'].get(key, None)

        # Add summary metrics
        for key in ['val/param_type_acc', 'val/loss', 'train/loss', 'epoch']:
            row[key] = run['summary'].get(key, None)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / 'per_run_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved: {summary_csv_path.name}")

    # Aggregate curves
    print("\n" + "="*80)
    print("AGGREGATING TRAINING CURVES")
    print("="*80)

    aggregate_result = aggregate_curves(all_runs)

    if isinstance(aggregate_result, pd.DataFrame):
        # Save aggregate dataframe
        aggregate_csv_path = output_dir / 'aggregate_curves.csv'
        aggregate_result.to_csv(aggregate_csv_path, index=False)
        print(f"  Saved: {aggregate_csv_path.name}")

        # Plot curves
        plot_curves(aggregate_result, output_dir)

        # Save statistics
        stats = {}
        for col in aggregate_result.columns:
            if col != 'epoch' and '_mean' in col:
                metric_name = col.replace('_mean', '')
                stats[metric_name] = {
                    'final_mean': float(aggregate_result[col].iloc[-1]) if not pd.isna(aggregate_result[col].iloc[-1]) else None,
                    'final_std': float(aggregate_result[col.replace('_mean', '_std')].iloc[-1]) if f'{col.replace("_mean", "_std")}' in aggregate_result.columns else None,
                    'best_mean': float(aggregate_result[col].min()) if 'loss' in col else float(aggregate_result[col].max()),
                }

        with open(output_dir / 'aggregate_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: aggregate_statistics.json")

    elif isinstance(aggregate_result, dict) and 'summary_statistics' in aggregate_result:
        # Only summary statistics available
        print("  Full curves not available, using summary statistics only")

        with open(output_dir / 'aggregate_statistics.json', 'w') as f:
            json.dump(aggregate_result['summary_statistics'], f, indent=2)
        print(f"  Saved: aggregate_statistics.json")

        if 'per_run' in aggregate_result:
            per_run_df = aggregate_result['per_run']
            per_run_csv = output_dir / 'per_run_final_metrics.csv'
            per_run_df.to_csv(per_run_csv, index=False)
            print(f"  Saved: {per_run_csv.name}")

    print("\n" + "="*80)
    print("✓ CURVE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
