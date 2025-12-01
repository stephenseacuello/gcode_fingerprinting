#!/usr/bin/env python3
"""
Extract metrics from all local W&B runs.

This script reads all local wandb run directories and extracts metrics
without needing to access the W&B cloud.
"""
import json
import pandas as pd
from pathlib import Path
import wandb
from typing import Dict, List
import sys

def extract_run_metrics(run_dir: Path) -> Dict:
    """Extract metrics from a single wandb run directory."""
    try:
        # Read run files
        files_dir = run_dir / 'files'

        # Try to find wandb-summary.json (contains best metrics)
        summary_file = files_dir / 'wandb-summary.json'
        config_file = files_dir / 'config.yaml'

        metrics = {
            'run_id': run_dir.name.split('-')[-1],
            'run_name': run_dir.name,
            'run_path': str(run_dir.relative_to(run_dir.parent.parent)),
        }

        # Load summary (final/best metrics)
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

                # Extract key metrics
                metrics['val_g_command_acc'] = summary.get('val/g_command_acc', 0.0)
                metrics['val_overall_acc'] = summary.get('val/overall_acc', 0.0)
                metrics['val_m_command_acc'] = summary.get('val/m_command_acc', 0.0)
                metrics['val_numeric_acc'] = summary.get('val/numeric_acc', 0.0)
                metrics['val_loss'] = summary.get('val/loss', float('inf'))

                metrics['train_g_command_acc'] = summary.get('train/g_command_acc', 0.0)
                metrics['train_overall_acc'] = summary.get('train/overall_acc', 0.0)
                metrics['train_loss'] = summary.get('train/loss', float('inf'))

                metrics['epoch'] = summary.get('epoch', 0)
                metrics['_step'] = summary.get('_step', 0)

        # Load config
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

                # Extract hyperparameters (handle both dict and nested dict structures)
                if isinstance(config, dict):
                    # Get values, handling both direct values and {'value': x} format
                    def get_value(key, default=None):
                        val = config.get(key, default)
                        if isinstance(val, dict) and 'value' in val:
                            return val['value']
                        return val

                    metrics['batch_size'] = get_value('batch_size')
                    metrics['hidden_dim'] = get_value('hidden_dim')
                    metrics['num_layers'] = get_value('num_layers')
                    metrics['num_heads'] = get_value('num_heads')
                    metrics['learning_rate'] = get_value('learning_rate')
                    metrics['weight_decay'] = get_value('weight_decay')
                    metrics['class_weight_alpha'] = get_value('class_weight_alpha')
                    metrics['use_focal_loss'] = get_value('use_focal_loss')
                    metrics['label_smoothing'] = get_value('label_smoothing')
                    metrics['optimizer'] = get_value('optimizer')
                    metrics['scheduler'] = get_value('scheduler')

        return metrics

    except Exception as e:
        print(f"  ⚠️  Error processing {run_dir.name}: {e}")
        return None

def main():
    """Main function to extract metrics from all wandb runs."""
    base_dir = Path(__file__).parent
    wandb_dir = base_dir / 'wandb'

    print("=" * 80)
    print("EXTRACTING METRICS FROM LOCAL W&B RUNS")
    print("=" * 80)

    # Find all wandb run directories
    print(f"\n1. Finding all W&B runs in {wandb_dir}...")
    run_dirs = [d for d in wandb_dir.iterdir()
                if d.is_dir() and d.name.startswith('run-')]
    print(f"   Found {len(run_dirs)} W&B run directories")

    # Extract metrics from each run
    print("\n2. Extracting metrics from each run...")
    all_metrics = []

    for i, run_dir in enumerate(run_dirs, 1):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(run_dirs)} runs processed...")

        metrics = extract_run_metrics(run_dir)
        if metrics:
            all_metrics.append(metrics)

    print(f"   Successfully extracted metrics from {len(all_metrics)} runs")

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Sort by validation G-command accuracy
    df = df.sort_values('val_g_command_acc', ascending=False)

    # Save to CSV
    output_path = base_dir / 'outputs' / 'wandb_sweep_analysis.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n3. ✅ Saved analysis to: {output_path}")

    # Print top 10
    print("\n" + "=" * 80)
    print("TOP 10 MODELS BY VALIDATION G-COMMAND ACCURACY")
    print("=" * 80)

    top_10 = df.head(10)
    for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
        print(f"\n{rank}. Run: {row['run_id']}")
        print(f"   Val G-Acc: {row['val_g_command_acc']*100:.2f}% | "
              f"Val Overall: {row['val_overall_acc']*100:.2f}% | "
              f"Val Loss: {row['val_loss']:.4f}")

        if pd.notna(row.get('hidden_dim')):
            print(f"   Config: hidden_dim={int(row['hidden_dim'])}, "
                  f"batch_size={int(row.get('batch_size', 0))}, "
                  f"lr={row.get('learning_rate', 0):.6f}")
            print(f"   Layers: {int(row.get('num_layers', 0))}, "
                  f"Heads: {int(row.get('num_heads', 0))}, "
                  f"Optimizer: {row.get('optimizer', 'N/A')}, "
                  f"Scheduler: {row.get('scheduler', 'N/A')}")

    # Print statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total runs: {len(df)}")
    print(f"Runs with validation data: {(df['val_g_command_acc'] > 0).sum()}")

    print(f"\nG-Command Accuracy (Validation):")
    print(f"  Best: {df['val_g_command_acc'].max()*100:.2f}%")
    print(f"  Mean: {df['val_g_command_acc'].mean()*100:.2f}%")
    print(f"  Median: {df['val_g_command_acc'].median()*100:.2f}%")
    print(f"  Std: {df['val_g_command_acc'].std()*100:.2f}%")

    print(f"\nOverall Accuracy (Validation):")
    print(f"  Best: {df['val_overall_acc'].max()*100:.2f}%")
    print(f"  Mean: {df['val_overall_acc'].mean()*100:.2f}%")
    print(f"  Median: {df['val_overall_acc'].median()*100:.2f}%")

    # Save top 5 configs with proper structure
    print("\n" + "=" * 80)
    print("SAVING TOP 5 CONFIGURATIONS")
    print("=" * 80)

    configs_dir = base_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)

    for rank, (idx, row) in enumerate(top_10.head(5).iterrows(), 1):
        # Create config dict
        config = {
            'hidden_dim': int(row.get('hidden_dim', 256)),
            'num_layers': int(row.get('num_layers', 3)),
            'num_heads': int(row.get('num_heads', 4)),
            'batch_size': int(row.get('batch_size', 16)),
            'learning_rate': float(row.get('learning_rate', 0.001)),
            'weight_decay': float(row.get('weight_decay', 0.0)),
            'class_weight_alpha': float(row.get('class_weight_alpha', 2.0)),
            'use_focal_loss': bool(row.get('use_focal_loss', False)),
            'label_smoothing': float(row.get('label_smoothing', 0.0)),
            'optimizer': str(row.get('optimizer', 'adamw')),
            'scheduler': str(row.get('scheduler', 'cosine')),

            '_metadata': {
                'rank': rank,
                'run_id': row['run_id'],
                'val_g_command_acc': float(row['val_g_command_acc']),
                'val_overall_acc': float(row['val_overall_acc']),
                'val_loss': float(row['val_loss']),
                'epoch': int(row.get('epoch', 0)),
            }
        }

        # Save config
        output_file = configs_dir / f'phase1_rank{rank}.json'
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✅ Rank {rank}: {output_file.name}")
        print(f"      Val G-Acc: {row['val_g_command_acc']*100:.2f}%")

    # Save best config
    if len(top_10) > 0:
        best_row = top_10.iloc[0]
        best_config = {
            'hidden_dim': int(best_row.get('hidden_dim', 256)),
            'num_layers': int(best_row.get('num_layers', 3)),
            'num_heads': int(best_row.get('num_heads', 4)),
            'batch_size': int(best_row.get('batch_size', 16)),
            'learning_rate': float(best_row.get('learning_rate', 0.001)),
            'weight_decay': float(best_row.get('weight_decay', 0.0)),
            'class_weight_alpha': float(best_row.get('class_weight_alpha', 2.0)),
            'use_focal_loss': bool(best_row.get('use_focal_loss', False)),
            'label_smoothing': float(best_row.get('label_smoothing', 0.0)),
            'optimizer': str(best_row.get('optimizer', 'adamw')),
            'scheduler': str(best_row.get('scheduler', 'cosine')),

            '_metadata': {
                'rank': 1,
                'run_id': best_row['run_id'],
                'val_g_command_acc': float(best_row['val_g_command_acc']),
                'val_overall_acc': float(best_row['val_overall_acc']),
                'val_loss': float(best_row['val_loss']),
                'epoch': int(best_row.get('epoch', 0)),
            }
        }

        best_output = configs_dir / 'phase1_best.json'
        with open(best_output, 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"\n  ✅ Best model config: {best_output.name}")

    print("\n" + "=" * 80)
    print("✅ WANDB METRICS EXTRACTION COMPLETE!")
    print("=" * 80)

    return df

if __name__ == '__main__':
    df = main()
