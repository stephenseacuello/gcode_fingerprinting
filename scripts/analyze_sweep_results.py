#!/usr/bin/env python3
"""
Analyze all sweep results and extract proper metrics from history.json files.

This script:
1. Finds all training runs in outputs/
2. Loads history.json from each run
3. Extracts best validation metrics
4. Creates a corrected sweep_analysis.csv
5. Identifies top 5 models
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def load_history(history_path: Path) -> Optional[Dict]:
    """Load history.json file."""
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load {history_path}: {e}")
        return None

def extract_best_metrics(history: Dict) -> Dict:
    """Extract best metrics from history."""
    metrics = {
        'best_epoch': -1,
        'best_val_loss': float('inf'),
        'best_val_g_acc': 0.0,
        'best_val_overall_acc': 0.0,
        'final_train_loss': float('inf'),
        'final_train_g_acc': 0.0,
        'final_train_overall_acc': 0.0,
        'total_epochs': 0,
    }

    # Get validation metrics
    if 'val' in history and len(history['val']) > 0:
        val_epochs = history['val']
        metrics['total_epochs'] = len(val_epochs)

        # Find best epoch by G-command accuracy
        best_idx = 0
        best_g_acc = 0.0

        for idx, epoch_data in enumerate(val_epochs):
            g_acc = epoch_data.get('g_command_acc', 0.0)
            if g_acc > best_g_acc:
                best_g_acc = g_acc
                best_idx = idx

        # Extract best metrics
        best_epoch = val_epochs[best_idx]
        metrics['best_epoch'] = best_epoch.get('epoch', best_idx)
        metrics['best_val_loss'] = best_epoch.get('loss', float('inf'))
        metrics['best_val_g_acc'] = best_epoch.get('g_command_acc', 0.0)
        metrics['best_val_overall_acc'] = best_epoch.get('overall_acc', 0.0)

    # Get final training metrics
    if 'train' in history and len(history['train']) > 0:
        final_train = history['train'][-1]
        metrics['final_train_loss'] = final_train.get('loss', float('inf'))
        metrics['final_train_g_acc'] = final_train.get('g_command_acc', 0.0)
        metrics['final_train_overall_acc'] = final_train.get('overall_acc', 0.0)

    return metrics

def find_all_runs(base_dir: Path) -> List[Path]:
    """Find all training runs with history.json files."""
    runs = []

    # Search in outputs/training and outputs/wandb_sweeps
    for subdir in ['training', 'wandb_sweeps']:
        search_dir = base_dir / 'outputs' / subdir
        if not search_dir.exists():
            continue

        # Find all history.json files
        for history_file in search_dir.rglob('history.json'):
            runs.append(history_file.parent)

    return runs

def analyze_all_runs():
    """Analyze all training runs and create comprehensive report."""
    base_dir = Path(__file__).parent
    print("=" * 80)
    print("ANALYZING ALL TRAINING RUNS")
    print("=" * 80)

    # Find all runs
    print("\n1. Finding all training runs...")
    runs = find_all_runs(base_dir)
    print(f"   Found {len(runs)} runs with history.json")

    # Analyze each run
    print("\n2. Extracting metrics from each run...")
    results = []

    for i, run_dir in enumerate(runs, 1):
        history_path = run_dir / 'history.json'
        run_name = run_dir.name

        print(f"   [{i}/{len(runs)}] {run_name}")

        # Load history
        history = load_history(history_path)
        if history is None:
            continue

        # Extract metrics
        metrics = extract_best_metrics(history)

        # Add run info
        result = {
            'run_name': run_name,
            'run_dir': str(run_dir.relative_to(base_dir)),
            **metrics
        }

        # Try to load config if available
        config_path = run_dir / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    result['d_model'] = config.get('d_model', None)
                    result['lstm_layers'] = config.get('lstm_layers', None)
                    result['batch_size'] = config.get('batch_size', None)
                    result['learning_rate'] = config.get('learning_rate', None)
            except:
                pass

        results.append(result)

        # Print quick summary
        if metrics['best_val_g_acc'] > 0:
            print(f"       Best Val G-Acc: {metrics['best_val_g_acc']*100:.2f}% "
                  f"(epoch {metrics['best_epoch']}, loss: {metrics['best_val_loss']:.4f})")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by best validation G-command accuracy
    df = df.sort_values('best_val_g_acc', ascending=False)

    # Save to CSV
    output_path = base_dir / 'outputs' / 'sweep_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"\n3. ✅ Saved analysis to: {output_path}")

    # Print top 10 models
    print("\n" + "=" * 80)
    print("TOP 10 MODELS BY VALIDATION G-COMMAND ACCURACY")
    print("=" * 80)

    top_10 = df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{row.name + 1}. {row['run_name']}")
        print(f"   Val G-Acc: {row['best_val_g_acc']*100:.2f}% | "
              f"Val Overall: {row['best_val_overall_acc']*100:.2f}% | "
              f"Val Loss: {row['best_val_loss']:.4f}")
        print(f"   Epoch: {row['best_epoch']} / {row['total_epochs']}")
        if pd.notna(row.get('d_model')):
            print(f"   d_model: {int(row['d_model'])}, "
                  f"batch_size: {int(row.get('batch_size', 0))}, "
                  f"lr: {row.get('learning_rate', 0):.6f}")
        print(f"   Path: {row['run_dir']}")

    # Statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total runs analyzed: {len(df)}")
    print(f"Runs with validation data: {(df['best_val_g_acc'] > 0).sum()}")
    print(f"\nG-Command Accuracy (Validation):")
    print(f"  Best: {df['best_val_g_acc'].max()*100:.2f}%")
    print(f"  Mean: {df['best_val_g_acc'].mean()*100:.2f}%")
    print(f"  Median: {df['best_val_g_acc'].median()*100:.2f}%")
    print(f"  Std: {df['best_val_g_acc'].std()*100:.2f}%")

    print(f"\nOverall Accuracy (Validation):")
    print(f"  Best: {df['best_val_overall_acc'].max()*100:.2f}%")
    print(f"  Mean: {df['best_val_overall_acc'].mean()*100:.2f}%")
    print(f"  Median: {df['best_val_overall_acc'].median()*100:.2f}%")

    print(f"\nValidation Loss:")
    valid_losses = df[df['best_val_loss'] != float('inf')]['best_val_loss']
    if len(valid_losses) > 0:
        print(f"  Best: {valid_losses.min():.4f}")
        print(f"  Mean: {valid_losses.mean():.4f}")
        print(f"  Median: {valid_losses.median():.4f}")

    # Save top 5 configs
    print("\n" + "=" * 80)
    print("SAVING TOP 5 CONFIGURATIONS")
    print("=" * 80)

    configs_dir = base_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)

    for rank, (idx, row) in enumerate(top_10.head(5).iterrows(), 1):
        run_dir = base_dir / row['run_dir']
        config_path = run_dir / 'config.json'

        if config_path.exists():
            # Load and save with rank
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Add metadata
            config['_metadata'] = {
                'rank': rank,
                'run_name': row['run_name'],
                'best_val_g_acc': float(row['best_val_g_acc']),
                'best_val_overall_acc': float(row['best_val_overall_acc']),
                'best_val_loss': float(row['best_val_loss']),
                'best_epoch': int(row['best_epoch']),
            }

            output_config = configs_dir / f'phase1_rank{rank}.json'
            with open(output_config, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"  ✅ Rank {rank}: {output_config.name}")
            print(f"      Val G-Acc: {row['best_val_g_acc']*100:.2f}%")

    # Also save the best as phase1_best.json
    if len(top_10) > 0:
        best_row = top_10.iloc[0]
        best_run_dir = base_dir / best_row['run_dir']
        best_config_path = best_run_dir / 'config.json'

        if best_config_path.exists():
            with open(best_config_path, 'r') as f:
                best_config = json.load(f)

            best_config['_metadata'] = {
                'rank': 1,
                'run_name': best_row['run_name'],
                'best_val_g_acc': float(best_row['best_val_g_acc']),
                'best_val_overall_acc': float(best_row['best_val_overall_acc']),
                'best_val_loss': float(best_row['best_val_loss']),
                'best_epoch': int(best_row['best_epoch']),
            }

            best_output = configs_dir / 'phase1_best.json'
            with open(best_output, 'w') as f:
                json.dump(best_config, f, indent=2)

            print(f"\n  ✅ Best model config: {best_output.name}")

    return df

if __name__ == '__main__':
    df = analyze_all_runs()
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review outputs/sweep_analysis.csv for detailed metrics")
    print("  2. Check configs/phase1_best.json for best hyperparameters")
    print("  3. Use top configs for baseline comparison and further experiments")
