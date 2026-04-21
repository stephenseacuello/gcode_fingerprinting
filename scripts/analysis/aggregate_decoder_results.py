#!/usr/bin/env python3
"""
Aggregate decoder sweep results across configs and folds.

Usage:
    python scripts/analysis/aggregate_decoder_results.py \
        --results_dir outputs/decoder20260303 \
        --configs feat110 feat98 feat56
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def collect_metrics(results_dir, configs, n_folds=5):
    """Collect metrics.json from each config/fold combination."""
    all_results = {}

    for config in configs:
        config_results = []
        for fold in range(1, n_folds + 1):
            metrics_path = results_dir / config / f"fold_{fold}" / "results" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                config_results.append({
                    'fold': fold,
                    'best_epoch': data.get('best_epoch'),
                    'test': data.get('test_metrics', {}),
                    'val': data.get('val_metrics', {}),
                })
            else:
                print(f"  WARNING: Missing {metrics_path}")
        all_results[config] = config_results

    return all_results


def compute_summary(all_results):
    """Compute mean +/- std for each metric across folds."""
    summary = {}

    for config, folds in all_results.items():
        if not folds:
            continue

        # Collect all metric keys from test results
        metric_keys = set()
        for f in folds:
            metric_keys.update(f['test'].keys())

        config_summary = {
            'n_folds': len(folds),
            'best_epochs': [f['best_epoch'] for f in folds],
        }

        for split in ['test', 'val']:
            split_summary = {}
            for key in sorted(metric_keys):
                values = [f[split].get(key) for f in folds if f[split].get(key) is not None]
                if values:
                    split_summary[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values,
                    }
            config_summary[split] = split_summary

        summary[config] = config_summary

    return summary


def format_markdown_table(summary, configs):
    """Generate a markdown comparison table."""
    lines = []
    lines.append("# Decoder Sweep Results\n")

    # Feature count labels
    feat_labels = {
        'feat110': '110 features',
        'feat98': '98 features',
        'feat56': '56 features',
    }

    # Test metrics table
    lines.append("## Test Set Performance (mean +/- std across folds)\n")

    metrics_order = [
        ('token_accuracy', 'Token Acc'),
        ('sequence_accuracy', 'Seq Acc'),
        ('type_accuracy', 'Type Acc'),
        ('command_accuracy', 'Cmd Acc'),
        ('param_type_accuracy', 'Param Acc'),
        ('loss', 'Loss'),
    ]

    # Header
    header = "| Metric |"
    sep = "|--------|"
    for config in configs:
        label = feat_labels.get(config, config)
        header += f" {label} |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for key, label in metrics_order:
        row = f"| {label} |"
        for config in configs:
            if config in summary and 'test' in summary[config]:
                m = summary[config]['test'].get(key)
                if m:
                    if key == 'loss':
                        row += f" {m['mean']:.3f} +/- {m['std']:.3f} |"
                    else:
                        row += f" {m['mean']:.1%} +/- {m['std']:.1%} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        lines.append(row)

    # Best epochs
    row = "| Best Epoch |"
    for config in configs:
        if config in summary:
            epochs = summary[config].get('best_epochs', [])
            if epochs:
                row += f" {np.mean(epochs):.0f} +/- {np.std(epochs):.0f} |"
            else:
                row += " N/A |"
        else:
            row += " N/A |"
    lines.append(row)

    row = "| Folds |"
    for config in configs:
        if config in summary:
            row += f" {summary[config]['n_folds']}/5 |"
        else:
            row += " 0/5 |"
    lines.append(row)

    # Val metrics table
    lines.append("\n## Validation Set Performance\n")
    header = "| Metric |"
    sep = "|--------|"
    for config in configs:
        label = feat_labels.get(config, config)
        header += f" {label} |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for key, label in metrics_order:
        row = f"| {label} |"
        for config in configs:
            if config in summary and 'val' in summary[config]:
                m = summary[config]['val'].get(key)
                if m:
                    if key == 'loss':
                        row += f" {m['mean']:.3f} +/- {m['std']:.3f} |"
                    else:
                        row += f" {m['mean']:.1%} +/- {m['std']:.1%} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        lines.append(row)

    # Per-fold detail
    lines.append("\n## Per-Fold Token Accuracy (Test)\n")
    header = "| Fold |"
    sep = "|------|"
    for config in configs:
        label = feat_labels.get(config, config)
        header += f" {label} |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for fold in range(5):
        row = f"| {fold + 1} |"
        for config in configs:
            if config in summary and 'test' in summary[config]:
                vals = summary[config]['test'].get('token_accuracy', {}).get('values', [])
                if fold < len(vals):
                    row += f" {vals[fold]:.1%} |"
                else:
                    row += " - |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate decoder sweep results")
    parser.add_argument("--results_dir", type=str, default="outputs/decoder20260303")
    parser.add_argument("--configs", nargs="+", default=["feat110", "feat98", "feat56"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print(f"Collecting results from {results_dir}")
    all_results = collect_metrics(results_dir, args.configs)

    for config, folds in all_results.items():
        print(f"  {config}: {len(folds)} folds collected")

    summary = compute_summary(all_results)

    # Save summary JSON
    summary_json_path = results_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_json_path}")

    # Save summary markdown
    md = format_markdown_table(summary, args.configs)
    summary_md_path = results_dir / "summary.md"
    summary_md_path.write_text(md)
    print(f"Saved: {summary_md_path}")

    # Print to console
    print(f"\n{md}")


if __name__ == "__main__":
    main()
