#!/usr/bin/env python3
"""
Compare multiple checkpoints side-by-side on the same test set.

Usage:
    python scripts/compare_checkpoints.py \
        --checkpoints outputs/checkpoint1.pt outputs/checkpoint2.pt \
        --test-data outputs/processed_quick/test_sequences.npz \
        --vocab-path data/vocabulary.json \
        --output outputs/checkpoint_comparison
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import evaluation function
from evaluate_checkpoint import load_checkpoint, load_test_data, evaluate
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device


def compare_checkpoints(checkpoint_paths, test_sequences, test_sensor, vocab_path, device):
    """
    Evaluate multiple checkpoints and return comparison data.
    """
    results = []

    decomposer = TokenDecomposer(vocab_path)

    for idx, checkpoint_path in enumerate(checkpoint_paths, 1):
        print(f"\n{'='*80}")
        print(f"EVALUATING CHECKPOINT {idx}/{len(checkpoint_paths)}")
        print(f"{'='*80}")

        try:
            # Load checkpoint
            backbone, multihead_lm, config = load_checkpoint(checkpoint_path, decomposer, device)

            # Evaluate
            metrics = evaluate(
                backbone, multihead_lm, decomposer,
                test_sequences, test_sensor, device, decomposer.vocab_size
            )

            # Add to results
            results.append({
                'checkpoint': checkpoint_path.name,
                'checkpoint_path': str(checkpoint_path),
                **metrics
            })

        except Exception as e:
            print(f"❌ Failed to evaluate {checkpoint_path}: {e}")
            continue

    return results


def generate_comparison_table(results, output_dir: Path):
    """Generate comparison table."""
    df = pd.DataFrame(results)

    # Sort by overall accuracy
    df = df.sort_values('overall_acc', ascending=False)

    # Save to CSV
    csv_path = output_dir / 'checkpoint_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison table to: {csv_path}")

    # Print table
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON TABLE")
    print("=" * 80)
    print(df[['checkpoint', 'command_acc', 'param_type_acc', 'param_value_acc', 'overall_acc']].to_string(index=False))
    print("=" * 80)

    return df


def generate_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Generate comparison bar chart."""
    # Prepare data for plotting
    metrics = ['command_acc', 'param_type_acc', 'param_value_acc', 'overall_acc']
    metric_names = ['Command', 'Param Type', 'Param Value', 'Overall']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.2

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        offset = (i - 1.5) * width
        ax.bar([p + offset for p in x], df[metric], width, label=name)

    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Checkpoint Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['checkpoint'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plot_path = output_dir / 'checkpoint_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison plot to: {plot_path}")


def generate_markdown_report(df: pd.DataFrame, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Checkpoint Comparison Report\n")
    report.append(f"**Number of checkpoints evaluated**: {len(df)}\n")
    report.append("## Summary\n")

    best_checkpoint = df.iloc[0]
    report.append(f"**Best checkpoint**: `{best_checkpoint['checkpoint']}`\n")
    report.append(f"- Overall Accuracy: **{best_checkpoint['overall_acc']:.4f}** ({best_checkpoint['overall_acc']*100:.2f}%)\n")
    report.append(f"- Command Accuracy: {best_checkpoint['command_acc']:.4f}\n")
    report.append(f"- Parameter Type Accuracy: {best_checkpoint['param_type_acc']:.4f}\n")
    report.append(f"- Parameter Value Accuracy: {best_checkpoint['param_value_acc']:.4f}\n")

    report.append("\n## Detailed Comparison\n")
    report.append("| Rank | Checkpoint | Command Acc | Param Type Acc | Param Value Acc | Overall Acc |\n")
    report.append("|------|-----------|-------------|----------------|-----------------|-------------|\n")

    for idx, row in df.iterrows():
        rank = list(df.index).index(idx) + 1
        report.append(f"| {rank} | {row['checkpoint']} | {row['command_acc']:.4f} | "
                     f"{row['param_type_acc']:.4f} | {row['param_value_acc']:.4f} | "
                     f"{row['overall_acc']:.4f} |\n")

    report.append("\n## Visualization\n")
    report.append("![Checkpoint Comparison](checkpoint_comparison.png)\n")

    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"✓ Saved comparison report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to checkpoint files')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test sequences (.npz)')
    parser.add_argument('--vocab-path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='outputs/checkpoint_comparison',
                       help='Output directory')

    args = parser.parse_args()

    # Validate inputs
    checkpoint_paths = [Path(p) for p in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1

    if not Path(args.test_data).exists():
        print(f"❌ Test data not found: {args.test_data}")
        return 1

    if not Path(args.vocab_path).exists():
        print(f"❌ Vocabulary not found: {args.vocab_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 80)
        print("CHECKPOINT COMPARISON")
        print("=" * 80)
        print(f"Comparing {len(checkpoint_paths)} checkpoints")
        print()

        # Setup device
        device = get_device()
        print(f"Using device: {device}\n")

        # Load test data
        test_sequences, test_sensor = load_test_data(Path(args.test_data))

        # Compare checkpoints
        results = compare_checkpoints(checkpoint_paths, test_sequences, test_sensor, args.vocab_path, device)

        if len(results) == 0:
            print("\n❌ No checkpoints could be evaluated")
            return 1

        # Generate comparison outputs
        df = generate_comparison_table(results, output_dir)
        generate_comparison_plot(df, output_dir)
        generate_markdown_report(df, output_dir)

        print("\n✅ Comparison complete!")
        print(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
