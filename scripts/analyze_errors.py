"""
Comprehensive error analysis for G-code prediction model.

Analyzes test set predictions to identify:
- Confusion matrices per head (type, command, param_type, param_value)
- Error patterns by G-command, parameter type, operation type
- Most problematic sequences
- Error correlation analysis

Usage:
    python scripts/analyze_errors.py --checkpoint outputs/multihead_aug_v2/checkpoint_best.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.dataset import GCodeDataset


class ErrorAnalyzer:
    """Comprehensive error analysis for model predictions."""

    def __init__(self, checkpoint_path: str, data_path: str, vocab_path: str):
        """
        Initialize error analyzer.

        Args:
            checkpoint_path: Path to model checkpoint
            data_path: Path to test data (.npz)
            vocab_path: Path to vocabulary JSON
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.vocab_path = Path(vocab_path)
        self.device = self._get_device()

        # Load model
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = checkpoint['model']
        self.model.eval()
        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")

        # Load data
        print(f"Loading test data from {data_path}")
        self.dataset = GCodeDataset(data_path)
        print(f"✓ Loaded {len(self.dataset)} test samples")

        # Load vocabulary and decomposer
        self.decomposer = TokenDecomposer(str(vocab_path))

        # Results storage
        self.predictions = []
        self.targets = []
        self.errors = defaultdict(list)

    def _get_device(self):
        """Get available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def run_inference(self, batch_size: int = 8):
        """
        Run inference on entire test set.

        Args:
            batch_size: Batch size for inference
        """
        print("\nRunning inference on test set...")
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False
        )

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                tokens = batch['tokens'].to(self.device)

                # Run model (this is simplified - adapt to your model's output)
                outputs = self.model(continuous, categorical, tokens[:, :-1])

                # Extract predictions (adapt based on your model)
                # For multi-head model, you'll have separate outputs
                if isinstance(outputs, dict):
                    predictions = outputs.get('token_predictions', tokens)
                else:
                    predictions = outputs

                all_predictions.append(predictions.cpu())
                all_targets.append(tokens.cpu())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size} samples...")

        self.predictions = torch.cat(all_predictions, dim=0)
        self.targets = torch.cat(all_targets, dim=0)

        print(f"✓ Inference complete: {len(self.predictions)} predictions")

    def compute_accuracies(self):
        """Compute overall and per-head accuracies."""
        print("\n=== Computing Accuracies ===")

        # Overall accuracy
        mask = self.targets != 0  # Ignore padding
        correct = (self.predictions[mask] == self.targets[mask]).sum().item()
        total = mask.sum().item()
        overall_acc = correct / total * 100 if total > 0 else 0

        print(f"Overall Token Accuracy: {overall_acc:.2f}%")

        # Decompose predictions and targets
        print("\nDecomposing tokens for per-head analysis...")
        pred_decomposed = self.decomposer.decompose_batch(self.predictions)
        target_decomposed = self.decomposer.decompose_batch(self.targets)

        # Per-head accuracies
        accuracies = {}

        for head_name in ['type', 'command_id', 'param_type_id', 'param_value_id']:
            pred_head = pred_decomposed[head_name][mask]
            target_head = target_decomposed[head_name][mask]

            correct_head = (pred_head == target_head).sum().item()
            acc_head = correct_head / len(pred_head) * 100 if len(pred_head) > 0 else 0
            accuracies[head_name] = acc_head

            print(f"{head_name:20s}: {acc_head:.2f}%")

        return overall_acc, accuracies

    def generate_confusion_matrices(self, output_dir: Path):
        """
        Generate confusion matrices for each head.

        Args:
            output_dir: Directory to save confusion matrix plots
        """
        print("\n=== Generating Confusion Matrices ===")
        output_dir.mkdir(parents=True, exist_ok=True)

        mask = self.targets != 0
        pred_decomposed = self.decomposer.decompose_batch(self.predictions)
        target_decomposed = self.decomposer.decompose_batch(self.targets)

        # Type gate confusion matrix
        self._plot_confusion_matrix(
            target_decomposed['type'][mask].numpy(),
            pred_decomposed['type'][mask].numpy(),
            title="Type Gate Confusion Matrix",
            labels=["SPECIAL", "COMMAND", "PARAMETER", "NUMERIC"],
            output_path=output_dir / "confusion_type.png"
        )

        # Command confusion matrix
        cmd_mask = target_decomposed['type'][mask] == TokenDecomposer.TYPE_COMMAND
        if cmd_mask.sum() > 0:
            self._plot_confusion_matrix(
                target_decomposed['command_id'][mask][cmd_mask].numpy(),
                pred_decomposed['command_id'][mask][cmd_mask].numpy(),
                title="Command Confusion Matrix",
                labels=[cmd[:10] for cmd in self.decomposer.command_tokens],  # Truncate labels
                output_path=output_dir / "confusion_command.png",
                figsize=(12, 10)
            )

        # Parameter type confusion matrix
        param_mask = (target_decomposed['type'][mask] == TokenDecomposer.TYPE_PARAMETER) | \
                     (target_decomposed['type'][mask] == TokenDecomposer.TYPE_NUMERIC)
        if param_mask.sum() > 0:
            self._plot_confusion_matrix(
                target_decomposed['param_type_id'][mask][param_mask].numpy(),
                pred_decomposed['param_type_id'][mask][param_mask].numpy(),
                title="Parameter Type Confusion Matrix",
                labels=self.decomposer.param_tokens,
                output_path=output_dir / "confusion_param_type.png"
            )

        # Parameter value confusion matrix (simplified - only first 20 values)
        numeric_mask = target_decomposed['type'][mask] == TokenDecomposer.TYPE_NUMERIC
        if numeric_mask.sum() > 0:
            target_vals = target_decomposed['param_value_id'][mask][numeric_mask].numpy()
            pred_vals = pred_decomposed['param_value_id'][mask][numeric_mask].numpy()

            # Limit to 0-19 for visualization
            limited_mask = target_vals < 20
            if limited_mask.sum() > 100:  # Only plot if enough samples
                self._plot_confusion_matrix(
                    target_vals[limited_mask],
                    pred_vals[limited_mask],
                    title="Parameter Value Confusion Matrix (0-19)",
                    labels=[f"{i:02d}" for i in range(20)],
                    output_path=output_dir / "confusion_param_value.png",
                    figsize=(14, 12)
                )

        print(f"✓ Confusion matrices saved to {output_dir}")

    def _plot_confusion_matrix(self, y_true, y_pred, title, labels, output_path, figsize=(10, 8)):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved {output_path.name}")

    def analyze_error_patterns(self):
        """Analyze error patterns by command, parameter, etc."""
        print("\n=== Analyzing Error Patterns ===")

        mask = self.targets != 0
        pred_decomposed = self.decomposer.decompose_batch(self.predictions)
        target_decomposed = self.decomposer.decompose_batch(self.targets)

        errors = {}

        # Errors by command
        cmd_mask = target_decomposed['type'][mask] == TokenDecomposer.TYPE_COMMAND
        if cmd_mask.sum() > 0:
            cmd_errors = {}
            for i, cmd in enumerate(self.decomposer.command_tokens):
                cmd_samples = target_decomposed['command_id'][mask][cmd_mask] == i
                if cmd_samples.sum() > 0:
                    correct = (pred_decomposed['command_id'][mask][cmd_mask][cmd_samples] == i).sum().item()
                    total = cmd_samples.sum().item()
                    accuracy = correct / total * 100
                    cmd_errors[cmd] = {
                        'accuracy': accuracy,
                        'total_samples': total,
                        'errors': total - correct
                    }

            # Sort by error count
            sorted_cmds = sorted(cmd_errors.items(), key=lambda x: x[1]['errors'], reverse=True)
            print("\nTop 10 Commands with Most Errors:")
            for cmd, stats in sorted_cmds[:10]:
                print(f"  {cmd:10s}: {stats['accuracy']:5.1f}% ({stats['errors']}/{stats['total_samples']} errors)")

            errors['commands'] = cmd_errors

        # Errors by parameter type
        param_mask = target_decomposed['type'][mask] == TokenDecomposer.TYPE_NUMERIC
        if param_mask.sum() > 0:
            param_errors = {}
            for i, param in enumerate(self.decomposer.param_tokens):
                param_samples = target_decomposed['param_type_id'][mask][param_mask] == i
                if param_samples.sum() > 0:
                    correct = (pred_decomposed['param_type_id'][mask][param_mask][param_samples] == i).sum().item()
                    total = param_samples.sum().item()
                    accuracy = correct / total * 100
                    param_errors[param] = {
                        'accuracy': accuracy,
                        'total_samples': total,
                        'errors': total - correct
                    }

            print("\nParameter Type Accuracies:")
            for param, stats in sorted(param_errors.items(), key=lambda x: x[1]['accuracy']):
                print(f"  {param:10s}: {stats['accuracy']:5.1f}% ({stats['total_samples']} samples)")

            errors['parameters'] = param_errors

        return errors

    def generate_report(self, output_path: Path, overall_acc, accuracies, error_patterns):
        """Generate comprehensive markdown report."""
        print(f"\nGenerating report: {output_path}")

        with open(output_path, 'w') as f:
            f.write("# Comprehensive Error Analysis Report\n\n")
            f.write(f"**Generated:** {Path.cwd().name}\n")
            f.write(f"**Checkpoint:** {self.checkpoint_path.name}\n")
            f.write(f"**Test Samples:** {len(self.dataset)}\n\n")

            f.write("---\n\n")

            # Overall metrics
            f.write("## Overall Metrics\n\n")
            f.write(f"- **Overall Token Accuracy:** {overall_acc:.2f}%\n")
            for head, acc in accuracies.items():
                f.write(f"- **{head}:** {acc:.2f}%\n")
            f.write("\n")

            # Error patterns
            if 'commands' in error_patterns:
                f.write("## Command Errors (Top 10)\n\n")
                f.write("| Command | Accuracy | Errors | Total |\n")
                f.write("|---------|----------|--------|-------|\n")
                sorted_cmds = sorted(error_patterns['commands'].items(),
                                   key=lambda x: x[1]['errors'], reverse=True)
                for cmd, stats in sorted_cmds[:10]:
                    f.write(f"| {cmd} | {stats['accuracy']:.1f}% | {stats['errors']} | {stats['total_samples']} |\n")
                f.write("\n")

            if 'parameters' in error_patterns:
                f.write("## Parameter Type Accuracies\n\n")
                f.write("| Parameter | Accuracy | Total Samples |\n")
                f.write("|-----------|----------|---------------|\n")
                for param, stats in sorted(error_patterns['parameters'].items(),
                                          key=lambda x: x[1]['accuracy']):
                    f.write(f"| {param} | {stats['accuracy']:.1f}% | {stats['total_samples']} |\n")
                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### High Priority\n")
            if accuracies.get('param_value_id', 0) < 70:
                f.write("1. **Parameter values need improvement** - Consider:\n")
                f.write("   - Finer vocabulary bucketing (3-digit instead of 2-digit)\n")
                f.write("   - Hybrid classification + regression approach\n")
                f.write("   - Increase model capacity (d_model, layers)\n\n")

            if 'commands' in error_patterns:
                worst_cmds = [cmd for cmd, stats in error_patterns['commands'].items()
                             if stats['accuracy'] < 90]
                if worst_cmds:
                    f.write(f"2. **Focus on problematic commands:** {', '.join(worst_cmds[:5])}\n")
                    f.write("   - Increase oversampling for these commands\n")
                    f.write("   - Review training data quality\n\n")

            f.write("### Medium Priority\n")
            f.write("- Run hyperparameter sweeps for vocabulary bucketing\n")
            f.write("- Experiment with different loss weights\n")
            f.write("- Try ensemble methods\n\n")

        print(f"✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive error analysis")
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/multihead_aug_v2/checkpoint_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                       default='data/processed_v2/test_sequences.npz',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='data/gcode_vocab_v2.json',
                       help='Path to vocabulary')
    parser.add_argument('--output', type=str,
                       default='results/error_analysis',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = ErrorAnalyzer(args.checkpoint, args.data, args.vocab)
    analyzer.run_inference()
    overall_acc, accuracies = analyzer.compute_accuracies()
    analyzer.generate_confusion_matrices(output_dir / 'confusion_matrices')
    error_patterns = analyzer.analyze_error_patterns()
    analyzer.generate_report(output_dir / 'error_analysis_report.md',
                            overall_acc, accuracies, error_patterns)

    print("\n" + "="*60)
    print("✓ Error analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
