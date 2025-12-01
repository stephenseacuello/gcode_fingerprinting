#!/usr/bin/env python3
"""
Enhanced Performance Analysis Script for G-code Fingerprinting Model
Provides detailed insights into model performance, error patterns, and improvement opportunities
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device
from torch.utils.data import DataLoader
from tqdm import tqdm

# Try importing wandb for fetching training history
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Training history features will be limited.")


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for G-code fingerprinting model."""

    def __init__(self, checkpoint_path: Path, data_dir: Path, vocab_path: Path, output_dir: Path = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.vocab_path = Path(vocab_path)
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Load vocabulary and decomposer
        self.decomposer = TokenDecomposer(vocab_path)
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            self.vocab = vocab_data['vocab'] if isinstance(vocab_data, dict) and 'vocab' in vocab_data else vocab_data

        # Load model
        self.backbone, self.model = self._load_model()

        # Initialize analysis results
        self.results = defaultdict(dict)

    def _load_model(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load model from checkpoint with dynamic dimension detection."""
        print(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract config
        config = checkpoint.get('config', {
            'hidden_dim': 256,
            'num_layers': 5,
            'num_heads': 8
        })

        # Infer sensor dimensions
        sensor_dims = [219, 4]  # Default
        if 'backbone_state_dict' in checkpoint:
            backbone_state = checkpoint['backbone_state_dict']
            if 'encoders.0.proj.0.weight' in backbone_state:
                sensor_dims[0] = backbone_state['encoders.0.proj.0.weight'].shape[1]
            if 'encoders.1.proj.0.weight' in backbone_state:
                sensor_dims[1] = backbone_state['encoders.1.proj.0.weight'].shape[1]

        print(f"Detected sensor dimensions: {sensor_dims}")

        # Create models
        backbone_config = ModelConfig(
            sensor_dims=sensor_dims,
            d_model=config['hidden_dim'],
            lstm_layers=config['num_layers'],
            gcode_vocab=len(self.vocab),
            n_heads=config['num_heads'],
            dropout=0.1
        )
        backbone = MM_DTAE_LSTM(backbone_config).to(self.device)

        model = MultiHeadGCodeLM(
            d_model=config['hidden_dim'],
            n_commands=self.decomposer.n_commands,
            n_param_types=self.decomposer.n_param_types,
            n_param_values=self.decomposer.n_param_values,
            nhead=config['num_heads'],
            num_layers=config['num_layers'],
            vocab_size=len(self.vocab)
        ).to(self.device)

        # Load weights
        if 'backbone_state_dict' in checkpoint:
            backbone.load_state_dict(checkpoint['backbone_state_dict'])
        if 'multihead_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['multihead_state_dict'])

        backbone.eval()
        model.eval()

        return backbone, model

    def analyze_dataset(self, dataloader: DataLoader) -> Dict:
        """Analyze dataset statistics and model predictions."""
        print("\n📊 Analyzing dataset and predictions...")

        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        all_confidences = defaultdict(list)
        token_errors = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing"):
                # Get inputs
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device).float()
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                # Forward pass
                tgt_in = tokens[:, :-1]
                tgt_out = tokens[:, 1:]

                mods = [continuous, categorical]
                backbone_out = self.backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
                memory = backbone_out['memory']

                logits = self.model(memory, tgt_in)

                # Decompose targets
                tgt_decomposed = self.decomposer.decompose_batch(tgt_out)

                # Get predictions
                B, T = tgt_out.shape
                valid_mask = tgt_decomposed['type'] != 0

                # Command predictions
                if 'command_logits' in logits:
                    cmd_pred = torch.argmax(logits['command_logits'], dim=-1)
                    cmd_conf = torch.softmax(logits['command_logits'], dim=-1).max(dim=-1)[0]
                    cmd_mask = (tgt_decomposed['type'] == 1) & valid_mask

                    if cmd_mask.any():
                        all_predictions['command'].extend(cmd_pred[cmd_mask].cpu().numpy())
                        all_targets['command'].extend(tgt_decomposed['command_id'][cmd_mask].cpu().numpy())
                        all_confidences['command'].extend(cmd_conf[cmd_mask].cpu().numpy())

                # Parameter type predictions
                if 'param_type_logits' in logits:
                    pt_pred = torch.argmax(logits['param_type_logits'], dim=-1)
                    pt_conf = torch.softmax(logits['param_type_logits'], dim=-1).max(dim=-1)[0]
                    pt_mask = ((tgt_decomposed['type'] == 2) | (tgt_decomposed['type'] == 3)) & valid_mask

                    if pt_mask.any():
                        all_predictions['param_type'].extend(pt_pred[pt_mask].cpu().numpy())
                        all_targets['param_type'].extend(tgt_decomposed['param_type_id'][pt_mask].cpu().numpy())
                        all_confidences['param_type'].extend(pt_conf[pt_mask].cpu().numpy())

                # Parameter value predictions
                if 'param_value_logits' in logits:
                    pv_pred = torch.argmax(logits['param_value_logits'], dim=-1)
                    pv_conf = torch.softmax(logits['param_value_logits'], dim=-1).max(dim=-1)[0]
                    pv_mask = (tgt_decomposed['type'] == 3) & valid_mask

                    if pv_mask.any():
                        all_predictions['param_value'].extend(pv_pred[pv_mask].cpu().numpy())
                        all_targets['param_value'].extend(tgt_decomposed['param_value_id'][pv_mask].cpu().numpy())
                        all_confidences['param_value'].extend(pv_conf[pv_mask].cpu().numpy())

                # Track token-level errors
                for i in range(B):
                    for j in range(T):
                        if valid_mask[i, j]:
                            token_id = tgt_out[i, j].item()
                            token_str = self.vocab.get(str(token_id), f'UNK_{token_id}')

                            # Check if prediction is correct
                            reconstructed = self._reconstruct_token(
                                tgt_decomposed['type'][i, j].item(),
                                cmd_pred[i, j].item() if 'command_logits' in logits else 0,
                                pt_pred[i, j].item() if 'param_type_logits' in logits else 0,
                                pv_pred[i, j].item() if 'param_value_logits' in logits else 0
                            )

                            token_errors[token_str]['total'] += 1
                            if reconstructed == token_id:
                                token_errors[token_str]['correct'] += 1
                            else:
                                pred_str = self.vocab.get(str(reconstructed), f'UNK_{reconstructed}')
                                token_errors[token_str]['errors'].append(pred_str)

        return {
            'predictions': dict(all_predictions),
            'targets': dict(all_targets),
            'confidences': dict(all_confidences),
            'token_errors': dict(token_errors)
        }

    def _reconstruct_token(self, token_type: int, cmd: int, pt: int, pv: int) -> int:
        """Reconstruct token ID from decomposed predictions."""
        # Simplified reconstruction - would need full decomposer logic
        if token_type == 1:  # Command
            return cmd + 3  # Offset for special tokens
        elif token_type == 2:  # Parameter type
            return pt + 14  # Offset for commands
        elif token_type == 3:  # Parameter value
            return pv + 21  # Offset for param types
        else:
            return 0  # PAD

    def plot_confusion_matrices(self, predictions: Dict, targets: Dict) -> None:
        """Plot confusion matrices for each prediction head."""
        print("\n📊 Generating confusion matrices...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        heads = ['command', 'param_type', 'param_value']
        titles = ['Command Predictions', 'Parameter Type Predictions', 'Parameter Value Predictions']

        for idx, (head, title) in enumerate(zip(heads, titles)):
            if head in predictions and head in targets:
                preds = predictions[head]
                tgts = targets[head]

                # Create confusion matrix
                cm = confusion_matrix(tgts, preds)

                # Normalize
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                # Plot
                ax = axes[idx]
                sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
                           cbar_kws={'label': 'Accuracy'})
                ax.set_title(title)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrices saved to {self.output_dir / 'confusion_matrices.png'}")

    def plot_confidence_analysis(self, confidences: Dict) -> None:
        """Analyze and plot prediction confidence distributions."""
        print("\n📊 Analyzing prediction confidence...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (head, confs) in enumerate(confidences.items()):
            if idx >= 6:
                break

            ax = axes[idx]

            # Histogram of confidences
            ax.hist(confs, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(confs), color='red', linestyle='--', label=f'Mean: {np.mean(confs):.3f}')
            ax.axvline(np.median(confs), color='green', linestyle='--', label=f'Median: {np.median(confs):.3f}')

            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title(f'{head.replace("_", " ").title()} Confidence')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(confidences), 6):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Confidence analysis saved to {self.output_dir / 'confidence_analysis.png'}")

    def analyze_error_patterns(self, token_errors: Dict) -> pd.DataFrame:
        """Analyze error patterns and create detailed report."""
        print("\n📊 Analyzing error patterns...")

        # Create error DataFrame
        error_data = []
        for token, stats in token_errors.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                error_rate = 1 - accuracy

                # Find most common errors
                if stats['errors']:
                    error_counter = Counter(stats['errors'])
                    most_common_errors = error_counter.most_common(3)
                    top_error = most_common_errors[0][0] if most_common_errors else 'None'
                    top_error_count = most_common_errors[0][1] if most_common_errors else 0
                else:
                    top_error = 'None'
                    top_error_count = 0

                error_data.append({
                    'token': token,
                    'occurrences': stats['total'],
                    'accuracy': accuracy,
                    'error_rate': error_rate,
                    'most_common_error': top_error,
                    'error_frequency': top_error_count / max(stats['total'] - stats['correct'], 1)
                })

        df = pd.DataFrame(error_data)
        df = df.sort_values('error_rate', ascending=False)

        # Save detailed report
        df.to_csv(self.output_dir / 'error_analysis.csv', index=False)
        print(f"✅ Error analysis saved to {self.output_dir / 'error_analysis.csv'}")

        # Plot top errors
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Top 20 hardest tokens
        top_errors = df.head(20)
        ax = axes[0]
        bars = ax.barh(range(len(top_errors)), top_errors['error_rate'].values)
        ax.set_yticks(range(len(top_errors)))
        ax.set_yticklabels(top_errors['token'].values)
        ax.set_xlabel('Error Rate')
        ax.set_title('Top 20 Hardest Tokens to Predict')
        ax.grid(True, alpha=0.3)

        # Color bars by error rate
        for bar, val in zip(bars, top_errors['error_rate'].values):
            bar.set_color(plt.cm.Reds(val))

        # Token frequency vs accuracy
        ax = axes[1]
        scatter = ax.scatter(df['occurrences'], df['accuracy'],
                           c=df['error_rate'], cmap='RdYlGn_r',
                           alpha=0.6, s=50)
        ax.set_xlabel('Token Occurrences (log scale)')
        ax.set_ylabel('Accuracy')
        ax.set_xscale('log')
        ax.set_title('Token Frequency vs Prediction Accuracy')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Error Rate')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Error patterns saved to {self.output_dir / 'error_patterns.png'}")

        return df

    def fetch_wandb_history(self, run_id: str = None) -> Optional[pd.DataFrame]:
        """Fetch training history from W&B if available."""
        if not WANDB_AVAILABLE:
            return None

        try:
            api = wandb.Api()

            # Get run
            if run_id:
                run = api.run(f"seacuello-university-of-rhode-island/gcode-fingerprinting/{run_id}")
            else:
                # Try to find the most recent run
                runs = api.runs("seacuello-university-of-rhode-island/gcode-fingerprinting",
                               {"$and": [{"state": "finished"}]},
                               order="-created_at", per_page=1)
                if not runs:
                    return None
                run = runs[0]

            print(f"📊 Fetching history from W&B run: {run.name} ({run.id})")

            # Get history
            history = run.history()
            return history

        except Exception as e:
            print(f"⚠️  Could not fetch W&B history: {e}")
            return None

    def plot_training_history(self, history: pd.DataFrame) -> None:
        """Plot training history from W&B data."""
        if history is None:
            return

        print("\n📊 Plotting training history...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax = axes[0, 0]
        if 'train/loss' in history.columns:
            ax.plot(history.index, history['train/loss'], label='Train Loss', alpha=0.7)
        if 'val/loss' in history.columns:
            ax.plot(history.index, history['val/loss'], label='Val Loss', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[0, 1]
        metrics = ['command_acc', 'param_type_acc', 'param_value_acc', 'overall_acc']
        colors = ['blue', 'green', 'orange', 'red']
        for metric, color in zip(metrics, colors):
            val_metric = f'val/{metric}'
            if val_metric in history.columns:
                ax.plot(history.index, history[val_metric],
                       label=metric.replace('_', ' ').title(),
                       alpha=0.7, color=color)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracies')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 0]
        if 'learning_rate' in history.columns:
            ax.plot(history.index, history['learning_rate'])
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Composite metrics
        ax = axes[1, 1]
        if 'val/composite_acc' in history.columns:
            ax.plot(history.index, history['val/composite_acc'],
                   label='Composite Accuracy', color='purple', linewidth=2)
        if 'val/param_value_mae' in history.columns:
            ax2 = ax.twinx()
            ax2.plot(history.index, history['val/param_value_mae'],
                    label='Param Value MAE', color='brown', alpha=0.7)
            ax2.set_ylabel('MAE', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy', color='purple')
        ax.tick_params(axis='y', labelcolor='purple')
        ax.set_title('Composite Metrics')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Training history saved to {self.output_dir / 'training_history.png'}")

    def generate_report(self, results: Dict) -> None:
        """Generate comprehensive analysis report."""
        print("\n📝 Generating analysis report...")

        report = []
        report.append("=" * 80)
        report.append("G-CODE FINGERPRINTING MODEL - PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.checkpoint_path}")
        report.append("")

        # Overall accuracies
        report.append("OVERALL PERFORMANCE METRICS:")
        report.append("-" * 40)

        for head in ['command', 'param_type', 'param_value']:
            if head in results['predictions']:
                preds = results['predictions'][head]
                tgts = results['targets'][head]
                accuracy = np.mean(np.array(preds) == np.array(tgts))
                report.append(f"  {head.replace('_', ' ').title()} Accuracy: {accuracy:.2%}")

        # Confidence statistics
        report.append("")
        report.append("PREDICTION CONFIDENCE:")
        report.append("-" * 40)

        for head, confs in results['confidences'].items():
            mean_conf = np.mean(confs)
            std_conf = np.std(confs)
            low_conf = np.sum(np.array(confs) < 0.5) / len(confs)
            report.append(f"  {head.replace('_', ' ').title()}:")
            report.append(f"    Mean: {mean_conf:.3f} ± {std_conf:.3f}")
            report.append(f"    Low confidence (<0.5): {low_conf:.1%}")

        # Error patterns
        report.append("")
        report.append("TOP ERROR-PRONE TOKENS:")
        report.append("-" * 40)

        if 'error_df' in results:
            top_errors = results['error_df'].head(10)
            for _, row in top_errors.iterrows():
                report.append(f"  {row['token']:10s} Error: {row['error_rate']:.1%} "
                            f"(→ {row['most_common_error']} {row['error_frequency']:.0%})")

        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS FOR IMPROVEMENT:")
        report.append("-" * 40)

        # Analyze weaknesses
        weaknesses = []
        for head in ['command', 'param_type', 'param_value']:
            if head in results['predictions']:
                acc = np.mean(np.array(results['predictions'][head]) ==
                            np.array(results['targets'][head]))
                if acc < 0.7:
                    weaknesses.append((head, acc))

        if weaknesses:
            report.append("  Focus areas for improvement:")
            for head, acc in weaknesses:
                report.append(f"    - {head.replace('_', ' ').title()} (current: {acc:.1%})")
                if head == 'param_value':
                    report.append("      Consider: regression approach, larger hidden dim, more data augmentation")
                elif head == 'param_type':
                    report.append("      Consider: increased weight in loss, dedicated attention layers")

        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✅ Report saved to {self.output_dir / 'analysis_report.txt'}")

    def run_full_analysis(self, batch_size: int = 32, run_id: str = None) -> None:
        """Run complete performance analysis."""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Load validation data
        print(f"\nLoading validation data from {self.data_dir}")
        val_dataset = GCodeDataset(self.data_dir / 'val_sequences.npz')
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"✅ Loaded {len(val_dataset)} validation sequences")

        # Analyze predictions
        analysis_results = self.analyze_dataset(val_loader)

        # Generate visualizations
        self.plot_confusion_matrices(
            analysis_results['predictions'],
            analysis_results['targets']
        )

        self.plot_confidence_analysis(analysis_results['confidences'])

        error_df = self.analyze_error_patterns(analysis_results['token_errors'])
        analysis_results['error_df'] = error_df

        # Fetch and plot W&B history if available
        history = self.fetch_wandb_history(run_id)
        if history is not None:
            self.plot_training_history(history)

        # Generate report
        self.generate_report(analysis_results)

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze G-code fingerprinting model performance')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/best_config_training/checkpoint_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='outputs/processed_hybrid',
                       help='Path to processed data directory')
    parser.add_argument('--vocab-path', type=str,
                       default='data/vocabulary_1digit_hybrid.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for analysis')
    parser.add_argument('--wandb-run', type=str, default=None,
                       help='W&B run ID to fetch training history')

    args = parser.parse_args()

    # Create analyzer
    analyzer = PerformanceAnalyzer(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        vocab_path=Path(args.vocab_path),
        output_dir=Path(args.output_dir)
    )

    # Run analysis
    analyzer.run_full_analysis(
        batch_size=args.batch_size,
        run_id=args.wandb_run
    )


if __name__ == '__main__':
    main()