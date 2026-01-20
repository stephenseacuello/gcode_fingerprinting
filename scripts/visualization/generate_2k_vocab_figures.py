#!/usr/bin/env python3
"""
Generate visualization figures for the 2k vocabulary experiments.

Generates:
1. Confusion matrix for token predictions
2. Per-token-type accuracy breakdown
3. Per-operation accuracy breakdown
4. Sensor ablation comparison bar chart
5. Modality ablation comparison bar chart
6. Architecture ablation comparison
7. Baseline comparison bar chart
8. Ensemble comparison

Usage:
    python scripts/visualization/generate_2k_vocab_figures.py \
        --model-path outputs/full_pipeline_2k_vocab/ensemble/seed_123/best_model.pt \
        --data-dir outputs/stratified_splits_2k_vocab \
        --vocab-path data/vocabulary_4digit_full.json \
        --encoder-path outputs/encoder_model.pt \
        --results-dir outputs/full_pipeline_2k_vocab \
        --output-dir outputs/paper_figures_2k_vocab
"""

import os
import sys
import json
import argparse
import platform
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import torch
from torch.utils.data import DataLoader

# Try seaborn for prettier plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

NUM_WORKERS = 0 if platform.system() == 'Windows' else 4


def load_model_and_data(model_path, data_dir, vocab_path, encoder_path, config_path=None, device='cuda'):
    """Load model and create data loaders."""
    from miracle.model.model import EnhancedEncoder
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    idx_to_token = {v: k for k, v in vocab.items()}

    # Load metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 155)
    else:
        input_dim = 155

    # Load model checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    # Override with config file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    max_seq_len = config.get('max_seq_len', 40)

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
        pooling_n_heads=config.get('pooling_n_heads', 64),
        pooling_n_queries=config.get('pooling_n_queries', 64),
    )

    if os.path.exists(encoder_path):
        enc_ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.load_state_dict(enc_ckpt.get('model_state_dict', enc_ckpt), strict=False)
    encoder.to(device)
    encoder.eval()

    # Create decoder
    d_model = config.get('d_model', 1280)
    n_heads = config.get('n_heads', 32)
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        sensor_dim=128,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config.get('n_layers', 7),
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_model * config.get('ffn_multiplier', 3),
        dropout=config.get('dropout', 0.0),
        max_seq_len=max_seq_len,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config.get('embed_dropout', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0),
        use_sensor_prior=config.get('use_sensor_prior', False),
        sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
    )
    decoder.load_state_dict(ckpt['model_state_dict'], strict=False)
    decoder.to(device)
    decoder.eval()

    # Load test dataset
    test_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='test',
        max_token_len=max_seq_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )

    return encoder, decoder, test_loader, vocab, idx_to_token, config


def compute_detailed_metrics(encoder, decoder, test_loader, idx_to_token, device='cuda'):
    """Compute detailed metrics including confusion data."""
    decoder.eval()
    encoder.eval()

    # For confusion matrix
    all_preds = []
    all_targets = []

    # Per-token-type tracking
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    # Per-operation tracking
    op_names = ['G0', 'G1', 'G2', 'G3', 'G28', 'M104', 'M106', 'M109', 'M140']
    op_correct = defaultdict(int)
    op_total = defaultdict(int)

    # Overall
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)
            mask = target_tokens != 0

            # Overall accuracy
            correct = ((preds == target_tokens) & mask)
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Collect for confusion matrix
            for i in range(target_tokens.size(0)):
                for j in range(target_tokens.size(1)):
                    if mask[i, j]:
                        pred_idx = preds[i, j].item()
                        target_idx = target_tokens[i, j].item()
                        all_preds.append(pred_idx)
                        all_targets.append(target_idx)

                        # Classify token type
                        target_tok = idx_to_token.get(target_idx, '<UNK>')
                        if target_tok.startswith('G') or target_tok.startswith('M'):
                            tok_type = 'Command'
                        elif target_tok in ['X', 'Y', 'Z', 'E', 'F', 'I', 'J', 'R', 'S']:
                            tok_type = 'Param Type'
                        elif target_tok in ['+', '-']:
                            tok_type = 'Sign'
                        elif target_tok == '.':
                            tok_type = 'Decimal'
                        elif target_tok.isdigit():
                            tok_type = 'Digit'
                        else:
                            tok_type = 'Other'

                        type_total[tok_type] += 1
                        if correct[i, j]:
                            type_correct[tok_type] += 1

            # Per-operation accuracy
            for i in range(operations.size(0)):
                op = operations[i].item()
                if op < len(op_names):
                    op_name = op_names[op]
                else:
                    op_name = f'Op{op}'
                sample_mask = mask[i]
                sample_correct = correct[i]
                op_correct[op_name] += sample_correct.sum().item()
                op_total[op_name] += sample_mask.sum().item()

    # Compute accuracies
    overall_acc = total_correct / total_tokens if total_tokens > 0 else 0
    type_accs = {t: type_correct[t] / type_total[t] if type_total[t] > 0 else 0 for t in type_total}
    op_accs = {op: op_correct[op] / op_total[op] if op_total[op] > 0 else 0 for op in op_total}

    return {
        'overall_acc': overall_acc,
        'type_accs': type_accs,
        'type_totals': dict(type_total),
        'op_accs': op_accs,
        'op_totals': dict(op_total),
        'all_preds': all_preds,
        'all_targets': all_targets,
    }


def plot_confusion_matrix(all_preds, all_targets, idx_to_token, output_path, top_k=30):
    """Plot confusion matrix for top-k most common tokens."""
    from sklearn.metrics import confusion_matrix

    # Count token frequencies
    target_counts = defaultdict(int)
    for t in all_targets:
        target_counts[t] += 1

    # Get top-k tokens
    top_tokens = sorted(target_counts.keys(), key=lambda x: target_counts[x], reverse=True)[:top_k]
    top_set = set(top_tokens)

    # Filter predictions and targets to top tokens
    filtered_preds = []
    filtered_targets = []
    for p, t in zip(all_preds, all_targets):
        if t in top_set:
            filtered_preds.append(p if p in top_set else -1)
            filtered_targets.append(t)

    # Create labels
    labels = [idx_to_token.get(t, f'<{t}>') for t in top_tokens]

    # Compute confusion matrix
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_tokens)

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    if HAS_SEABORN:
        sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(cm_norm, cmap='Blues')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax)

    ax.set_xlabel('Predicted Token')
    ax.set_ylabel('True Token')
    ax.set_title(f'Confusion Matrix (Top {top_k} Tokens, Normalized)')

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved confusion matrix to: {output_path}")
    plt.close(fig)


def plot_per_type_accuracy(type_accs, type_totals, output_path):
    """Plot per-token-type accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by accuracy
    sorted_types = sorted(type_accs.keys(), key=lambda x: type_accs[x], reverse=True)
    accs = [type_accs[t] * 100 for t in sorted_types]
    totals = [type_totals[t] for t in sorted_types]

    # Create bars with color gradient
    colors = plt.cm.RdYlGn(np.array(accs) / 100)
    bars = ax.barh(sorted_types, accs, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, acc, total in zip(bars, accs, totals):
        width = bar.get_width()
        ax.annotate(f'{acc:.1f}% (n={total:,})',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)

    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Per-Token-Type Accuracy')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved per-type accuracy to: {output_path}")
    plt.close(fig)


def plot_per_operation_accuracy(op_accs, op_totals, output_path):
    """Plot per-operation accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort operations
    sorted_ops = sorted(op_accs.keys())
    accs = [op_accs[op] * 100 for op in sorted_ops]
    totals = [op_totals.get(op, 0) for op in sorted_ops]

    x = np.arange(len(sorted_ops))
    colors = plt.cm.RdYlGn(np.array(accs) / 100)
    bars = ax.bar(x, accs, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, acc, total in zip(bars, accs, totals):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%\n(n={total:,})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_ops, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('G-code Operation')
    ax.set_title('Per-Operation Token Accuracy')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved per-operation accuracy to: {output_path}")
    plt.close(fig)


def plot_ablation_comparison(results_dir, ablation_type, output_path):
    """Plot ablation study results."""
    ablation_dir = Path(results_dir) / ablation_type
    if not ablation_dir.exists():
        print(f"Ablation directory not found: {ablation_dir}")
        return

    results = {}
    for subdir in ablation_dir.iterdir():
        if subdir.is_dir():
            results_file = subdir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                val_acc = data.get('val_acc', data.get('best_val_metric', 0))
                test_acc = data.get('test_acc', 0)
                results[subdir.name] = {
                    'val_acc': val_acc * 100 if val_acc < 1 else val_acc,
                    'test_acc': test_acc * 100 if test_acc < 1 else test_acc,
                }

    if not results:
        print(f"No results found in {ablation_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sorted_names = sorted(results.keys())
    x = np.arange(len(sorted_names))
    width = 0.35

    val_accs = [results[n]['val_acc'] for n in sorted_names]
    test_accs = [results[n]['test_acc'] for n in sorted_names]

    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='#3498db', alpha=0.8)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in sorted_names], rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{ablation_type.replace("_", " ").title()}')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved {ablation_type} ablation plot to: {output_path}")
    plt.close(fig)


def plot_baseline_comparison(results_dir, output_path):
    """Plot baseline comparison."""
    baselines_dir = Path(results_dir) / 'baselines'
    if not baselines_dir.exists():
        print(f"Baselines directory not found: {baselines_dir}")
        return

    # Load results
    results = {}
    for results_file in baselines_dir.glob('*/results.json'):
        name = results_file.parent.name
        with open(results_file) as f:
            data = json.load(f)
        test_acc = data.get('test_acc', data.get('best_val_metric', 0))
        results[name] = test_acc * 100 if test_acc < 1 else test_acc

    if not results:
        print(f"No baseline results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_names = sorted(results.keys(), key=lambda x: results[x])
    accs = [results[n] for n in sorted_names]

    # Color code by performance
    colors = ['#e74c3c' if acc < 50 else '#f39c12' if acc < 70 else '#27ae60' for acc in accs]

    bars = ax.barh(sorted_names, accs, color=colors, edgecolor='black', linewidth=0.5)

    for bar, acc in zip(bars, accs):
        width = bar.get_width()
        ax.annotate(f'{acc:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Baseline Comparison')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved baseline comparison to: {output_path}")
    plt.close(fig)


def plot_ensemble_comparison(results_dir, output_path):
    """Plot ensemble results vs individual seeds."""
    ensemble_dir = Path(results_dir) / 'ensemble'
    summary_file = ensemble_dir / 'ensemble_summary.json'

    if not summary_file.exists():
        print(f"Ensemble summary not found: {summary_file}")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual seeds
    individual = summary.get('individual_results', [])
    seed_names = [f"Seed {r['seed']}" for r in individual]
    seed_val_accs = [r['val_acc'] * 100 if r['val_acc'] < 1 else r['val_acc'] for r in individual]
    seed_test_accs = [r.get('test_acc', 0) * 100 if r.get('test_acc', 0) < 1 else r.get('test_acc', 0) for r in individual]

    # Ensemble
    ensemble_val = summary.get('ensemble_val_acc', 0) * 100 if summary.get('ensemble_val_acc', 0) < 1 else summary.get('ensemble_val_acc', 0)

    # Plot
    x = np.arange(len(seed_names) + 1)
    width = 0.35

    all_val = seed_val_accs + [ensemble_val]
    all_test = seed_test_accs + [0]
    all_names = seed_names + ['Ensemble']

    bars1 = ax.bar(x - width/2, all_val, width, label='Validation', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, all_test, width, label='Test', color='#3498db', alpha=0.8)

    # Highlight ensemble
    bars1[-1].set_color('#e74c3c')
    bars1[-1].set_alpha(1.0)

    for bar, val in zip(bars1, all_val):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, all_test):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(all_names)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ensemble vs Individual Seeds')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.png'))
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved ensemble comparison to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate 2k vocab experiment figures')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint (optional)')
    parser.add_argument('--data-dir', type=str, help='Path to data splits (required if model-path given)')
    parser.add_argument('--vocab-path', type=str, help='Path to vocabulary (required if model-path given)')
    parser.add_argument('--encoder-path', type=str, help='Path to encoder (required if model-path given)')
    parser.add_argument('--config-path', type=str, default=None, help='Path to config file')
    parser.add_argument('--results-dir', type=str, required=True, help='Results directory with experiment outputs')
    parser.add_argument('--output-dir', type=str, default='outputs/paper_figures_2k_vocab')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-model-eval', action='store_true', help='Skip model evaluation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and compute metrics if requested
    if args.model_path and not args.skip_model_eval:
        if not all([args.data_dir, args.vocab_path, args.encoder_path]):
            print("Error: --data-dir, --vocab-path, --encoder-path required when using --model-path")
            return

        print("\nLoading model and computing metrics...")
        encoder, decoder, test_loader, vocab, idx_to_token, config = load_model_and_data(
            model_path=args.model_path,
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            encoder_path=args.encoder_path,
            config_path=args.config_path,
            device=device,
        )

        metrics = compute_detailed_metrics(encoder, decoder, test_loader, idx_to_token, device)

        # Save metrics
        metrics_file = output_dir / 'detailed_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'overall_acc': metrics['overall_acc'],
                'type_accs': metrics['type_accs'],
                'type_totals': metrics['type_totals'],
                'op_accs': metrics['op_accs'],
                'op_totals': metrics['op_totals'],
            }, f, indent=2)
        print(f"Saved metrics to: {metrics_file}")
        print(f"Overall test accuracy: {metrics['overall_acc']*100:.2f}%")

        # Generate model-based plots
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(
            metrics['all_preds'], metrics['all_targets'], idx_to_token,
            output_dir / 'confusion_matrix', top_k=30
        )

        print("\nGenerating per-type accuracy plot...")
        plot_per_type_accuracy(
            metrics['type_accs'], metrics['type_totals'],
            output_dir / 'per_type_accuracy'
        )

        print("\nGenerating per-operation accuracy plot...")
        plot_per_operation_accuracy(
            metrics['op_accs'], metrics['op_totals'],
            output_dir / 'per_operation_accuracy'
        )

    # Generate ablation plots from results directory
    print("\nGenerating ablation comparison plots...")
    for ablation_type in ['sensor_ablations', 'modality_ablations', 'architecture_ablations']:
        plot_ablation_comparison(
            args.results_dir, ablation_type,
            output_dir / f'{ablation_type}_comparison'
        )

    # Generate baseline comparison
    print("\nGenerating baseline comparison...")
    plot_baseline_comparison(args.results_dir, output_dir / 'baseline_comparison')

    # Generate ensemble comparison
    print("\nGenerating ensemble comparison...")
    plot_ensemble_comparison(args.results_dir, output_dir / 'ensemble_comparison')

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
