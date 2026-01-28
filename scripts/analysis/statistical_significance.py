#!/usr/bin/env python3
"""
Statistical significance analysis for ablation experiments.

Computes:
1. Bootstrap 95% confidence intervals for each ablation
2. McNemar's test for pairwise model comparisons
3. Summary statistics for paper

Usage:
    python scripts/analysis/statistical_significance.py \
        --ablation-dir outputs/jan26/sensor_ablations \
        --baseline-dir outputs/jan23_followup/no_leakage/training_tuned \
        --data-dir outputs/jan23_followup/no_leakage \
        --encoder-path outputs/production/encoder/best_model.pt \
        --vocab-path data/vocabulary_4digit_full.json \
        --output-dir outputs/jan26/statistical_analysis
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn


class MaskedDataset:
    """Wrapper dataset that applies a sensor mask to the features."""

    def __init__(self, base_dataset, mask: np.ndarray):
        self.base_dataset = base_dataset
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['sensor_features'] = item['sensor_features'] * self.mask
        return item


def get_predictions(
    encoder: nn.Module,
    decoder: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get per-token predictions and labels for the entire dataset.

    Returns:
        predictions: (n_samples, seq_len) predicted token indices
        labels: (n_samples, seq_len) ground truth token indices
    """
    all_preds = []
    all_labels = []

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch in data_loader:
            sensors = batch['sensor_features'].to(device)
            tokens = batch['tokens'].to(device)
            operations = batch['operation_labels'].to(device)

            # Encode
            latent, _ = encoder(sensors, operations)

            # Decode (teacher forcing for consistent comparison)
            logits = decoder(tokens[:, :-1], latent, operations)

            # Get predictions (greedy)
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            labels = tokens[:, 1:]  # Shift for teacher forcing

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    pad_token: int = 0
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for token accuracy.

    Args:
        predictions: (n_samples, seq_len) predicted tokens
        labels: (n_samples, seq_len) ground truth tokens
        n_bootstrap: number of bootstrap samples
        ci: confidence interval (default 0.95 for 95% CI)
        pad_token: token ID to ignore (padding)

    Returns:
        Dict with mean, std, ci_low, ci_high
    """
    n_samples = len(predictions)
    accuracies = []

    for _ in range(n_bootstrap):
        # Sample with replacement at sequence level
        idx = np.random.choice(n_samples, n_samples, replace=True)

        boot_pred = predictions[idx].flatten()
        boot_label = labels[idx].flatten()

        # Mask out padding
        valid_mask = boot_label != pad_token

        if valid_mask.sum() > 0:
            acc = (boot_pred[valid_mask] == boot_label[valid_mask]).mean()
            accuracies.append(acc)

    accuracies = np.array(accuracies)
    alpha = 1 - ci

    return {
        'mean': float(np.mean(accuracies)),
        'std': float(np.std(accuracies)),
        'ci_low': float(np.percentile(accuracies, alpha/2 * 100)),
        'ci_high': float(np.percentile(accuracies, (1 - alpha/2) * 100)),
    }


def friedman_test(
    all_predictions: Dict[str, np.ndarray],
    labels: np.ndarray,
    pad_token: int = 0
) -> Dict[str, float]:
    """
    Perform Friedman test for comparing multiple models.

    Non-parametric alternative to repeated-measures ANOVA.
    Treats each test sample as a 'block' and compares model rankings.

    Args:
        all_predictions: Dict mapping model name to predictions
        labels: Ground truth labels
        pad_token: Token ID to ignore

    Returns:
        Dict with test statistic, p-value, and post-hoc results
    """
    from scipy import stats

    model_names = list(all_predictions.keys())
    n_models = len(model_names)

    if n_models < 3:
        return {'error': 'Friedman test requires at least 3 groups'}

    # Compute per-sample accuracy for each model
    # Shape: (n_samples, n_models)
    n_samples = len(labels)
    sample_accuracies = np.zeros((n_samples, n_models))

    for j, name in enumerate(model_names):
        preds = all_predictions[name]
        for i in range(n_samples):
            valid_mask = labels[i] != pad_token
            if valid_mask.sum() > 0:
                sample_accuracies[i, j] = (preds[i][valid_mask] == labels[i][valid_mask]).mean()

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*[sample_accuracies[:, j] for j in range(n_models)])

    result = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'n_models': n_models,
        'n_samples': n_samples,
        'significant': p_value < 0.05,
        'model_names': model_names,
        'mean_ranks': {},
    }

    # Compute mean ranks for each model
    ranks = np.zeros_like(sample_accuracies)
    for i in range(n_samples):
        ranks[i] = stats.rankdata(-sample_accuracies[i])  # Negative so higher acc = lower rank
    mean_ranks = ranks.mean(axis=0)

    for j, name in enumerate(model_names):
        result['mean_ranks'][name] = float(mean_ranks[j])

    return result


def one_way_anova(
    group_accuracies: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Perform one-way ANOVA for comparing multiple groups.

    Requires multiple observations per group (e.g., multiple seeds).

    Args:
        group_accuracies: Dict mapping group name to list of accuracy values

    Returns:
        Dict with F-statistic, p-value, and effect size
    """
    from scipy import stats

    groups = list(group_accuracies.values())
    group_names = list(group_accuracies.keys())

    # Check we have enough data
    min_obs = min(len(g) for g in groups)
    if min_obs < 2:
        return {'error': 'ANOVA requires at least 2 observations per group'}

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    result = {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'eta_squared': float(eta_squared),
        'n_groups': len(groups),
        'observations_per_group': [len(g) for g in groups],
        'group_means': {name: float(np.mean(acc)) for name, acc in group_accuracies.items()},
        'group_stds': {name: float(np.std(acc)) for name, acc in group_accuracies.items()},
        'significant': p_value < 0.05,
    }

    # Post-hoc Tukey HSD if significant
    if p_value < 0.05:
        try:
            from scipy.stats import tukey_hsd
            tukey_result = tukey_hsd(*groups)
            result['tukey_pvalues'] = {}
            for i, name_i in enumerate(group_names):
                for j, name_j in enumerate(group_names):
                    if i < j:
                        result['tukey_pvalues'][f'{name_i}_vs_{name_j}'] = float(tukey_result.pvalue[i, j])
        except ImportError:
            result['tukey_note'] = 'Tukey HSD requires scipy >= 1.8'

    return result


def mcnemar_test(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: np.ndarray,
    pad_token: int = 0
) -> Dict[str, float]:
    """
    Perform McNemar's test comparing two models.

    Tests whether the models have significantly different error rates.
    """
    from scipy import stats

    # Flatten
    pred_a = pred_a.flatten()
    pred_b = pred_b.flatten()
    labels = labels.flatten()

    # Valid tokens only
    valid_mask = labels != pad_token
    pred_a = pred_a[valid_mask]
    pred_b = pred_b[valid_mask]
    labels = labels[valid_mask]

    # Correct/incorrect for each model
    a_correct = (pred_a == labels)
    b_correct = (pred_b == labels)

    # Contingency table
    n11 = np.sum(a_correct & b_correct)      # both right
    n10 = np.sum(a_correct & ~b_correct)     # A right, B wrong
    n01 = np.sum(~a_correct & b_correct)     # A wrong, B right
    n00 = np.sum(~a_correct & ~b_correct)    # both wrong

    if n01 + n10 == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'n_both_correct': int(n11),
            'n_a_only': int(n10),
            'n_b_only': int(n01),
            'n_both_wrong': int(n00),
            'significant': False
        }

    # Chi-square with continuity correction
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'statistic': float(chi2),
        'p_value': float(p_value),
        'n_both_correct': int(n11),
        'n_a_only': int(n10),
        'n_b_only': int(n01),
        'n_both_wrong': int(n00),
        'significant': p_value < 0.05
    }


def load_encoder_decoder(
    encoder_path: str,
    decoder_path: str,
    vocab_path: str,
    input_dim: int,
    config: dict,
    device: str
) -> Tuple[EnhancedEncoder, SensorMultiHeadDecoder]:
    """Load encoder and decoder models."""

    # Load vocabulary
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    vocab_size = len(vocab['token_to_id'])

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
        pooling_n_heads=config.get('pooling_n_heads', 8),
        pooling_n_queries=config.get('pooling_n_queries', 16),
    )

    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        encoder.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    encoder.to(device)
    encoder.eval()

    # Create decoder
    d_model = config.get('d_model', 192)
    n_heads = config.get('n_heads', 8)
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config.get('n_layers', 4),
        sensor_dim=128,
        n_operations=9,
        dropout=0.0,
        embed_dropout=0.0,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        max_seq_len=config.get('max_seq_len', 32),
    ).to(device)

    decoder_ckpt = torch.load(decoder_path, map_location='cpu', weights_only=False)
    decoder.load_state_dict(decoder_ckpt.get('model_state_dict', decoder_ckpt), strict=False)
    decoder.eval()

    return encoder, decoder


def parse_sensor_mask(ablation_name: str, master_columns: List[str]) -> Optional[np.ndarray]:
    """Parse ablation name to determine sensor mask."""
    if ablation_name.startswith('only_'):
        sensor_name = ablation_name[5:]
        mask = np.array([sensor_name in col for col in master_columns], dtype=np.float32)
        return mask
    elif ablation_name.startswith('without_'):
        sensor_name = ablation_name[8:]
        mask = np.array([sensor_name not in col for col in master_columns], dtype=np.float32)
        return mask
    return None


def main():
    parser = argparse.ArgumentParser(description='Statistical significance analysis')
    parser.add_argument('--ablation-dir', type=str, required=True,
                        help='Directory containing ablation results')
    parser.add_argument('--baseline-dir', type=str, required=True,
                        help='Directory containing baseline model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing train/val/test splits')
    parser.add_argument('--encoder-path', type=str, required=True,
                        help='Path to encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file for model architecture')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {'d_model': 192, 'n_heads': 8, 'n_layers': 4, 'max_seq_len': 32, 'batch_size': 32}

    # Load metadata
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 155)
        master_columns = metadata.get('master_columns', [])
    else:
        input_dim = 155
        master_columns = []

    print(f"Input dim: {input_dim}, Columns: {len(master_columns)}")

    # Find ablation directories
    ablation_dir = Path(args.ablation_dir)
    ablation_dirs = sorted([d for d in ablation_dir.iterdir() if d.is_dir()])
    print(f"Found {len(ablation_dirs)} ablation experiments")

    all_predictions = {}
    all_labels = None
    bootstrap_results = {}

    max_seq_len = config.get('max_seq_len', 32)
    batch_size = config.get('batch_size', 32)

    # Load baseline
    print("\n" + "="*60)
    print("Loading baseline model...")
    print("="*60)

    baseline_dir = Path(args.baseline_dir)
    baseline_decoder_path = baseline_dir / 'best_model.pt'

    if baseline_decoder_path.exists():
        test_dataset = DecoderDatasetFromSplits(
            split_dir=str(data_dir), split='test', max_token_len=max_seq_len
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=decoder_collate_fn
        )

        encoder, decoder = load_encoder_decoder(
            args.encoder_path, str(baseline_decoder_path), args.vocab_path, input_dim, config, device
        )

        baseline_preds, baseline_labels = get_predictions(encoder, decoder, test_loader, device)
        all_predictions['baseline'] = baseline_preds
        all_labels = baseline_labels

        bootstrap_results['baseline'] = bootstrap_confidence_interval(
            baseline_preds, baseline_labels, args.n_bootstrap
        )
        bs = bootstrap_results['baseline']
        print(f"  Baseline: {bs['mean']*100:.2f}% 95% CI [{bs['ci_low']*100:.2f}, {bs['ci_high']*100:.2f}]")
    else:
        print(f"  WARNING: Baseline not found at {baseline_decoder_path}")

    # Process ablations
    print("\n" + "="*60)
    print("Processing ablations...")
    print("="*60)

    for abl_dir in tqdm(ablation_dirs, desc="Ablations"):
        decoder_path = abl_dir / 'best_model.pt'
        results_path = abl_dir / 'results.json'

        if not decoder_path.exists() or not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        ablation_name = results.get('ablation_name', abl_dir.name)
        sensor_mask = parse_sensor_mask(ablation_name, master_columns)

        test_base = DecoderDatasetFromSplits(
            split_dir=str(data_dir), split='test', max_token_len=max_seq_len
        )
        test_dataset = MaskedDataset(test_base, sensor_mask) if sensor_mask is not None else test_base
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=decoder_collate_fn
        )

        try:
            encoder, decoder = load_encoder_decoder(
                args.encoder_path, str(decoder_path), args.vocab_path, input_dim, config, device
            )
            preds, labels = get_predictions(encoder, decoder, test_loader, device)
            all_predictions[ablation_name] = preds

            if all_labels is None:
                all_labels = labels

            bootstrap_results[ablation_name] = bootstrap_confidence_interval(preds, labels, args.n_bootstrap)
        except Exception as e:
            print(f"  Error processing {ablation_name}: {e}")
            continue

    # Save bootstrap results
    with open(output_dir / 'bootstrap_results.json', 'w') as f:
        json.dump(bootstrap_results, f, indent=2)

    # McNemar's tests
    print("\n" + "="*60)
    print("McNemar's Tests (vs Baseline)")
    print("="*60)

    mcnemar_results = {}

    if 'baseline' in all_predictions:
        for name, preds in all_predictions.items():
            if name == 'baseline':
                continue

            result = mcnemar_test(all_predictions['baseline'], preds, all_labels)
            mcnemar_results[f"baseline_vs_{name}"] = result

            sig = "***" if result['p_value'] < 0.001 else \
                  "**" if result['p_value'] < 0.01 else \
                  "*" if result['p_value'] < 0.05 else ""
            print(f"  vs {name}: chi2={result['statistic']:.1f}, p={result['p_value']:.4f} {sig}")

    with open(output_dir / 'mcnemar_results.json', 'w') as f:
        json.dump(mcnemar_results, f, indent=2)

    # Friedman test (non-parametric ANOVA alternative)
    print("\n" + "="*60)
    print("Friedman Test (Non-parametric ANOVA)")
    print("="*60)

    if len(all_predictions) >= 3:
        friedman_result = friedman_test(all_predictions, all_labels)
        if 'error' not in friedman_result:
            sig = "***" if friedman_result['p_value'] < 0.001 else \
                  "**" if friedman_result['p_value'] < 0.01 else \
                  "*" if friedman_result['p_value'] < 0.05 else ""
            print(f"  Chi-square = {friedman_result['statistic']:.2f}, "
                  f"p = {friedman_result['p_value']:.4f} {sig}")
            print(f"  Conclusion: {'Significant' if friedman_result['significant'] else 'Not significant'} "
                  f"differences among {friedman_result['n_models']} models")
            print("\n  Mean Ranks (lower = better):")
            for name, rank in sorted(friedman_result['mean_ranks'].items(), key=lambda x: x[1]):
                print(f"    {name}: {rank:.2f}")

            with open(output_dir / 'friedman_results.json', 'w') as f:
                json.dump(friedman_result, f, indent=2)
        else:
            print(f"  {friedman_result['error']}")
    else:
        print("  Skipped: Need at least 3 models for Friedman test")

    # Check for multi-seed data for proper ANOVA
    print("\n" + "="*60)
    print("One-Way ANOVA (if multi-seed data available)")
    print("="*60)

    # Look for multi-seed results
    multi_seed_accuracies = {}
    for abl_dir in ablation_dirs:
        # Check for seed subdirectories
        seed_dirs = [d for d in abl_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        if len(seed_dirs) >= 2:
            ablation_name = abl_dir.name
            accs = []
            for seed_dir in seed_dirs:
                results_path = seed_dir / 'results.json'
                if results_path.exists():
                    with open(results_path) as f:
                        seed_results = json.load(f)
                    accs.append(seed_results.get('test_token_acc', seed_results.get('test_metrics', {}).get('token_acc', 0)))
            if len(accs) >= 2:
                multi_seed_accuracies[ablation_name] = accs

    if len(multi_seed_accuracies) >= 2:
        anova_result = one_way_anova(multi_seed_accuracies)
        if 'error' not in anova_result:
            sig = "***" if anova_result['p_value'] < 0.001 else \
                  "**" if anova_result['p_value'] < 0.01 else \
                  "*" if anova_result['p_value'] < 0.05 else ""
            print(f"  F({anova_result['n_groups']-1}, {sum(anova_result['observations_per_group'])-anova_result['n_groups']}) = "
                  f"{anova_result['f_statistic']:.2f}, p = {anova_result['p_value']:.4f} {sig}")
            print(f"  Effect size (eta-squared): {anova_result['eta_squared']:.3f}")
            print(f"  Conclusion: {'Significant' if anova_result['significant'] else 'Not significant'} differences")

            print("\n  Group Statistics:")
            for name in sorted(anova_result['group_means'].keys()):
                mean = anova_result['group_means'][name]
                std = anova_result['group_stds'][name]
                print(f"    {name}: {mean*100:.2f}% +/- {std*100:.2f}%")

            with open(output_dir / 'anova_results.json', 'w') as f:
                json.dump(anova_result, f, indent=2)
        else:
            print(f"  {anova_result['error']}")
    else:
        print("  Skipped: No multi-seed data found")
        print("  To enable ANOVA, run ablations with --n-seeds 3")

    # LaTeX table
    print("\n" + "="*60)
    print("LaTeX Table for Paper")
    print("="*60)

    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{Token Acc.} & \\textbf{95\\% CI} & \\textbf{p-value} \\\\")
    print("\\midrule")

    for name, bs in sorted(bootstrap_results.items(), key=lambda x: x[1]['mean'], reverse=True):
        ci_str = f"[{bs['ci_low']*100:.1f}, {bs['ci_high']*100:.1f}]"

        if name == 'baseline':
            p_str = "---"
        elif f"baseline_vs_{name}" in mcnemar_results:
            p = mcnemar_results[f"baseline_vs_{name}"]['p_value']
            p_str = "$<$0.001***" if p < 0.001 else f"{p:.3f}**" if p < 0.01 else f"{p:.2f}*" if p < 0.05 else f"{p:.2f}"
        else:
            p_str = "---"

        print(f"{name} & {bs['mean']*100:.2f}\\% & {ci_str} & {p_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")

    # Save summary
    with open(output_dir / 'statistical_summary.json', 'w') as f:
        json.dump({
            'bootstrap_cis': bootstrap_results,
            'mcnemar_tests': mcnemar_results,
            'n_bootstrap': args.n_bootstrap,
            'n_test_samples': len(all_labels) if all_labels is not None else 0
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
