#!/usr/bin/env python3
"""
Compute accurate statistics for the paper from actual data.

This script:
1. Computes sensor correlation statistics
2. Verifies test set size
3. Outputs corrected values for the paper
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'outputs' / 'stratified_splits_2k_vocab'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sensor_correlations():
    """Compute actual pairwise correlations between sensors."""
    print("=" * 60)
    print("Computing Sensor Correlation Statistics")
    print("=" * 60)

    # Load test data
    test_data = np.load(DATA_DIR / 'test_sequences.npz', allow_pickle=True)
    X = test_data['continuous']  # (n_samples, seq_len, n_features)

    print(f"Test data shape: {X.shape}")
    n_samples, seq_len, n_features = X.shape

    # Flatten to (n_samples * seq_len, n_features) for correlation
    X_flat = X.reshape(-1, n_features)
    print(f"Flattened shape: {X_flat.shape}")

    # Compute correlation matrix
    # Remove any constant columns first
    std = np.std(X_flat, axis=0)
    valid_cols = std > 1e-10
    X_valid = X_flat[:, valid_cols]
    print(f"Valid features (non-constant): {X_valid.shape[1]}")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_valid.T)

    # Get upper triangle (excluding diagonal)
    upper_tri = np.triu_indices(corr_matrix.shape[0], k=1)
    correlations = corr_matrix[upper_tri]

    # Compute statistics
    max_corr = np.nanmax(np.abs(correlations))
    mean_abs_corr = np.nanmean(np.abs(correlations))
    pairs_above_05 = np.sum(np.abs(correlations) > 0.5)
    pairs_above_08 = np.sum(np.abs(correlations) > 0.8)
    total_pairs = len(correlations)

    print(f"\nCorrelation Statistics:")
    print(f"  Total feature pairs: {total_pairs}")
    print(f"  Maximum pairwise correlation: {max_corr:.2f}")
    print(f"  Mean absolute correlation: {mean_abs_corr:.2f}")
    print(f"  Pairs with |r| > 0.5: {pairs_above_05}")
    print(f"  Pairs with |r| > 0.8: {pairs_above_08}")

    # Save results
    results = {
        'total_features': int(n_features),
        'valid_features': int(X_valid.shape[1]),
        'total_pairs': int(total_pairs),
        'max_pairwise_correlation': float(max_corr),
        'mean_absolute_correlation': float(mean_abs_corr),
        'pairs_above_0.5': int(pairs_above_05),
        'pairs_above_0.8': int(pairs_above_08)
    }

    with open(OUTPUT_DIR / 'correlation_statistics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {OUTPUT_DIR / 'correlation_statistics.json'}")

    return results


def verify_dataset_sizes():
    """Verify actual dataset split sizes."""
    print("\n" + "=" * 60)
    print("Verifying Dataset Sizes")
    print("=" * 60)

    splits = {}
    for split_name in ['train', 'val', 'test']:
        data = np.load(DATA_DIR / f'{split_name}_sequences.npz', allow_pickle=True)
        n_samples = data['continuous'].shape[0]
        splits[split_name] = n_samples

        # Get operation distribution
        op_types = data['operation_type']
        unique, counts = np.unique(op_types, return_counts=True)
        print(f"\n{split_name.upper()}: {n_samples} samples")
        print(f"  Operations: {dict(zip(unique, counts))}")

    total = sum(splits.values())
    print(f"\nTotal: {total} samples")
    print(f"Split ratios: train={splits['train']/total:.1%}, val={splits['val']/total:.1%}, test={splits['test']/total:.1%}")

    return splits


def get_component_ablation_values():
    """Extract component ablation values with consistent settings."""
    print("\n" + "=" * 60)
    print("Component Ablation Analysis (All with augment=false)")
    print("=" * 60)

    ablation_dir = BASE_DIR / 'outputs' / 'full_pipeline_2k_vocab' / 'architecture_ablations'

    # All these experiments should have augment=false for fair comparison
    experiments = {
        'baseline': ('cross_attention', 'cross_attn_ON_seed42'),
        'no_cross_attention': ('cross_attention', 'cross_attn_OFF_seed42'),
        'no_positional': ('positional', 'pos_enc_OFF_seed42'),
        'no_operation_cond': ('operation', 'op_cond_OFF_seed42'),
    }

    results = {}

    for name, (subdir, exp_name) in experiments.items():
        result_path = ablation_dir / subdir / exp_name / 'results.json'
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)

            token_acc = data['test_metrics']['token_acc']
            augment = data['args'].get('augment', False)

            results[name] = {
                'token_acc': token_acc,
                'token_acc_pct': token_acc * 100,
                'augment': augment
            }
            print(f"  {name}: {token_acc*100:.2f}% (augment={augment})")

    # Calculate deltas from baseline
    if 'baseline' in results:
        baseline_acc = results['baseline']['token_acc_pct']
        print(f"\nDeltas from baseline ({baseline_acc:.2f}%):")
        for name, data in results.items():
            if name != 'baseline':
                delta = data['token_acc_pct'] - baseline_acc
                print(f"  {name}: {delta:+.2f}%")
                results[name]['delta'] = delta

    return results


def main():
    print("=" * 60)
    print("Computing Paper Statistics from Actual Data")
    print("=" * 60)

    # 1. Compute correlation statistics
    corr_stats = compute_sensor_correlations()

    # 2. Verify dataset sizes
    splits = verify_dataset_sizes()

    # 3. Get component ablation values
    ablation_results = get_component_ablation_values()

    # Summary for paper
    print("\n" + "=" * 60)
    print("VALUES FOR PAPER")
    print("=" * 60)

    print("\n[Table 5 - Correlation Statistics]")
    print(f"  Maximum pairwise correlation: {corr_stats['max_pairwise_correlation']:.2f}")
    print(f"  Mean absolute correlation: {corr_stats['mean_absolute_correlation']:.2f}")
    print(f"  Pairs with |r| > 0.5: {corr_stats['pairs_above_0.5']}")
    print(f"  Pairs with |r| > 0.8: {corr_stats['pairs_above_0.8']}")

    print(f"\n[Dataset]")
    print(f"  Test set size: {splits['test']} samples")
    print(f"  Total samples: {sum(splits.values())}")

    print(f"\n[Table 6 - Component Ablation]")
    if 'baseline' in ablation_results:
        print(f"  Full Model: {ablation_results['baseline']['token_acc_pct']:.2f}%")
    if 'no_cross_attention' in ablation_results:
        print(f"  -Cross-Attention: {ablation_results['no_cross_attention']['delta']:+.1f}%")
    if 'no_positional' in ablation_results:
        print(f"  -Positional Encoding: {ablation_results['no_positional']['delta']:+.1f}%")
    if 'no_operation_cond' in ablation_results:
        print(f"  -Operation Conditioning: {ablation_results['no_operation_cond']['delta']:+.1f}%")


if __name__ == '__main__':
    main()
