#!/usr/bin/env python3
"""
One-Way ANOVA for Modality/Feature Group Ablation Studies.

Trains models for each of 10 feature groups with multiple seeds,
then runs one-way ANOVA to test if modality differences are statistically significant.

Groups (10):
  Sensor modalities (6):
    - Accelerometer: Ax, Ay, Az across 12 operational sensors (36 features)
    - Gyroscope: Gx, Gy, Gz across 12 sensors (36 features)
    - Magnetometer: Mx, My, Mz across 12 sensors (36 features)
    - Environmental: Pressure, Temperature, Proximity across 12 sensors (36 features)
    - Color: ColorR, ColorG, ColorB, ColorA across 12 sensors (48 features)
    - RMS: RMS across 12 sensors (12 features)

  Machine features (3):
    - Motor electrical: spindle, x/y/z motor voltage + current (8 features)
    - Positions: desired + measured XYZ + line (7 features)
    - Controller: status, units, feed, plane, etc. (9 features)

  Baseline (1):
    - All features (296 total)

Seeds: 3 per group (default: 42, 123, 456)
Total runs: 10 × 3 = 30

Usage:
    python scripts/experiments/run_modality_anova.py \
        --config configs/decoder_config.json \
        --data-dir outputs/jan23_followup/no_leakage \
        --encoder-path outputs/jan26/ensemble/seed_42/best_model.pt \
        --vocab-path data/vocabulary_4digit_full.json \
        --output-dir outputs/anova/modalities

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import platform

NUM_WORKERS = 0 if platform.system() in ('Windows', 'Darwin') else 4

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from torch.utils.data import DataLoader
from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# 12 operational sensors
OPERATIONAL_SENSORS = [
    'xa_motor', 'frame_r1', 'frame_r2', 'frame_b2',
    'frame_l3', 'frame_l2', 'spindle1', 'spindle2',
    'y_bed__1', 'y_bed__2', 'y_bed__3', 'y_bed__4',
]

# Grouped sensor modalities
MODALITY_GROUPS = {
    'accelerometer': ['Ax', 'Ay', 'Az'],
    'gyroscope': ['Gx', 'Gy', 'Gz'],
    'magnetometer': ['Mx', 'My', 'Mz'],
    'environmental': ['Pressure', 'Temperature', 'Proximity'],
    'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
    'rms': ['RMS'],
}

# Machine feature groups
MACHINE_GROUPS = {
    'motor_electrical': [
        'spindle', 'spindle_A', 'x_motor', 'x_motor_A',
        'y_motor', 'y_motor_A', 'z_motor', 'z_motor_A',
    ],
    'positions': ['posx', 'posy', 'posz', 'mpox', 'mpoy', 'mpoz', 'line'],
    'controller': ['coor', 'dist', 'feed', 'gcode_line', 'momo', 'plane', 'stat', 'unit', 'vel'],
}

# All machine features combined (for single-modality runs that need machine context)
ALL_MACHINE_FEATURES = (
    MACHINE_GROUPS['motor_electrical'] +
    MACHINE_GROUPS['positions'] +
    MACHINE_GROUPS['controller']
)


def get_modality_group_mask(
    master_columns: List[str],
    modality_group: str,
    include_machine_features: bool = True,
) -> np.ndarray:
    """Create mask for a sensor modality group (e.g., accelerometer across all sensors)."""
    mask = np.zeros(len(master_columns), dtype=np.float32)

    if modality_group in MODALITY_GROUPS:
        # Sensor modality group
        modalities = MODALITY_GROUPS[modality_group]
        for i, col in enumerate(master_columns):
            # Check if this is a sensor modality column
            for sensor in OPERATIONAL_SENSORS:
                for mod in modalities:
                    if col == f'{sensor}.{mod}':
                        mask[i] = 1.0

            # Include machine features if requested
            if include_machine_features and col in ALL_MACHINE_FEATURES:
                mask[i] = 1.0

    elif modality_group in MACHINE_GROUPS:
        # Machine feature group only
        features = MACHINE_GROUPS[modality_group]
        for i, col in enumerate(master_columns):
            if col in features:
                mask[i] = 1.0

    return mask


def get_all_features_mask(master_columns: List[str]) -> np.ndarray:
    """Create mask that keeps all features (baseline)."""
    return np.ones(len(master_columns), dtype=np.float32)


class MaskedDataset:
    """Wrapper dataset that applies a feature mask."""

    def __init__(self, base_dataset, mask: np.ndarray):
        self.base_dataset = base_dataset
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['sensor_features'] = item['sensor_features'] * self.mask
        return item


def evaluate_model(encoder, decoder, data_loader, device):
    """Evaluate model accuracy."""
    encoder.eval()
    decoder.eval()

    total_correct = 0
    total_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    with torch.no_grad():
        for batch in data_loader:
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
            total_correct += ((preds == target_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()

            for i in range(preds.size(0)):
                seq_mask = mask[i]
                if (preds[i][seq_mask] == target_tokens[i][seq_mask]).all():
                    correct_sequences += 1
                total_sequences += 1

    return {
        'token_accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'sequence_accuracy': correct_sequences / total_sequences if total_sequences > 0 else 0,
    }


def train_single_config(
    config: dict,
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    feature_mask: np.ndarray,
    config_name: str,
    device: str,
    seed: int,
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict:
    """Train a single configuration with a specific seed."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    print(f"\n  Training: {config_name} (seed={seed})")
    print(f"    Features active: {feature_mask.sum():.0f}/{len(feature_mask)}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    input_dim = metadata.get('n_continuous_features', 296)

    # Load datasets
    max_seq_len = config.get('max_seq_len', 32)
    train_base = DecoderDatasetFromSplits(data_dir, 'train', max_token_len=max_seq_len)
    val_base = DecoderDatasetFromSplits(data_dir, 'val', max_token_len=max_seq_len)
    test_base = DecoderDatasetFromSplits(data_dir, 'test', max_token_len=max_seq_len)

    train_dataset = MaskedDataset(train_base, feature_mask)
    val_dataset = MaskedDataset(val_base, feature_mask)
    test_dataset = MaskedDataset(test_base, feature_mask)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS)

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
    )
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        encoder.load_state_dict(ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', {})), strict=False)
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
        sensor_dim=128,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config.get('n_layers', 4),
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_model * config.get('ffn_multiplier', 4),
        dropout=config.get('dropout', 0.3),
        max_seq_len=max_seq_len,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config.get('embed_dropout', 0.1),
    )
    decoder.to(device)

    optimizer = AdamW(decoder.parameters(), lr=config.get('learning_rate', 1e-4),
                      weight_decay=config.get('weight_decay', 0.01))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.get('restart_period', 20))
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.get('label_smoothing', 0.1))

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    run_output_dir = Path(output_dir) / config_name / f'seed_{seed}'
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        decoder.train()
        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            with torch.no_grad():
                encoder_out = encoder(sensor_data)
                sensor_memory = encoder_out['memory']

            optimizer.zero_grad()
            outputs = decoder(tokens=input_tokens, sensor_embeddings=sensor_memory, operation_type=operations)
            logits = outputs['legacy_logits']
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()

        scheduler.step()

        # Validation
        val_metrics = evaluate_model(encoder, decoder, val_loader, device)
        val_acc = val_metrics['token_accuracy']

        if epoch % 20 == 0:
            print(f"      Epoch {epoch}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'val_acc': val_acc,
            }, run_output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch}")
                break

    # Test evaluation
    decoder.load_state_dict(torch.load(run_output_dir / 'best_model.pt', weights_only=False)['model_state_dict'])
    test_metrics = evaluate_model(encoder, decoder, test_loader, device)

    result = {
        'config_name': config_name,
        'seed': seed,
        'best_val_acc': best_val_acc,
        'test_token_acc': test_metrics['token_accuracy'],
        'test_seq_acc': test_metrics['sequence_accuracy'],
        'features_active': int(feature_mask.sum()),
    }

    with open(run_output_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"      Final: val={best_val_acc:.4f}, test={test_metrics['token_accuracy']:.4f}")
    return result


def run_anova(results_by_group: Dict[str, List[float]], output_dir: Path):
    """Run one-way ANOVA and post-hoc tests."""
    from scipy import stats

    print("\n" + "=" * 70)
    print("ONE-WAY ANOVA: MODALITY/FEATURE GROUP COMPARISON")
    print("=" * 70)

    # Prepare data for ANOVA
    groups = list(results_by_group.keys())
    data = [results_by_group[g] for g in groups]

    # Run one-way ANOVA
    f_stat, p_value = stats.f_oneway(*data)

    print(f"\nANOVA Results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  Significant at α=0.01: {'YES' if p_value < 0.01 else 'NO'}")

    # Descriptive statistics
    print(f"\nGroup Statistics:")
    print(f"  {'Group':<20} {'Mean':>10} {'Std':>10} {'N':>5}")
    print("-" * 50)

    # Sort by mean accuracy
    sorted_groups = sorted(groups, key=lambda g: -np.mean(results_by_group[g]))
    for group in sorted_groups:
        vals = results_by_group[group]
        print(f"  {group:<20} {np.mean(vals)*100:>9.2f}% {np.std(vals)*100:>9.2f}% {len(vals):>5}")

    # Post-hoc: Tukey HSD (if significant)
    posthoc_results = None
    if p_value < 0.05:
        print("\n--- Post-hoc: Tukey HSD ---")
        try:
            from scipy.stats import tukey_hsd
            res = tukey_hsd(*data)
            posthoc_results = {
                'statistic': res.statistic.tolist(),
                'pvalue': res.pvalue.tolist(),
                'groups': groups,
            }

            # Find significant pairwise differences
            sig_pairs = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if res.pvalue[i, j] < 0.05:
                        diff = np.mean(results_by_group[groups[i]]) - np.mean(results_by_group[groups[j]])
                        sig_pairs.append((groups[i], groups[j], diff, res.pvalue[i, j]))

            if sig_pairs:
                print(f"\n  Significant pairwise differences (p < 0.05):")
                sig_pairs.sort(key=lambda x: -abs(x[2]))
                for g1, g2, diff, pval in sig_pairs[:15]:  # Top 15
                    print(f"    {g1} vs {g2}: diff={diff*100:+.2f}%, p={pval:.4f}")
            else:
                print("  No significant pairwise differences found.")
        except ImportError:
            print("  (tukey_hsd requires scipy >= 1.8)")

    # Save ANOVA results
    anova_results = {
        'test': 'one-way ANOVA',
        'groups': groups,
        'n_groups': len(groups),
        'n_per_group': [len(results_by_group[g]) for g in groups],
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'group_means': {g: float(np.mean(results_by_group[g])) for g in groups},
        'group_stds': {g: float(np.std(results_by_group[g])) for g in groups},
        'ranking': sorted_groups,
        'posthoc': posthoc_results,
    }

    with open(output_dir / 'anova_results.json', 'w') as f:
        json.dump(anova_results, f, indent=2)

    print(f"\nANOVA results saved to: {output_dir / 'anova_results.json'}")
    return anova_results


def main():
    parser = argparse.ArgumentParser(description='One-way ANOVA for modality/feature group comparison')
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data splits')
    parser.add_argument('--encoder-path', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated seeds')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--include-machine-features', action='store_true', default=True,
                        help='Include machine features when testing sensor modalities')
    args = parser.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Device: {args.device}")

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Load master columns
    metadata_path = Path(args.data_dir) / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    master_columns = metadata.get('continuous_columns', [])
    print(f"Features: {len(master_columns)}")

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Seeds: {seeds} ({len(seeds)} per group)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build configurations
    configs = {}

    # 6 sensor modality groups
    for group_name in MODALITY_GROUPS.keys():
        mask = get_modality_group_mask(
            master_columns, group_name,
            include_machine_features=args.include_machine_features
        )
        configs[f'mod_{group_name}'] = mask
        print(f"  {group_name}: {mask.sum():.0f} features")

    # 3 machine feature groups
    for group_name in MACHINE_GROUPS.keys():
        mask = get_modality_group_mask(master_columns, group_name, include_machine_features=False)
        configs[f'machine_{group_name}'] = mask
        print(f"  {group_name}: {mask.sum():.0f} features")

    # Baseline: all features
    configs['all_features'] = get_all_features_mask(master_columns)
    print(f"  all_features: {configs['all_features'].sum():.0f} features")

    print(f"\nConfigurations to run: {len(configs)}")
    print(f"Total training runs: {len(configs)} × {len(seeds)} = {len(configs) * len(seeds)}")

    # Train all configurations
    results_by_group = defaultdict(list)
    all_results = []

    for config_name, mask in configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name} ({mask.sum():.0f} features)")
        print(f"{'='*60}")

        for seed in seeds:
            result = train_single_config(
                config=config,
                data_dir=args.data_dir,
                encoder_path=args.encoder_path,
                vocab_path=args.vocab_path,
                output_dir=str(output_dir),
                feature_mask=mask,
                config_name=config_name,
                device=args.device,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            results_by_group[config_name].append(result['test_token_acc'])
            all_results.append(result)

    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump({
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'seeds': seeds,
                'n_configs': len(configs),
                'configs': list(configs.keys()),
                'modality_groups': {k: v for k, v in MODALITY_GROUPS.items()},
                'machine_groups': {k: v for k, v in MACHINE_GROUPS.items()},
            },
            'results': all_results,
            'results_by_group': {k: v for k, v in results_by_group.items()},
        }, f, indent=2)

    # Run ANOVA
    run_anova(results_by_group, output_dir)

    print(f"\n{'='*60}")
    print("MODALITY ANOVA COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
