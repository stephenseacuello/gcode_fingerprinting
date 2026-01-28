#!/usr/bin/env python3
"""
Run Individual Modality Ablation Studies for G-code Fingerprinting.

This script runs ablation studies for INDIVIDUAL modalities:
- Leave-one-out: Remove each individual modality and measure performance drop
- Leave-one-in: Use only single modality and measure performance

Individual Modalities (20 total):

  Sensor Modalities (17) - each from 12 sensors:
    - Ax, Ay, Az (accelerometer axes)
    - Gx, Gy, Gz (gyroscope axes)
    - Mx, My, Mz (magnetometer axes)
    - Pressure, Temperature, Proximity (environmental)
    - ColorR, ColorG, ColorB, ColorA (color)
    - RMS (computed)

  Machine Feature Groups (3) - treated as individual groups:
    - Motor Electrical: current/voltage from 4 motors (8 features)
    - Positions: X/Y/Z position, velocity, spindle speed (7 features)
    - Controller: feed rate, tool number, program state, etc. (9 features)

Usage:
    python scripts/experiments/run_individual_modality_ablations.py \
        --output-dir outputs/individual_modality_ablations \
        --config outputs/best_config.json \
        --data-dir outputs/processed_v3 \
        --encoder-path outputs/encoder/best_model.pt \
        --vocab-path data/vocabulary_4digit_full.json

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import platform
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

# Windows and macOS have issues with multiprocessing DataLoader workers
NUM_WORKERS = 0 if platform.system() in ('Windows', 'Darwin') else 4

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# =============================================================================
# INDIVIDUAL MODALITY DEFINITIONS
# =============================================================================

# Individual sensor modalities (sensor_id.modality format)
SENSOR_MODALITIES = [
    'Ax', 'Ay', 'Az',           # Accelerometer
    'Gx', 'Gy', 'Gz',           # Gyroscope
    'Mx', 'My', 'Mz',           # Magnetometer
    'Pressure', 'Temperature', 'Proximity',  # Environmental
    'ColorR', 'ColorG', 'ColorB', 'ColorA',  # Color
    'RMS',                       # Computed
]

# Machine feature column name patterns (treated as groups since they're not per-sensor)
MACHINE_FEATURE_PATTERNS = {
    'motor_electrical': [
        'spindle', 'spindle_a', 'x_motor', 'x_motor_a',
        'y_motor', 'y_motor_a', 'z_motor', 'z_motor_a',
    ],
    'positions': [
        'posx', 'posy', 'posz', 'mpox', 'mpoy', 'mpoz', 'line',
    ],
    'controller': [
        'coor', 'dist', 'feed', 'gcode_line', 'momo', 'plane', 'stat', 'unit', 'vel',
    ],
}

# All individual modalities (17 sensor + 3 machine groups = 20)
ALL_INDIVIDUAL_MODALITIES = SENSOR_MODALITIES + list(MACHINE_FEATURE_PATTERNS.keys())


def get_individual_modality_mask(
    master_columns: List[str],
    modalities_to_ablate: Optional[List[str]] = None,
    modalities_to_keep: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Create a mask for individual modality ablation.

    Args:
        master_columns: List of all column names
        modalities_to_ablate: If set, mask these modalities (leave-one-out)
        modalities_to_keep: If set, only keep these modalities (leave-one-in)

    Returns:
        Binary mask [D] where 1 means keep, 0 means mask
    """
    mask = np.ones(len(master_columns), dtype=np.float32)

    for i, col in enumerate(master_columns):
        col_lower = col.lower()
        matched_modality = None

        # Check sensor modalities (format: sensor_id.modality)
        parts = col.split('.')
        if len(parts) >= 2:
            modality = parts[1]
            if modality in SENSOR_MODALITIES:
                matched_modality = modality

        # Check machine features (column name patterns)
        if matched_modality is None:
            for group_name, patterns in MACHINE_FEATURE_PATTERNS.items():
                for pattern in patterns:
                    if col_lower == pattern or col_lower.startswith(pattern):
                        matched_modality = group_name
                        break
                if matched_modality:
                    break

        # Apply mask based on ablation type
        if matched_modality:
            if modalities_to_ablate is not None:
                if matched_modality in modalities_to_ablate:
                    mask[i] = 0.0
            elif modalities_to_keep is not None:
                if matched_modality not in modalities_to_keep:
                    mask[i] = 0.0

    return mask


class MaskedDataset:
    """Wrapper dataset that applies a mask to the features."""

    def __init__(self, base_dataset, mask: np.ndarray):
        self.base_dataset = base_dataset
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['sensor_features'] = item['sensor_features'] * self.mask
        return item


def evaluate_model(
    encoder: nn.Module,
    decoder: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
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


def train_ablation_model(
    config: dict,
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    feature_mask: np.ndarray,
    ablation_name: str,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict:
    """Train a model with feature ablation."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    print(f"\n{'='*60}")
    print(f"Training: {ablation_name}")
    print(f"Mask sum: {feature_mask.sum():.0f}/{len(feature_mask)} features active")
    print(f"{'='*60}")

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load metadata to get input_dim
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 296)
    else:
        input_dim = config.get('sensor_input_dim', 296)

    max_seq_len = config.get('max_seq_len', 32)
    train_base = DecoderDatasetFromSplits(
        split_dir=data_dir, split='train', max_token_len=max_seq_len,
    )
    val_base = DecoderDatasetFromSplits(
        split_dir=data_dir, split='val', max_token_len=max_seq_len,
    )
    test_base = DecoderDatasetFromSplits(
        split_dir=data_dir, split='test', max_token_len=max_seq_len,
    )

    train_dataset = MaskedDataset(train_base, feature_mask)
    val_dataset = MaskedDataset(val_base, feature_mask)
    test_dataset = MaskedDataset(test_base, feature_mask)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS
    )

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
    d_model = config['d_model']
    n_heads = config['n_heads']
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        sensor_dim=128,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config['n_layers'],
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_model * config.get('ffn_multiplier', 4),
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config.get('embed_dropout', 0.1),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_sensor_prior=config.get('use_sensor_prior', True),
        sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
    )
    decoder.to(device)

    optimizer = AdamW(
        decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('restart_period', 20),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config.get('label_smoothing', 0.1)
    )

    best_val_acc = 0.0
    patience_counter = 0
    ablation_output_dir = Path(output_dir) / ablation_name
    ablation_output_dir.mkdir(parents=True, exist_ok=True)

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
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()

        scheduler.step()

        val_metrics = evaluate_model(encoder, decoder, val_loader, device)
        val_acc = val_metrics['token_accuracy']

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: val_token_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'val_acc': val_acc,
                'ablation_name': ablation_name,
            }, ablation_output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    decoder.load_state_dict(
        torch.load(ablation_output_dir / 'best_model.pt', weights_only=False)['model_state_dict']
    )
    test_metrics = evaluate_model(encoder, decoder, test_loader, device)

    results = {
        'ablation_name': ablation_name,
        'best_val_token_acc': best_val_acc,
        'test_token_acc': test_metrics['token_accuracy'],
        'test_sequence_acc': test_metrics['sequence_accuracy'],
        'features_active': int(feature_mask.sum()),
        'features_total': len(feature_mask),
    }

    with open(ablation_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Final: val={best_val_acc:.4f}, test={test_metrics['token_accuracy']:.4f}")
    return results


def run_individual_ablations(
    config: dict,
    master_columns: List[str],
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
    ablation_type: str = 'both',
) -> Dict[str, Dict]:
    """Run ablations for individual modalities."""
    results = {}

    # Baseline (all features)
    baseline_mask = np.ones(len(master_columns), dtype=np.float32)
    results['baseline'] = train_ablation_model(
        config=config,
        data_dir=data_dir,
        encoder_path=encoder_path,
        vocab_path=vocab_path,
        output_dir=output_dir,
        feature_mask=baseline_mask,
        ablation_name='baseline_all_features',
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )

    # Leave-one-out for each individual modality
    if ablation_type in ['both', 'leave_one_out']:
        print("\n" + "=" * 70)
        print("LEAVE-ONE-OUT: Removing each individual modality")
        print("=" * 70)
        for modality in ALL_INDIVIDUAL_MODALITIES:
            mask = get_individual_modality_mask(master_columns, modalities_to_ablate=[modality])
            results[f'without_{modality}'] = train_ablation_model(
                config=config,
                data_dir=data_dir,
                encoder_path=encoder_path,
                vocab_path=vocab_path,
                output_dir=output_dir,
                feature_mask=mask,
                ablation_name=f'without_{modality}',
                device=device,
                max_epochs=max_epochs,
                patience=patience,
            )

    # Leave-one-in for each individual modality
    if ablation_type in ['both', 'leave_one_in']:
        print("\n" + "=" * 70)
        print("LEAVE-ONE-IN: Using only each individual modality")
        print("=" * 70)
        for modality in ALL_INDIVIDUAL_MODALITIES:
            mask = get_individual_modality_mask(master_columns, modalities_to_keep=[modality])
            results[f'only_{modality}'] = train_ablation_model(
                config=config,
                data_dir=data_dir,
                encoder_path=encoder_path,
                vocab_path=vocab_path,
                output_dir=output_dir,
                feature_mask=mask,
                ablation_name=f'only_{modality}',
                device=device,
                max_epochs=max_epochs,
                patience=patience,
            )

    return results


def generate_report(all_results: Dict, output_dir: str):
    """Generate comprehensive ablation report."""
    output_path = Path(output_dir)

    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'sensor_modalities': SENSOR_MODALITIES,
            'machine_feature_groups': list(MACHINE_FEATURE_PATTERNS.keys()),
            'all_individual_modalities': ALL_INDIVIDUAL_MODALITIES,
        },
        'results': all_results,
    }

    # Compute importance scores
    baseline_acc = all_results.get('baseline', {}).get('test_token_acc', 0)

    # Individual importance (leave-one-out)
    modality_importance_loo = {}
    for modality in ALL_INDIVIDUAL_MODALITIES:
        key = f'without_{modality}'
        if key in all_results:
            ablated_acc = all_results[key]['test_token_acc']
            modality_importance_loo[modality] = baseline_acc - ablated_acc

    # Individual importance (leave-one-in)
    modality_importance_loi = {}
    for modality in ALL_INDIVIDUAL_MODALITIES:
        key = f'only_{modality}'
        if key in all_results:
            modality_importance_loi[modality] = all_results[key]['test_token_acc']

    report['importance'] = {
        'leave_one_out': modality_importance_loo,
        'leave_one_in': modality_importance_loi,
    }

    # Save report
    with open(output_path / 'individual_modality_ablation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODALITY ABLATION SUMMARY")
    print("=" * 80)

    print(f"\nBaseline (all features): {baseline_acc*100:.2f}%")

    if modality_importance_loo:
        print("\n--- LEAVE-ONE-OUT (importance = accuracy drop when removed) ---")
        print("Top 10 most important:")
        sorted_loo = sorted(modality_importance_loo.items(), key=lambda x: -x[1])[:10]
        for modality, importance in sorted_loo:
            sign = "+" if importance < 0 else "-"
            print(f"  {modality:20s}: {sign}{abs(importance)*100:.2f}%")

    if modality_importance_loi:
        print("\n--- LEAVE-ONE-IN (accuracy with only this modality) ---")
        print("Top 10 best performing:")
        sorted_loi = sorted(modality_importance_loi.items(), key=lambda x: -x[1])[:10]
        for modality, acc in sorted_loi:
            print(f"  {modality:20s}: {acc*100:.2f}%")

    print(f"\nFull report saved to: {output_path / 'individual_modality_ablation_report.json'}")


def main():
    parser = argparse.ArgumentParser(description='Run individual modality ablation studies')
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data splits')
    parser.add_argument('--encoder-path', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--ablation-type', type=str, default='both',
                        choices=['both', 'leave_one_out', 'leave_one_in'],
                        help='Type of ablation to run')
    parser.add_argument('--max-epochs', type=int, default=100, help='Max epochs per ablation')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detected)')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")

    with open(args.config) as f:
        config = json.load(f)

    # Load master columns from metadata
    metadata_path = Path(args.data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        master_columns = metadata.get('continuous_columns', metadata.get('master_columns', []))
        print(f"Loaded {len(master_columns)} columns from metadata")
    else:
        raise FileNotFoundError(f"metadata.json not found in {args.data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ablations
    all_results = run_individual_ablations(
        config=config,
        master_columns=master_columns,
        data_dir=args.data_dir,
        encoder_path=args.encoder_path,
        vocab_path=args.vocab_path,
        output_dir=str(output_dir),
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        ablation_type=args.ablation_type,
    )

    # Generate report
    generate_report(all_results, str(output_dir))

    print(f"\n{'='*60}")
    print("INDIVIDUAL MODALITY ABLATION STUDY COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
