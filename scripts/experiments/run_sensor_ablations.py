#!/usr/bin/env python3
"""
Run Sensor ID Ablation Studies for G-code Fingerprinting.

This script runs ablation studies to measure the contribution of each sensor:
- Leave-one-out ablation: Remove each sensor and measure performance drop
- Leave-one-in ablation: Use only single sensor and measure performance

12 Sensor IDs:
  xa_motor, frame_r1, frame_r2, frame_b2, frame_l3, frame_l2,
  spindle1, spindle2, y_bed__1, y_bed__2, y_bed__3, y_bed__4

Each sensor has 17 modalities (Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz,
Pressure, Temperature, Proximity, ColorR, ColorG, ColorB, ColorA, RMS)

Usage:
    python scripts/experiments/run_sensor_ablations.py \
        --output-dir outputs/sensor_ablations \
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
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# Sensor IDs from the dataset
SENSOR_IDS = [
    'xa_motor', 'frame_r1', 'frame_r2', 'frame_b2',
    'frame_l3', 'frame_l2', 'spindle1', 'spindle2',
    'y_bed__1', 'y_bed__2', 'y_bed__3', 'y_bed__4',
]

# Modalities per sensor
MODALITIES = [
    'Ax', 'Ay', 'Az',          # Accelerometer
    'Gx', 'Gy', 'Gz',          # Gyroscope
    'Mx', 'My', 'Mz',          # Magnetometer
    'Pressure', 'Temperature', 'Proximity',  # Environmental
    'ColorR', 'ColorG', 'ColorB', 'ColorA',  # Color
    'RMS',                     # Computed
]


def get_sensor_mask(
    master_columns: List[str],
    sensor_to_ablate: Optional[str] = None,
    sensors_to_keep: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Create a mask for sensor ablation.

    Args:
        master_columns: List of all column names
        sensor_to_ablate: If set, mask this sensor (leave-one-out)
        sensors_to_keep: If set, only keep these sensors (leave-one-in)

    Returns:
        Binary mask [D] where 1 means keep, 0 means mask
    """
    mask = np.ones(len(master_columns), dtype=np.float32)

    for i, col in enumerate(master_columns):
        # Parse sensor ID from column name (format: sensor_id.modality)
        parts = col.split('.')
        if len(parts) >= 2:
            sensor_id = parts[0]

            if sensor_to_ablate is not None:
                # Leave-one-out: mask this sensor
                if sensor_id == sensor_to_ablate:
                    mask[i] = 0.0

            elif sensors_to_keep is not None:
                # Leave-one-in: mask all except these sensors
                if sensor_id not in sensors_to_keep:
                    mask[i] = 0.0

    return mask


class MaskedDataset:
    """Wrapper dataset that applies a sensor mask to the features."""

    def __init__(self, base_dataset, mask: np.ndarray):
        self.base_dataset = base_dataset
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # Apply mask to sensor features
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

            # Encode
            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']

            # Decode
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)

            # Token accuracy
            mask = target_tokens != 0
            total_correct += ((preds == target_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Sequence accuracy
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
    sensor_mask: np.ndarray,
    ablation_name: str,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict:
    """Train a model with sensor ablation."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    print(f"\n{'='*60}")
    print(f"Training: {ablation_name}")
    print(f"Mask sum: {sensor_mask.sum():.0f}/{len(sensor_mask)} features active")
    print(f"{'='*60}")

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load datasets with mask applied
    max_seq_len = config.get('max_seq_len', 32)
    train_base = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='train',
        max_token_len=max_seq_len,
    )
    val_base = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='val',
        max_token_len=max_seq_len,
    )
    test_base = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='test',
        max_token_len=max_seq_len,
    )

    train_dataset = MaskedDataset(train_base, sensor_mask)
    val_dataset = MaskedDataset(val_base, sensor_mask)
    test_dataset = MaskedDataset(test_base, sensor_mask)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=2
    )

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=config.get('sensor_input_dim', 155),
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
        pooling_n_heads=config.get('pooling_n_heads', 8),
        pooling_n_queries=config.get('pooling_n_queries', 16),
    )

    # Load pretrained encoder
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
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

    # Optimizer and scheduler
    optimizer = AdamW(
        decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('restart_period', 20),
    )

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config.get('label_smoothing', 0.1)
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    ablation_output_dir = Path(output_dir) / ablation_name
    ablation_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        # Training
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

        # Validation
        val_metrics = evaluate_model(encoder, decoder, val_loader, device)
        val_acc = val_metrics['token_accuracy']

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: val_token_acc={val_acc:.4f}")

        # Early stopping and checkpointing
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

    # Evaluate on test set
    decoder.load_state_dict(
        torch.load(ablation_output_dir / 'best_model.pt', weights_only=False)['model_state_dict']
    )
    test_metrics = evaluate_model(encoder, decoder, test_loader, device)

    results = {
        'ablation_name': ablation_name,
        'best_val_token_acc': best_val_acc,
        'test_token_acc': test_metrics['token_accuracy'],
        'test_sequence_acc': test_metrics['sequence_accuracy'],
        'features_active': int(sensor_mask.sum()),
        'features_total': len(sensor_mask),
    }

    # Save results
    with open(ablation_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Final: val={best_val_acc:.4f}, test={test_metrics['token_accuracy']:.4f}")

    return results


def run_leave_one_out_ablations(
    config: dict,
    master_columns: List[str],
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict[str, Dict]:
    """Run leave-one-out ablations for each sensor."""
    results = {}

    # First train baseline (all sensors)
    baseline_mask = np.ones(len(master_columns), dtype=np.float32)
    results['baseline'] = train_ablation_model(
        config=config,
        data_dir=data_dir,
        encoder_path=encoder_path,
        vocab_path=vocab_path,
        output_dir=output_dir,
        sensor_mask=baseline_mask,
        ablation_name='baseline_all_sensors',
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )

    # Leave-one-out for each sensor
    for sensor_id in SENSOR_IDS:
        mask = get_sensor_mask(master_columns, sensor_to_ablate=sensor_id)
        results[f'without_{sensor_id}'] = train_ablation_model(
            config=config,
            data_dir=data_dir,
            encoder_path=encoder_path,
            vocab_path=vocab_path,
            output_dir=output_dir,
            sensor_mask=mask,
            ablation_name=f'without_{sensor_id}',
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )

    return results


def run_leave_one_in_ablations(
    config: dict,
    master_columns: List[str],
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict[str, Dict]:
    """Run leave-one-in ablations (single sensor only)."""
    results = {}

    for sensor_id in SENSOR_IDS:
        mask = get_sensor_mask(master_columns, sensors_to_keep=[sensor_id])
        results[f'only_{sensor_id}'] = train_ablation_model(
            config=config,
            data_dir=data_dir,
            encoder_path=encoder_path,
            vocab_path=vocab_path,
            output_dir=output_dir,
            sensor_mask=mask,
            ablation_name=f'only_{sensor_id}',
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )

    return results


def generate_sensor_ablation_report(all_results: Dict, output_dir: str):
    """Generate comprehensive ablation report."""
    output_path = Path(output_dir)

    # Add metadata
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'sensor_ids': SENSOR_IDS,
            'modalities': MODALITIES,
        },
        'results': all_results,
    }

    # Compute importance scores
    baseline_acc = all_results.get('baseline', {}).get('test_token_acc', 0)

    importance_leave_one_out = {}
    for sensor_id in SENSOR_IDS:
        key = f'without_{sensor_id}'
        if key in all_results:
            ablated_acc = all_results[key]['test_token_acc']
            # Importance = how much performance drops when removed
            importance_leave_one_out[sensor_id] = baseline_acc - ablated_acc

    importance_leave_one_in = {}
    for sensor_id in SENSOR_IDS:
        key = f'only_{sensor_id}'
        if key in all_results:
            # Importance = how much accuracy when only this sensor is used
            importance_leave_one_in[sensor_id] = all_results[key]['test_token_acc']

    report['importance'] = {
        'leave_one_out': importance_leave_one_out,
        'leave_one_in': importance_leave_one_in,
    }

    # Save full report
    with open(output_path / 'sensor_ablation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("SENSOR ABLATION SUMMARY")
    print("=" * 80)

    print(f"\nBaseline (all sensors): {baseline_acc*100:.2f}%")

    print("\n--- LEAVE-ONE-OUT (importance = accuracy drop when removed) ---")
    sorted_loo = sorted(importance_leave_one_out.items(), key=lambda x: -x[1])
    for sensor_id, importance in sorted_loo:
        sign = "+" if importance < 0 else "-"
        print(f"  {sensor_id:12s}: {sign}{abs(importance)*100:.2f}%")

    if importance_leave_one_in:
        print("\n--- LEAVE-ONE-IN (accuracy with only this sensor) ---")
        sorted_loi = sorted(importance_leave_one_in.items(), key=lambda x: -x[1])
        for sensor_id, acc in sorted_loi:
            print(f"  {sensor_id:12s}: {acc*100:.2f}%")

    # Find most/least important sensors
    if importance_leave_one_out:
        most_important = max(importance_leave_one_out.items(), key=lambda x: x[1])
        least_important = min(importance_leave_one_out.items(), key=lambda x: x[1])
        print(f"\nMost important sensor:  {most_important[0]} (drop: {most_important[1]*100:.2f}%)")
        print(f"Least important sensor: {least_important[0]} (drop: {least_important[1]*100:.2f}%)")

    print(f"\nFull report saved to: {output_path / 'sensor_ablation_report.json'}")


def main():
    parser = argparse.ArgumentParser(description='Run sensor ID ablation studies')
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data splits')
    parser.add_argument('--encoder-path', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--master-columns', type=str, help='Path to master columns JSON (optional, auto-detected from data)')
    parser.add_argument('--ablation-type', type=str, default='both',
                        choices=['leave_one_out', 'leave_one_in', 'both'],
                        help='Type of ablation to run')
    parser.add_argument('--max-epochs', type=int, default=100, help='Max epochs per ablation')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Load master columns - try multiple sources
    master_columns = None

    # 1. Try explicit path
    if args.master_columns and os.path.exists(args.master_columns):
        with open(args.master_columns) as f:
            master_columns = json.load(f)
        print(f"Loaded master columns from: {args.master_columns}")

    # 2. Try metadata file in data directory
    if master_columns is None:
        metadata_path = Path(args.data_dir) / 'train_sequences_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            if 'master_columns' in metadata:
                master_columns = metadata['master_columns']
                print(f"Loaded master columns from metadata: {len(master_columns)} features")

    # 3. Fallback to default
    if master_columns is None:
        master_columns = []
        for sensor_id in SENSOR_IDS:
            for modality in MODALITIES:
                master_columns.append(f'{sensor_id}.{modality}')
        print(f"Using default column list: {len(master_columns)} features")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run ablations
    if args.ablation_type in ['leave_one_out', 'both']:
        loo_results = run_leave_one_out_ablations(
            config=config,
            master_columns=master_columns,
            data_dir=args.data_dir,
            encoder_path=args.encoder_path,
            vocab_path=args.vocab_path,
            output_dir=str(output_dir),
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        all_results.update(loo_results)

    if args.ablation_type in ['leave_one_in', 'both']:
        loi_results = run_leave_one_in_ablations(
            config=config,
            master_columns=master_columns,
            data_dir=args.data_dir,
            encoder_path=args.encoder_path,
            vocab_path=args.vocab_path,
            output_dir=str(output_dir),
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        all_results.update(loi_results)

    # Generate report
    generate_sensor_ablation_report(all_results, str(output_dir))

    print(f"\n{'='*60}")
    print("SENSOR ABLATION STUDY COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
