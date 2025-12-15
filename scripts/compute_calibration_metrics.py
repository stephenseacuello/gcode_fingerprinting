#!/usr/bin/env python3
"""
Compute calibration metrics (ECE, Brier Score) for the sensor multihead decoder.

Expected Calibration Error (ECE) measures how well model confidence matches accuracy.
Brier Score measures the accuracy of probabilistic predictions.

Usage:
    PYTHONPATH=src python scripts/compute_calibration_metrics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


# ============================================================================
# MM-DTAE-LSTM Encoder (copied from training script for consistency)
# ============================================================================

class MM_DTAE_LSTM(nn.Module):
    """MM-DTAE-LSTM encoder for sensor feature extraction."""

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_classes: int = 9,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        noise_factor: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.noise_factor = noise_factor
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        encoder_output_dim = hidden_dim * self.num_directions
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=1)
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input sequence to latent representation."""
        x = self.input_proj(x)
        encoded, (h_n, c_n) = self.encoder_lstm(x)
        latent = self.bottleneck(encoded)
        return latent, (h_n, c_n)

    def classify(self, latent: torch.Tensor) -> tuple:
        """Classify operation type from latent."""
        attn_weights = self.temporal_attention(latent)
        pooled = (latent * attn_weights).sum(dim=1)
        logits = self.classification_head(pooled)
        return logits, attn_weights


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|

    Perfect calibration: ECE = 0
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Brier Score for multi-class predictions.

    Brier = (1/N) * sum_i sum_j (p_ij - y_ij)^2

    where p_ij is predicted probability for class j, y_ij is 1 if true class is j.
    Range: [0, 2], lower is better. Perfect prediction: 0
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), targets] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def load_model_and_data(model_dir: Path, split_dir: Path, vocab_path: Path, device):
    """Load trained model and test data."""

    # Load args
    with open(model_dir / 'args.json') as f:
        args = json.load(f)

    # Load vocabulary
    decomposer = TokenDecomposer(str(vocab_path))

    # Load encoder - uses the same structure as training script
    encoder_path = Path(args['encoder_path'])
    encoder_checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    # Create encoder with same config as training
    encoder = MM_DTAE_LSTM(
        input_dim=155,  # Continuous sensor features only
        hidden_dim=256,
        latent_dim=args['sensor_dim'],  # 128
        n_classes=args['n_operations'],  # 9
    ).to(device)

    # Handle checkpoint format
    if isinstance(encoder_checkpoint, dict) and 'model_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_checkpoint)
    encoder.eval()

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        vocab_size=decomposer.vocab_size,
        d_model=args['d_model'],
        n_heads=args['n_heads'],
        n_layers=args['n_layers'],
        sensor_dim=args['sensor_dim'],
        n_operations=args['n_operations'],
        n_types=args['n_types'],
        n_commands=args['n_commands'],
        n_param_types=args['n_param_types'],
        max_int_digits=2,
        n_decimal_digits=4,
        dropout=args['dropout'],
        embed_dropout=args['embed_dropout'],
        max_seq_len=args['max_seq_len'],
    ).to(device)

    # Load decoder weights
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device, weights_only=False)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()

    # Load test data using the same dataset class as training
    test_dataset = DecoderDatasetFromSplits(split_dir, 'test', max_token_len=args['max_seq_len'])

    return encoder, decoder, decomposer, test_dataset, args


def evaluate_calibration(encoder, decoder, decomposer, test_dataset, device, n_bins=15):
    """
    Evaluate calibration metrics across the main structural prediction heads.
    """
    print("Evaluating calibration metrics...")

    # Storage for predictions per head (focus on main structural heads)
    head_results = {
        'type': {'confidences': [], 'correct': []},
        'command': {'confidences': [], 'correct': []},
        'param_type': {'confidences': [], 'correct': []},
    }

    # Also collect for Brier score (need full probability distributions)
    brier_data = {
        'type': {'probs': [], 'targets': []},
        'command': {'probs': [], 'targets': []},
        'param_type': {'probs': [], 'targets': []},
    }

    n_samples = len(test_dataset)

    with torch.no_grad():
        for idx in range(n_samples):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{n_samples}...")

            # Get sample from dataset (already has BOS/EOS handled correctly)
            sample = test_dataset[idx]

            # Prepare inputs
            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)  # [1, T_s, 155]
            input_tokens = sample['input_tokens'].unsqueeze(0).to(device)  # [1, max_len] with BOS prefix
            target_tokens = sample['target_tokens'].unsqueeze(0).to(device)  # [1, max_len] for targets
            op = sample['operation_type'].unsqueeze(0).to(device)
            length = sample['length'].item()

            # Get encoder output: sensor_emb [B, T_s, latent_dim=128]
            sensor_emb, _ = encoder.encode(sensor_features)

            # Decoder forward with teacher forcing - input excludes last position
            # input_tokens is [BOS, t1, t2, ..., PAD], we use all non-pad positions
            decoder_out = decoder(
                tokens=input_tokens[:, :length],  # [BOS, t1, t2, ..., tn-1]
                sensor_embeddings=sensor_emb,
                operation_type=op,
            )

            # Decompose target tokens for ground truth
            decomposed = decomposer.decompose_batch(target_tokens[:, :length])

            # Get targets - these are the actual tokens model should predict
            type_targets = decomposed['type'][0].cpu().numpy()
            command_targets = decomposed['command_id'][0].cpu().numpy()
            param_type_targets = decomposed['param_type_id'][0].cpu().numpy()

            # Process type head
            type_logits = decoder_out['type_logits'][0]  # [T, n_types]
            type_probs = F.softmax(type_logits, dim=-1).cpu().numpy()
            type_preds = type_probs.argmax(axis=-1)
            type_confs = type_probs.max(axis=-1)

            head_results['type']['confidences'].extend(type_confs.tolist())
            head_results['type']['correct'].extend((type_preds == type_targets).astype(float).tolist())
            brier_data['type']['probs'].append(type_probs)
            brier_data['type']['targets'].append(type_targets)

            # Process command head (only for COMMAND tokens, type=1)
            cmd_mask = type_targets == 1
            if cmd_mask.any():
                cmd_logits = decoder_out['command_logits'][0]
                cmd_probs = F.softmax(cmd_logits, dim=-1).cpu().numpy()
                cmd_preds = cmd_probs.argmax(axis=-1)
                cmd_confs = cmd_probs.max(axis=-1)

                head_results['command']['confidences'].extend(cmd_confs[cmd_mask].tolist())
                head_results['command']['correct'].extend((cmd_preds[cmd_mask] == command_targets[cmd_mask]).astype(float).tolist())
                brier_data['command']['probs'].append(cmd_probs[cmd_mask])
                brier_data['command']['targets'].append(command_targets[cmd_mask])

            # Process param_type head (only for PARAM tokens, type=2)
            param_mask = type_targets == 2
            if param_mask.any():
                pt_logits = decoder_out['param_type_logits'][0]
                pt_probs = F.softmax(pt_logits, dim=-1).cpu().numpy()
                pt_preds = pt_probs.argmax(axis=-1)
                pt_confs = pt_probs.max(axis=-1)

                head_results['param_type']['confidences'].extend(pt_confs[param_mask].tolist())
                head_results['param_type']['correct'].extend((pt_preds[param_mask] == param_type_targets[param_mask]).astype(float).tolist())
                brier_data['param_type']['probs'].append(pt_probs[param_mask])
                brier_data['param_type']['targets'].append(param_type_targets[param_mask])

    # Compute ECE and accuracy per head
    calibration_results = {}
    for head_name, data in head_results.items():
        if len(data['confidences']) == 0:
            continue

        confs = np.array(data['confidences'])
        correct = np.array(data['correct'])

        ece = compute_ece(confs, correct, n_bins=n_bins)
        accuracy = correct.mean()
        mean_confidence = confs.mean()

        calibration_results[head_name] = {
            'ece': float(ece),
            'accuracy': float(accuracy),
            'mean_confidence': float(mean_confidence),
            'n_predictions': len(confs),
            'overconfidence': float(mean_confidence - accuracy),  # positive = overconfident
        }

    # Compute Brier scores for heads with full probability data
    for head_name, data in brier_data.items():
        if len(data['probs']) == 0:
            continue
        probs = np.vstack(data['probs'])
        targets = np.concatenate(data['targets'])
        brier = compute_brier_score(probs, targets)
        calibration_results[head_name]['brier_score'] = float(brier)

    return calibration_results


def compute_aggregate_metrics(calibration_results):
    """Compute aggregate calibration metrics."""

    # Weight by number of predictions
    total_predictions = sum(r['n_predictions'] for r in calibration_results.values())

    weighted_ece = sum(
        r['ece'] * r['n_predictions'] / total_predictions
        for r in calibration_results.values()
    )

    weighted_accuracy = sum(
        r['accuracy'] * r['n_predictions'] / total_predictions
        for r in calibration_results.values()
    )

    weighted_confidence = sum(
        r['mean_confidence'] * r['n_predictions'] / total_predictions
        for r in calibration_results.values()
    )

    # Average Brier for heads that have it
    brier_heads = [r for r in calibration_results.values() if 'brier_score' in r]
    avg_brier = np.mean([r['brier_score'] for r in brier_heads]) if brier_heads else None

    return {
        'weighted_ece': weighted_ece,
        'weighted_accuracy': weighted_accuracy,
        'weighted_confidence': weighted_confidence,
        'weighted_overconfidence': weighted_confidence - weighted_accuracy,
        'average_brier_score': avg_brier,
        'total_predictions': total_predictions,
    }


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'outputs' / 'sensor_multihead_v3'
    split_dir = base_dir / 'outputs' / 'stratified_splits_v2'
    vocab_path = base_dir / 'data' / 'vocabulary_4digit_hybrid.json'

    device = get_device()
    print(f"Using device: {device}")

    # Load model and data
    print("\nLoading model and test data...")
    encoder, decoder, decomposer, test_dataset, args = load_model_and_data(
        model_dir, split_dir, vocab_path, device
    )
    print(f"  Loaded {len(test_dataset)} test samples")

    # Compute calibration metrics
    print("\n" + "="*60)
    calibration_results = evaluate_calibration(
        encoder, decoder, decomposer, test_dataset, device, n_bins=15
    )

    # Compute aggregates
    aggregate = compute_aggregate_metrics(calibration_results)

    # Print results
    print("\n" + "="*60)
    print("CALIBRATION METRICS")
    print("="*60)

    print("\nPer-Head Results:")
    print("-"*60)
    for head, metrics in calibration_results.items():
        print(f"\n{head.upper()}:")
        print(f"  ECE:            {metrics['ece']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']*100:.2f}%")
        print(f"  Mean Conf:      {metrics['mean_confidence']*100:.2f}%")
        print(f"  Overconfidence: {metrics['overconfidence']*100:+.2f}%")
        if 'brier_score' in metrics:
            print(f"  Brier Score:    {metrics['brier_score']:.4f}")
        print(f"  N predictions:  {metrics['n_predictions']:,}")

    print("\n" + "="*60)
    print("AGGREGATE METRICS")
    print("="*60)
    print(f"  Weighted ECE:            {aggregate['weighted_ece']:.4f}")
    print(f"  Weighted Accuracy:       {aggregate['weighted_accuracy']*100:.2f}%")
    print(f"  Weighted Confidence:     {aggregate['weighted_confidence']*100:.2f}%")
    print(f"  Weighted Overconfidence: {aggregate['weighted_overconfidence']*100:+.2f}%")
    if aggregate['average_brier_score']:
        print(f"  Average Brier Score:     {aggregate['average_brier_score']:.4f}")
    print(f"  Total Predictions:       {aggregate['total_predictions']:,}")

    # Save results
    output_file = model_dir / 'calibration_metrics.json'
    results = {
        'per_head': calibration_results,
        'aggregate': aggregate,
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {output_file}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    ece = aggregate['weighted_ece']
    if ece < 0.05:
        print(f"ECE = {ece:.4f}: Well-calibrated (< 0.05)")
    elif ece < 0.10:
        print(f"ECE = {ece:.4f}: Reasonably calibrated (0.05-0.10)")
    elif ece < 0.15:
        print(f"ECE = {ece:.4f}: Moderate calibration (0.10-0.15)")
    else:
        print(f"ECE = {ece:.4f}: Poor calibration (> 0.15)")

    overconf = aggregate['weighted_overconfidence']
    if overconf > 0.05:
        print(f"Overconfidence = {overconf*100:+.2f}%: Model is overconfident")
    elif overconf < -0.05:
        print(f"Overconfidence = {overconf*100:+.2f}%: Model is underconfident")
    else:
        print(f"Overconfidence = {overconf*100:+.2f}%: Well-balanced")

    return 0


if __name__ == '__main__':
    sys.exit(main())
