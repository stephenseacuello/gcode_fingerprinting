#!/usr/bin/env python3
"""
Error analysis for G-code fingerprinting model.

Analyzes prediction errors to identify systematic failure patterns.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.model import EnhancedEncoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


# Local MM_DTAE_LSTM class (same as in train_sensor_multihead.py)
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


def load_model_and_data(checkpoint_dir: Path, split_dir: Path, vocab_path: Path, device):
    """Load the trained model and test data."""
    # Load vocabulary
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    # Build token_to_id and id_to_token mappings
    token_to_id = vocab_data['vocab']
    id_to_token = {str(v): k for k, v in token_to_id.items()}
    vocab_size = len(token_to_id)

    vocab = {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'vocab_size': vocab_size
    }

    # Load test dataset
    test_dataset = DecoderDatasetFromSplits(split_dir, 'test', max_token_len=32)

    # Load args from results.json
    results_path = checkpoint_dir / 'results.json'
    with open(results_path) as f:
        results = json.load(f)
    args = argparse.Namespace(**results['args'])

    # Get sensor dim from dataset
    sensor_dim = test_dataset.get_sensor_dim()

    # Create encoder based on args
    if args.use_enhanced_encoder:
        encoder = EnhancedEncoder(
            input_dim=sensor_dim,
            hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.sensor_dim,
            n_operations=args.n_operations,
            use_multiscale=args.use_multiscale_encoder,
            n_scales=args.encoder_n_scales if hasattr(args, 'encoder_n_scales') else 4,
            kernel_sizes=args.encoder_kernel_sizes if hasattr(args, 'encoder_kernel_sizes') else [3, 5, 7, 11],
            dilations=args.encoder_dilations if hasattr(args, 'encoder_dilations') else [1, 2, 4, 8],
            lstm_layers=args.encoder_lstm_layers if hasattr(args, 'encoder_lstm_layers') else 2,
            use_multihead_pooling=args.use_multihead_pooling if hasattr(args, 'use_multihead_pooling') else False,
            pooling_n_heads=args.pooling_n_heads if hasattr(args, 'pooling_n_heads') else 4,
            pooling_n_queries=args.pooling_n_queries if hasattr(args, 'pooling_n_queries') else 8,
            dropout=args.encoder_dropout if hasattr(args, 'encoder_dropout') else 0.3,
            use_auxiliary_heads=args.use_auxiliary_heads if hasattr(args, 'use_auxiliary_heads') else False,
        )
    else:
        encoder = MM_DTAE_LSTM(
            input_dim=sensor_dim,
            hidden_dim=256,
            latent_dim=args.sensor_dim,
            n_classes=args.n_operations,
        )

    # Load pretrained encoder weights
    if args.encoder_path and Path(args.encoder_path).exists():
        checkpoint = torch.load(args.encoder_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        encoder.load_state_dict(state_dict, strict=False)

    encoder.to(device)
    encoder.eval()

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_operations=args.n_operations,
        n_types=args.n_types,
        n_commands=args.n_commands,
        n_param_types=args.n_param_types,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        sensor_dim=args.sensor_dim,
        max_int_digits=args.max_int_digits,
        n_decimal_digits=args.n_decimal_digits,
    ).to(device)

    # Load decoder checkpoint
    decoder_checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device, weights_only=False)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder.eval()

    return encoder, decoder, test_dataset, vocab, args


def analyze_predictions(encoder, decoder, test_dataset, vocab, device, num_samples=None):
    """Run inference and collect prediction statistics."""

    id_to_token = vocab['id_to_token']

    # Error tracking
    errors_by_type = defaultdict(list)
    errors_by_position = defaultdict(int)
    errors_by_token_type = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    confusion_pairs = Counter()

    def get_token_type(token_str):
        if token_str.startswith('G') or token_str.startswith('M'):
            return 'COMMAND'
        elif token_str.startswith('NUM_'):
            parts = token_str.split('_')
            if len(parts) >= 2:
                return f'NUM_{parts[1]}'
            return 'NUM_OTHER'
        elif token_str in ['PAD', 'BOS', 'EOS', 'UNK', 'MASK']:
            return 'SPECIAL'
        elif token_str in ['X', 'Y', 'Z', 'R', 'F', 'I', 'J', 'K', 'S']:
            return 'PARAM_TYPE'
        else:
            return 'OTHER'

    num_samples = num_samples or len(test_dataset)
    total_tokens = 0
    correct_tokens = 0
    sequence_correct = 0

    print(f"Analyzing {num_samples} test samples...")

    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{num_samples}...")

            sample = test_dataset[idx]

            # Move to device
            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)
            input_tokens = sample['input_tokens'].unsqueeze(0).to(device)
            target_tokens = sample['target_tokens'].to(device)
            operation = sample.get('operation_type', torch.tensor(0)).unsqueeze(0).to(device)

            # Encode - handle different encoder return formats
            enc_result = encoder.encode(sensor_features)

            # EnhancedEncoder returns (features, memory) both tensors
            # MM_DTAE_LSTM returns (latent, (h_n, c_n))
            if isinstance(enc_result[1], tuple):
                # MM_DTAE_LSTM: enc_result = (latent_seq, (h_n, c_n))
                memory = enc_result[0]  # latent sequence (B, T, latent_dim)
                # Get operation from classifier
                op_logits, _ = encoder.classify(memory)
                operation = op_logits.argmax(-1)
            else:
                # EnhancedEncoder: enc_result = (features, memory)
                features, memory = enc_result
                # Get operation from classifier
                op_logits, _ = encoder.classify(features)
                operation = op_logits.argmax(-1)

            # Decode (teacher forcing)
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=memory,
                operation_type=operation,
            )

            # Get predictions from legacy_logits (vocab-level predictions)
            pred_tokens = outputs['legacy_logits'].argmax(dim=-1)[0]

            # Analyze errors - ONLY up to first EOS in target (fixes EOS->PAD false errors)
            seq_len = min(len(pred_tokens), len(target_tokens))
            seq_correct_flag = True

            # Find EOS token ID
            eos_token_id = None
            pad_token_id = None
            for tok_str, tok_id in vocab['token_to_id'].items():
                if tok_str == 'EOS':
                    eos_token_id = tok_id
                elif tok_str == 'PAD':
                    pad_token_id = tok_id

            # Find first EOS position in target
            eos_pos = seq_len  # Default to full length
            for pos in range(seq_len):
                if target_tokens[pos].item() == eos_token_id:
                    eos_pos = pos + 1  # Include EOS token itself
                    break

            for pos in range(eos_pos):
                pred_id = pred_tokens[pos].item()
                target_id = target_tokens[pos].item()

                # Skip PAD tokens in target
                if target_id == pad_token_id:
                    continue

                pred_str = id_to_token.get(str(pred_id), f'<ID_{pred_id}>')
                target_str = id_to_token.get(str(target_id), f'<ID_{target_id}>')

                token_type = get_token_type(target_str)
                total_tokens += 1

                if pred_id == target_id:
                    correct_tokens += 1
                    errors_by_token_type[token_type]['correct'] += 1
                else:
                    seq_correct_flag = False
                    errors_by_position[pos] += 1
                    errors_by_token_type[token_type]['incorrect'] += 1
                    confusion_pairs[(pred_str, target_str)] += 1

                    # Get context
                    context_start = max(0, pos - 2)
                    context_end = min(eos_pos, pos + 3)
                    context = [id_to_token.get(str(target_tokens[i].item()), '?')
                              for i in range(context_start, context_end)]

                    errors_by_type[token_type].append({
                        'pred': pred_str,
                        'target': target_str,
                        'position': pos,
                        'context': context
                    })

            if seq_correct_flag:
                sequence_correct += 1

    # Compute statistics
    results = {
        'overall': {
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens,
            'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0,
            'total_sequences': num_samples,
            'correct_sequences': sequence_correct,
            'sequence_accuracy': sequence_correct / num_samples if num_samples > 0 else 0,
        },
        'by_token_type': {},
        'by_position': dict(errors_by_position),
        'top_confusion_pairs': [],
        'errors_by_type_samples': {},
    }

    for token_type, counts in errors_by_token_type.items():
        total = counts['correct'] + counts['incorrect']
        results['by_token_type'][token_type] = {
            'total': total,
            'correct': counts['correct'],
            'incorrect': counts['incorrect'],
            'accuracy': counts['correct'] / total if total > 0 else 0
        }

    results['top_confusion_pairs'] = [
        {'pred': pred, 'target': target, 'count': count}
        for (pred, target), count in confusion_pairs.most_common(30)
    ]

    for token_type, errors in errors_by_type.items():
        results['errors_by_type_samples'][token_type] = errors[:10]

    return results


def print_report(results: Dict):
    """Print a formatted error analysis report."""

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)

    overall = results['overall']
    print(f"\n  OVERALL METRICS")
    print(f"   Token Accuracy:    {overall['token_accuracy']*100:.2f}% ({overall['correct_tokens']}/{overall['total_tokens']})")
    print(f"   Sequence Accuracy: {overall['sequence_accuracy']*100:.2f}% ({overall['correct_sequences']}/{overall['total_sequences']})")

    print(f"\n  ACCURACY BY TOKEN TYPE")
    print("-" * 60)
    by_type = results['by_token_type']
    sorted_types = sorted(by_type.items(), key=lambda x: x[1]['accuracy'])
    for token_type, stats in sorted_types:
        bar_len = int(stats['accuracy'] * 30)
        bar = '*' * bar_len + '.' * (30 - bar_len)
        print(f"   {token_type:12} {bar} {stats['accuracy']*100:5.1f}% ({stats['incorrect']:4} err / {stats['total']:5} tot)")

    print(f"\n  ERRORS BY POSITION IN SEQUENCE")
    print("-" * 60)
    by_pos = results['by_position']
    if by_pos:
        max_pos = max(by_pos.keys())
        for pos in range(min(max_pos + 1, 20)):
            count = by_pos.get(pos, 0)
            bar = '*' * min(count // 5, 40)
            print(f"   Position {pos:2}: {bar} ({count})")

    print(f"\n  TOP CONFUSION PAIRS (pred -> target)")
    print("-" * 60)
    for item in results['top_confusion_pairs'][:15]:
        print(f"   {item['pred']:20} -> {item['target']:20} ({item['count']} times)")

    print(f"\n  SAMPLE ERRORS BY TOKEN TYPE")
    print("-" * 60)
    for token_type, samples in results['errors_by_type_samples'].items():
        if samples:
            print(f"\n   [{token_type}]")
            for s in samples[:3]:
                context_str = ' '.join(s['context'])
                print(f"      pos={s['position']:2}: pred={s['pred']:15} target={s['target']:15} ctx=[{context_str}]")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Error analysis for G-code model')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/enhanced_encoder_full',
                       help='Directory containing best_model.pt and results.json')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3',
                       help='Directory containing train/val/test splits')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_full.json',
                       help='Path to vocabulary file')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to analyze (default: all)')
    parser.add_argument('--output', type=str, default='reports/error_analysis.json',
                       help='Output path for detailed results')

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    encoder, decoder, test_dataset, vocab, model_args = load_model_and_data(
        Path(args.checkpoint_dir),
        Path(args.split_dir),
        Path(args.vocab_path),
        device
    )

    print(f"Loaded model from {args.checkpoint_dir}")
    print(f"Test dataset: {len(test_dataset)} samples")

    results = analyze_predictions(
        encoder, decoder, test_dataset, vocab, device,
        num_samples=args.num_samples
    )

    print_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
