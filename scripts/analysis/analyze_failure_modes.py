#!/usr/bin/env python3
"""
Failure mode analysis for G-code prediction.

Analyzes prediction errors by:
1. Operation type (which operations are hardest?)
2. Token type (command, param, value errors)
3. Position in sequence (early vs late errors)
4. Parameter type (X, Y, Z, R, F errors)
5. Digit position (which digit positions are hardest?)

Outputs detailed breakdown and suggestions for improvement.

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.decoder_dataset import (
    DecoderDatasetFromSplits,
    decoder_collate_fn,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)


OPERATION_NAMES = [
    "adaptive", "adaptive150025", "face", "face150025",
    "pocket", "pocket150025", "damageadaptive", "damageface", "damagepocket"
]

TOKEN_TYPES = ['SPECIAL', 'COMMAND', 'PARAM', 'NUMERIC']


class FailureModeAnalyzer:
    """Analyzes and categorizes prediction failures."""

    def __init__(self, vocab, id2token):
        self.vocab = vocab
        self.id2token = id2token

        # Build token type mapping
        self.token_types = {}
        for token, tid in vocab.items():
            if token in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                self.token_types[tid] = 'SPECIAL'
            elif token.startswith('G') or token.startswith('M'):
                self.token_types[tid] = 'COMMAND'
            elif token in ['X', 'Y', 'Z', 'R', 'F', 'I', 'J', 'K', 'A', 'B', 'C', 'S']:
                self.token_types[tid] = 'PARAM'
            elif token.startswith('NUM_'):
                self.token_types[tid] = 'NUMERIC'
            else:
                self.token_types[tid] = 'OTHER'

        # Extract param type from NUM_X_1234 format
        self.token_params = {}
        for token, tid in vocab.items():
            if token.startswith('NUM_'):
                parts = token.split('_')
                if len(parts) >= 2:
                    self.token_params[tid] = parts[1]

        # Error counters
        self.errors_by_operation = defaultdict(lambda: {'total': 0, 'errors': 0})
        self.errors_by_token_type = defaultdict(lambda: {'total': 0, 'errors': 0})
        self.errors_by_position = defaultdict(lambda: {'total': 0, 'errors': 0})
        self.errors_by_param = defaultdict(lambda: {'total': 0, 'errors': 0})

        # Confusion matrices
        self.command_confusion = defaultdict(Counter)
        self.param_confusion = defaultdict(Counter)
        self.type_confusion = defaultdict(Counter)

        # Specific error patterns
        self.error_patterns = []

    def analyze_prediction(self, pred_tokens, target_tokens, operation_type, type_preds=None, type_targets=None):
        """
        Analyze a single prediction for failure modes.

        Args:
            pred_tokens: [L] predicted token IDs
            target_tokens: [L] target token IDs
            operation_type: Operation type index
            type_preds: [L] predicted token types (optional)
            type_targets: [L] target token types (optional)
        """
        op_name = OPERATION_NAMES[operation_type] if operation_type < len(OPERATION_NAMES) else f'Op{operation_type}'

        for pos in range(min(len(pred_tokens), len(target_tokens))):
            pred = pred_tokens[pos].item() if isinstance(pred_tokens[pos], torch.Tensor) else pred_tokens[pos]
            tgt = target_tokens[pos].item() if isinstance(target_tokens[pos], torch.Tensor) else target_tokens[pos]

            # Skip padding
            if tgt == PAD_TOKEN_ID:
                continue

            # Update operation stats
            self.errors_by_operation[op_name]['total'] += 1

            # Update position stats
            self.errors_by_position[pos]['total'] += 1

            # Update token type stats
            tgt_type = self.token_types.get(tgt, 'OTHER')
            self.errors_by_token_type[tgt_type]['total'] += 1

            # Update param stats (for numeric tokens)
            tgt_param = self.token_params.get(tgt, None)
            if tgt_param:
                self.errors_by_param[tgt_param]['total'] += 1

            # Check for error
            if pred != tgt:
                self.errors_by_operation[op_name]['errors'] += 1
                self.errors_by_position[pos]['errors'] += 1
                self.errors_by_token_type[tgt_type]['errors'] += 1

                if tgt_param:
                    self.errors_by_param[tgt_param]['errors'] += 1

                # Track confusion
                pred_token = self.id2token.get(pred, f'UNK{pred}')
                tgt_token = self.id2token.get(tgt, f'UNK{tgt}')

                if tgt_type == 'COMMAND':
                    self.command_confusion[tgt_token][pred_token] += 1
                elif tgt_type == 'PARAM':
                    self.param_confusion[tgt_token][pred_token] += 1

                # Store error pattern
                if len(self.error_patterns) < 1000:  # Limit storage
                    self.error_patterns.append({
                        'position': pos,
                        'operation': op_name,
                        'target_token': tgt_token,
                        'pred_token': pred_token,
                        'target_type': tgt_type,
                    })

    def get_report(self):
        """Generate analysis report."""
        report = {
            'summary': {},
            'by_operation': {},
            'by_token_type': {},
            'by_position': {},
            'by_param': {},
            'command_confusion': {},
            'param_confusion': {},
            'top_error_patterns': [],
        }

        # Compute error rates
        total_tokens = sum(v['total'] for v in self.errors_by_token_type.values())
        total_errors = sum(v['errors'] for v in self.errors_by_token_type.values())

        report['summary'] = {
            'total_tokens': total_tokens,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_tokens, 1),
        }

        # By operation
        for op, stats in sorted(self.errors_by_operation.items()):
            rate = stats['errors'] / max(stats['total'], 1)
            report['by_operation'][op] = {
                'total': stats['total'],
                'errors': stats['errors'],
                'error_rate': rate,
            }

        # By token type
        for tt, stats in sorted(self.errors_by_token_type.items()):
            rate = stats['errors'] / max(stats['total'], 1)
            report['by_token_type'][tt] = {
                'total': stats['total'],
                'errors': stats['errors'],
                'error_rate': rate,
            }

        # By position (first 20 positions)
        for pos in range(min(20, max(self.errors_by_position.keys()) + 1)):
            if pos in self.errors_by_position:
                stats = self.errors_by_position[pos]
                rate = stats['errors'] / max(stats['total'], 1)
                report['by_position'][pos] = {
                    'total': stats['total'],
                    'errors': stats['errors'],
                    'error_rate': rate,
                }

        # By param
        for param, stats in sorted(self.errors_by_param.items()):
            rate = stats['errors'] / max(stats['total'], 1)
            report['by_param'][param] = {
                'total': stats['total'],
                'errors': stats['errors'],
                'error_rate': rate,
            }

        # Command confusion (top errors)
        for tgt_cmd, preds in self.command_confusion.items():
            top_preds = preds.most_common(3)
            if top_preds:
                report['command_confusion'][tgt_cmd] = [
                    {'predicted': p, 'count': c} for p, c in top_preds
                ]

        # Param confusion
        for tgt_param, preds in self.param_confusion.items():
            top_preds = preds.most_common(3)
            if top_preds:
                report['param_confusion'][tgt_param] = [
                    {'predicted': p, 'count': c} for p, c in top_preds
                ]

        # Top error patterns
        pattern_counts = Counter(
            (p['target_type'], p['operation'], p['target_token'])
            for p in self.error_patterns
        )
        report['top_error_patterns'] = [
            {'type': t, 'operation': o, 'token': tok, 'count': c}
            for (t, o, tok), c in pattern_counts.most_common(20)
        ]

        return report

    def print_report(self):
        """Print formatted report."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print("FAILURE MODE ANALYSIS")
        print("=" * 70)

        # Summary
        s = report['summary']
        print(f"\nOverall: {s['total_errors']:,} errors / {s['total_tokens']:,} tokens "
              f"({s['error_rate']*100:.2f}% error rate)")

        # By operation
        print("\n" + "-" * 70)
        print("ERRORS BY OPERATION")
        print("-" * 70)
        print(f"{'Operation':<20} {'Total':<10} {'Errors':<10} {'Error Rate':<12}")
        print("-" * 70)

        sorted_ops = sorted(
            report['by_operation'].items(),
            key=lambda x: x[1]['error_rate'],
            reverse=True
        )
        for op, stats in sorted_ops:
            print(f"{op:<20} {stats['total']:<10} {stats['errors']:<10} {stats['error_rate']*100:>8.2f}%")

        # By token type
        print("\n" + "-" * 70)
        print("ERRORS BY TOKEN TYPE")
        print("-" * 70)
        print(f"{'Type':<15} {'Total':<10} {'Errors':<10} {'Error Rate':<12}")
        print("-" * 70)

        sorted_types = sorted(
            report['by_token_type'].items(),
            key=lambda x: x[1]['error_rate'],
            reverse=True
        )
        for tt, stats in sorted_types:
            print(f"{tt:<15} {stats['total']:<10} {stats['errors']:<10} {stats['error_rate']*100:>8.2f}%")

        # By position
        print("\n" + "-" * 70)
        print("ERRORS BY POSITION")
        print("-" * 70)
        print(f"{'Position':<10} {'Total':<10} {'Errors':<10} {'Error Rate':<12}")
        print("-" * 70)

        for pos in range(min(15, len(report['by_position']))):
            if pos in report['by_position']:
                stats = report['by_position'][pos]
                print(f"{pos:<10} {stats['total']:<10} {stats['errors']:<10} {stats['error_rate']*100:>8.2f}%")

        # By param
        print("\n" + "-" * 70)
        print("ERRORS BY PARAMETER")
        print("-" * 70)
        print(f"{'Param':<10} {'Total':<10} {'Errors':<10} {'Error Rate':<12}")
        print("-" * 70)

        sorted_params = sorted(
            report['by_param'].items(),
            key=lambda x: x[1]['error_rate'],
            reverse=True
        )
        for param, stats in sorted_params:
            print(f"{param:<10} {stats['total']:<10} {stats['errors']:<10} {stats['error_rate']*100:>8.2f}%")

        # Top error patterns
        print("\n" + "-" * 70)
        print("TOP ERROR PATTERNS")
        print("-" * 70)

        for i, pattern in enumerate(report['top_error_patterns'][:10]):
            print(f"{i+1}. [{pattern['type']}] {pattern['operation']}: {pattern['token']} "
                  f"({pattern['count']} errors)")

        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        recommendations = self._generate_recommendations(report)
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. {rec}")

    def _generate_recommendations(self, report):
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Check token type error rates
        type_errors = report['by_token_type']
        if type_errors.get('NUMERIC', {}).get('error_rate', 0) > 0.1:
            recommendations.append(
                "High NUMERIC error rate detected. Consider:\n"
                "   - Increasing digit_weight in loss\n"
                "   - Using stronger focal_gamma for digit positions\n"
                "   - Adding consistency loss between digit and regression heads"
            )

        if type_errors.get('COMMAND', {}).get('error_rate', 0) > 0.05:
            recommendations.append(
                "High COMMAND error rate detected. Consider:\n"
                "   - Increasing command_weight in loss\n"
                "   - Adding class balancing for rare commands (G02, G03)\n"
                "   - Oversampling sequences with arc commands"
            )

        # Check operation error rates
        op_errors = report['by_operation']
        high_error_ops = [
            op for op, stats in op_errors.items()
            if stats['error_rate'] > 0.15
        ]
        if high_error_ops:
            recommendations.append(
                f"High error rate for operations: {', '.join(high_error_ops)}\n"
                "   - Consider operation-specific fine-tuning\n"
                "   - Use temperature sampling to upweight these operations\n"
                "   - Check if these operations have sufficient training data"
            )

        # Check position-based errors
        pos_errors = report['by_position']
        late_errors = [
            pos for pos, stats in pos_errors.items()
            if int(pos) > 10 and stats['error_rate'] > 0.2
        ]
        if late_errors:
            recommendations.append(
                "High error rate at later positions. Consider:\n"
                "   - Increasing scheduled sampling decay\n"
                "   - Adding EOS calibration\n"
                "   - Using curriculum learning with longer sequences"
            )

        # Check param-specific errors
        param_errors = report['by_param']
        high_error_params = [
            p for p, stats in param_errors.items()
            if stats['error_rate'] > 0.15
        ]
        if high_error_params:
            recommendations.append(
                f"High error rate for params: {', '.join(high_error_params)}\n"
                "   - Consider param-specific loss weighting\n"
                "   - Check value distribution for these parameters\n"
                "   - Ensure bucketing covers the full value range"
            )

        if not recommendations:
            recommendations.append(
                "No critical issues detected. Consider:\n"
                "   - Fine-tuning with SCST for sequence-level optimization\n"
                "   - Ensemble multiple decoding strategies"
            )

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze prediction failure modes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_full.json')
    parser.add_argument('--encoder-path', type=str, default='outputs/mm_dtae_lstm_v2/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='reports/failure_analysis')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--max-length', type=int, default=32)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    id2token = {v: k for k, v in vocab.items()}

    # Create analyzer
    analyzer = FailureModeAnalyzer(vocab, id2token)

    # Import and load models (similar to comparison script)
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    import torch.nn as nn

    class MM_DTAE_LSTM(nn.Module):
        def __init__(self, input_dim=155, hidden_dim=256, latent_dim=128, n_classes=9):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=0.3)
            self.bottleneck = nn.Sequential(
                nn.Linear(hidden_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.classification_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(latent_dim, n_classes)
            )
            self.temporal_attention = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softmax(dim=1))

        def encode(self, x):
            x = self.input_proj(x)
            encoded, states = self.encoder_lstm(x)
            latent = self.bottleneck(encoded)
            return latent, states

        def classify(self, latent):
            attn_weights = self.temporal_attention(latent)
            pooled = (latent * attn_weights).sum(dim=1)
            logits = self.classification_head(pooled)
            return logits, attn_weights

    # Load encoder
    print("Loading encoder...")
    encoder = MM_DTAE_LSTM()
    encoder_state = torch.load(args.encoder_path, map_location='cpu')
    encoder.load_state_dict(encoder_state['model_state_dict'], strict=False)
    encoder = encoder.to(device)
    encoder.eval()

    # Load decoder
    print("Loading decoder...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab),
        d_model=config.get('d_model', 192),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 4),
        sensor_dim=128,
        n_operations=9,
        dropout=0.0,
    )
    decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = decoder.to(device)
    decoder.eval()

    # Load test data
    print("Loading test data...")
    test_dataset = DecoderDatasetFromSplits(
        args.split_dir,
        args.vocab_path,
        split='test',
        max_length=args.max_length,
    )

    if args.n_samples < len(test_dataset):
        indices = list(range(args.n_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=decoder_collate_fn,
        num_workers=0,
    )

    # Run analysis
    print(f"\nAnalyzing {len(test_dataset)} samples...")

    for batch in tqdm(test_loader, desc="Analyzing"):
        sensor_features = batch['sensor_features'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        operation_type = batch.get('operation_type', torch.zeros(sensor_features.size(0), dtype=torch.long)).to(device)

        with torch.no_grad():
            # Encode
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            op_pred = op_logits.argmax(-1)

            # Decode (greedy)
            generated = torch.full((sensor_features.size(0), 1), BOS_TOKEN_ID, dtype=torch.long, device=device)

            for _ in range(args.max_length - 1):
                outputs = decoder(
                    tokens=generated,
                    sensor_embeddings=sensor_emb,
                    operation_type=op_pred,
                )
                next_logits = outputs['legacy_logits'][:, -1, :]
                next_token = next_logits.argmax(-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        # Analyze each sample
        for b in range(sensor_features.size(0)):
            analyzer.analyze_prediction(
                pred_tokens=generated[b],
                target_tokens=target_tokens[b],
                operation_type=operation_type[b].item(),
            )

    # Print report
    analyzer.print_report()

    # Save report
    report = analyzer.get_report()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'failure_analysis_{timestamp}.json'

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
