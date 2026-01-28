#!/usr/bin/env python3
"""
Top-k accuracy evaluation for G-code token prediction.

Measures whether the correct token appears in the model's top-k predictions,
broken down by token type (command vs numeric). With a 2,498-token vocabulary
including fine-grained numeric buckets, many "wrong" top-1 predictions may be
near-misses that appear in top-3 or top-5.

Usage:
    python scripts/evaluation/evaluate_topk_accuracy.py
    python scripts/evaluation/evaluate_topk_accuracy.py --model-dir outputs/jan26/ensemble/seed_42
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


def classify_token(token_str):
    """Classify a token string into a category."""
    if token_str.startswith('<'):
        return 'special'
    # Command tokens: G0, G1, G2, G3, M3, M5, etc.
    if token_str[0] in ('G', 'M', 'S', 'F', 'T', 'N'):
        return 'command'
    # Parameter letters: X, Y, Z, I, J, K, R, P, Q
    if len(token_str) == 1 and token_str.isalpha():
        return 'parameter'
    # Sign tokens
    if token_str in ('+', '-'):
        return 'sign'
    # Numeric tokens (digit buckets, decimal values)
    try:
        float(token_str)
        return 'numeric'
    except ValueError:
        pass
    # Digit tokens like D0, D1, ...
    if token_str.startswith('D') and len(token_str) > 1:
        return 'digit'
    return 'other'


def load_models(model_dir, vocab_path, split_dir, device):
    """Load encoder and decoder from a training checkpoint."""
    model_dir = Path(model_dir)

    # Load checkpoint
    checkpoint_path = model_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    # Load args
    args = {}
    args_path = model_dir / 'args.json'
    if args_path.exists():
        with open(args_path) as f:
            args = json.load(f)
    elif 'args' in checkpoint:
        args = checkpoint['args']

    # Load vocab
    decomposer = TokenDecomposer(str(vocab_path))

    # Get input dim from metadata
    input_dim = 296
    metadata_path = Path(split_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 296)

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=args.get('encoder_hidden_dim', 256),
        latent_dim=args.get('sensor_dim', 128),
        n_operations=args.get('n_operations', 9),
        n_scales=args.get('encoder_n_scales', 4),
        kernel_sizes=args.get('encoder_kernel_sizes', [3, 5, 7, 11]),
        dilations=args.get('encoder_dilations', [1, 2, 4, 8]),
        dropout=args.get('encoder_dropout', 0.3),
        lstm_layers=args.get('encoder_lstm_layers', 2),
    ).to(device)

    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    encoder.eval()

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        vocab_size=decomposer.vocab_size,
        d_model=args.get('d_model', 192),
        n_heads=args.get('n_heads', 8),
        n_layers=args.get('n_layers', 4),
        sensor_dim=128,
        n_operations=args.get('n_operations', 9),
        n_types=args.get('n_types', 4),
        n_commands=args.get('n_commands', 6),
        n_param_types=args.get('n_param_types', 10),
        max_int_digits=2,
        n_decimal_digits=4,
        dropout=args.get('dropout', 0.3),
        embed_dropout=args.get('embed_dropout', 0.1),
        max_seq_len=args.get('max_seq_len', 32),
    ).to(device)

    decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder.eval()

    return encoder, decoder, decomposer, args


def evaluate_topk(encoder, decoder, test_dataset, decomposer, device, ks=(1, 3, 5, 10)):
    """Evaluate top-k accuracy on test set using teacher forcing."""
    # Build id-to-token map
    id_to_token = {v: k for k, v in decomposer.vocab.items()}
    pad_id = decomposer.vocab.get('<PAD>', 0)

    # Accumulators: overall and per-category
    overall = {k: {'correct': 0, 'total': 0} for k in ks}
    per_category = {cat: {k: {'correct': 0, 'total': 0} for k in ks}
                    for cat in ['command', 'parameter', 'sign', 'numeric', 'digit', 'other', 'special']}

    n_samples = len(test_dataset)
    print(f"Evaluating top-k accuracy on {n_samples} test samples...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Top-k eval"):
            sample = test_dataset[idx]

            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)
            target_tokens = sample['target_tokens'].to(device)  # [seq_len]
            op_type = sample['operation_type'].unsqueeze(0).to(device)

            # Encode
            pooled, sensor_emb = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(pooled)
            pred_op = op_logits.argmax(-1)

            # Teacher-forced forward pass: feed ground truth tokens, get logits
            # target_tokens is [t1, t2, ..., EOS]
            # Decoder input should be [BOS, t1, t2, ...] (shift right)
            bos_id = decomposer.vocab.get('<BOS>', 1)
            decoder_input = torch.cat([
                torch.tensor([bos_id], device=device),
                target_tokens[:-1]  # all but last
            ]).unsqueeze(0)  # [1, seq_len]

            decoder_out = decoder(decoder_input, sensor_emb, pred_op)

            # Get logits: [1, seq_len, vocab_size]
            logits = decoder_out.get('legacy_logits', decoder_out.get('logits'))
            if logits is None:
                # Try composing from heads
                logits = decoder_out.get('composed_logits')
            if logits is None:
                print(f"WARNING: No logits found. Keys: {list(decoder_out.keys())}")
                continue

            logits = logits.squeeze(0)  # [seq_len, vocab_size]

            # Compare against target_tokens [seq_len]
            for t in range(target_tokens.size(0)):
                target_id = target_tokens[t].item()
                if target_id == pad_id:
                    continue

                token_str = id_to_token.get(target_id, '<UNK>')
                category = classify_token(token_str)

                # Get top-k predictions
                token_logits = logits[t]  # [vocab_size]
                for k in ks:
                    topk_ids = torch.topk(token_logits, k).indices
                    hit = int(target_id in topk_ids)
                    overall[k]['correct'] += hit
                    overall[k]['total'] += 1
                    per_category[category][k]['correct'] += hit
                    per_category[category][k]['total'] += 1

    return overall, per_category, id_to_token


def main():
    parser = argparse.ArgumentParser(description="Top-k accuracy evaluation")
    parser.add_argument('--model-dir', type=str,
                        default='outputs/jan26/ensemble/seed_42',
                        help='Path to model directory with best_model.pt')
    parser.add_argument('--vocab-path', type=str,
                        default='data/vocabulary_4digit_full.json')
    parser.add_argument('--split-dir', type=str,
                        default='outputs/jan23_followup/no_leakage')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='Values of k to evaluate')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/analysis/topk_accuracy')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    encoder, decoder, decomposer, model_args = load_models(
        args.model_dir, args.vocab_path, args.split_dir, device
    )

    # Load test data
    print("\nLoading test data...")
    test_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'test',
        max_token_len=model_args.get('max_seq_len', 32)
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Evaluate
    ks = tuple(sorted(args.ks))
    overall, per_category, id_to_token = evaluate_topk(
        encoder, decoder, test_dataset, decomposer, device, ks=ks
    )

    # Print results
    print("\n" + "=" * 70)
    print("TOP-K ACCURACY RESULTS")
    print("=" * 70)

    print(f"\n{'k':>5}  {'Accuracy':>10}  {'Correct':>10}  {'Total':>10}")
    print("-" * 45)
    for k in ks:
        acc = overall[k]['correct'] / max(overall[k]['total'], 1)
        print(f"{k:>5}  {acc:>10.2%}  {overall[k]['correct']:>10}  {overall[k]['total']:>10}")

    print("\n\nPER-CATEGORY BREAKDOWN:")
    print(f"{'Category':>12}", end="")
    for k in ks:
        print(f"  {'Top-'+str(k):>8}", end="")
    print(f"  {'Count':>8}")
    print("-" * (14 + 10 * len(ks) + 10))

    for cat in ['command', 'parameter', 'sign', 'numeric', 'digit', 'other', 'special']:
        total = per_category[cat][ks[0]]['total']
        if total == 0:
            continue
        print(f"{cat:>12}", end="")
        for k in ks:
            acc = per_category[cat][k]['correct'] / max(per_category[cat][k]['total'], 1)
            print(f"  {acc:>8.2%}", end="")
        print(f"  {total:>8}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_dir': str(args.model_dir),
        'ks': list(ks),
        'overall': {str(k): {
            'accuracy': overall[k]['correct'] / max(overall[k]['total'], 1),
            'correct': overall[k]['correct'],
            'total': overall[k]['total'],
        } for k in ks},
        'per_category': {cat: {str(k): {
            'accuracy': per_category[cat][k]['correct'] / max(per_category[cat][k]['total'], 1),
            'correct': per_category[cat][k]['correct'],
            'total': per_category[cat][k]['total'],
        } for k in ks} for cat in per_category if per_category[cat][ks[0]]['total'] > 0},
    }

    output_path = output_dir / 'topk_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
