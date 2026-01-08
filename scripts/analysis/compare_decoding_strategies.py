#!/usr/bin/env python3
"""
Systematic comparison of decoding strategies for G-code generation.

Compares:
1. Greedy decoding (argmax at each step)
2. Beam search (various beam widths)
3. Grammar-constrained decoding
4. Beam search + reranking

Outputs detailed metrics and examples for analysis.

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Enable MPS fallback for Mac
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


def load_models(checkpoint_path, vocab_path, encoder_path, device):
    """Load encoder and decoder models."""
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    vocab_size = len(vocab)
    id2token = {v: k for k, v in vocab.items()}

    # Define encoder architecture (MM-DTAE-LSTM)
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
    encoder = MM_DTAE_LSTM()
    encoder_state = torch.load(encoder_path, map_location='cpu')
    encoder.load_state_dict(encoder_state['model_state_dict'], strict=False)
    encoder = encoder.to(device)
    encoder.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get decoder config from checkpoint
    config = checkpoint.get('config', {})
    d_model = config.get('d_model', 192)
    n_heads = config.get('n_heads', 8)
    n_layers = config.get('n_layers', 4)

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        sensor_dim=128,
        n_operations=9,
        dropout=0.0,  # No dropout for inference
    )

    # Load decoder weights
    decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = decoder.to(device)
    decoder.eval()

    return encoder, decoder, vocab, id2token


def greedy_decode(decoder, sensor_emb, operation_type, max_length=32, device='cpu'):
    """Greedy decoding - select argmax at each step."""
    B = sensor_emb.size(0)

    # Start with BOS
    generated = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )

        # Get next token from legacy logits
        next_logits = outputs['legacy_logits'][:, -1, :]  # [B, V]
        next_token = next_logits.argmax(-1, keepdim=True)  # [B, 1]

        # Don't update finished sequences
        next_token = next_token.masked_fill(finished.unsqueeze(-1), PAD_TOKEN_ID)

        generated = torch.cat([generated, next_token], dim=1)

        # Update finished
        finished = finished | (next_token.squeeze(-1) == EOS_TOKEN_ID)

        if finished.all():
            break

    return generated


def beam_search_decode(decoder, sensor_emb, operation_type, beam_width=5, max_length=32,
                       length_penalty=0.6, device='cpu'):
    """Beam search decoding."""
    B = sensor_emb.size(0)
    V = decoder.vocab_size

    # Expand for beam search
    sensor_emb_beam = sensor_emb.repeat_interleave(beam_width, dim=0)  # [B*beam, T, D]
    op_type_beam = operation_type.repeat_interleave(beam_width, dim=0)  # [B*beam]

    # Initialize beams
    beams = torch.full((B * beam_width, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
    beam_scores = torch.zeros(B * beam_width, device=device)
    beam_scores[1::beam_width] = float('-inf')  # Only first beam is active initially

    finished_seqs = [[] for _ in range(B)]

    for step in range(max_length - 1):
        # Forward pass
        outputs = decoder(
            tokens=beams,
            sensor_embeddings=sensor_emb_beam,
            operation_type=op_type_beam,
        )

        # Get logits for last position
        logits = outputs['legacy_logits'][:, -1, :]  # [B*beam, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # Add to beam scores
        next_scores = beam_scores.unsqueeze(-1) + log_probs  # [B*beam, V]

        # Reshape for per-example beam selection
        next_scores = next_scores.view(B, beam_width * V)  # [B, beam*V]

        # Select top-k
        topk_scores, topk_indices = next_scores.topk(beam_width, dim=-1)  # [B, beam]

        # Decode indices
        beam_indices = topk_indices // V  # Which beam
        token_indices = topk_indices % V  # Which token

        # Update beams
        new_beams = []
        new_scores = []

        for b in range(B):
            for k in range(beam_width):
                beam_idx = beam_indices[b, k].item()
                token_idx = token_indices[b, k].item()
                score = topk_scores[b, k].item()

                prev_beam = beams[b * beam_width + beam_idx]
                new_beam = torch.cat([prev_beam, torch.tensor([token_idx], device=device)])

                if token_idx == EOS_TOKEN_ID:
                    # Finished sequence
                    length = new_beam.size(0)
                    lp = ((5 + length) / 6) ** length_penalty
                    final_score = score / lp
                    finished_seqs[b].append((new_beam, final_score))
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)

        if not new_beams:
            break

        # Pad to same length
        max_len = max(b.size(0) for b in new_beams)
        padded_beams = torch.full((len(new_beams), max_len), PAD_TOKEN_ID, device=device)
        for i, b in enumerate(new_beams):
            padded_beams[i, :b.size(0)] = b

        beams = padded_beams
        beam_scores = torch.tensor(new_scores, device=device)

        # Expand sensor embeddings if needed
        if beams.size(0) != sensor_emb_beam.size(0):
            # Rebuild with correct batch size
            sensor_emb_beam = sensor_emb.repeat_interleave(beams.size(0) // B, dim=0)
            op_type_beam = operation_type.repeat_interleave(beams.size(0) // B, dim=0)

    # Select best from finished sequences
    results = []
    for b in range(B):
        if finished_seqs[b]:
            best_seq, best_score = max(finished_seqs[b], key=lambda x: x[1])
            results.append(best_seq)
        elif beams.size(0) > 0:
            # No finished sequence, return best beam
            results.append(beams[b * min(beam_width, beams.size(0) // B)])
        else:
            results.append(torch.tensor([BOS_TOKEN_ID, EOS_TOKEN_ID], device=device))

    # Pad results
    max_len = max(r.size(0) for r in results)
    padded = torch.full((B, max_len), PAD_TOKEN_ID, device=device)
    for i, r in enumerate(results):
        padded[i, :r.size(0)] = r

    return padded


def constrained_decode(decoder, sensor_emb, operation_type, max_length=32, device='cpu'):
    """Grammar-constrained greedy decoding."""
    from miracle.training.grammar_constraints import GCodeGrammarConstraints

    B = sensor_emb.size(0)
    constraints = GCodeGrammarConstraints()

    # Start with BOS
    generated = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_length - 1):
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )

        # Get next token logits
        next_logits = outputs['legacy_logits'][:, -1, :]  # [B, V]

        # Apply grammar constraints
        constrained_logits = constraints.apply_hard_constraints(
            logits=next_logits,
            current_tokens=generated,
            position=step,
        )

        next_token = constrained_logits.argmax(-1, keepdim=True)  # [B, 1]
        next_token = next_token.masked_fill(finished.unsqueeze(-1), PAD_TOKEN_ID)

        generated = torch.cat([generated, next_token], dim=1)
        finished = finished | (next_token.squeeze(-1) == EOS_TOKEN_ID)

        if finished.all():
            break

    return generated


def compute_metrics(predictions, targets, id2token):
    """Compute accuracy metrics."""
    B = predictions.size(0)

    token_correct = 0
    token_total = 0
    seq_correct = 0

    for b in range(B):
        pred = predictions[b]
        tgt = targets[b]

        # Find sequence length (up to EOS)
        pred_len = (pred == EOS_TOKEN_ID).nonzero()
        pred_len = pred_len[0].item() + 1 if len(pred_len) > 0 else pred.size(0)

        tgt_len = (tgt == EOS_TOKEN_ID).nonzero()
        tgt_len = tgt_len[0].item() + 1 if len(tgt_len) > 0 else tgt.size(0)

        # Token accuracy (up to min length)
        min_len = min(pred_len, tgt_len)
        for i in range(min_len):
            if tgt[i] != PAD_TOKEN_ID:
                token_total += 1
                if pred[i] == tgt[i]:
                    token_correct += 1

        # Sequence accuracy
        if pred_len == tgt_len:
            if torch.equal(pred[:pred_len], tgt[:tgt_len]):
                seq_correct += 1

    return {
        'token_accuracy': token_correct / max(token_total, 1),
        'sequence_accuracy': seq_correct / B,
        'token_correct': token_correct,
        'token_total': token_total,
        'seq_correct': seq_correct,
        'n_samples': B,
    }


def tokens_to_gcode(token_ids, id2token):
    """Convert token IDs to G-code string."""
    parts = []
    for tid in token_ids:
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        if tid in (PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID):
            continue
        token = id2token.get(tid, f"UNK{tid}")
        parts.append(token)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description='Compare decoding strategies')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3', help='Data split directory')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_full.json')
    parser.add_argument('--encoder-path', type=str, default='outputs/mm_dtae_lstm_v2/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='reports/decoding_comparison')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples to evaluate')
    parser.add_argument('--beam-widths', type=int, nargs='+', default=[3, 5, 10])
    parser.add_argument('--max-length', type=int, default=32)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    encoder, decoder, vocab, id2token = load_models(
        args.checkpoint, args.vocab_path, args.encoder_path, device
    )

    # Load test data
    print("Loading test data...")
    test_dataset = DecoderDatasetFromSplits(
        args.split_dir,
        args.vocab_path,
        split='test',
        max_length=args.max_length,
    )

    # Limit samples
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

    # Strategies to compare
    strategies = {
        'greedy': lambda dec, sem, op: greedy_decode(dec, sem, op, args.max_length, device),
    }

    # Add beam search variants
    for bw in args.beam_widths:
        strategies[f'beam_{bw}'] = lambda dec, sem, op, bw=bw: beam_search_decode(
            dec, sem, op, bw, args.max_length, 0.6, device
        )

    # Add constrained decoding
    strategies['constrained'] = lambda dec, sem, op: constrained_decode(
        dec, sem, op, args.max_length, device
    )

    # Run comparison
    results = {name: defaultdict(list) for name in strategies}
    examples = []

    print(f"\nComparing {len(strategies)} decoding strategies...")

    for batch in tqdm(test_loader, desc="Evaluating"):
        sensor_features = batch['sensor_features'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        operation_type = batch.get('operation_type', torch.zeros(sensor_features.size(0), dtype=torch.long)).to(device)

        # Encode
        with torch.no_grad():
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            op_pred = op_logits.argmax(-1)

        # Decode with each strategy
        predictions = {}
        for name, decode_fn in strategies.items():
            with torch.no_grad():
                pred = decode_fn(decoder, sensor_emb, op_pred)
                predictions[name] = pred

                # Compute metrics
                metrics = compute_metrics(pred, target_tokens, id2token)
                for k, v in metrics.items():
                    results[name][k].append(v)

        # Store examples (first batch only)
        if len(examples) < 10:
            for b in range(min(5, sensor_features.size(0))):
                example = {
                    'target': tokens_to_gcode(target_tokens[b], id2token),
                    'operation': operation_type[b].item(),
                }
                for name in strategies:
                    example[name] = tokens_to_gcode(predictions[name][b], id2token)
                examples.append(example)

    # Aggregate results
    summary = {}
    for name in strategies:
        total_correct = sum(results[name]['token_correct'])
        total_tokens = sum(results[name]['token_total'])
        total_seq_correct = sum(results[name]['seq_correct'])
        total_samples = sum(results[name]['n_samples'])

        summary[name] = {
            'token_accuracy': total_correct / max(total_tokens, 1),
            'sequence_accuracy': total_seq_correct / max(total_samples, 1),
            'n_samples': total_samples,
        }

    # Print results
    print("\n" + "=" * 70)
    print("DECODING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\nDataset: {args.split_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {summary[list(strategies.keys())[0]]['n_samples']}")
    print("\n" + "-" * 70)
    print(f"{'Strategy':<20} {'Token Acc':<15} {'Seq Acc':<15}")
    print("-" * 70)

    for name in strategies:
        s = summary[name]
        print(f"{name:<20} {s['token_accuracy']*100:>10.2f}%    {s['sequence_accuracy']*100:>10.2f}%")

    print("-" * 70)

    # Find best strategy
    best_token = max(summary.items(), key=lambda x: x[1]['token_accuracy'])
    best_seq = max(summary.items(), key=lambda x: x[1]['sequence_accuracy'])
    print(f"\nBest token accuracy: {best_token[0]} ({best_token[1]['token_accuracy']*100:.2f}%)")
    print(f"Best sequence accuracy: {best_seq[0]} ({best_seq[1]['sequence_accuracy']*100:.2f}%)")

    # Show examples
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    for i, ex in enumerate(examples[:5]):
        print(f"\nExample {i+1} (Op: {ex['operation']}):")
        print(f"  Target:      {ex['target']}")
        for name in strategies:
            match = "✓" if ex[name] == ex['target'] else "✗"
            print(f"  {name:<12}: {ex[name]} {match}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'comparison_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'examples': examples,
            'config': vars(args),
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
