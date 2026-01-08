#!/usr/bin/env python3
"""
Evaluate beam search vs greedy decoding for sequence-level accuracy.

This script compares:
1. Greedy decoding (argmax at each step)
2. Beam search with various beam widths

Key insight: With 90% token accuracy, greedy achieves ~60% sequence accuracy
(0.9^5 ≈ 0.59 for 5-token sequences). Beam search can recover from early errors
by maintaining multiple hypotheses.

Usage:
    python scripts/evaluate_beam_search.py

Author: Claude Code
Date: December 2025
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


# ============================================================================
# MM-DTAE-LSTM Encoder (copied from training script for consistency)
# ============================================================================

class MM_DTAE_LSTM(nn.Module):
    """MM-DTAE-LSTM encoder for sensor feature extraction.

    This must match the architecture from train_sensor_multihead.py exactly.
    """

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


# ============================================================================
# Beam Search Implementation for Multi-Head Decoder
# ============================================================================

def greedy_decode(decoder, encoder, sensor_features, operation_type,
                  bos_id, eos_id, pad_id, max_len, device):
    """
    Greedy decoding (baseline) - take argmax at each step.

    Returns:
        generated: [B, L] token IDs
        probs: [B, L] probabilities of selected tokens
    """
    B = sensor_features.size(0)

    # Get sensor embeddings
    sensor_emb, _ = encoder.encode(sensor_features)

    # Initialize
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    all_probs = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        # Forward through decoder
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )

        # Get logits for last position
        logits = outputs['legacy_logits'][:, -1, :]  # [B, vocab_size]
        probs = F.softmax(logits, dim=-1)

        # Greedy selection
        next_token = logits.argmax(dim=-1)  # [B]
        next_prob = probs.gather(1, next_token.unsqueeze(1)).squeeze(1)  # [B]

        # Handle finished sequences
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)

        # Append
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        all_probs.append(next_prob)

        # Check for EOS
        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    return generated, torch.stack(all_probs, dim=1) if all_probs else None


def beam_search_decode(decoder, encoder, sensor_features, operation_type,
                       bos_id, eos_id, pad_id, max_len, beam_width,
                       length_penalty=0.6, device='cpu'):
    """
    Beam search decoding for multi-head decoder.

    Maintains top-k hypotheses at each step to find globally better sequences.

    Args:
        decoder: SensorMultiHeadDecoder
        encoder: MM_DTAE_LSTM encoder
        sensor_features: [B, T_s, 155] sensor data
        operation_type: [B] operation type indices
        bos_id, eos_id, pad_id: special token IDs
        max_len: maximum sequence length
        beam_width: number of beams (k)
        length_penalty: α for length normalization (score / length^α)
        device: computation device

    Returns:
        best_sequences: [B, L] best sequence for each batch item
        best_scores: [B] normalized scores
    """
    B = sensor_features.size(0)

    # Get sensor embeddings (shared across all beams)
    sensor_emb, _ = encoder.encode(sensor_features)

    # Process each batch item separately (beam search is per-sample)
    all_best_sequences = []
    all_best_scores = []

    for b in range(B):
        # Get embeddings for this sample
        sample_sensor_emb = sensor_emb[b:b+1]  # [1, T_s, latent_dim]
        sample_op_type = operation_type[b:b+1]  # [1]

        # Expand for beam search
        beam_sensor_emb = sample_sensor_emb.expand(beam_width, -1, -1)  # [beam, T_s, latent_dim]
        beam_op_type = sample_op_type.expand(beam_width)  # [beam]

        # Initialize beams
        # Each beam: (sequence, log_prob, finished)
        beams = [(torch.tensor([[bos_id]], device=device), 0.0, False)]

        completed_beams = []

        for step in range(max_len - 1):
            if len(beams) == 0:
                break

            all_candidates = []

            for seq, score, finished in beams:
                if finished:
                    completed_beams.append((seq, score, True))
                    continue

                # Forward through decoder (single sequence)
                outputs = decoder(
                    tokens=seq,
                    sensor_embeddings=sample_sensor_emb,
                    operation_type=sample_op_type,
                )

                # Get logits for last position
                logits = outputs['legacy_logits'][0, -1, :]  # [vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)

                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    token_id = topk_indices[k].item()
                    token_log_prob = topk_log_probs[k].item()

                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    new_score = score + token_log_prob
                    new_finished = (token_id == eos_id)

                    all_candidates.append((new_seq, new_score, new_finished))

            # Sort candidates by score and keep top beam_width
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # Move completed beams to completed list
            new_beams = []
            for seq, score, finished in beams:
                if finished:
                    completed_beams.append((seq, score, True))
                else:
                    new_beams.append((seq, score, finished))
            beams = new_beams

        # Add any remaining unfinished beams
        completed_beams.extend(beams)

        # Select best sequence with length normalization
        best_seq = None
        best_normalized_score = float('-inf')

        for seq, score, _ in completed_beams:
            length = seq.size(1)
            normalized_score = score / (length ** length_penalty)
            if normalized_score > best_normalized_score:
                best_normalized_score = normalized_score
                best_seq = seq

        if best_seq is None:
            # Fallback: just BOS
            best_seq = torch.tensor([[bos_id]], device=device)
            best_normalized_score = 0.0

        all_best_sequences.append(best_seq.squeeze(0))  # Remove batch dim
        all_best_scores.append(best_normalized_score)

    # Pad sequences to same length
    max_seq_len = max(seq.size(0) for seq in all_best_sequences)
    padded_sequences = torch.full((B, max_seq_len), pad_id, dtype=torch.long, device=device)
    for b, seq in enumerate(all_best_sequences):
        padded_sequences[b, :seq.size(0)] = seq

    return padded_sequences, torch.tensor(all_best_scores, device=device)


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_token_accuracy(pred_tokens, target_tokens, pad_id):
    """Compute token-level accuracy ignoring padding."""
    valid_mask = target_tokens != pad_id
    if valid_mask.sum() == 0:
        return 0.0
    correct = (pred_tokens[valid_mask] == target_tokens[valid_mask]).float()
    return correct.mean().item()


def compute_sequence_accuracy(pred_tokens, target_tokens, pad_id, eos_id):
    """
    Compute exact sequence match accuracy.

    Two sequences match if all non-padding tokens are identical.
    """
    B = pred_tokens.size(0)
    matches = 0

    for b in range(B):
        # Get actual sequence lengths (up to EOS or first PAD)
        pred_seq = pred_tokens[b]
        target_seq = target_tokens[b]

        # Find sequence ends
        pred_end = (pred_seq == eos_id).nonzero()
        if len(pred_end) > 0:
            pred_end = pred_end[0].item() + 1
        else:
            pred_end = (pred_seq != pad_id).sum().item()

        target_end = (target_seq == eos_id).nonzero()
        if len(target_end) > 0:
            target_end = target_end[0].item() + 1
        else:
            target_end = (target_seq != pad_id).sum().item()

        # Compare sequences
        if pred_end == target_end:
            if torch.equal(pred_seq[:pred_end], target_seq[:target_end]):
                matches += 1

    return matches / B


# ============================================================================
# Main Evaluation
# ============================================================================

def load_models(model_dir, encoder_path, vocab_path, device):
    """Load encoder and decoder models."""
    # Load vocabulary - TokenDecomposer expects path, not dict
    decomposer = TokenDecomposer(str(vocab_path))

    # Load decoder config
    results_path = model_dir / 'results.json'
    with open(results_path) as f:
        results = json.load(f)
    args = results.get('args', {})

    # Create encoder
    encoder = MM_DTAE_LSTM(
        input_dim=155,
        hidden_dim=256,
        latent_dim=128,
        num_lstm_layers=2,
        n_classes=9,
        dropout=0.3,
    ).to(device)

    # Load encoder weights
    encoder_ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
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

    # Load decoder weights
    decoder_ckpt = torch.load(model_dir / 'best_model.pt', map_location=device, weights_only=False)
    decoder.load_state_dict(decoder_ckpt['model_state_dict'])
    decoder.eval()

    return encoder, decoder, decomposer, args


def evaluate_decoding_methods(encoder, decoder, test_dataset, decomposer, device, max_samples=None, verbose=False):
    """
    Evaluate greedy vs beam search decoding.
    """
    # Get special token IDs
    bos_id = decomposer.vocab.get('<BOS>', 1)
    eos_id = decomposer.vocab.get('<EOS>', 2)
    pad_id = decomposer.vocab.get('<PAD>', 0)

    # Get token-to-id mapping for verbose output
    id_to_token = {v: k for k, v in decomposer.vocab.items()}

    # Results storage
    results = {
        'greedy': {'token_acc': [], 'seq_acc': []},
        'beam_2': {'token_acc': [], 'seq_acc': []},
        'beam_3': {'token_acc': [], 'seq_acc': []},
        'beam_5': {'token_acc': [], 'seq_acc': []},
        'beam_6': {'token_acc': [], 'seq_acc': []},
    }

    n_samples = min(len(test_dataset), max_samples) if max_samples else len(test_dataset)
    print(f"Evaluating {n_samples} test samples...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Evaluating"):
            sample = test_dataset[idx]

            # Prepare inputs
            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)
            target_tokens = sample['target_tokens'].unsqueeze(0).to(device)
            op_type = sample['operation_type'].unsqueeze(0).to(device)

            # Get encoder operation classification
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

            max_len = target_tokens.size(1)

            # Greedy decoding
            greedy_pred, _ = greedy_decode(
                decoder, encoder, sensor_features, operation_type,
                bos_id, eos_id, pad_id, max_len + 1, device  # +1 to account for BOS
            )

            # Remove BOS from predictions (target_tokens doesn't have BOS)
            # greedy_pred is [BOS, p1, p2, ...], target_tokens is [t1, t2, ..., EOS]
            greedy_pred = greedy_pred[:, 1:]  # Now [p1, p2, ...]

            # Align lengths for comparison
            pred_len = greedy_pred.size(1)
            target_len = target_tokens.size(1)
            if pred_len < target_len:
                greedy_pred = F.pad(greedy_pred, (0, target_len - pred_len), value=pad_id)
            elif pred_len > target_len:
                greedy_pred = greedy_pred[:, :target_len]

            token_acc = compute_token_accuracy(greedy_pred, target_tokens, pad_id)
            seq_acc = compute_sequence_accuracy(greedy_pred, target_tokens, pad_id, eos_id)
            results['greedy']['token_acc'].append(token_acc)
            results['greedy']['seq_acc'].append(seq_acc)

            # Verbose output for first few samples
            if verbose and idx < 5:
                print(f"\n--- Sample {idx} ---")
                pred_toks = greedy_pred[0].cpu().tolist()
                tgt_toks = target_tokens[0].cpu().tolist()
                pred_str = ' '.join([id_to_token.get(t, f'<{t}>') for t in pred_toks if t != pad_id])
                tgt_str = ' '.join([id_to_token.get(t, f'<{t}>') for t in tgt_toks if t != pad_id])
                print(f"  Target: {tgt_str}")
                print(f"  Pred:   {pred_str}")
                print(f"  Token acc: {token_acc:.3f}, Seq acc: {seq_acc}")

            # Beam search with different beam widths
            for beam_width, key in [(2, 'beam_2'), (3, 'beam_3'), (5, 'beam_5'), (6, 'beam_6')]:
                beam_pred, _ = beam_search_decode(
                    decoder, encoder, sensor_features, operation_type,
                    bos_id, eos_id, pad_id, max_len + 1, beam_width,  # +1 to account for BOS
                    length_penalty=0.6, device=device
                )

                # Remove BOS from predictions
                beam_pred = beam_pred[:, 1:]

                # Align lengths
                pred_len = beam_pred.size(1)
                if pred_len < target_len:
                    beam_pred = F.pad(beam_pred, (0, target_len - pred_len), value=pad_id)
                elif pred_len > target_len:
                    beam_pred = beam_pred[:, :target_len]

                token_acc = compute_token_accuracy(beam_pred, target_tokens, pad_id)
                seq_acc = compute_sequence_accuracy(beam_pred, target_tokens, pad_id, eos_id)
                results[key]['token_acc'].append(token_acc)
                results[key]['seq_acc'].append(seq_acc)

    return results


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'outputs' / 'sensor_multihead_v3'
    encoder_path = base_dir / 'outputs' / 'mm_dtae_lstm_v2' / 'best_model.pt'
    split_dir = base_dir / 'outputs' / 'stratified_splits_v2'
    vocab_path = base_dir / 'data' / 'vocabulary_4digit_hybrid.json'

    device = get_device()
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    encoder, decoder, decomposer, args = load_models(
        model_dir, encoder_path, vocab_path, device
    )
    print(f"  Encoder: MM-DTAE-LSTM (128-dim latent)")
    print(f"  Decoder: SensorMultiHeadDecoder ({args.get('d_model', 192)}-dim)")
    print(f"  Vocab size: {decomposer.vocab_size}")

    # Load test data
    print("\nLoading test data...")
    test_dataset = DecoderDatasetFromSplits(split_dir, 'test', max_token_len=args.get('max_seq_len', 32))
    print(f"  Test samples: {len(test_dataset)}")

    # Check for debug mode
    import sys
    debug_mode = '--debug' in sys.argv
    max_samples = 20 if debug_mode else None

    # Evaluate
    print("\n" + "="*60)
    print("COMPARING DECODING METHODS")
    print("="*60)

    results = evaluate_decoding_methods(
        encoder, decoder, test_dataset, decomposer, device,
        max_samples=max_samples, verbose=debug_mode
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n{'Method':<15} {'Token Acc':<12} {'Seq Acc':<12}")
    print("-"*40)

    for method, metrics in results.items():
        token_acc = np.mean(metrics['token_acc']) * 100
        seq_acc = np.mean(metrics['seq_acc']) * 100
        print(f"{method:<15} {token_acc:>10.2f}%  {seq_acc:>10.2f}%")

    # Save results
    output_path = model_dir / 'beam_search_results.json'
    summary = {
        method: {
            'token_accuracy': float(np.mean(metrics['token_acc'])),
            'sequence_accuracy': float(np.mean(metrics['seq_acc'])),
            'token_accuracy_std': float(np.std(metrics['token_acc'])),
            'sequence_accuracy_std': float(np.std(metrics['seq_acc'])),
        }
        for method, metrics in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    greedy_seq = np.mean(results['greedy']['seq_acc']) * 100
    beam5_seq = np.mean(results['beam_5']['seq_acc']) * 100
    improvement = beam5_seq - greedy_seq

    print(f"\nSequence accuracy improvement from beam search (k=5):")
    print(f"  Greedy: {greedy_seq:.2f}%")
    print(f"  Beam-5: {beam5_seq:.2f}%")
    print(f"  Improvement: +{improvement:.2f}% ({improvement/greedy_seq*100:.1f}% relative)")

    # Theoretical vs actual
    greedy_token = np.mean(results['greedy']['token_acc'])
    avg_seq_len = 4.5  # Approximate average sequence length
    theoretical_seq = (greedy_token ** avg_seq_len) * 100
    print(f"\nTheoretical sequence accuracy (token_acc^{avg_seq_len:.1f}):")
    print(f"  Expected: {theoretical_seq:.2f}%")
    print(f"  Actual:   {greedy_seq:.2f}%")


if __name__ == '__main__':
    main()
