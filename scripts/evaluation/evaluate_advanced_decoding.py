#!/usr/bin/env python3
"""
Advanced Decoding Methods for Sequence-Level Accuracy Improvement.

Implements multiple decoding strategies beyond greedy/beam search:
1. Constrained Decoding - Enforce G-code grammar during generation
2. MBR (Minimum Bayes Risk) - Sample multiple sequences, pick consensus
3. Temperature Sampling with Voting - Sample with temperature, majority vote
4. Top-k/Top-p (Nucleus) Sampling with Voting

Usage:
    python scripts/evaluate_advanced_decoding.py [--debug] [--method METHOD]

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
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


# ============================================================================
# MM-DTAE-LSTM Encoder
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
        x = self.input_proj(x)
        encoded, (h_n, c_n) = self.encoder_lstm(x)
        latent = self.bottleneck(encoded)
        return latent, (h_n, c_n)

    def classify(self, latent: torch.Tensor) -> tuple:
        attn_weights = self.temporal_attention(latent)
        pooled = (latent * attn_weights).sum(dim=1)
        logits = self.classification_head(pooled)
        return logits, attn_weights


# ============================================================================
# G-code Grammar for Constrained Decoding
# ============================================================================

class GCodeGrammar:
    """
    G-code grammar constraints for valid sequence generation.

    Grammar rules:
    1. Sequence starts with: command (G0/G1/G2/G3/G53/M30) OR parameter (X/Y/Z/R/F)
    2. After command: parameter OR EOS
    3. After parameter (X/Y/Z/R/F): NUM_* value for that parameter OR another parameter OR EOS
    4. After NUM_*: another parameter OR EOS
    5. EOS is always valid (allows early termination)
    """

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

        # Categorize tokens by type
        self.special_ids = {vocab.get(t, -1) for t in ['PAD', 'BOS', 'EOS', 'UNK', 'MASK']}
        self.command_ids = {vocab[t] for t in vocab if t.startswith('G') or t.startswith('M')
                           if t not in ['MASK']}
        self.param_ids = {vocab[t] for t in ['X', 'Y', 'Z', 'R', 'F'] if t in vocab}

        # Build NUM token mapping: param -> set of valid NUM tokens
        self.param_to_nums = defaultdict(set)
        for token, tid in vocab.items():
            if token.startswith('NUM_'):
                # Extract parameter type: NUM_X_1234 -> X
                parts = token.split('_')
                if len(parts) >= 2:
                    param = parts[1]
                    self.param_to_nums[param].add(tid)

        # Special tokens
        self.bos_id = vocab.get('BOS', 1)
        self.eos_id = vocab.get('EOS', 2)
        self.pad_id = vocab.get('PAD', 0)
        self.unk_id = vocab.get('UNK', 3)

    def get_valid_next_tokens(self, generated_sequence: list) -> set:
        """
        Given the generated sequence so far, return set of valid next token IDs.

        Args:
            generated_sequence: List of token IDs generated so far (including BOS)

        Returns:
            Set of valid token IDs for the next position
        """
        if len(generated_sequence) == 0:
            return {self.bos_id}

        # After BOS: command or parameter
        if len(generated_sequence) == 1 and generated_sequence[0] == self.bos_id:
            return self.command_ids | self.param_ids | {self.eos_id}

        last_token_id = generated_sequence[-1]
        last_token = self.id_to_token.get(last_token_id, '')

        # After EOS: only PAD
        if last_token_id == self.eos_id:
            return {self.pad_id}

        # After PAD: only PAD
        if last_token_id == self.pad_id:
            return {self.pad_id}

        # After command (G0, G1, etc): parameter or EOS
        if last_token_id in self.command_ids:
            return self.param_ids | {self.eos_id}

        # After parameter (X, Y, Z, R, F): NUM value for that param, or another param, or EOS
        if last_token_id in self.param_ids:
            valid = self.param_to_nums.get(last_token, set()).copy()
            # Also allow other parameters or EOS (in case of implicit values)
            valid |= self.param_ids | {self.eos_id, self.unk_id}
            return valid

        # After NUM_*: another parameter or EOS
        if last_token.startswith('NUM_'):
            return self.param_ids | {self.eos_id}

        # Default: allow everything except special
        return self.command_ids | self.param_ids | {self.eos_id}

    def create_mask(self, generated_sequence: list, vocab_size: int, device: str) -> torch.Tensor:
        """
        Create a binary mask where 1 = valid token, 0 = invalid token.
        """
        valid_ids = self.get_valid_next_tokens(generated_sequence)
        mask = torch.zeros(vocab_size, device=device)
        for tid in valid_ids:
            if 0 <= tid < vocab_size:
                mask[tid] = 1.0
        return mask


# ============================================================================
# Decoding Methods
# ============================================================================

def greedy_decode(decoder, encoder, sensor_features, operation_type,
                  bos_id, eos_id, pad_id, max_len, device):
    """Standard greedy decoding."""
    B = sensor_features.size(0)
    sensor_emb, _ = encoder.encode(sensor_features)

    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )
        logits = outputs['legacy_logits'][:, -1, :]
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    return generated


def constrained_greedy_decode(decoder, encoder, sensor_features, operation_type,
                              grammar, bos_id, eos_id, pad_id, max_len, device):
    """
    Greedy decoding with grammar constraints.
    Invalid tokens are masked to -inf before argmax.
    """
    B = sensor_features.size(0)
    sensor_emb, _ = encoder.encode(sensor_features)
    vocab_size = decoder.vocab_size

    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )
        logits = outputs['legacy_logits'][:, -1, :]  # [B, vocab_size]

        # Apply grammar constraints per batch item
        for b in range(B):
            if not finished[b]:
                seq = generated[b].cpu().tolist()
                mask = grammar.create_mask(seq, vocab_size, device)
                # Mask invalid tokens to -inf
                logits[b] = logits[b].masked_fill(mask == 0, float('-inf'))

        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    return generated


def sample_decode(decoder, encoder, sensor_features, operation_type,
                  bos_id, eos_id, pad_id, max_len, device,
                  temperature=1.0, top_k=0, top_p=0.0):
    """
    Sample-based decoding with temperature, top-k, and top-p (nucleus) sampling.
    """
    B = sensor_features.size(0)
    sensor_emb, _ = encoder.encode(sensor_features)

    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        outputs = decoder(
            tokens=generated,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )
        logits = outputs['legacy_logits'][:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    return generated


def mbr_decode(decoder, encoder, sensor_features, operation_type,
               bos_id, eos_id, pad_id, max_len, device,
               n_samples=10, temperature=0.8):
    """
    Minimum Bayes Risk (MBR) decoding.

    1. Sample n_samples sequences
    2. Compute pairwise similarities (token overlap)
    3. Return the sequence with highest average similarity to others
    """
    B = sensor_features.size(0)

    # Generate multiple samples
    samples = []
    for _ in range(n_samples):
        sample = sample_decode(
            decoder, encoder, sensor_features, operation_type,
            bos_id, eos_id, pad_id, max_len, device,
            temperature=temperature, top_k=50
        )
        samples.append(sample[:, 1:])  # Remove BOS

    # For each batch item, find the MBR sequence
    best_sequences = []
    for b in range(B):
        batch_samples = [s[b].cpu().tolist() for s in samples]

        # Compute pairwise token overlap (Jaccard-like similarity)
        def similarity(seq1, seq2):
            # Remove padding
            s1 = [t for t in seq1 if t != pad_id]
            s2 = [t for t in seq2 if t != pad_id]
            if len(s1) == 0 or len(s2) == 0:
                return 0.0
            # Token-level F1
            matches = sum(1 for i, t in enumerate(s1) if i < len(s2) and s2[i] == t)
            precision = matches / len(s1) if len(s1) > 0 else 0
            recall = matches / len(s2) if len(s2) > 0 else 0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Find sequence with highest average similarity
        best_idx = 0
        best_score = -1
        for i, seq_i in enumerate(batch_samples):
            avg_sim = np.mean([similarity(seq_i, seq_j) for j, seq_j in enumerate(batch_samples) if i != j])
            if avg_sim > best_score:
                best_score = avg_sim
                best_idx = i

        best_sequences.append(samples[best_idx][b])

    # Pad to same length
    max_seq_len = max(seq.size(0) for seq in best_sequences)
    result = torch.full((B, max_seq_len), pad_id, dtype=torch.long, device=device)
    for b, seq in enumerate(best_sequences):
        result[b, :seq.size(0)] = seq

    return result


def voting_decode(decoder, encoder, sensor_features, operation_type,
                  bos_id, eos_id, pad_id, max_len, device,
                  n_samples=5, temperature=0.7):
    """
    Voting-based decoding: sample multiple sequences, vote on each position.
    """
    B = sensor_features.size(0)

    # Generate multiple samples
    samples = []
    for _ in range(n_samples):
        sample = sample_decode(
            decoder, encoder, sensor_features, operation_type,
            bos_id, eos_id, pad_id, max_len, device,
            temperature=temperature, top_k=30
        )
        samples.append(sample[:, 1:])  # Remove BOS

    # Vote on each position
    max_sample_len = max(s.size(1) for s in samples)
    result = torch.full((B, max_sample_len), pad_id, dtype=torch.long, device=device)

    for b in range(B):
        for pos in range(max_sample_len):
            votes = []
            for sample in samples:
                if pos < sample.size(1):
                    votes.append(sample[b, pos].item())
            if votes:
                # Majority vote
                vote_counts = Counter(votes)
                result[b, pos] = vote_counts.most_common(1)[0][0]

    return result


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_token_accuracy(pred_tokens, target_tokens, pad_id):
    valid_mask = target_tokens != pad_id
    if valid_mask.sum() == 0:
        return 0.0
    correct = (pred_tokens[valid_mask] == target_tokens[valid_mask]).float()
    return correct.mean().item()


def compute_sequence_accuracy(pred_tokens, target_tokens, pad_id, eos_id):
    B = pred_tokens.size(0)
    matches = 0

    for b in range(B):
        pred_seq = pred_tokens[b]
        target_seq = target_tokens[b]

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

        if pred_end == target_end:
            if torch.equal(pred_seq[:pred_end], target_seq[:target_end]):
                matches += 1

    return matches / B


def check_grammar_validity(pred_tokens, grammar, pad_id, eos_id):
    """Check if generated sequence follows grammar rules."""
    B = pred_tokens.size(0)
    valid_count = 0

    for b in range(B):
        seq = pred_tokens[b].cpu().tolist()
        is_valid = True

        # Check each transition
        for i in range(len(seq) - 1):
            if seq[i] == pad_id or seq[i] == eos_id:
                break
            current_seq = [grammar.bos_id] + seq[:i+1]
            valid_next = grammar.get_valid_next_tokens(current_seq)
            if seq[i+1] not in valid_next and seq[i+1] != pad_id:
                is_valid = False
                break

        if is_valid:
            valid_count += 1

    return valid_count / B


# ============================================================================
# Main Evaluation
# ============================================================================

def load_models(model_dir, encoder_path, vocab_path, device):
    decomposer = TokenDecomposer(str(vocab_path))

    results_path = model_dir / 'results.json'
    with open(results_path) as f:
        results = json.load(f)
    args = results.get('args', {})

    encoder = MM_DTAE_LSTM(
        input_dim=155,
        hidden_dim=256,
        latent_dim=128,
        num_lstm_layers=2,
        n_classes=9,
        dropout=0.3,
    ).to(device)

    encoder_ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    encoder.eval()

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
        use_sensor_prior=args.get('use_sensor_prior', True),
    ).to(device)

    decoder_ckpt = torch.load(model_dir / 'best_model.pt', map_location=device, weights_only=False)
    # Use strict=False to handle type_token_mask buffer
    decoder.load_state_dict(decoder_ckpt['model_state_dict'], strict=False)

    # Set vocab to rebuild type_token_mask if the model has that method
    if hasattr(decoder, 'set_vocab'):
        decoder.set_vocab(decomposer.vocab)

    decoder.eval()

    return encoder, decoder, decomposer, args


def evaluate_all_methods(encoder, decoder, test_dataset, decomposer, grammar, device, max_samples=None):
    """Evaluate all decoding methods."""
    bos_id = decomposer.vocab.get('<BOS>', decomposer.vocab.get('BOS', 1))
    eos_id = decomposer.vocab.get('<EOS>', decomposer.vocab.get('EOS', 2))
    pad_id = decomposer.vocab.get('<PAD>', decomposer.vocab.get('PAD', 0))

    methods = {
        'greedy': {},
        'constrained_greedy': {},
        'sample_t0.7': {},
        'sample_t0.5': {},
        'voting_n5': {},
        'voting_n10': {},
        'mbr_n5': {},
        'mbr_n10': {},
    }

    for method in methods:
        methods[method] = {'token_acc': [], 'seq_acc': [], 'grammar_valid': []}

    n_samples = min(len(test_dataset), max_samples) if max_samples else len(test_dataset)
    print(f"Evaluating {n_samples} test samples with {len(methods)} methods...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Evaluating"):
            sample = test_dataset[idx]

            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)
            target_tokens = sample['target_tokens'].unsqueeze(0).to(device)
            op_type = sample['operation_type'].unsqueeze(0).to(device)

            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

            max_len = target_tokens.size(1)
            target_len = target_tokens.size(1)

            # 1. Greedy
            pred = greedy_decode(decoder, encoder, sensor_features, operation_type,
                                bos_id, eos_id, pad_id, max_len + 1, device)
            pred = pred[:, 1:]  # Remove BOS
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['greedy']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['greedy']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['greedy']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 2. Constrained Greedy
            pred = constrained_greedy_decode(decoder, encoder, sensor_features, operation_type,
                                             grammar, bos_id, eos_id, pad_id, max_len + 1, device)
            pred = pred[:, 1:]
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['constrained_greedy']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['constrained_greedy']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['constrained_greedy']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 3. Sampling with temperature 0.7
            pred = sample_decode(decoder, encoder, sensor_features, operation_type,
                                bos_id, eos_id, pad_id, max_len + 1, device,
                                temperature=0.7, top_k=50)
            pred = pred[:, 1:]
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['sample_t0.7']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['sample_t0.7']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['sample_t0.7']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 4. Sampling with temperature 0.5
            pred = sample_decode(decoder, encoder, sensor_features, operation_type,
                                bos_id, eos_id, pad_id, max_len + 1, device,
                                temperature=0.5, top_k=30)
            pred = pred[:, 1:]
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['sample_t0.5']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['sample_t0.5']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['sample_t0.5']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 5. Voting (n=5)
            pred = voting_decode(decoder, encoder, sensor_features, operation_type,
                                bos_id, eos_id, pad_id, max_len + 1, device,
                                n_samples=5, temperature=0.7)
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['voting_n5']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['voting_n5']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['voting_n5']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 6. Voting (n=10)
            pred = voting_decode(decoder, encoder, sensor_features, operation_type,
                                bos_id, eos_id, pad_id, max_len + 1, device,
                                n_samples=10, temperature=0.7)
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['voting_n10']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['voting_n10']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['voting_n10']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 7. MBR (n=5)
            pred = mbr_decode(decoder, encoder, sensor_features, operation_type,
                             bos_id, eos_id, pad_id, max_len + 1, device,
                             n_samples=5, temperature=0.8)
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['mbr_n5']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['mbr_n5']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['mbr_n5']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

            # 8. MBR (n=10)
            pred = mbr_decode(decoder, encoder, sensor_features, operation_type,
                             bos_id, eos_id, pad_id, max_len + 1, device,
                             n_samples=10, temperature=0.8)
            if pred.size(1) < target_len:
                pred = F.pad(pred, (0, target_len - pred.size(1)), value=pad_id)
            elif pred.size(1) > target_len:
                pred = pred[:, :target_len]
            methods['mbr_n10']['token_acc'].append(compute_token_accuracy(pred, target_tokens, pad_id))
            methods['mbr_n10']['seq_acc'].append(compute_sequence_accuracy(pred, target_tokens, pad_id, eos_id))
            methods['mbr_n10']['grammar_valid'].append(check_grammar_validity(pred, grammar, pad_id, eos_id))

    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run on small subset')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    parser.add_argument('--encoder-path', type=str, default=None, help='Encoder checkpoint path')
    parser.add_argument('--split-dir', type=str, default=None, help='Split directory')
    parser.add_argument('--vocab-path', type=str, default=None, help='Vocabulary path')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Use command line args or defaults (updated to full vocab paths)
    model_dir = Path(args.model_dir) if args.model_dir else base_dir / 'outputs' / 'breakthrough_full_vocab'
    encoder_path = Path(args.encoder_path) if args.encoder_path else base_dir / 'outputs' / 'mm_dtae_lstm_v2' / 'best_model.pt'
    split_dir = Path(args.split_dir) if args.split_dir else base_dir / 'outputs' / 'stratified_splits_full_vocab'
    vocab_path = Path(args.vocab_path) if args.vocab_path else base_dir / 'data' / 'vocabulary_4digit_full.json'

    device = get_device()
    print(f"Using device: {device}")

    print("\nLoading models...")
    encoder, decoder, decomposer, model_args = load_models(
        model_dir, encoder_path, vocab_path, device
    )

    # Create grammar
    grammar = GCodeGrammar(decomposer.vocab)
    print(f"Grammar initialized with {len(grammar.command_ids)} commands, {len(grammar.param_ids)} params")

    print("\nLoading test data...")
    test_dataset = DecoderDatasetFromSplits(split_dir, 'test', max_token_len=model_args.get('max_seq_len', 32))
    print(f"  Test samples: {len(test_dataset)}")

    max_samples = args.samples or (20 if args.debug else None)

    print("\n" + "="*70)
    print("EVALUATING ADVANCED DECODING METHODS")
    print("="*70)

    results = evaluate_all_methods(
        encoder, decoder, test_dataset, decomposer, grammar, device,
        max_samples=max_samples
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'Method':<20} {'Token Acc':<12} {'Seq Acc':<12} {'Grammar %':<12}")
    print("-"*56)

    for method, metrics in results.items():
        token_acc = np.mean(metrics['token_acc']) * 100
        seq_acc = np.mean(metrics['seq_acc']) * 100
        grammar_valid = np.mean(metrics['grammar_valid']) * 100
        print(f"{method:<20} {token_acc:>10.2f}%  {seq_acc:>10.2f}%  {grammar_valid:>10.1f}%")

    # Save results
    reports_dir = base_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / 'advanced_decoding_results.json'
    summary = {
        method: {
            'token_accuracy': float(np.mean(metrics['token_acc'])),
            'sequence_accuracy': float(np.mean(metrics['seq_acc'])),
            'grammar_validity': float(np.mean(metrics['grammar_valid'])),
            'token_accuracy_std': float(np.std(metrics['token_acc'])),
            'sequence_accuracy_std': float(np.std(metrics['seq_acc'])),
        }
        for method, metrics in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    greedy_seq = np.mean(results['greedy']['seq_acc']) * 100
    best_method = max(results.keys(), key=lambda m: np.mean(results[m]['seq_acc']))
    best_seq = np.mean(results[best_method]['seq_acc']) * 100

    print(f"\nBest method: {best_method}")
    print(f"  Greedy seq acc: {greedy_seq:.2f}%")
    print(f"  Best seq acc:   {best_seq:.2f}%")
    print(f"  Improvement:    +{best_seq - greedy_seq:.2f}%")


if __name__ == '__main__':
    main()
