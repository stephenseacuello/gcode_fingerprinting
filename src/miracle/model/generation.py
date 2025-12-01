"""
Enhanced generation methods for G-code language model.

Implements:
- Temperature sampling
- Nucleus (top-p) sampling
- Beam search decoding

These methods fix the autoregressive generation issue where the model
gets stuck in <SOS> loops during inference.

Author: Claude Code
Date: November 2025
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> torch.Tensor:
    """
    Sample next token with temperature and nucleus (top-p) sampling.

    Args:
        logits: [vocab_size] unnormalized logits
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold (keep top tokens with cumulative prob >= top_p)

    Returns:
        Sampled token ID [1]
    """
    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Nucleus (top-p) sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)

        # Find cutoff index
        cutoff_idx = torch.where(cumsum_probs >= top_p)[0]
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0].item() + 1
        else:
            cutoff_idx = len(sorted_probs)

        # Zero out probabilities below threshold
        sorted_probs[cutoff_idx:] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample from filtered distribution
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices[sampled_idx]
    else:
        # Regular temperature sampling
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate_with_sampling(
    model_generate_fn,
    memory: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, List[float]]:
    """
    Generate tokens using temperature and nucleus sampling.

    This is a drop-in replacement for greedy generation that adds
    randomness to prevent getting stuck in loops.

    Args:
        model_generate_fn: Function that computes logits given current sequence
        memory: Encoder memory [B, T, d_model]
        max_len: Maximum sequence length
        bos_id: Beginning-of-sequence token ID
        eos_id: End-of-sequence token ID
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        generated: [B, seq_len] generated token IDs
        entropies: List of entropy values at each step (for analysis)
    """
    batch_size = memory.size(0)
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    entropies = []

    for step in range(max_len):
        # Get logits for next token
        logits = model_generate_fn(generated, memory)  # [B, vocab_size]

        # Calculate entropy (for monitoring)
        probs = F.softmax(logits / temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropies.append(entropy.mean().item())

        # Sample next token for each batch item
        next_tokens = []
        for b in range(batch_size):
            next_token = sample_with_temperature(
                logits[b],
                temperature=temperature,
                top_p=top_p
            )
            next_tokens.append(next_token)

        next_tokens = torch.stack(next_tokens, dim=0)  # [B, 1]
        generated = torch.cat([generated, next_tokens], dim=1)

        # Stop if all sequences generated EOS
        if (next_tokens == eos_id).all():
            break

    return generated, entropies


def beam_search_decode(
    model_generate_fn,
    memory: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    beam_size: int = 5,
    length_penalty: float = 0.6,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Beam search decoding for better generation quality.

    Maintains top-k most likely sequences at each step.

    Args:
        model_generate_fn: Function that computes logits given current sequence
        memory: Encoder memory [B, T, d_model]
        max_len: Maximum sequence length
        bos_id: Beginning-of-sequence token ID
        eos_id: End-of-sequence token ID
        beam_size: Number of beams to maintain
        length_penalty: Penalty for longer sequences (< 1.0 encourages longer)
        device: Device to run on

    Returns:
        best_sequence: [B, seq_len] best generated sequence
    """
    batch_size = memory.size(0)

    # For simplicity, only support batch_size=1 for now
    if batch_size > 1:
        # Fall back to greedy for batched inference
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = model_generate_fn(generated, memory)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated

    # Initialize beams
    sequences = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(beam_size, device=device)
    completed_sequences = []
    completed_scores = []

    # Expand memory for beam search
    memory_expanded = memory.repeat(beam_size, 1, 1)  # [beam_size, T, d_model]

    for step in range(max_len):
        # Get logits for all beams
        logits = model_generate_fn(sequences, memory_expanded)  # [beam_size, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate scores for all possible next tokens
        vocab_size = log_probs.size(-1)

        if step == 0:
            # First step: all beams are identical, only use first beam
            next_scores = scores[0].unsqueeze(0) + log_probs[0]  # [vocab_size]
            next_scores = next_scores.view(-1)  # [vocab_size]
        else:
            # Subsequent steps: consider all beam * vocab combinations
            next_scores = scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
            next_scores = next_scores.view(-1)  # [beam_size * vocab_size]

        # Select top beam_size candidates
        top_scores, top_indices = torch.topk(next_scores, beam_size)

        # Convert flat indices to (beam_idx, token_idx)
        if step == 0:
            beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
            token_indices = top_indices
        else:
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

        # Build new sequences
        new_sequences = []
        new_scores = []

        for i in range(beam_size):
            beam_idx = beam_indices[i]
            token_idx = token_indices[i]

            # Extend sequence
            if step == 0:
                seq = torch.cat([sequences[0:1], token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
            else:
                seq = torch.cat([sequences[beam_idx:beam_idx+1], token_idx.unsqueeze(0).unsqueeze(0)], dim=1)

            # Check if sequence is completed
            if token_idx == eos_id:
                # Apply length penalty
                length = seq.size(1)
                score = top_scores[i] / (length ** length_penalty)
                completed_sequences.append(seq)
                completed_scores.append(score)
            else:
                new_sequences.append(seq)
                new_scores.append(top_scores[i])

        # Update beams
        if len(new_sequences) == 0:
            # All beams completed
            break

        # Pad new sequences to same length if necessary
        max_seq_len = max(seq.size(1) for seq in new_sequences)
        sequences = torch.zeros(len(new_sequences), max_seq_len, dtype=torch.long, device=device)
        for i, seq in enumerate(new_sequences):
            sequences[i, :seq.size(1)] = seq

        scores = torch.tensor(new_scores, device=device)

        # Adjust beam size if we have fewer beams
        if sequences.size(0) < beam_size:
            beam_size = sequences.size(0)
            memory_expanded = memory.repeat(beam_size, 1, 1)

    # Select best completed sequence
    if completed_sequences:
        best_idx = torch.argmax(torch.tensor(completed_scores))
        return completed_sequences[best_idx].unsqueeze(0)
    else:
        # No completed sequences, return best partial sequence
        if len(new_sequences) > 0:
            best_idx = torch.argmax(scores)
            return sequences[best_idx:best_idx+1]
        else:
            # Fallback: return initial token
            return torch.full((1, 1), bos_id, dtype=torch.long, device=device)
