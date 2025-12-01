"""
Advanced sampling strategies for text generation.

Implements:
- Beam search with length penalty
- Temperature sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Repetition penalty
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BeamHypothesis:
    """Single hypothesis in beam search."""
    tokens: List[int]
    score: float
    log_prob: float

    def __lt__(self, other):
        """Compare by normalized score."""
        return self.score < other.score


class BeamSearchGenerator:
    """
    Beam search generation with length normalization.

    Maintains top-k hypotheses at each step and returns best sequence.
    """

    def __init__(self, beam_width: int = 5, max_length: int = 64,
                 length_penalty: float = 0.6, early_stopping: bool = True):
        """
        Initialize beam search generator.

        Args:
            beam_width: Number of beams to maintain
            max_length: Maximum sequence length
            length_penalty: Length normalization factor (Wu et al., 2016)
            early_stopping: Stop when all beams finish
        """
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def generate(self, model, encoder_output, start_token: int,
                 end_token: int, pad_token: int = 0) -> Tuple[List[int], float]:
        """
        Generate sequence using beam search.

        Args:
            model: Model with forward method
            encoder_output: Encoder hidden states [1, T, D]
            start_token: Start token ID
            end_token: End token ID
            pad_token: Padding token ID

        Returns:
            (best_sequence, best_score)
        """
        device = encoder_output.device
        batch_size = encoder_output.size(0)

        # Initialize beams
        beams = [BeamHypothesis([start_token], 0.0, 0.0) for _ in range(self.beam_width)]
        finished_beams = []

        for step in range(self.max_length):
            if len(beams) == 0:
                break

            # Prepare inputs for all active beams
            beam_tokens = []
            beam_indices = []

            for beam_idx, beam in enumerate(beams):
                if beam.tokens[-1] == end_token:
                    # Beam finished
                    finished_beams.append(beam)
                    continue

                beam_tokens.append(beam.tokens)
                beam_indices.append(beam_idx)

            if len(beam_tokens) == 0:
                break  # All beams finished

            # Get model predictions
            # (This is simplified - actual implementation depends on your model)
            max_len = max(len(tokens) for tokens in beam_tokens)
            input_tensor = torch.zeros(len(beam_tokens), max_len, dtype=torch.long, device=device)
            for i, tokens in enumerate(beam_tokens):
                input_tensor[i, :len(tokens)] = torch.tensor(tokens, device=device)

            with torch.no_grad():
                # Model forward pass (simplified)
                logits = model.decode_step(encoder_output.repeat(len(beam_tokens), 1, 1), input_tensor)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [num_beams, vocab_size]

            # Expand beams
            all_candidates = []

            for i, beam_idx in enumerate(beam_indices):
                beam = beams[beam_idx]
                beam_log_probs = log_probs[i]

                # Get top-k tokens
                topk_log_probs, topk_ids = beam_log_probs.topk(self.beam_width)

                for k in range(self.beam_width):
                    token_id = topk_ids[k].item()
                    token_log_prob = topk_log_probs[k].item()

                    new_tokens = beam.tokens + [token_id]
                    new_log_prob = beam.log_prob + token_log_prob

                    # Compute normalized score
                    new_score = self._normalize_score(new_log_prob, len(new_tokens))

                    all_candidates.append(BeamHypothesis(new_tokens, new_score, new_log_prob))

            # Select top beams
            all_candidates.sort(reverse=True)  # Sort by score
            beams = all_candidates[:self.beam_width]

        # Combine finished and active beams
        all_beams = finished_beams + beams
        all_beams.sort(reverse=True)

        # Return best beam
        if len(all_beams) > 0:
            best_beam = all_beams[0]
            return best_beam.tokens, best_beam.score
        else:
            return [start_token, end_token], 0.0

    def _normalize_score(self, log_prob: float, length: int) -> float:
        """
        Length normalization (Wu et al., 2016).

        score = log_prob / ((5 + length) / 6) ^ alpha
        """
        if self.length_penalty == 0:
            return log_prob

        length_penalty = ((5 + length) / 6) ** self.length_penalty
        return log_prob / length_penalty


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Temperature-based sampling.

    Higher temperature (>1.0) = more random
    Lower temperature (<1.0) = more deterministic

    Args:
        logits: [batch_size, vocab_size] logits
        temperature: Temperature parameter

    Returns:
        Sampled token indices [batch_size]
    """
    if temperature == 0:
        # Greedy decoding
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """
    Top-k sampling.

    Sample from top-k highest probability tokens.

    Args:
        logits: [batch_size, vocab_size] logits
        k: Number of top tokens to consider
        temperature: Temperature scaling

    Returns:
        Sampled token indices [batch_size]
    """
    # Scale by temperature
    scaled_logits = logits / temperature

    # Get top-k
    topk_logits, topk_indices = scaled_logits.topk(k, dim=-1)

    # Sample from top-k
    probs = F.softmax(topk_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original vocabulary
    return topk_indices.gather(-1, sampled_idx).squeeze(-1)


def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.

    Sample from smallest set of tokens with cumulative probability >= p.

    Args:
        logits: [batch_size, vocab_size] logits
        p: Cumulative probability threshold
        temperature: Temperature scaling

    Returns:
        Sampled token indices [batch_size]
    """
    # Scale by temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort probabilities
    sorted_probs, sorted_indices = probs.sort(descending=True, dim=-1)
    cumsum_probs = sorted_probs.cumsum(dim=-1)

    # Find cutoff index
    cutoff_mask = cumsum_probs >= p
    # Keep at least one token
    cutoff_mask[:, 0] = False

    # Set probabilities outside nucleus to zero
    sorted_probs[cutoff_mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    # Map back to original indices
    return sorted_indices.gather(-1, sampled_sorted_idx).squeeze(-1)


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: List[int],
                            penalty: float = 1.2) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Reduces probability of already-generated tokens.

    Args:
        logits: [vocab_size] logits
        generated_tokens: List of previously generated token IDs
        penalty: Penalty factor (>1.0 reduces repetition)

    Returns:
        Modified logits
    """
    if penalty == 1.0 or len(generated_tokens) == 0:
        return logits

    # Apply penalty to repeated tokens
    for token in set(generated_tokens):
        if 0 <= token < logits.size(-1):
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty

    return logits


class MultiHeadSampler:
    """
    Sampling for multi-head architecture with per-head temperatures.

    Allows different temperature/sampling strategies for each prediction head.
    """

    def __init__(self,
                 type_temp: float = 1.0,
                 command_temp: float = 1.0,
                 param_type_temp: float = 1.0,
                 param_value_temp: float = 1.0,
                 method: str = 'greedy'):
        """
        Initialize multi-head sampler.

        Args:
            type_temp: Temperature for type gate
            command_temp: Temperature for command head
            param_type_temp: Temperature for parameter type head
            param_value_temp: Temperature for parameter value head
            method: Sampling method ('greedy', 'temperature', 'top_k', 'nucleus')
        """
        self.type_temp = type_temp
        self.command_temp = command_temp
        self.param_type_temp = param_type_temp
        self.param_value_temp = param_value_temp
        self.method = method

    def sample(self, logits_dict: dict) -> dict:
        """
        Sample from multi-head logits.

        Args:
            logits_dict: Dictionary with keys 'type', 'command', 'param_type', 'param_value'

        Returns:
            Dictionary with sampled indices
        """
        samples = {}

        # Type gate (typically low temperature for confident predictions)
        if 'type' in logits_dict:
            samples['type'] = self._sample_with_temp(
                logits_dict['type'],
                self.type_temp
            )

        # Command head (low temperature for precise commands)
        if 'command' in logits_dict:
            samples['command'] = self._sample_with_temp(
                logits_dict['command'],
                self.command_temp
            )

        # Parameter type (medium temperature)
        if 'param_type' in logits_dict:
            samples['param_type'] = self._sample_with_temp(
                logits_dict['param_type'],
                self.param_type_temp
            )

        # Parameter value (higher temperature for diversity)
        if 'param_value' in logits_dict:
            samples['param_value'] = self._sample_with_temp(
                logits_dict['param_value'],
                self.param_value_temp
            )

        return samples

    def _sample_with_temp(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample using specified method and temperature."""
        if self.method == 'greedy' or temperature == 0:
            return logits.argmax(dim=-1)
        elif self.method == 'temperature':
            return temperature_sampling(logits, temperature)
        elif self.method == 'top_k':
            return top_k_sampling(logits, k=50, temperature=temperature)
        elif self.method == 'nucleus':
            return nucleus_sampling(logits, p=0.9, temperature=temperature)
        else:
            # Default to temperature sampling
            return temperature_sampling(logits, temperature)


# Example usage
if __name__ == '__main__':
    # Test temperature sampling
    logits = torch.randn(2, 100)  # [batch=2, vocab=100]

    print("Testing sampling strategies...")

    # Greedy
    greedy = temperature_sampling(logits, temperature=0.0)
    print(f"Greedy: {greedy}")

    # Temperature
    temp_high = temperature_sampling(logits, temperature=2.0)
    print(f"Temperature=2.0: {temp_high}")

    # Top-k
    topk = top_k_sampling(logits, k=10, temperature=1.0)
    print(f"Top-k (k=10): {topk}")

    # Nucleus
    nucleus = nucleus_sampling(logits, p=0.9, temperature=1.0)
    print(f"Nucleus (p=0.9): {nucleus}")

    # Multi-head sampling
    sampler = MultiHeadSampler(
        type_temp=0.5,
        command_temp=0.7,
        param_type_temp=1.0,
        param_value_temp=1.5,
        method='temperature'
    )

    logits_dict = {
        'type': torch.randn(1, 4),
        'command': torch.randn(1, 15),
        'param_type': torch.randn(1, 10),
        'param_value': torch.randn(1, 100),
    }

    samples = sampler.sample(logits_dict)
    print(f"\nMulti-head samples: {samples}")
