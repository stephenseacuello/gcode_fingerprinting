"""
Enhanced generation methods for G-code language model.

Implements:
- Temperature sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Combined sampling (temperature + top-k + top-p)
- Beam search decoding
- FSM-constrained generation (grammar-enforced)
- Digit-by-digit value generation

These methods fix the autoregressive generation issue where the model
gets stuck in <SOS> loops during inference.

Author: Claude Code
Date: November 2025, Updated December 2025
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

from .grammar_fsm import (
    GCodeGrammarFSM,
    BatchGCodeGrammarFSM,
    apply_hard_grammar_masks,
    apply_batch_grammar_masks,
    TYPE_SPECIAL,
    TYPE_COMMAND,
    TYPE_PARAMETER,
    TYPE_NUMERIC,
)

# Import digit utilities for value reconstruction
try:
    from ..dataset.target_utils import compose_digits_to_value, SIGN_POSITIVE, SIGN_NEGATIVE
except ImportError:
    # Fallback definitions if import fails
    SIGN_POSITIVE = 0
    SIGN_NEGATIVE = 1

    def compose_digits_to_value(sign, digits, max_int_digits=2, n_decimal_digits=4):
        if sign == 2:  # PAD
            return 0.0
        total_digits = max_int_digits + n_decimal_digits
        if len(digits) < total_digits:
            digits = digits + [0] * (total_digits - len(digits))
        digits = [min(max(0, d if d != 10 else 0), 9) for d in digits]
        int_value = sum(d * (10 ** (max_int_digits - 1 - i)) for i, d in enumerate(digits[:max_int_digits]))
        dec_value = sum(d * (10 ** -(i + 1)) for i, d in enumerate(digits[max_int_digits:]))
        value = int_value + dec_value
        return -value if sign == SIGN_NEGATIVE else value


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


@torch.no_grad()
def generate_with_grammar_fsm(
    model,
    memory: torch.Tensor,
    decomposer,
    max_len: int = 50,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    allow_modal_commands: bool = False,
    boost_encouraged: bool = True,
    use_digit_head: bool = True,
    max_int_digits: int = 2,
    n_decimal_digits: int = 4,
    operation_type: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate G-code tokens with hard grammar enforcement using FSM.

    This function guarantees that generated sequences follow valid G-code
    grammar by using a finite state machine to mask invalid transitions.

    Grammar rules enforced:
    1. Sequence starts with COMMAND (G0, G1, etc.)
    2. COMMAND → PARAMETER (X, Y, Z, etc.)
    3. PARAMETER → NUMERIC (value) - MANDATORY, cannot skip!
    4. NUMERIC → PARAMETER (more params) or EOS
    5. Single letter rule: each param used only once per line
    6. Command-specific rules: G0 cannot have F, etc.

    Args:
        model: MultiHeadGCodeLM model instance
        memory: Encoder memory from sensor data [B, T, d_model]
        decomposer: TokenDecomposer for token info
        max_len: Maximum sequence length
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 = disabled, typical: 40-100)
        top_p: Nucleus sampling threshold (1.0 = disabled, typical: 0.9-0.95)
        allow_modal_commands: If True, allows starting with parameters
        boost_encouraged: If True, boost encouraged params (soft boost)
        use_digit_head: If True, use digit-by-digit value prediction
        max_int_digits: Number of integer digits for digit prediction
        n_decimal_digits: Number of decimal digits for digit prediction
        operation_type: Operation type ID for conditioning value prediction
        device: Device to run on

    Returns:
        generated_tokens: [B, seq_len] generated token IDs
        generation_info: Dictionary with generation statistics
    """
    model.eval()
    B = memory.size(0)

    # Get special token IDs
    bos_id = decomposer.vocab.get('<BOS>', 1)
    eos_id = decomposer.vocab.get('<EOS>', 2)
    pad_id = decomposer.vocab.get('<PAD>', 0)

    # Initialize FSMs for each batch item
    batch_fsm = BatchGCodeGrammarFSM(B, decomposer, allow_modal_commands)

    # Start with BOS token
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    # Track generation stats
    stats = {
        'steps': 0,
        'finished': torch.zeros(B, dtype=torch.bool, device=device),
        'errors': torch.zeros(B, dtype=torch.bool, device=device),
        'type_predictions': [],
        'grammar_violations': 0,
        'numeric_values': [],  # Track generated numeric values for debugging
        'used_digit_head': use_digit_head and hasattr(model, 'digit_value_head') and model.digit_value_head is not None,
    }

    for step in range(max_len):
        stats['steps'] = step + 1

        # Check if all sequences are done
        if stats['finished'].all():
            break

        # Forward pass through model
        # Embed and position encode
        x = model.pos(model.embed(generated))

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = model.causal_mask(seq_len, device)

        # Decode
        dec = model.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)

        # Get logits from last position
        last_hidden = dec[:, -1:, :]  # [B, 1, d_model]

        # Get predictions from all heads
        logits = {
            'type_logits': model.type_gate(last_hidden),  # [B, 1, 4]
            'command_logits': model.command_head(last_hidden),  # [B, 1, n_commands]
            'param_type_logits': model.param_type_head(last_hidden),  # [B, 1, n_param_types]
        }

        # Apply hard grammar masks using FSM
        constrained_logits = apply_batch_grammar_masks(
            logits,
            batch_fsm,
            boost_encouraged=boost_encouraged,
        )

        # Decode token type first
        type_logits = constrained_logits['type_logits'].squeeze(1)  # [B, 4]

        # Apply temperature and sample
        if temperature != 1.0:
            type_logits = type_logits / temperature

        type_probs = F.softmax(type_logits, dim=-1)
        type_pred = torch.argmax(type_probs, dim=-1)  # [B]

        stats['type_predictions'].append(type_pred.cpu().tolist())

        # Now decode the actual token based on type
        next_tokens = torch.full((B,), pad_id, dtype=torch.long, device=device)

        for b in range(B):
            if stats['finished'][b]:
                next_tokens[b] = pad_id
                continue

            token_type = type_pred[b].item()

            if token_type == TYPE_SPECIAL:
                # EOS or other special token
                next_tokens[b] = eos_id
                stats['finished'][b] = True
                batch_fsm.fsms[b].transition(TYPE_SPECIAL)

            elif token_type == TYPE_COMMAND:
                # Sample command using unified sampling function
                cmd_logits = constrained_logits['command_logits'][b, 0, :]  # [n_commands]
                greedy = (temperature <= 0.1 and top_k == 0 and top_p >= 1.0)
                cmd_id = sample_with_options(
                    cmd_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy
                )

                # Compose token
                token_id = decomposer.compose_token(TYPE_COMMAND, cmd_id, 0, 0)
                next_tokens[b] = token_id

                # Update FSM
                batch_fsm.fsms[b].transition(TYPE_COMMAND, command_id=cmd_id)

            elif token_type == TYPE_PARAMETER:
                # Sample parameter type using unified sampling function
                param_logits = constrained_logits['param_type_logits'][b, 0, :]  # [n_params]
                greedy = (temperature <= 0.1 and top_k == 0 and top_p >= 1.0)
                param_id = sample_with_options(
                    param_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy
                )

                # Compose token (parameter letter only)
                token_id = decomposer.compose_token(TYPE_PARAMETER, 0, param_id, 0)
                next_tokens[b] = token_id

                # Update FSM
                batch_fsm.fsms[b].transition(TYPE_PARAMETER, param_id=param_id)

            elif token_type == TYPE_NUMERIC:
                # Get the current param type from FSM context
                current_param = batch_fsm.fsms[b].current_param
                param_id = decomposer.param2id.get(current_param, 0) if current_param else 0

                # Use digit-by-digit prediction if available and enabled
                if use_digit_head and hasattr(model, 'digit_value_head') and model.digit_value_head is not None:
                    value = decode_digit_value(
                        model,
                        last_hidden[b:b+1],
                        operation_type=operation_type,
                        param_type=param_id,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        max_int_digits=max_int_digits,
                        n_decimal_digits=n_decimal_digits,
                    )
                elif hasattr(model, 'param_value_regression_head'):
                    # Fall back to regression head
                    value_pred = model.param_value_regression_head(last_hidden[b:b+1])  # [1, 1, 1]
                    value = value_pred.squeeze().item()
                else:
                    # Last resort fallback: use a default value
                    value = 1.5

                # Quantize value to bucket for token composition
                bucket_digits = decomposer.vocab_data.get('config', {}).get('bucket_digits', 2)
                n_buckets = 10 ** bucket_digits

                # Convert value to bucket ID
                value_id = int(abs(value * 10)) % n_buckets
                value_id = max(0, min(n_buckets - 1, value_id))

                # Compose numeric token
                token_id = decomposer.compose_token(TYPE_NUMERIC, 0, param_id, value_id)
                next_tokens[b] = token_id

                # Track generated value for debugging
                stats['numeric_values'].append({
                    'batch': b,
                    'step': step,
                    'param': current_param,
                    'value': value,
                    'bucket_id': value_id
                })

                # Update FSM
                batch_fsm.fsms[b].transition(TYPE_NUMERIC)

            else:
                # Unknown type - this shouldn't happen with proper masking
                stats['grammar_violations'] += 1
                stats['errors'][b] = True
                next_tokens[b] = eos_id
                stats['finished'][b] = True

        # Append next tokens
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

    return generated, stats


def _sample_nucleus(logits: torch.Tensor, top_p: float) -> int:
    """Sample from logits using nucleus (top-p) sampling."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)

    # Find cutoff
    cutoff_idx = torch.where(cumsum_probs >= top_p)[0]
    if len(cutoff_idx) > 0:
        cutoff_idx = cutoff_idx[0].item() + 1
    else:
        cutoff_idx = len(sorted_probs)

    # Zero out and renormalize
    sorted_probs[cutoff_idx:] = 0.0
    if sorted_probs.sum() > 0:
        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices[sampled_idx].item()
    else:
        return sorted_indices[0].item()


def sample_with_options(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> int:
    """
    Sample from logits with temperature, top-k, and top-p (nucleus) filtering.

    The filtering is applied in order: temperature → top-k → top-p

    Args:
        logits: [vocab_size] unnormalized logits
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
                    Values < 1.0 make distribution sharper, > 1.0 make it flatter
        top_k: Keep only top K tokens (0 = disabled)
               Typical values: 40-100
        top_p: Nucleus sampling - keep tokens with cumulative prob >= p (1.0 = disabled)
               Typical values: 0.9-0.95
        greedy: If True, ignore sampling and return argmax

    Returns:
        Sampled token index

    Examples:
        # Greedy decoding (most likely token)
        sample_with_options(logits, greedy=True)

        # Temperature sampling only
        sample_with_options(logits, temperature=0.7)

        # Top-k only
        sample_with_options(logits, top_k=50)

        # Nucleus (top-p) only
        sample_with_options(logits, top_p=0.9)

        # Combined (recommended for quality + diversity)
        sample_with_options(logits, temperature=0.8, top_k=50, top_p=0.9)
    """
    if greedy:
        return torch.argmax(logits).item()

    # 1. Apply temperature scaling
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # 2. Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Get the k-th largest value as threshold
        threshold = torch.topk(logits, top_k)[0][..., -1]
        # Mask out tokens below threshold
        logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)

    # 3. Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff - keep tokens until cumsum >= top_p
        sorted_mask = cumsum_probs > top_p
        # Shift right to keep at least one token
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        # Scatter mask back to original order
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        logits = torch.where(mask, torch.full_like(logits, float('-inf')), logits)

    # 4. Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)

    # Handle edge case where all probs are 0 (shouldn't happen but safety check)
    if probs.sum() <= 0:
        return torch.argmax(logits).item()

    return torch.multinomial(probs, num_samples=1).item()


def decode_digit_value(
    model,
    hidden: torch.Tensor,
    operation_type: Optional[int] = None,
    param_type: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_int_digits: int = 2,
    n_decimal_digits: int = 4,
) -> float:
    """
    Decode a numeric value using the digit-by-digit prediction head.

    Args:
        model: MultiHeadGCodeLM model with digit_value_head
        hidden: [1, 1, d_model] hidden state from decoder
        operation_type: Operation type ID for conditioning
        param_type: Parameter type ID for conditioning
        temperature: Sampling temperature for digit selection
        top_k: Top-k filtering for digit selection
        top_p: Top-p (nucleus) filtering for digit selection
        max_int_digits: Number of integer digit positions
        n_decimal_digits: Number of decimal digit positions

    Returns:
        Decoded numeric value as float
    """
    device = hidden.device

    # Check if model has digit value head
    if not hasattr(model, 'digit_value_head') or model.digit_value_head is None:
        # Fallback to regression head if available
        if hasattr(model, 'param_value_regression_head'):
            value_pred = model.param_value_regression_head(hidden)
            return value_pred.squeeze().item()
        return 0.0

    # Prepare conditioning tensors
    op_type_tensor = None
    param_type_tensor = None

    if operation_type is not None:
        op_type_tensor = torch.tensor([[operation_type]], device=device)
    if param_type is not None:
        param_type_tensor = torch.tensor([[param_type]], device=device)

    # Forward through digit value head
    digit_output = model.digit_value_head(
        hidden,
        operation_type=op_type_tensor,
        param_type=param_type_tensor
    )

    # Extract sign and digit logits
    sign_logits = digit_output['sign_logits'].squeeze()  # [3]
    digit_logits = digit_output['digit_logits'].squeeze()  # [n_digits, 10]

    # Decode sign
    sign_pred = sample_with_options(
        sign_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=(temperature <= 0.1)  # Use greedy for very low temp
    )

    # Decode each digit
    n_digits = digit_logits.size(0)
    digit_preds = []
    for d in range(n_digits):
        digit_pred = sample_with_options(
            digit_logits[d],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=(temperature <= 0.1)
        )
        digit_preds.append(digit_pred)

    # Compose digits back to value
    value = compose_digits_to_value(sign_pred, digit_preds, max_int_digits, n_decimal_digits)

    return value


def tokens_to_gcode_string(
    tokens: torch.Tensor,
    decomposer,
    skip_special: bool = True,
) -> str:
    """
    Convert token IDs to human-readable G-code string.

    Args:
        tokens: [seq_len] token IDs
        decomposer: TokenDecomposer instance
        skip_special: If True, skip BOS/EOS/PAD tokens

    Returns:
        G-code string like "G0 X1.5750 Y1.8000"
    """
    special_ids = {
        decomposer.vocab.get('<PAD>', 0),
        decomposer.vocab.get('<BOS>', 1),
        decomposer.vocab.get('<EOS>', 2),
        decomposer.vocab.get('<UNK>', 3),
        decomposer.vocab.get('<MASK>', 4),
    }

    parts = []
    current_param = None

    for token_id in tokens.tolist():
        if skip_special and token_id in special_ids:
            continue

        token_str = decomposer.id2token.get(token_id, '<UNK>')

        # Parse token type
        if token_str.startswith('G') or token_str.startswith('M'):
            parts.append(token_str)
            current_param = None
        elif token_str in decomposer.param2id:
            current_param = token_str
        elif token_str.startswith('NUM_'):
            # Parse: NUM_X_15 -> value
            token_parts = token_str.split('_')
            if len(token_parts) >= 3:
                param = token_parts[1]
                value_str = token_parts[2]
                try:
                    value = int(value_str)
                    # Convert bucket back to approximate value
                    bucket_digits = decomposer.vocab_data.get('config', {}).get('bucket_digits', 2)
                    # This is an approximation - real value would need more info
                    approx_value = value / (10 ** (bucket_digits - 1))
                    parts.append(f"{param}{approx_value:.4f}")
                except ValueError:
                    parts.append(f"{param}?")
        else:
            # Unknown token
            parts.append(token_str)

    return ' '.join(parts)
