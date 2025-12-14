"""
Target decomposition utilities for multi-head G-code prediction.

Decomposes tokens into hierarchical targets:
1. Token type (command vs parameter)
2. Command ID (G0, G1, M3, etc.) OR Parameter type (X, Y, Z, F, R, S)
3. Parameter value (00-99 for numeric buckets) OR Hybrid (coarse 0-9 + residual 0.0-9.9)
4. Digit-by-digit decomposition for precise numeric value prediction

Supports multiple modes:
- Standard: param_value_id (single bucket)
- Hybrid: param_value_coarse_id (tens digit) + param_value_residual (continuous)
- Digit: sign + individual digit predictions for precise numeric values

This enables separate prediction heads to reduce gradient competition and improve generalization.
"""
from typing import Dict, Tuple, List, Optional, Union
import torch
import json
from pathlib import Path
import math


# Constants for digit-by-digit prediction
SIGN_POSITIVE = 0
SIGN_NEGATIVE = 1
SIGN_PAD = 2
DIGIT_PAD = 10  # For padding digit positions


def decompose_value_to_digits(
    value: float,
    max_int_digits: int = 2,
    n_decimal_digits: int = 4
) -> Tuple[int, List[int]]:
    """
    Decompose a numeric value into sign and individual digits.

    Format: ±XX.XXXX (sign + 2 integer digits + 4 decimal digits)
    Example: +1.5750 → sign=0, digits=[0, 1, 5, 7, 5, 0]
    Example: -12.3456 → sign=1, digits=[1, 2, 3, 4, 5, 6]

    Args:
        value: The numeric value to decompose
        max_int_digits: Number of integer digit positions (default 2 for 0-99)
        n_decimal_digits: Number of decimal digit positions (default 4)

    Returns:
        (sign, digits) where:
        - sign: 0=positive, 1=negative, 2=pad
        - digits: List of digit values (0-9), length = max_int_digits + n_decimal_digits
    """
    # Handle NaN or inf
    if math.isnan(value) or math.isinf(value):
        return (SIGN_PAD, [DIGIT_PAD] * (max_int_digits + n_decimal_digits))

    # Determine sign
    sign = SIGN_NEGATIVE if value < 0 else SIGN_POSITIVE
    abs_value = abs(value)

    # Clamp to maximum representable value
    max_value = (10 ** max_int_digits) - (10 ** -n_decimal_digits)
    abs_value = min(abs_value, max_value)

    # Split into integer and decimal parts
    int_part = int(abs_value)
    dec_part = abs_value - int_part

    # Extract integer digits (right-padded with leading zeros)
    int_digits = []
    remaining = int_part
    for _ in range(max_int_digits):
        int_digits.append(remaining % 10)
        remaining //= 10
    int_digits.reverse()  # Most significant first

    # Extract decimal digits
    dec_digits = []
    for _ in range(n_decimal_digits):
        dec_part *= 10
        digit = int(dec_part)
        dec_digits.append(min(digit, 9))  # Clamp to 0-9
        dec_part -= digit

    return (sign, int_digits + dec_digits)


def compose_digits_to_value(
    sign: int,
    digits: List[int],
    max_int_digits: int = 2,
    n_decimal_digits: int = 4
) -> float:
    """
    Compose individual digits back into a numeric value.

    Args:
        sign: 0=positive, 1=negative, 2=pad (returns 0.0)
        digits: List of digit values (0-9), length = max_int_digits + n_decimal_digits
        max_int_digits: Number of integer digit positions
        n_decimal_digits: Number of decimal digit positions

    Returns:
        The reconstructed float value
    """
    if sign == SIGN_PAD:
        return 0.0

    # Validate and clamp digits
    total_digits = max_int_digits + n_decimal_digits
    if len(digits) < total_digits:
        digits = digits + [0] * (total_digits - len(digits))

    # Clamp any invalid digit values
    digits = [min(max(0, d if d != DIGIT_PAD else 0), 9) for d in digits]

    # Reconstruct integer part
    int_value = 0
    for i in range(max_int_digits):
        int_value = int_value * 10 + digits[i]

    # Reconstruct decimal part
    dec_value = 0.0
    for i in range(n_decimal_digits):
        dec_value = dec_value * 10 + digits[max_int_digits + i]
    dec_value /= (10 ** n_decimal_digits)

    # Combine
    value = int_value + dec_value

    # Apply sign
    if sign == SIGN_NEGATIVE:
        value = -value

    return value


def decompose_values_to_digits_batch(
    values: torch.Tensor,
    max_int_digits: int = 2,
    n_decimal_digits: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a batch of numeric values into signs and digits.

    Args:
        values: [B, T] or [N] tensor of numeric values
        max_int_digits: Number of integer digit positions
        n_decimal_digits: Number of decimal digit positions

    Returns:
        (signs, digits) where:
        - signs: Same shape as values, values 0/1/2
        - digits: Shape [..., n_digits] where n_digits = max_int_digits + n_decimal_digits
    """
    original_shape = values.shape
    flat_values = values.flatten()
    n_values = flat_values.numel()
    n_digits = max_int_digits + n_decimal_digits

    signs = torch.zeros(n_values, dtype=torch.long, device=values.device)
    digits = torch.zeros(n_values, n_digits, dtype=torch.long, device=values.device)

    for i in range(n_values):
        v = flat_values[i].item()
        sign, digit_list = decompose_value_to_digits(v, max_int_digits, n_decimal_digits)
        signs[i] = sign
        for j, d in enumerate(digit_list):
            digits[i, j] = d

    # Reshape back
    signs = signs.view(*original_shape)
    digits = digits.view(*original_shape, n_digits)

    return signs, digits


def compose_digits_to_values_batch(
    signs: torch.Tensor,
    digits: torch.Tensor,
    max_int_digits: int = 2,
    n_decimal_digits: int = 4
) -> torch.Tensor:
    """
    Compose batched digit predictions back into numeric values.

    Args:
        signs: [B, T] or [N] tensor of sign predictions (0/1/2)
        digits: [..., n_digits] tensor of digit predictions (0-9)
        max_int_digits: Number of integer digit positions
        n_decimal_digits: Number of decimal digit positions

    Returns:
        Tensor of reconstructed float values with same shape as signs
    """
    original_shape = signs.shape
    flat_signs = signs.flatten()
    n_values = flat_signs.numel()

    # Flatten digits to [N, n_digits]
    n_digits = max_int_digits + n_decimal_digits
    flat_digits = digits.view(n_values, n_digits)

    values = torch.zeros(n_values, dtype=torch.float32, device=signs.device)

    for i in range(n_values):
        sign = int(flat_signs[i].item())
        digit_list = [int(flat_digits[i, j].item()) for j in range(n_digits)]
        values[i] = compose_digits_to_value(sign, digit_list, max_int_digits, n_decimal_digits)

    return values.view(*original_shape)


class TokenDecomposer:
    """
    Decomposes G-code tokens into hierarchical targets for multi-head prediction.

    Token Types:
    - COMMAND: G-commands (G0, G1, G2, G3, G90, etc.), M-commands (M3, M5, etc.)
    - PARAMETER: Axis tokens (X, Y, Z) and parameter tokens (F, R, S)
    - NUMERIC: Numeric value tokens (NUM_X_15, NUM_Y_23, etc.)
    - SPECIAL: PAD, BOS, EOS, UNK, MASK

    Decomposition:
    - Commands: type=COMMAND, command_id, param_type=0, param_value=0
    - Parameters: type=PARAMETER, command_id=0, param_type, param_value=0
    - Numerics: type=NUMERIC, command_id=0, param_type, param_value
    - Special: type=SPECIAL, command_id=0, param_type=0, param_value=0
    """

    # Token type IDs
    TYPE_SPECIAL = 0
    TYPE_COMMAND = 1
    TYPE_PARAMETER = 2
    TYPE_NUMERIC = 3

    def __init__(self, vocab_path: str):
        """
        Args:
            vocab_path: Path to vocabulary JSON file
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.vocab_data = vocab_data  # Save for later use
        self.vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Build token type mappings
        self.special_tokens = {'<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>'}
        self.command_tokens = []  # G0, G1, M3, etc.
        self.param_tokens = []    # X, Y, Z, F, R, S
        self.numeric_tokens = []  # NUM_X_15, NUM_Y_23, etc.

        # Parse vocabulary
        for token, token_id in self.vocab.items():
            if token in self.special_tokens:
                continue
            elif token.startswith('G') and len(token) > 1 and token[1:].isdigit():
                self.command_tokens.append(token)
            elif token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                self.command_tokens.append(token)
            elif token.startswith('NUM_'):
                self.numeric_tokens.append(token)
            else:
                # Single-letter axis/parameter tokens (X, Y, Z, F, R, S, etc.)
                if len(token) <= 2:
                    self.param_tokens.append(token)

        # Create command vocab: [G0, G1, G2, ..., M3, M5, ...]
        self.command_tokens = sorted(self.command_tokens)
        self.command2id = {cmd: i for i, cmd in enumerate(self.command_tokens)}
        self.n_commands = len(self.command_tokens)

        # Create parameter type vocab: [X, Y, Z, F, R, S, ...]
        self.param_tokens = sorted(self.param_tokens)
        self.param2id = {param: i for i, param in enumerate(self.param_tokens)}
        self.n_param_types = len(self.param_tokens)

        # Parameter value vocab: infer from bucket_digits config
        # bucket_digits=2 → 100 values (00-99)
        # bucket_digits=3 → 1000 values (000-999)
        # bucket_digits=4 → 10000 values (0000-9999)
        bucket_digits = self.vocab_data.get('config', {}).get('bucket_digits', 3)
        self.n_param_values = 10 ** bucket_digits

        print(f"TokenDecomposer initialized:")
        print(f"  Total vocab: {self.vocab_size}")
        print(f"  Commands: {self.n_commands} ({', '.join(self.command_tokens[:5])}...)")
        print(f"  Param types: {self.n_param_types} ({', '.join(self.param_tokens[:5])}...)")
        print(f"  Param values: {self.n_param_values} ({bucket_digits}-digit bucketing)")

    def decompose_token(self, token_id: int) -> Tuple[int, int, int, int]:
        """
        Decompose a token ID into (type, command_id, param_type_id, param_value_id).

        Args:
            token_id: Token ID from vocabulary

        Returns:
            (type, command_id, param_type_id, param_value_id)
        """
        if token_id >= self.vocab_size:
            # Out of vocab -> UNK
            return (self.TYPE_SPECIAL, 0, 0, 0)

        token = self.id2token.get(token_id, '<UNK>')

        # Special tokens
        if token in self.special_tokens:
            return (self.TYPE_SPECIAL, 0, 0, 0)

        # Command tokens (G0, G1, M3, etc.)
        if token in self.command2id:
            command_id = self.command2id[token]
            return (self.TYPE_COMMAND, command_id, 0, 0)

        # Parameter tokens (X, Y, Z, F, R, S)
        if token in self.param2id:
            param_type_id = self.param2id[token]
            return (self.TYPE_PARAMETER, 0, param_type_id, 0)

        # Numeric tokens (NUM_X_15, NUM_Y_23, etc.)
        if token.startswith('NUM_'):
            # Parse: NUM_X_15 -> param_type=X, param_value=15
            parts = token.split('_')
            if len(parts) >= 3:
                param_type = parts[1]  # X, Y, Z, etc.
                param_value_str = parts[2]  # "15", "-1", etc.

                if param_type in self.param2id:
                    param_type_id = self.param2id[param_type]
                else:
                    param_type_id = 0

                # Parse numeric value
                try:
                    param_value = int(param_value_str)
                    # Clip to valid range based on bucket_digits from vocab config
                    param_value_id = max(0, min(self.n_param_values - 1, abs(param_value)))
                except ValueError:
                    param_value_id = 0

                return (self.TYPE_NUMERIC, 0, param_type_id, param_value_id)

        # Unknown token
        return (self.TYPE_SPECIAL, 0, 0, 0)

    def decompose_token_hybrid(self, token_id: int) -> Tuple[int, int, int, int]:
        """
        Decompose a token ID for hybrid bucketing (coarse only - residuals come from preprocessing).

        For 1-digit bucketing:
        - NUM_X_3 → param_type=X, param_value_coarse=3 (represents values 30-39)

        Args:
            token_id: Token ID from vocabulary

        Returns:
            (type, command_id, param_type_id, param_value_coarse_id)
        """
        type_id, cmd_id, param_type_id, param_value_id = self.decompose_token(token_id)

        # For hybrid bucketing, if this is a numeric token, extract tens digit as coarse bucket
        if type_id == self.TYPE_NUMERIC:
            # param_value_id is the bucketed value (e.g., 3 for NUM_X_3)
            # For 1-digit bucketing, this IS the coarse bucket (0-9)
            param_value_coarse_id = param_value_id % 10  # Ensure 0-9 range
            return (type_id, cmd_id, param_type_id, param_value_coarse_id)

        # For non-numeric tokens, coarse bucket is 0
        return (type_id, cmd_id, param_type_id, 0)

    def decompose_batch(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose a batch of token IDs into hierarchical targets.

        Args:
            token_ids: [B, T] token IDs

        Returns:
            Dictionary with:
            - 'type': [B, T] token type (0-3)
            - 'command_id': [B, T] command ID (0 to n_commands-1)
            - 'param_type_id': [B, T] parameter type ID (0 to n_param_types-1)
            - 'param_value_id': [B, T] parameter value ID (0-99)
        """
        B, T = token_ids.shape
        device = token_ids.device

        types = torch.zeros(B, T, dtype=torch.long, device=device)
        command_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        param_type_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        param_value_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        # Decompose each token
        for b in range(B):
            for t in range(T):
                token_id = int(token_ids[b, t].item())
                type_id, cmd_id, param_type_id, param_value_id = self.decompose_token(token_id)
                types[b, t] = type_id
                command_ids[b, t] = cmd_id
                param_type_ids[b, t] = param_type_id
                param_value_ids[b, t] = param_value_id

        return {
            'type': types,
            'command_id': command_ids,
            'param_type_id': param_type_ids,
            'param_value_id': param_value_ids,
        }

    def decompose_batch_hybrid(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose a batch of token IDs for hybrid bucketing (coarse buckets only).

        For 1-digit bucketing:
        - NUM_X_3 → coarse_bucket=3 (tens digit)
        - Residual values must be provided separately from preprocessing

        Args:
            token_ids: [B, T] token IDs

        Returns:
            Dictionary with:
            - 'type': [B, T] token type (0-3)
            - 'command_id': [B, T] command ID
            - 'param_type_id': [B, T] parameter type ID
            - 'param_value_coarse_id': [B, T] coarse bucket ID (0-9)
        """
        B, T = token_ids.shape
        device = token_ids.device

        types = torch.zeros(B, T, dtype=torch.long, device=device)
        command_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        param_type_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        param_value_coarse_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        # Decompose each token
        for b in range(B):
            for t in range(T):
                token_id = int(token_ids[b, t].item())
                type_id, cmd_id, param_type_id, coarse_id = self.decompose_token_hybrid(token_id)
                types[b, t] = type_id
                command_ids[b, t] = cmd_id
                param_type_ids[b, t] = param_type_id
                param_value_coarse_ids[b, t] = coarse_id

        return {
            'type': types,
            'command_id': command_ids,
            'param_type_id': param_type_ids,
            'param_value_coarse_id': param_value_coarse_ids,
        }

    def decompose_batch_with_digits(
        self,
        token_ids: torch.Tensor,
        raw_values: Optional[torch.Tensor] = None,
        max_int_digits: int = 2,
        n_decimal_digits: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose a batch of token IDs with digit-by-digit value targets.

        Args:
            token_ids: [B, T] token IDs
            raw_values: [B, T] raw numeric values (optional - if not provided, uses bucket centers)
            max_int_digits: Number of integer digit positions
            n_decimal_digits: Number of decimal digit positions

        Returns:
            Dictionary with:
            - 'type': [B, T] token type (0-3)
            - 'command_id': [B, T] command ID
            - 'param_type_id': [B, T] parameter type ID
            - 'param_value_id': [B, T] parameter value ID (bucket)
            - 'sign_targets': [B, T] sign targets (0=+, 1=-, 2=pad)
            - 'digit_targets': [B, T, n_digits] digit targets (0-9, 10=pad)
        """
        B, T = token_ids.shape
        device = token_ids.device
        n_digits = max_int_digits + n_decimal_digits

        # First get standard decomposition
        decomposed = self.decompose_batch(token_ids)

        # Initialize digit targets
        sign_targets = torch.full((B, T), SIGN_PAD, dtype=torch.long, device=device)
        digit_targets = torch.full((B, T, n_digits), DIGIT_PAD, dtype=torch.long, device=device)

        # If raw values provided, use them directly
        if raw_values is not None:
            # Only decompose for NUMERIC tokens
            numeric_mask = decomposed['type'] == self.TYPE_NUMERIC

            for b in range(B):
                for t in range(T):
                    if numeric_mask[b, t]:
                        value = raw_values[b, t].item()
                        sign, digits = decompose_value_to_digits(value, max_int_digits, n_decimal_digits)
                        sign_targets[b, t] = sign
                        for d, digit in enumerate(digits):
                            digit_targets[b, t, d] = digit

        decomposed['sign_targets'] = sign_targets
        decomposed['digit_targets'] = digit_targets

        return decomposed

    def extract_raw_value_from_token(self, token: str) -> Optional[float]:
        """
        Extract the raw numeric value from a token like NUM_X_1234.

        For 4-digit bucketing with value_range=[-100, 100]:
        - NUM_X_0000 → -100.0
        - NUM_X_5000 → 0.0
        - NUM_X_9999 → ~100.0

        Args:
            token: Token string (e.g., "NUM_X_1234")

        Returns:
            Estimated raw value, or None if not a numeric token
        """
        if not token.startswith('NUM_'):
            return None

        parts = token.split('_')
        if len(parts) < 3:
            return None

        try:
            bucket_id = int(parts[2])
        except ValueError:
            return None

        # Get value range from vocab config
        config = self.vocab_data.get('config', {})
        value_range = config.get('value_range', [-100, 100])
        bucket_digits = config.get('bucket_digits', 4)
        n_buckets = 10 ** bucket_digits

        # Convert bucket to value
        min_val, max_val = value_range
        value = min_val + (bucket_id / n_buckets) * (max_val - min_val)

        return value

    def compose_token(self, type_id: int, command_id: int, param_type_id: int, param_value_id: int) -> int:
        """
        Compose hierarchical targets back into a token ID.

        Args:
            type_id: Token type (0-3)
            command_id: Command ID
            param_type_id: Parameter type ID
            param_value_id: Parameter value ID (00-99)

        Returns:
            Token ID from vocabulary (best match)
        """
        # Special tokens -> PAD
        if type_id == self.TYPE_SPECIAL:
            return self.vocab.get('<PAD>', 0)

        # Command tokens
        if type_id == self.TYPE_COMMAND:
            if command_id < len(self.command_tokens):
                token = self.command_tokens[command_id]
                return self.vocab.get(token, 0)
            return self.vocab.get('<UNK>', 0)

        # Parameter tokens
        if type_id == self.TYPE_PARAMETER:
            if param_type_id < len(self.param_tokens):
                token = self.param_tokens[param_type_id]
                return self.vocab.get(token, 0)
            return self.vocab.get('<UNK>', 0)

        # Numeric tokens
        if type_id == self.TYPE_NUMERIC:
            if param_type_id < len(self.param_tokens):
                param_type = self.param_tokens[param_type_id]
                # Find closest numeric token: NUM_X_157 (3-digit bucketing)
                token = f"NUM_{param_type}_{param_value_id:03d}"
                if token in self.vocab:
                    return self.vocab[token]
                # Try without zero-padding
                token = f"NUM_{param_type}_{param_value_id}"
                if token in self.vocab:
                    return self.vocab[token]
            return self.vocab.get('<UNK>', 0)

        return self.vocab.get('<UNK>', 0)

    def compose_batch(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compose hierarchical targets back into token IDs.

        Args:
            targets: Dictionary with 'type', 'command_id', 'param_type_id', 'param_value_id'

        Returns:
            token_ids: [B, T] token IDs
        """
        B, T = targets['type'].shape
        device = targets['type'].device

        token_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        for b in range(B):
            for t in range(T):
                type_id = int(targets['type'][b, t].item())
                cmd_id = int(targets['command_id'][b, t].item())
                param_type_id = int(targets['param_type_id'][b, t].item())
                param_value_id = int(targets['param_value_id'][b, t].item())

                token_id = self.compose_token(type_id, cmd_id, param_type_id, param_value_id)
                token_ids[b, t] = token_id

        return token_ids


# Example usage
if __name__ == '__main__':
    from pathlib import Path

    # Test digit decomposition functions
    print("=" * 60)
    print("Testing digit decomposition functions")
    print("=" * 60)

    test_values = [1.5750, -12.3456, 0.0, 99.9999, -0.0001, 42.0]
    print("\nValue → Digits → Value roundtrip:")
    for val in test_values:
        sign, digits = decompose_value_to_digits(val)
        reconstructed = compose_digits_to_value(sign, digits)
        sign_str = '+' if sign == SIGN_POSITIVE else '-' if sign == SIGN_NEGATIVE else 'P'
        digits_str = ''.join(str(d) for d in digits[:2]) + '.' + ''.join(str(d) for d in digits[2:])
        print(f"  {val:10.4f} → {sign_str}{digits_str} → {reconstructed:10.4f}  (diff: {abs(val - reconstructed):.6f})")

    # Test batch operations
    print("\nBatch digit decomposition:")
    values_tensor = torch.tensor(test_values)
    signs, digits = decompose_values_to_digits_batch(values_tensor)
    reconstructed_values = compose_digits_to_values_batch(signs, digits)
    print(f"  Original: {values_tensor.tolist()}")
    print(f"  Signs: {signs.tolist()}")
    print(f"  Digits shape: {digits.shape}")
    print(f"  Reconstructed: {reconstructed_values.tolist()}")

    # Test decomposer
    print("\n" + "=" * 60)
    print("Testing TokenDecomposer")
    print("=" * 60)

    vocab_path = Path('data/gcode_vocab_v2.json')
    if vocab_path.exists():
        decomposer = TokenDecomposer(str(vocab_path))

        # Test token decomposition
        print("\nExample decompositions:")
        test_tokens = ['<PAD>', 'G0', 'G1', 'X', 'Y', 'NUM_X_15', 'NUM_Y_23', 'M3']

        for token in test_tokens:
            if token in decomposer.vocab:
                token_id = decomposer.vocab[token]
                type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

                type_names = ['SPECIAL', 'COMMAND', 'PARAMETER', 'NUMERIC']
                print(f"  {token:12s} -> type={type_names[type_id]:9s}, cmd={cmd_id:2d}, param_type={param_type_id:2d}, param_value={param_value_id:2d}")

                # Test composition
                reconstructed_id = decomposer.compose_token(type_id, cmd_id, param_type_id, param_value_id)
                reconstructed_token = decomposer.id2token.get(reconstructed_id, '<UNK>')
                if reconstructed_token != token:
                    print(f"    WARNING: Reconstruction mismatch! {token} -> {reconstructed_token}")

        # Test batch decomposition
        print("\nBatch decomposition test:")
        test_batch = torch.tensor([
            [decomposer.vocab.get('G0', 0), decomposer.vocab.get('X', 0), decomposer.vocab.get('NUM_X_15', 0)],
            [decomposer.vocab.get('G1', 0), decomposer.vocab.get('Y', 0), decomposer.vocab.get('NUM_Y_23', 0)],
        ])

        decomposed = decomposer.decompose_batch(test_batch)
        print(f"  Input shape: {test_batch.shape}")
        print(f"  Types: {decomposed['type']}")
        print(f"  Command IDs: {decomposed['command_id']}")
        print(f"  Param type IDs: {decomposed['param_type_id']}")
        print(f"  Param value IDs: {decomposed['param_value_id']}")

        # Test composition
        reconstructed = decomposer.compose_batch(decomposed)
        print(f"  Reconstructed: {reconstructed}")
        print(f"  Match: {torch.equal(test_batch, reconstructed)}")

        # Test digit decomposition with raw values
        print("\nDigit decomposition with raw values:")
        raw_values = torch.tensor([
            [0.0, 0.0, 15.5],
            [0.0, 0.0, 23.75],
        ], dtype=torch.float32)
        decomposed_with_digits = decomposer.decompose_batch_with_digits(test_batch, raw_values)
        print(f"  Sign targets: {decomposed_with_digits['sign_targets']}")
        print(f"  Digit targets shape: {decomposed_with_digits['digit_targets'].shape}")
        print(f"  Digit targets [0,2]: {decomposed_with_digits['digit_targets'][0, 2].tolist()}")
        print(f"  Digit targets [1,2]: {decomposed_with_digits['digit_targets'][1, 2].tolist()}")
    else:
        print(f"\nVocab file not found at {vocab_path}, skipping TokenDecomposer tests")
