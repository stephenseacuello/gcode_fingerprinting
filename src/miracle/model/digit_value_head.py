"""
Digit-by-digit numeric value prediction head for G-code generation.

Instead of predicting a single regression value or bucket ID,
this module predicts each character of the numeric value sequentially:
    +1.5750 → ['+', '1', '.', '5', '7', '5', '0']

This approach:
1. Eliminates mode collapse to mean values
2. Allows fine-grained control over each digit
3. Leverages operation-type and parameter-type conditioning
4. Supports variable precision per parameter

Architecture:
    Input: [B, T, d_model] hidden states from decoder
    + Operation embedding (which operation: FACE, POCKET, etc.)
    + Parameter type embedding (which param: X, Y, Z, F, R)
    Output: Sign logits + Digit logits for each position

Author: Claude Code
Date: December 2025
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Value format configuration per parameter type
# Format: (max_int_digits, n_decimal_digits, typical_range)
PARAM_VALUE_CONFIG = {
    'X': {'max_int_digits': 2, 'n_decimal_digits': 4, 'min': -1.0, 'max': 5.0},
    'Y': {'max_int_digits': 2, 'n_decimal_digits': 4, 'min': -1.0, 'max': 5.0},
    'Z': {'max_int_digits': 1, 'n_decimal_digits': 4, 'min': -0.5, 'max': 1.0},
    'F': {'max_int_digits': 2, 'n_decimal_digits': 1, 'min': 0.0, 'max': 100.0},
    'R': {'max_int_digits': 1, 'n_decimal_digits': 4, 'min': 0.0, 'max': 5.0},
    'S': {'max_int_digits': 5, 'n_decimal_digits': 0, 'min': 0.0, 'max': 30000.0},
    'I': {'max_int_digits': 1, 'n_decimal_digits': 4, 'min': -5.0, 'max': 5.0},
    'J': {'max_int_digits': 1, 'n_decimal_digits': 4, 'min': -5.0, 'max': 5.0},
    'K': {'max_int_digits': 1, 'n_decimal_digits': 4, 'min': -5.0, 'max': 5.0},
}

# Default config for unknown parameters
DEFAULT_PARAM_CONFIG = {'max_int_digits': 2, 'n_decimal_digits': 4, 'min': -10.0, 'max': 10.0}


class DigitByDigitValueHead(nn.Module):
    """
    Predicts numeric values digit-by-digit with operation and parameter conditioning.

    For a value like 1.5750:
    - Sign prediction: 3 classes (+, -, neutral/positive)
    - Integer part: max_int_digits heads, each 10 classes (0-9)
    - Decimal part: n_decimal_digits heads, each 10 classes (0-9)

    Total predictions per value: 1 (sign) + max_int_digits + n_decimal_digits

    The model learns to predict each position independently (parallel decoding)
    rather than autoregressively, for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        n_operations: int = 9,
        n_param_types: int = 10,
        max_int_digits: int = 2,
        n_decimal_digits: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Hidden dimension from decoder
            n_operations: Number of operation types (for conditioning)
            n_param_types: Number of parameter types (X, Y, Z, etc.)
            max_int_digits: Maximum integer digits (e.g., 2 for values up to 99)
            n_decimal_digits: Number of decimal digits to predict
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_operations = n_operations
        self.n_param_types = n_param_types
        self.max_int_digits = max_int_digits
        self.n_decimal_digits = n_decimal_digits
        self.n_digit_positions = max_int_digits + n_decimal_digits

        # Conditioning embeddings
        self.operation_embed = nn.Embedding(n_operations, d_model // 4)
        self.param_type_embed = nn.Embedding(n_param_types, d_model // 4)

        # Position embedding for digit positions
        self.digit_position_embed = nn.Embedding(self.n_digit_positions, d_model // 4)

        # Combined context projection
        # Input: d_model (hidden) + d_model//4 (op) + d_model//4 (param) = d_model * 1.5
        self.context_proj = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sign prediction head: 3 classes (positive=0, negative=1, zero=2)
        self.sign_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

        # Digit prediction heads - one per position
        # Each head predicts 10 classes (0-9)
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + d_model // 4, d_model // 2),  # +position embed
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 10),
            )
            for _ in range(self.n_digit_positions)
        ])

        # Optional: Auxiliary regression head for numerical supervision
        # Helps guide digit predictions toward correct magnitude
        self.aux_regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        operation_type: torch.Tensor,
        param_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for digit-by-digit prediction.

        Args:
            hidden: [B, T, d_model] hidden states from decoder
            operation_type: [B] operation type indices (per sequence)
            param_type: [B, T] parameter type indices (per position)

        Returns:
            Dictionary with:
            - sign_logits: [B, T, 3] sign prediction
            - digit_logits: [B, T, n_positions, 10] digit predictions
            - aux_value: [B, T, 1] auxiliary regression prediction
        """
        B, T, D = hidden.shape
        device = hidden.device

        # Get conditioning embeddings
        op_emb = self.operation_embed(operation_type)  # [B, d_model//4]
        op_emb = op_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model//4]

        param_emb = self.param_type_embed(param_type)  # [B, T, d_model//4]

        # Combine hidden state with conditioning
        context = torch.cat([hidden, op_emb, param_emb], dim=-1)  # [B, T, d_model + d_model//2]
        context = self.context_proj(context)  # [B, T, d_model]

        # Predict sign
        sign_logits = self.sign_head(context)  # [B, T, 3]

        # Predict each digit position
        digit_logits_list = []
        for pos in range(self.n_digit_positions):
            # Add position embedding
            pos_emb = self.digit_position_embed.weight[pos]  # [d_model//4]
            pos_emb = pos_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, d_model//4]

            pos_context = torch.cat([context, pos_emb], dim=-1)  # [B, T, d_model + d_model//4]
            digit_logits = self.digit_heads[pos](pos_context)  # [B, T, 10]
            digit_logits_list.append(digit_logits)

        digit_logits = torch.stack(digit_logits_list, dim=2)  # [B, T, n_positions, 10]

        # Auxiliary regression prediction
        aux_value = self.aux_regression_head(context)  # [B, T, 1]

        return {
            'sign_logits': sign_logits,
            'digit_logits': digit_logits,
            'aux_value': aux_value,
        }

    def decode_to_values(
        self,
        sign_logits: torch.Tensor,
        digit_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode sign and digit predictions to numeric values.

        Args:
            sign_logits: [B, T, 3] sign logits
            digit_logits: [B, T, n_positions, 10] digit logits

        Returns:
            values: [B, T] decoded numeric values
        """
        # Get predictions
        sign_pred = torch.argmax(sign_logits, dim=-1)  # [B, T], 0=+, 1=-, 2=zero
        digit_pred = torch.argmax(digit_logits, dim=-1)  # [B, T, n_positions]

        B, T = sign_pred.shape
        values = torch.zeros(B, T, device=sign_logits.device)

        # Vectorized decoding
        for pos in range(self.max_int_digits):
            # Integer part: each position is 10^(max_int_digits - 1 - pos)
            power = self.max_int_digits - 1 - pos
            values = values + digit_pred[:, :, pos].float() * (10 ** power)

        for pos in range(self.n_decimal_digits):
            # Decimal part: each position is 10^(-pos-1)
            idx = self.max_int_digits + pos
            power = -(pos + 1)
            values = values + digit_pred[:, :, idx].float() * (10 ** power)

        # Apply sign
        sign_multiplier = torch.where(sign_pred == 1, -1.0, 1.0)
        sign_multiplier = torch.where(sign_pred == 2, 0.0, sign_multiplier)  # Zero case
        values = values * sign_multiplier

        return values

    def encode_values_to_digits(
        self,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode numeric values to sign and digit targets.

        Args:
            values: [B, T] numeric values

        Returns:
            sign_targets: [B, T] sign indices (0=+, 1=-, 2=zero)
            digit_targets: [B, T, n_positions] digit indices (0-9)
        """
        B, T = values.shape
        device = values.device

        # Determine sign
        sign_targets = torch.zeros(B, T, dtype=torch.long, device=device)
        sign_targets = torch.where(values < 0, 1, sign_targets)  # Negative = 1
        sign_targets = torch.where(values == 0, 2, sign_targets)  # Zero = 2

        # Get absolute values for digit extraction
        abs_values = torch.abs(values)

        # Extract digits
        digit_targets = torch.zeros(B, T, self.n_digit_positions, dtype=torch.long, device=device)

        # Integer part
        int_part = abs_values.long()
        for pos in range(self.max_int_digits):
            power = self.max_int_digits - 1 - pos
            divisor = 10 ** power
            digit = (int_part // divisor) % 10
            digit_targets[:, :, pos] = digit

        # Decimal part
        dec_part = abs_values - int_part.float()
        for pos in range(self.n_decimal_digits):
            # Shift decimal to get the digit
            dec_part = dec_part * 10
            digit = dec_part.long() % 10
            digit_targets[:, :, self.max_int_digits + pos] = digit

        return sign_targets, digit_targets


class ParameterAwareDigitHead(nn.Module):
    """
    Parameter-aware digit prediction with per-parameter configuration.

    Different parameters have different value ranges and precision requirements:
    - X, Y: Position coordinates (e.g., 0.0 to 3.5, 4 decimal places)
    - Z: Depth (e.g., -0.125 to 0.6, 4 decimal places)
    - F: Feed rate (e.g., 7 to 40, 1 decimal place)
    - R: Radius (e.g., 0.75 to 3.75, 4 decimal places)
    - S: Spindle speed (e.g., 0 to 30000, no decimals)

    This module uses the appropriate configuration based on parameter type.
    """

    def __init__(
        self,
        d_model: int,
        n_operations: int = 9,
        param_names: List[str] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Hidden dimension
            n_operations: Number of operation types
            param_names: List of parameter names ['X', 'Y', 'Z', ...]
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_operations = n_operations
        self.param_names = param_names or ['X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K']
        self.n_param_types = len(self.param_names)

        # Get max dimensions across all parameters
        self.max_int_digits = max(
            PARAM_VALUE_CONFIG.get(p, DEFAULT_PARAM_CONFIG)['max_int_digits']
            for p in self.param_names
        )
        self.max_decimal_digits = max(
            PARAM_VALUE_CONFIG.get(p, DEFAULT_PARAM_CONFIG)['n_decimal_digits']
            for p in self.param_names
        )

        # Create shared digit head
        self.digit_head = DigitByDigitValueHead(
            d_model=d_model,
            n_operations=n_operations,
            n_param_types=self.n_param_types,
            max_int_digits=self.max_int_digits,
            n_decimal_digits=self.max_decimal_digits,
            dropout=dropout,
        )

        # Parameter name to index mapping
        self.param2idx = {name: i for i, name in enumerate(self.param_names)}

    def forward(
        self,
        hidden: torch.Tensor,
        operation_type: torch.Tensor,
        param_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden: [B, T, d_model] hidden states
            operation_type: [B] operation indices
            param_type: [B, T] parameter type indices

        Returns:
            Same as DigitByDigitValueHead.forward()
        """
        return self.digit_head(hidden, operation_type, param_type)

    def decode_to_values(
        self,
        sign_logits: torch.Tensor,
        digit_logits: torch.Tensor,
        param_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode to values, optionally applying parameter-specific rounding.

        Args:
            sign_logits: [B, T, 3]
            digit_logits: [B, T, n_positions, 10]
            param_type: [B, T] parameter types for precision masking

        Returns:
            values: [B, T] decoded values
        """
        values = self.digit_head.decode_to_values(sign_logits, digit_logits)

        # Apply parameter-specific precision if requested
        if param_type is not None:
            for param_name, param_idx in self.param2idx.items():
                config = PARAM_VALUE_CONFIG.get(param_name, DEFAULT_PARAM_CONFIG)
                n_dec = config['n_decimal_digits']

                # Mask for this parameter type
                mask = (param_type == param_idx)
                if mask.any():
                    # Round to appropriate precision
                    precision = 10 ** n_dec
                    values = torch.where(
                        mask,
                        torch.round(values * precision) / precision,
                        values
                    )

                    # Clamp to valid range
                    values = torch.where(
                        mask,
                        torch.clamp(values, config['min'], config['max']),
                        values
                    )

        return values


class DigitByDigitLoss(nn.Module):
    """
    Loss function for digit-by-digit value prediction.

    Combines:
    1. Cross-entropy loss for sign prediction
    2. Cross-entropy loss for each digit position
    3. Optional auxiliary MSE loss for numerical guidance
    """

    def __init__(
        self,
        n_digit_positions: int = 6,
        aux_loss_weight: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            n_digit_positions: Number of digit positions
            aux_loss_weight: Weight for auxiliary regression loss
            label_smoothing: Label smoothing for CE loss
        """
        super().__init__()

        self.n_digit_positions = n_digit_positions
        self.aux_loss_weight = aux_loss_weight

        self.sign_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )
        self.digit_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )
        self.aux_loss_fn = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        sign_targets: torch.Tensor,
        digit_targets: torch.Tensor,
        value_targets: torch.Tensor,
        numeric_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute digit-by-digit loss.

        Args:
            predictions: Dict with 'sign_logits', 'digit_logits', 'aux_value'
            sign_targets: [B, T] sign targets (0=+, 1=-, 2=zero)
            digit_targets: [B, T, n_positions] digit targets (0-9)
            value_targets: [B, T] raw numeric values (for aux loss)
            numeric_mask: [B, T] True for NUMERIC token positions

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        sign_logits = predictions['sign_logits']  # [B, T, 3]
        digit_logits = predictions['digit_logits']  # [B, T, n_positions, 10]
        aux_value = predictions['aux_value']  # [B, T, 1]

        B, T = numeric_mask.shape
        device = sign_logits.device

        loss_dict = {}

        # Only compute on NUMERIC positions
        if not numeric_mask.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {'sign_loss': 0.0, 'digit_loss': 0.0, 'aux_loss': 0.0}

        # 1. Sign loss
        sign_logits_flat = sign_logits[numeric_mask]  # [N, 3]
        sign_targets_flat = sign_targets[numeric_mask]  # [N]
        sign_loss = self.sign_loss_fn(sign_logits_flat, sign_targets_flat)
        loss_dict['sign_loss'] = sign_loss.item()

        # 2. Digit losses (per position)
        digit_loss = torch.tensor(0.0, device=device)
        n_positions = digit_logits.size(2)

        for pos in range(n_positions):
            pos_logits = digit_logits[:, :, pos, :][numeric_mask]  # [N, 10]
            pos_targets = digit_targets[:, :, pos][numeric_mask]  # [N]
            pos_loss = self.digit_loss_fn(pos_logits, pos_targets)
            digit_loss = digit_loss + pos_loss
            loss_dict[f'digit_{pos}_loss'] = pos_loss.item()

        digit_loss = digit_loss / n_positions
        loss_dict['digit_loss'] = digit_loss.item()

        # 3. Auxiliary regression loss
        aux_value_flat = aux_value[numeric_mask].squeeze(-1)  # [N]
        value_targets_flat = value_targets[numeric_mask]  # [N]
        aux_loss = self.aux_loss_fn(aux_value_flat, value_targets_flat)
        loss_dict['aux_loss'] = aux_loss.item()

        # Total loss
        total_loss = sign_loss + digit_loss + self.aux_loss_weight * aux_loss
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# Test code
if __name__ == '__main__':
    # Test the digit head
    print("Testing DigitByDigitValueHead...")

    B, T, D = 2, 10, 256
    n_ops = 9
    n_params = 10

    head = DigitByDigitValueHead(
        d_model=D,
        n_operations=n_ops,
        n_param_types=n_params,
        max_int_digits=2,
        n_decimal_digits=4,
    )

    # Random inputs
    hidden = torch.randn(B, T, D)
    op_type = torch.randint(0, n_ops, (B,))
    param_type = torch.randint(0, n_params, (B, T))

    # Forward pass
    out = head(hidden, op_type, param_type)
    print(f"  sign_logits: {out['sign_logits'].shape}")  # [B, T, 3]
    print(f"  digit_logits: {out['digit_logits'].shape}")  # [B, T, 6, 10]
    print(f"  aux_value: {out['aux_value'].shape}")  # [B, T, 1]

    # Test encoding/decoding
    print("\nTesting encode/decode...")
    test_values = torch.tensor([[1.5750, -0.125, 2.0], [0.0, 3.14159, -99.9]])
    sign_targets, digit_targets = head.encode_values_to_digits(test_values)
    print(f"  Original values: {test_values}")
    print(f"  Sign targets: {sign_targets}")
    print(f"  Digit targets shape: {digit_targets.shape}")

    # Create fake logits from targets
    sign_logits = F.one_hot(sign_targets, num_classes=3).float() * 10
    digit_logits = F.one_hot(digit_targets, num_classes=10).float() * 10

    decoded = head.decode_to_values(sign_logits, digit_logits)
    print(f"  Decoded values: {decoded}")
    print(f"  Max error: {(decoded - test_values).abs().max():.6f}")

    # Test loss
    print("\nTesting DigitByDigitLoss...")
    loss_fn = DigitByDigitLoss(n_digit_positions=6)

    numeric_mask = torch.ones(B, T, dtype=torch.bool)
    sign_targets_full = torch.zeros(B, T, dtype=torch.long)
    digit_targets_full = torch.zeros(B, T, 6, dtype=torch.long)
    value_targets = torch.randn(B, T)

    loss, loss_dict = loss_fn(out, sign_targets_full, digit_targets_full, value_targets, numeric_mask)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")

    print("\nAll tests passed!")
