"""
Sensor Multi-Head Decoder for G-code Token Generation.

This decoder improves upon SensorConditionedTokenDecoder by:
1. Adding operation conditioning (using 100% accurate operation_type from encoder)
2. Multi-head outputs for type, command, param_type instead of single vocab head
3. Digit-by-digit value prediction instead of 668 flat token classes

Architecture:
    Sensor Data [B, T_s, 155]
        ↓
    MM-DTAE-LSTM Encoder (FROZEN)
      - encode() → sensor_latent [B, T_s, 128]
      - classify() → operation_type [B] (100% accurate!)
        ↓
    SensorMultiHeadDecoder
      - Operation conditioning on sensor memory
      - TransformerDecoder with cross-attention
      - Multi-head outputs: type, command, param_type, digits
        ↓
    Structured G-code predictions

Author: Claude Code
Date: December 2025
"""

import math
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .digit_value_head import DigitByDigitValueHead


# ============================================================================
# Sensor Value Prior Module (Phase 2)
# ============================================================================

class SensorValuePrior(nn.Module):
    """
    Extract numeric value hints directly from sensor features.

    The problem: Model often collapses to predicting common values (NUM_X_-125,
    NUM_Y_2913) regardless of actual sensor input. This module provides a direct
    path from raw sensor features to digit predictions.

    Architecture:
        Sensor Memory [B, T_s, sensor_dim]
            ↓ (global pooling)
        Pooled Features [B, sensor_dim]
            ↓ (per-param-type heads)
        Digit Biases [B, n_param_types, n_digits, 10]
            ↓ (select based on predicted param_type)
        Position Biases [B, L, n_digits, 10]

    This provides a "shortcut" from sensor input directly to digit predictions,
    bypassing the transformer decoder's tendency to memorize common patterns.
    """

    def __init__(
        self,
        sensor_dim: int,
        n_param_types: int = 10,
        n_digits: int = 6,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            sensor_dim: Dimension of sensor embeddings from encoder
            n_param_types: Number of parameter types (X, Y, Z, F, R, etc.)
            n_digits: Total digit positions (max_int_digits + n_decimal_digits)
            hidden_dim: Hidden dimension for MLP (default: sensor_dim // 2)
            dropout: Dropout probability
        """
        super().__init__()
        self.sensor_dim = sensor_dim
        self.n_param_types = n_param_types
        self.n_digits = n_digits
        self.hidden_dim = hidden_dim or sensor_dim // 2

        # Multi-scale temporal pooling to capture different aspects of sensor signal
        self.pool_fc = nn.Sequential(
            nn.Linear(sensor_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-parameter-type value prediction heads
        # Each head predicts biases for all digit positions
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, n_digits * 10),  # n_digits × 10 classes
            )
            for _ in range(n_param_types)
        ])

        # Sign prediction head (per parameter type)
        self.sign_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim // 2, 3),  # positive, negative, zero
            )
            for _ in range(n_param_types)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to start as residual."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sensor_memory: torch.Tensor,
        param_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sensor-based priors for digit predictions.

        Args:
            sensor_memory: [B, T_s, sensor_dim] raw sensor embeddings
            param_type: [B, L] predicted parameter types per position

        Returns:
            Dictionary with:
            - digit_bias: [B, L, n_digits, 10] bias to add to digit logits
            - sign_bias: [B, L, 3] bias to add to sign logits
        """
        B, T_s, D = sensor_memory.shape
        L = param_type.shape[1]
        device = sensor_memory.device

        # Global pooling of sensor sequence
        pooled = sensor_memory.mean(dim=1)  # [B, sensor_dim]
        pooled = self.pool_fc(pooled)  # [B, hidden_dim]

        # Get predictions from all parameter-type heads
        # Stack: [B, n_param_types, n_digits * 10]
        all_digit_preds = torch.stack([
            head(pooled) for head in self.value_heads
        ], dim=1)  # [B, n_param_types, n_digits * 10]
        all_digit_preds = all_digit_preds.view(B, self.n_param_types, self.n_digits, 10)

        # Stack sign predictions: [B, n_param_types, 3]
        all_sign_preds = torch.stack([
            head(pooled) for head in self.sign_heads
        ], dim=1)  # [B, n_param_types, 3]

        # Select based on predicted param_type at each position
        # param_type: [B, L] -> indices into n_param_types
        param_type_clamped = param_type.clamp(0, self.n_param_types - 1)

        # Gather digit biases: [B, L, n_digits, 10]
        # Expand param_type to match dimensions
        param_idx = param_type_clamped.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        param_idx = param_idx.expand(-1, -1, self.n_digits, 10)  # [B, L, n_digits, 10]

        # Expand all_digit_preds to have L dimension for gather
        all_digit_expanded = all_digit_preds.unsqueeze(1).expand(-1, L, -1, -1, -1)  # [B, L, n_param_types, n_digits, 10]

        # Gather along param_type dimension
        digit_bias = torch.gather(
            all_digit_expanded,
            dim=2,
            index=param_idx.unsqueeze(2),  # [B, L, 1, n_digits, 10]
        ).squeeze(2)  # [B, L, n_digits, 10]

        # Gather sign biases: [B, L, 3]
        param_idx_sign = param_type_clamped.unsqueeze(-1)  # [B, L, 1]
        param_idx_sign = param_idx_sign.expand(-1, -1, 3)  # [B, L, 3]

        all_sign_expanded = all_sign_preds.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, n_param_types, 3]
        sign_bias = torch.gather(
            all_sign_expanded,
            dim=2,
            index=param_idx_sign.unsqueeze(2),  # [B, L, 1, 3]
        ).squeeze(2)  # [B, L, 3]

        return {
            'digit_bias': digit_bias,
            'sign_bias': sign_bias,
        }


# ============================================================================
# Stochastic Depth / DropPath
# ============================================================================

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample during training.

    Regularization technique that randomly drops entire residual branches
    during training. Helps prevent co-adaptation and improves generalization.

    References:
        - "Deep Networks with Stochastic Depth" (Huang et al., 2016)
        - "Going deeper with Image Transformers" (Touvron et al., 2021)
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob: Probability of dropping the path (0 = no drop, 1 = always drop)
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Generate random tensor with shape [batch_size, 1, 1, ...] to broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class TransformerDecoderLayerWithDropPath(nn.Module):
    """
    Transformer decoder layer with stochastic depth (DropPath).

    Wraps standard TransformerDecoderLayer with DropPath on residual connections.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # DropPath for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm_first = norm_first
        self.activation = F.gelu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with DropPath on residual connections."""

        if self.norm_first:
            # Pre-LN: norm before attention
            # Self-attention with DropPath
            x = tgt
            attn_out = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self.drop_path1(attn_out)

            # Cross-attention with DropPath
            cross_out = self._ca_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self.drop_path2(cross_out)

            # FFN with DropPath
            ffn_out = self._ff_block(self.norm3(x))
            x = x + self.drop_path3(ffn_out)
        else:
            # Post-LN: norm after attention
            x = tgt
            attn_out = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + self.drop_path1(attn_out))

            cross_out = self._ca_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + self.drop_path2(cross_out))

            ffn_out = self._ff_block(x)
            x = self.norm3(x + self.drop_path3(ffn_out))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ca_block(self, x, memory, attn_mask, key_padding_mask):
        """Cross-attention block."""
        x, _ = self.cross_attn(x, memory, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout2(x)

    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerEncoderLayerWithDropPath(nn.Module):
    """
    Transformer encoder layer with stochastic depth (DropPath).

    Used for self-attention-only ablation with DropPath.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # DropPath
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm_first = norm_first
        self.activation = F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with DropPath."""

        if self.norm_first:
            x = src
            attn_out = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self.drop_path1(attn_out)

            ffn_out = self._ff_block(self.norm2(x))
            x = x + self.drop_path2(ffn_out)
        else:
            x = src
            attn_out = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + self.drop_path1(attn_out))

            ffn_out = self._ff_block(x)
            x = self.norm2(x + self.drop_path2(ffn_out))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SensorMultiHeadDecoder(nn.Module):
    """
    Multi-head decoder with operation conditioning for G-code generation.

    Key improvements over SensorConditionedTokenDecoder:
    1. Operation embedding conditions the cross-attention memory
    2. Separate heads for type, command, param_type (structured output)
    3. Digit-by-digit value prediction (no 668-class bottleneck)

    Token Types:
    - Type 0: SPECIAL (PAD, BOS, EOS, UNK)
    - Type 1: COMMAND (G0, G1, G2, G3, G53)
    - Type 2: PARAM_LETTER (X, Y, Z, F, R)
    - Type 3: NUMERIC (predicted digit-by-digit)
    """

    def __init__(
        self,
        vocab_size: int = 668,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        sensor_dim: int = 128,
        n_operations: int = 9,
        n_types: int = 4,
        n_commands: int = 6,
        n_param_types: int = 10,
        max_int_digits: int = 2,
        n_decimal_digits: int = 4,
        d_ff: int = None,
        dropout: float = 0.3,
        embed_dropout: float = 0.1,
        max_seq_len: int = 64,
        # ============ ABLATION FLAGS ============
        no_operation_conditioning: bool = False,
        no_cross_attention: bool = False,
        no_positional_encoding: bool = False,
        # ============ SELF-CONDITIONING ============
        use_self_conditioning: bool = False,
        # ============ HIERARCHICAL CONDITIONING ============
        hierarchical: bool = False,
        hier_embed_dim: int = 48,
        # ============ MEMORY POSITIONAL ENCODING ============
        memory_pos_encoding: bool = False,
        # ============ REGULARIZATION ============
        drop_path_rate: float = 0.0,
        use_gradient_checkpointing: bool = False,
        # ============ SENSOR VALUE PRIOR (Phase 2) ============
        use_sensor_prior: bool = False,
        sensor_prior_weight: float = 0.5,
        # ============ NUMERIC REGRESSION HEAD ============
        use_regression_head: bool = False,
        # ============ WINDOW POSITION INPUT (Stage 2A) ============
        use_window_position: bool = False,
        max_windows_per_file: int = 32,
        # ============ SEQUENCE CLASSIFIER (Stage 2D) ============
        use_sequence_classifier: bool = False,
        n_unique_sequences: int = 34,
        # ============ POINTER NETWORK (V7) ============
        use_pointer_network: bool = False,
        axis_value_tables: dict = None,  # loaded from axis_value_tables.json
    ):
        """
        Args:
            vocab_size: Token vocabulary size (for embedding layer)
            d_model: Decoder hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            sensor_dim: MM-DTAE-LSTM latent dimension (128)
            n_operations: Number of operation types (9)
            n_types: Number of token types (4: SPECIAL, COMMAND, PARAM, NUMERIC)
            n_commands: Number of G-code commands (6: G0, G1, G2, G3, G53, OTHER)
            n_param_types: Number of parameter types (10: X, Y, Z, F, R, S, I, J, K, OTHER)
            max_int_digits: Max integer digits for value prediction
            n_decimal_digits: Number of decimal digits for value prediction
            d_ff: Feedforward dimension (default: 4 * d_model)
            dropout: Dropout rate
            embed_dropout: Embedding dropout rate
            max_seq_len: Maximum sequence length
            no_operation_conditioning: Ablation - disable operation embedding
            no_cross_attention: Ablation - use self-attention only (no sensor memory)
            no_positional_encoding: Ablation - disable positional encoding
            use_self_conditioning: Enable self-conditioning (feed previous logits as features)
            drop_path_rate: Stochastic depth drop rate (0 = disabled)
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            use_sensor_prior: Enable sensor value prior for direct sensor-to-digit prediction
            sensor_prior_weight: Weight for sensor prior bias (0-1, default 0.5)
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_operations = n_operations
        self.n_types = n_types
        self.n_commands = n_commands
        self.n_param_types = n_param_types

        # Store ablation flags
        self.no_operation_conditioning = no_operation_conditioning
        self.no_cross_attention = no_cross_attention
        self.no_positional_encoding = no_positional_encoding
        self.use_self_conditioning = use_self_conditioning

        # Regularization settings
        self.drop_path_rate = drop_path_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # ============ OPERATION CONDITIONING (KEY ADDITION) ============
        # Operation embedding to condition the decoder on known operation type
        if not no_operation_conditioning:
            self.operation_embed = nn.Embedding(n_operations, d_model // 4)
            sensor_proj_input_dim = sensor_dim + d_model // 4
        else:
            self.operation_embed = None
            sensor_proj_input_dim = sensor_dim

        # ============ SENSOR PROJECTION (with/without operation context) ============
        # Project sensor (+ operation) to decoder dimension
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_proj_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============ MEMORY POSITIONAL ENCODING ============
        # Adds positional info to sensor memory so decoder knows WHERE in the
        # toolpath each window comes from (disambiguates e.g. Y=0.175 vs Y=2.913)
        self.memory_pos_encoding_enabled = memory_pos_encoding
        if memory_pos_encoding:
            self.memory_pos_enc = SinusoidalPositionalEncoding(d_model, max_len=4096, dropout=dropout)
        else:
            self.memory_pos_enc = None

        # ============ HIERARCHICAL CONDITIONING ============
        # Cascaded heads: type → cmd/param → digits
        # Breaks the gradient competition between heads by giving downstream
        # heads explicit context from upstream predictions
        self.hierarchical = hierarchical
        self.hier_embed_dim = hier_embed_dim
        if hierarchical:
            # Embeddings for cascading context
            self.hier_type_embed = nn.Embedding(n_types, hier_embed_dim)
            self.hier_param_embed = nn.Embedding(n_param_types, hier_embed_dim)
            # Project embeddings to d_model for additive conditioning on digit head
            self.hier_type_proj = nn.Linear(hier_embed_dim, d_model)
            self.hier_param_proj = nn.Linear(hier_embed_dim, d_model)

        # ============ TOKEN EMBEDDING ============
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # ============ SELF-CONDITIONING ============
        # Project previous logits to conditioning features
        if use_self_conditioning:
            # Total logit dimensions: n_types + n_commands + n_param_types
            self_cond_dim = n_types + n_commands + n_param_types
            self.self_cond_proj = nn.Sequential(
                nn.Linear(self_cond_dim, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Combine token embedding with self-conditioning
            self.cond_combine = nn.Linear(d_model + d_model // 4, d_model)
        else:
            self.self_cond_proj = None
            self.cond_combine = None

        # ============ TRANSFORMER DECODER ============
        # Use stochastic depth with linearly increasing drop rate per layer
        # drop_path_rate is the max rate at the last layer
        dpr = [drop_path_rate * i / (n_layers - 1) if n_layers > 1 else 0.0
               for i in range(n_layers)]

        if not no_cross_attention:
            # Standard decoder with cross-attention to sensor memory
            if drop_path_rate > 0:
                # Use custom layers with DropPath
                self.decoder_layers = nn.ModuleList([
                    TransformerDecoderLayerWithDropPath(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_ff,
                        dropout=dropout,
                        drop_path=dpr[i],
                        batch_first=True,
                        norm_first=True,
                    )
                    for i in range(n_layers)
                ])
                self.decoder = None  # Use custom forward
            else:
                # Standard PyTorch decoder (no DropPath)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,  # Pre-LN for better stability
                )
                self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
                self.decoder_layers = None
            self.use_encoder_only = False
        else:
            # Ablation: Self-attention only (no sensor cross-attention)
            if drop_path_rate > 0:
                # Use custom layers with DropPath
                self.decoder_layers = nn.ModuleList([
                    TransformerEncoderLayerWithDropPath(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_ff,
                        dropout=dropout,
                        drop_path=dpr[i],
                        batch_first=True,
                        norm_first=True,
                    )
                    for i in range(n_layers)
                ])
                self.decoder = None
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerEncoder(encoder_layer, n_layers)
                self.decoder_layers = None
            self.use_encoder_only = True

        # ============ MULTI-HEAD OUTPUTS ============
        # Type head: SPECIAL, COMMAND, PARAM_LETTER, NUMERIC
        # Always takes raw hidden state (no conditioning needed - it's the root)
        self.type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_types),
        )

        # Command head: G0, G1, G2, G3, G53, OTHER
        # In hierarchical mode: conditioned on type embedding
        cmd_input_dim = d_model + hier_embed_dim if hierarchical else d_model
        self.command_head = nn.Sequential(
            nn.Linear(cmd_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_commands),
        )

        # Parameter type head: X, Y, Z, F, R, S, I, J, K, OTHER
        # In hierarchical mode: conditioned on type embedding
        param_input_dim = d_model + hier_embed_dim if hierarchical else d_model
        self.param_type_head = nn.Sequential(
            nn.Linear(param_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_param_types),
        )

        # ============ DIGIT VALUE HEAD ============
        # Digit-by-digit value prediction with operation/param conditioning
        self.digit_value_head = DigitByDigitValueHead(
            d_model=d_model,
            n_operations=n_operations,
            n_param_types=n_param_types,
            max_int_digits=max_int_digits,
            n_decimal_digits=n_decimal_digits,
            dropout=dropout,
        )

        # ============ SENSOR VALUE PRIOR (Phase 2) ============
        # Direct sensor-to-digit prediction to avoid mode collapse
        self.use_sensor_prior = use_sensor_prior
        self.sensor_prior_weight = sensor_prior_weight
        if use_sensor_prior:
            n_digits = max_int_digits + n_decimal_digits
            self.sensor_value_prior = SensorValuePrior(
                sensor_dim=sensor_dim,
                n_param_types=n_param_types,
                n_digits=n_digits,
                hidden_dim=sensor_dim,  # Use full sensor dim for hidden
                dropout=dropout,
            )
        else:
            self.sensor_value_prior = None

        # ============ OUTPUT NORMALIZATION ============
        self.output_norm = nn.LayerNorm(d_model)

        # ============ OPTIONAL: Legacy token head for comparison ============
        self.legacy_token_head = nn.Linear(d_model, vocab_size)

        # ============ TYPE CONSTRAINT MASK ============
        # Will be set via set_vocab() method
        self.vocab = None
        self.type_token_mask = None
        self.use_type_constraint = True  # Can be disabled for ablation

        # ============ GRAMMAR CONSTRAINT MASK ============
        # Transition mask: for each token, which tokens can follow it
        # Built in set_vocab() alongside type_token_mask
        self.grammar_mask = None
        self.use_grammar_constraint = True  # Can be toggled via CLI

        # ============ NUMERIC REGRESSION HEAD ============
        # Direct float prediction conditioned on param type
        self._use_regression_head = use_regression_head
        if use_regression_head:
            self.regression_head = nn.Sequential(
                nn.Linear(d_model + n_param_types, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.regression_head = None

        # ============ WINDOW POSITION INPUT (Stage 2A) ============
        self._use_window_position = use_window_position
        self._max_windows_per_file = max_windows_per_file
        if use_window_position:
            self.window_pos_embed = nn.Embedding(max_windows_per_file, d_model // 4)
            self.window_frac_proj = nn.Linear(1, d_model // 8)
            # Project the combined position features to d_model and add to memory
            self.window_pos_proj = nn.Linear(d_model // 4 + d_model // 8, d_model)

        # ============ SEQUENCE CLASSIFIER (Stage 2D) ============
        self._use_sequence_classifier = use_sequence_classifier
        self.n_unique_sequences = n_unique_sequences
        if use_sequence_classifier:
            self.sequence_classifier = nn.Sequential(
                nn.Linear(sensor_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_unique_sequences),
            )
            self.sequence_embed = nn.Embedding(n_unique_sequences, d_model)
            self.sequence_gate = nn.Linear(d_model, d_model)

        # ============ POINTER NETWORK (V7) ============
        self._use_pointer_network = use_pointer_network
        self.axis_value_tables = axis_value_tables
        if use_pointer_network and axis_value_tables is not None:
            # Per-axis classification heads
            # axis_value_tables: {"X": {"n_values": 203, ...}, "Y": {...}, ...}
            self.pointer_heads = nn.ModuleDict()
            self.axis_n_values = {}   # axis_name -> n_values
            self.axis_param_id = {}   # axis_name -> param_type_id (X=0, Y=1, etc.)
            param_names = ['X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K']
            for axis_name, axis_info in axis_value_tables.items():
                n_values = axis_info['n_values']
                if n_values < 2:
                    continue  # Skip axes with only 1 value (e.g., F=7.0)
                self.pointer_heads[axis_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.LayerNorm(d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, n_values),
                )
                self.axis_n_values[axis_name] = n_values
                if axis_name in param_names:
                    self.axis_param_id[axis_name] = param_names.index(axis_name)
            # Build reverse mapping: token_id -> (axis_name, axis_idx)
            self._build_pointer_token_maps()
        else:
            self.pointer_heads = None

        # ============ OPERATION-SEQUENCE PRIOR (Stage 2E) ============
        # Will be set via set_op_sequence_mask() method
        self.op_seq_mask = None

        # Initialize weights
        self._init_weights()

    def set_vocab(self, vocab: dict):
        """
        Set vocabulary and build type masks + grammar transition masks.

        Type masks: constrain token prediction by predicted token type.
        Grammar masks: constrain what tokens can follow a given token based on
        G-code grammar rules (e.g., after X only NUM_X_* is valid).

        Args:
            vocab: Dictionary mapping token strings to indices
        """
        self.vocab = vocab
        type_mask = self._build_type_masks(vocab)
        grammar_mask = self._build_grammar_masks(vocab)

        # Move to same device as model if already on device
        device = 'cpu'
        if hasattr(self, 'type_head') and next(self.type_head.parameters()).device.type != 'cpu':
            device = next(self.type_head.parameters()).device
            type_mask = type_mask.to(device)
            grammar_mask = grammar_mask.to(device)

        # Remove existing buffers if they exist, then register new ones
        for buf_name, buf_val in [('type_token_mask', type_mask), ('grammar_mask', grammar_mask)]:
            if hasattr(self, buf_name):
                delattr(self, buf_name)
            if buf_name in self._buffers:
                del self._buffers[buf_name]
            self.register_buffer(buf_name, buf_val)

        print(f"[SensorMultiHeadDecoder] Type constraint mask built: {type_mask.shape}")
        print(f"[SensorMultiHeadDecoder] Grammar mask built: {grammar_mask.shape}")

    def _build_type_masks(self, vocab: dict) -> torch.Tensor:
        """
        Build masks mapping token types to valid token indices.

        Type encoding:
        - Type 0: SPECIAL (PAD, BOS, EOS, UNK, MASK)
        - Type 1: COMMAND (G0, G1, G2, G3, G53, M30, NONE)
        - Type 2: PARAM_TYPE (X, Y, Z, F, R, S, I, J, K, A, B, C, P, Q, E)
        - Type 3: NUMERIC (NUM_X_*, NUM_Y_*, NUM_Z_*, NUM_F_*, NUM_R_*)

        Returns:
            mask: [n_types, vocab_size] where valid tokens have 0, invalid have large negative
        """
        # Use large negative value instead of -inf to avoid NaN/inf loss during training
        # when type predictions are incorrect (especially early in training)
        MASK_VALUE = -1e4  # Large enough to suppress but not cause inf loss
        mask = torch.full((self.n_types, len(vocab)), MASK_VALUE)

        # Special tokens
        special_tokens = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK'}

        # Parameter letters (single letter tokens)
        param_letters = {'X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K', 'A', 'B', 'C', 'P', 'Q', 'E'}

        for token, idx in vocab.items():
            if token in special_tokens:
                mask[0, idx] = 0  # Type 0: SPECIAL
            elif token.startswith('G') or token.startswith('M') or token == 'NONE':
                mask[1, idx] = 0  # Type 1: COMMAND
            elif token in param_letters:
                mask[2, idx] = 0  # Type 2: PARAM_TYPE
            elif token.startswith('NUM_'):
                mask[3, idx] = 0  # Type 3: NUMERIC
            else:
                # Unknown token type, allow in SPECIAL by default
                mask[0, idx] = 0

        # Log type distribution for debugging
        type_counts = [int((mask[i] == 0).sum()) for i in range(self.n_types)]
        print(f"[SensorMultiHeadDecoder] Type mask distribution: SPECIAL={type_counts[0]}, COMMAND={type_counts[1]}, PARAM={type_counts[2]}, NUMERIC={type_counts[3]}")

        return mask

    def _build_grammar_masks(self, vocab: dict) -> torch.Tensor:
        """
        Build grammar transition masks: for each token, which tokens can legally follow.

        G-code grammar rules:
        - BOS -> COMMAND (G0, G1, M30, ...) or PARAM (X, Y for continuation lines)
        - COMMAND (G0, G1, G2, G3, G53) -> PARAM (X, Y, Z, F, R, ...)
        - COMMAND (M30) -> EOS
        - PARAM (X) -> NUM_X_*
        - PARAM (Y) -> NUM_Y_*
        - PARAM (Z) -> NUM_Z_*
        - PARAM (R) -> NUM_R_*
        - PARAM (F) -> NUM_F_*
        - NUM_*_* -> PARAM (next axis) or EOS
        - EOS -> PAD
        - PAD -> PAD

        Returns:
            mask: [vocab_size, vocab_size] where valid transitions have 0,
                  invalid transitions have large negative value
        """
        MASK_VALUE = -1e4
        V = len(vocab)
        mask = torch.full((V, V), MASK_VALUE)

        # Build token sets
        special = {}
        move_commands = set()  # G0, G1, G2, G3, G53 — commands that take params
        end_commands = set()   # M30 — commands that end program
        params = {}            # letter -> vocab_id
        numerics_by_axis = {}  # axis -> set of vocab_ids
        other_tokens = set()

        param_letters = {'X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K'}

        for token, idx in vocab.items():
            if token in ('PAD', 'BOS', 'EOS', 'UNK', 'MASK'):
                special[token] = idx
            elif token in param_letters:
                params[token] = idx
            elif token.startswith('NUM_'):
                parts = token.split('_')
                if len(parts) >= 3:
                    axis = parts[1]
                    if axis not in numerics_by_axis:
                        numerics_by_axis[axis] = set()
                    numerics_by_axis[axis].add(idx)
            elif token.startswith('G'):
                move_commands.add(idx)
            elif token.startswith('M'):
                end_commands.add(idx)
            else:
                other_tokens.add(idx)

        all_param_ids = set(params.values())
        all_numeric_ids = set()
        for s in numerics_by_axis.values():
            all_numeric_ids.update(s)

        pad_id = special.get('PAD', 0)
        bos_id = special.get('BOS', 1)
        eos_id = special.get('EOS', 2)

        # Rule: BOS -> any COMMAND or any PARAM (for continuation lines)
        for cmd_id in (move_commands | end_commands):
            mask[bos_id, cmd_id] = 0
        for p_id in all_param_ids:
            mask[bos_id, p_id] = 0

        # Rule: move COMMAND (G0, G1, G2, G3, G53) -> any PARAM letter
        for cmd_id in move_commands:
            for p_id in all_param_ids:
                mask[cmd_id, p_id] = 0

        # Rule: end COMMAND (M30) -> EOS
        for cmd_id in end_commands:
            mask[cmd_id, eos_id] = 0

        # Rule: PARAM letter -> matching NUM tokens only
        for letter, p_id in params.items():
            if letter in numerics_by_axis:
                for num_id in numerics_by_axis[letter]:
                    mask[p_id, num_id] = 0
            else:
                # Axis with no numeric tokens in vocab — allow EOS as fallback
                mask[p_id, eos_id] = 0

        # Rule: NUM_*_* -> any PARAM letter or EOS
        for num_id in all_numeric_ids:
            for p_id in all_param_ids:
                mask[num_id, p_id] = 0
            mask[num_id, eos_id] = 0

        # Rule: NUM -> COMMAND (for multi-command lines like G53 G0)
        for num_id in all_numeric_ids:
            for cmd_id in (move_commands | end_commands):
                mask[num_id, cmd_id] = 0

        # Rule: COMMAND -> COMMAND (for G53 G0)
        for cmd_id in move_commands:
            for cmd_id2 in (move_commands | end_commands):
                mask[cmd_id, cmd_id2] = 0

        # Rule: EOS -> PAD
        mask[eos_id, pad_id] = 0

        # Rule: PAD -> PAD
        mask[pad_id, pad_id] = 0

        # Other/unknown tokens: allow anything (don't constrain)
        for oid in other_tokens:
            mask[oid, :] = 0
            mask[:, oid] = 0

        # UNK: allow anything
        unk_id = special.get('UNK', 3)
        mask[unk_id, :] = 0

        # Log stats
        valid_per_token = (mask == 0).sum(dim=1).float()
        print(f"[SensorMultiHeadDecoder] Grammar mask: avg valid next tokens = {valid_per_token.mean():.1f}, "
              f"min = {valid_per_token.min():.0f}, max = {valid_per_token.max():.0f}")

        return mask

    def set_op_sequence_mask(self, op_seq_mask: torch.Tensor):
        """
        Store a [n_op_types, n_unique_sequences] boolean mask for operation-type
        sequence prior (Stage 2E). True means the sequence is valid for that op type.

        Args:
            op_seq_mask: [n_op_types, n_unique_sequences] boolean tensor
        """
        if hasattr(self, 'op_seq_mask') and 'op_seq_mask' in self._buffers:
            del self._buffers['op_seq_mask']
        self.register_buffer('op_seq_mask', op_seq_mask)

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_pointer_token_maps(self):
        """Build mapping from vocab token IDs to pointer network axis indices."""
        self._token_to_pointer = {}  # token_id -> (axis_name, axis_idx)
        self._pointer_to_token = {}  # (axis_name, axis_idx) -> token_id
        if self.axis_value_tables is None:
            return
        for axis_name, axis_info in self.axis_value_tables.items():
            if axis_name not in self.axis_n_values:
                continue
            token_ids = axis_info.get('token_ids', [])
            for axis_idx, tok_id in enumerate(token_ids):
                self._token_to_pointer[tok_id] = (axis_name, axis_idx)
                self._pointer_to_token[(axis_name, axis_idx)] = tok_id

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        teacher_forcing_ratio: float = 1.0,
        prev_logits: Optional[Dict[str, torch.Tensor]] = None,
        window_index: Optional[torch.Tensor] = None,
        total_windows: Optional[torch.Tensor] = None,
        sequence_class: Optional[torch.Tensor] = None,
        op_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with optional scheduled sampling.

        Args:
            tokens: Input token IDs [B, L]
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B] (from 100% accurate encoder)
            tgt_mask: Optional causal mask [L, L]
            tgt_key_padding_mask: Optional padding mask for tokens [B, L]
            memory_key_padding_mask: Optional padding mask for sensor [B, T_s]
            return_hidden: Whether to return hidden states
            teacher_forcing_ratio: Probability of using ground truth token at each step.
                                   1.0 = full teacher forcing, 0.0 = full scheduled sampling.
            prev_logits: Optional dict with previous step logits for self-conditioning.
                        Keys: 'type_logits', 'command_logits', 'param_type_logits'

        Returns:
            Dictionary with:
            - type_logits: [B, L, n_types]
            - command_logits: [B, L, n_commands]
            - param_type_logits: [B, L, n_param_types]
            - sign_logits: [B, L, 3]
            - digit_logits: [B, L, n_positions, 10]
            - aux_value: [B, L, 1]
            - legacy_logits: [B, L, vocab_size] (optional, for comparison)
            - hidden: [B, L, d_model] (if return_hidden=True)
        """
        B, L = tokens.shape
        T_s = sensor_embeddings.size(1)
        device = tokens.device

        # ============ 1. OPERATION CONDITIONING ============
        # Get operation embedding and broadcast over sensor sequence
        if not self.no_operation_conditioning and self.operation_embed is not None:
            op_emb = self.operation_embed(operation_type)  # [B, d_model//4]
            op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)  # [B, T_s, d_model//4]
            # Concatenate sensor embeddings with operation context
            sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)
        else:
            # Ablation: No operation conditioning
            sensor_with_op = sensor_embeddings

        # ============ 2. SENSOR PROJECTION ============
        memory = self.sensor_projection(sensor_with_op)  # [B, T_s, d_model]

        # ============ 2a. WINDOW POSITION INPUT (Stage 2A) ============
        if self._use_window_position and window_index is not None:
            pos_emb = self.window_pos_embed(
                window_index.clamp(0, self._max_windows_per_file - 1)
            )  # [B, d_model//4]
            frac = (window_index.float() / total_windows.float().clamp(min=1)).unsqueeze(-1)  # [B, 1]
            frac_emb = self.window_frac_proj(frac)  # [B, d_model//8]
            win_pos = self.window_pos_proj(torch.cat([pos_emb, frac_emb], dim=-1))  # [B, d_model]
            memory = memory + win_pos.unsqueeze(1)  # broadcast over sequence length

        # ============ 2b. MEMORY POSITIONAL ENCODING ============
        if self.memory_pos_encoding_enabled and self.memory_pos_enc is not None:
            memory = self.memory_pos_enc(memory)

        # ============ 2c. SEQUENCE CLASSIFIER (Stage 2D) ============
        seq_logits = None
        if self._use_sequence_classifier:
            # Use raw sensor embeddings (before projection) for classifier input
            mem_pooled = sensor_embeddings.mean(dim=1)  # [B, sensor_dim]
            seq_logits = self.sequence_classifier(mem_pooled)  # [B, n_unique_sequences]

            # Apply operation-sequence prior mask (Stage 2E)
            if hasattr(self, 'op_seq_mask') and self.op_seq_mask is not None and op_pred is not None:
                valid_mask = self.op_seq_mask[op_pred]  # [B, n_sequences]
                seq_logits = seq_logits + (~valid_mask).float() * -1e4

            # Get class: use GT during training if provided, else argmax
            if self.training and sequence_class is not None:
                seq_class = sequence_class
            else:
                seq_class = seq_logits.argmax(-1)

            # Embed and gate
            seq_emb = self.sequence_embed(seq_class)  # [B, d_model]
            gate = torch.sigmoid(self.sequence_gate(seq_emb))
            memory = memory + (gate * seq_emb).unsqueeze(1)

        # ============ SCHEDULED SAMPLING (NEW) ============
        # If teacher_forcing_ratio < 1.0, mix ground truth with model predictions
        if self.training and teacher_forcing_ratio < 1.0:
            # Autoregressive generation with scheduled sampling
            return self._forward_with_scheduled_sampling(
                tokens=tokens,
                memory=memory,
                operation_type=operation_type,
                teacher_forcing_ratio=teacher_forcing_ratio,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_hidden=return_hidden,
            )

        # ============ 3. TOKEN EMBEDDING (full teacher forcing) ============
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)  # [B, L, d_model]
        tgt = self.embed_dropout(tgt)

        # ============ SELF-CONDITIONING ============
        # Incorporate previous step logits as additional features
        if self.use_self_conditioning and self.self_cond_proj is not None:
            if prev_logits is not None and 'type_logits' in prev_logits:
                # Shift prev_logits to align (prev[t] conditions current[t+1])
                # Concatenate type, command, param_type logits
                prev_type = prev_logits['type_logits']  # [B, L, n_types]
                prev_cmd = prev_logits['command_logits']  # [B, L, n_commands]
                prev_pt = prev_logits['param_type_logits']  # [B, L, n_param_types]

                # Concatenate and shift
                prev_concat = torch.cat([prev_type, prev_cmd, prev_pt], dim=-1)  # [B, L, self_cond_dim]

                # Shift: position t uses logits from position t-1
                # Pad with zeros for position 0
                shifted = torch.zeros_like(prev_concat)
                shifted[:, 1:, :] = prev_concat[:, :-1, :]

                # Project and combine with token embedding
                cond_features = self.self_cond_proj(shifted)  # [B, L, d_model//4]
                tgt = torch.cat([tgt, cond_features], dim=-1)  # [B, L, d_model + d_model//4]
                tgt = self.cond_combine(tgt)  # [B, L, d_model]

        # Ablation: Skip positional encoding if disabled
        if not self.no_positional_encoding:
            tgt = self.pos_encoding(tgt)

        # ============ 4. CAUSAL MASK ============
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, device)

        # ============ 5. TRANSFORMER DECODER ============
        if self.decoder_layers is not None:
            # Use custom layers with DropPath (and optional gradient checkpointing)
            hidden = tgt
            if not self.no_cross_attention:
                # Decoder layers with cross-attention
                for layer in self.decoder_layers:
                    if self.use_gradient_checkpointing and self.training:
                        hidden = checkpoint(
                            layer,
                            hidden,
                            memory,
                            tgt_mask,
                            None,  # memory_mask
                            tgt_key_padding_mask,
                            memory_key_padding_mask,
                            use_reentrant=False,
                        )
                    else:
                        hidden = layer(
                            hidden,
                            memory,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                        )
            else:
                # Encoder layers (self-attention only)
                for layer in self.decoder_layers:
                    if self.use_gradient_checkpointing and self.training:
                        hidden = checkpoint(
                            layer,
                            hidden,
                            tgt_mask,
                            tgt_key_padding_mask,
                            use_reentrant=False,
                        )
                    else:
                        hidden = layer(
                            hidden,
                            src_mask=tgt_mask,
                            src_key_padding_mask=tgt_key_padding_mask,
                        )
        elif not self.no_cross_attention:
            # Standard decoder with cross-attention to sensor memory
            hidden = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            # Ablation: Self-attention only (encoder-style)
            hidden = self.decoder(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=tgt_key_padding_mask,
            )
        hidden = self.output_norm(hidden)  # [B, L, d_model]

        # ============ 6. MULTI-HEAD PREDICTIONS ============
        type_logits = self.type_head(hidden)  # [B, L, n_types]

        if self.hierarchical:
            # ── Hierarchical conditioning: cascade type → cmd/param → digits ──
            # During training: teacher-force with ground truth type/param
            # During inference: use predicted type/param (97%+ accurate)
            type_pred = type_logits.argmax(-1)  # [B, L]
            type_emb = self.hier_type_embed(type_pred)  # [B, L, hier_embed_dim]

            # Command and param heads get type context
            hidden_with_type = torch.cat([hidden, type_emb], dim=-1)  # [B, L, d_model + hier_embed_dim]
            command_logits = self.command_head(hidden_with_type)  # [B, L, n_commands]
            param_type_logits = self.param_type_head(hidden_with_type)  # [B, L, n_param_types]

            # Digit head gets type + param context via the DigitByDigitValueHead
            # (it already takes param_type as input, plus we pass enriched hidden)
            param_type_pred = param_type_logits.argmax(-1)  # [B, L]
            param_emb = self.hier_param_embed(param_type_pred)  # [B, L, hier_embed_dim]

            # Enrich hidden for digit head with type + param embeddings (projected to d_model)
            hidden_for_digits = hidden + self.hier_type_proj(type_emb) + self.hier_param_proj(param_emb)
        else:
            command_logits = self.command_head(hidden)  # [B, L, n_commands]
            param_type_logits = self.param_type_head(hidden)  # [B, L, n_param_types]
            param_type_pred = param_type_logits.argmax(-1)  # [B, L]
            hidden_for_digits = hidden

        # ============ 7. DIGIT PREDICTIONS ============
        digit_outputs = self.digit_value_head(
            hidden=hidden_for_digits,
            operation_type=operation_type,
            param_type=param_type_pred,
        )

        # ============ 7b. APPLY SENSOR VALUE PRIOR (Phase 2) ============
        # Add sensor-derived biases to digit predictions to avoid mode collapse
        if self.use_sensor_prior and self.sensor_value_prior is not None:
            sensor_prior = self.sensor_value_prior(
                sensor_memory=sensor_embeddings,  # Raw sensor embeddings [B, T_s, sensor_dim]
                param_type=param_type_pred,  # [B, L]
            )
            # Add weighted bias to digit and sign logits
            digit_outputs['digit_logits'] = (
                digit_outputs['digit_logits'] +
                self.sensor_prior_weight * sensor_prior['digit_bias']
            )
            digit_outputs['sign_logits'] = (
                digit_outputs['sign_logits'] +
                self.sensor_prior_weight * sensor_prior['sign_bias']
            )

        # ============ 8. LEGACY LOGITS WITH TYPE + GRAMMAR CONSTRAINT ============
        raw_legacy_logits = self.legacy_token_head(hidden)  # [B, L, vocab_size]

        # Apply type constraint if mask is available and enabled
        constrained_logits = raw_legacy_logits
        if self.use_type_constraint and self.type_token_mask is not None:
            type_pred_for_mask = type_logits.argmax(-1)  # [B, L]
            type_mask = self.type_token_mask[type_pred_for_mask]  # [B, L, vocab_size]
            constrained_logits = constrained_logits + type_mask

        # Apply grammar constraint: mask based on previous token
        if self.use_grammar_constraint and self.grammar_mask is not None:
            # tokens[:, :] are the input tokens (shifted right by 1 from targets)
            # tokens[b, t] is the input at position t, constraining what can appear at position t
            prev_tokens = tokens  # [B, L] — the input tokens at each position
            grammar_constraint = self.grammar_mask[prev_tokens]  # [B, L, vocab_size]
            constrained_logits = constrained_logits + grammar_constraint

        # ============ 8b. POINTER NETWORK (V7) ============
        pointer_logits = None
        if self._use_pointer_network and self.pointer_heads is not None:
            # Compute per-axis logits for numeric token positions
            pointer_logits = {}
            for axis_name, head in self.pointer_heads.items():
                pointer_logits[axis_name] = head(hidden)  # [B, L, n_values_for_axis]

        # ============ 8c. NUMERIC REGRESSION HEAD ============
        regression_value = None
        if self._use_regression_head and self.regression_head is not None:
            # One-hot param type conditioning
            param_onehot = torch.nn.functional.one_hot(
                param_type_pred.clamp(0), self.n_param_types
            ).float()  # [B, L, n_param_types]
            reg_input = torch.cat([hidden_for_digits, param_onehot], dim=-1)
            regression_value = self.regression_head(reg_input).squeeze(-1)  # [B, L]

        # ============ 9. BUILD OUTPUT DICT ============
        outputs = {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'sign_logits': digit_outputs['sign_logits'],
            'digit_logits': digit_outputs['digit_logits'],
            'aux_value': digit_outputs['aux_value'],
            'legacy_logits': constrained_logits,  # Type + grammar constrained
            'raw_legacy_logits': raw_legacy_logits,  # Unconstrained for debugging
        }
        if regression_value is not None:
            outputs['regression_value'] = regression_value
        if seq_logits is not None:
            outputs['sequence_logits'] = seq_logits
        if pointer_logits is not None:
            outputs['pointer_logits'] = pointer_logits  # Dict[axis_name, [B, L, n_values]]

        if return_hidden:
            outputs['hidden'] = hidden

        return outputs

    def _forward_with_scheduled_sampling(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        operation_type: torch.Tensor,
        teacher_forcing_ratio: float,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with scheduled sampling to reduce exposure bias.

        At each timestep, with probability teacher_forcing_ratio, use ground truth token.
        Otherwise, use the model's own prediction from the previous timestep.

        This is more computationally expensive than full teacher forcing because
        we process one timestep at a time, but it helps the model learn to recover
        from its own mistakes, improving sequence-level accuracy.
        """
        B, L = tokens.shape
        device = tokens.device

        # Storage for all logits (need to preserve gradients)
        all_type_logits = []
        all_command_logits = []
        all_param_type_logits = []
        all_legacy_logits = []
        all_hidden = []

        # Start with just the BOS token (first column of tokens)
        current_tokens = tokens[:, :1]  # [B, 1]

        for t in range(L):
            # Embed current sequence
            tgt = self.token_embedding(current_tokens) * math.sqrt(self.d_model)
            tgt = self.embed_dropout(tgt)
            if not self.no_positional_encoding:
                tgt = self.pos_encoding(tgt)

            # Causal mask for current length
            tgt_mask = self._generate_causal_mask(current_tokens.size(1), device)

            # Decoder forward - handle both decoder_layers (DropPath) and standard decoder
            if self.decoder_layers is not None:
                hidden = tgt
                if not self.no_cross_attention:
                    for layer in self.decoder_layers:
                        hidden = layer(
                            hidden,
                            memory,
                            tgt_mask=tgt_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                        )
                else:
                    for layer in self.decoder_layers:
                        hidden = layer(hidden, src_mask=tgt_mask)
            elif not self.no_cross_attention:
                hidden = self.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            else:
                hidden = self.decoder(src=tgt, mask=tgt_mask)
            hidden = self.output_norm(hidden)

            # Get predictions for last position only
            last_hidden = hidden[:, -1:, :]  # [B, 1, d_model]

            # Multi-head predictions
            type_logits = self.type_head(last_hidden)  # [B, 1, n_types]
            command_logits = self.command_head(last_hidden)
            param_type_logits = self.param_type_head(last_hidden)
            legacy_logits = self.legacy_token_head(last_hidden)

            all_type_logits.append(type_logits)
            all_command_logits.append(command_logits)
            all_param_type_logits.append(param_type_logits)
            all_legacy_logits.append(legacy_logits)
            all_hidden.append(last_hidden)

            # Decide next token: teacher forcing vs model prediction
            if t < L - 1:  # Don't need to predict beyond sequence
                if random.random() < teacher_forcing_ratio:
                    # Use ground truth token
                    next_token = tokens[:, t+1:t+2]  # [B, 1]
                else:
                    # Sample from model's prediction (or argmax for stability)
                    next_token = legacy_logits.argmax(-1)  # [B, 1]

                # Append to current sequence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Concatenate all predictions
        type_logits = torch.cat(all_type_logits, dim=1)  # [B, L, n_types]
        command_logits = torch.cat(all_command_logits, dim=1)
        param_type_logits = torch.cat(all_param_type_logits, dim=1)
        legacy_logits = torch.cat(all_legacy_logits, dim=1)
        hidden = torch.cat(all_hidden, dim=1)  # [B, L, d_model]

        # Digit predictions (using predicted param_type)
        param_type_pred = param_type_logits.argmax(-1)
        digit_outputs = self.digit_value_head(
            hidden=hidden,
            operation_type=operation_type,
            param_type=param_type_pred,
        )

        outputs = {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'sign_logits': digit_outputs['sign_logits'],
            'digit_logits': digit_outputs['digit_logits'],
            'aux_value': digit_outputs['aux_value'],
            'legacy_logits': legacy_logits,
        }

        if return_hidden:
            outputs['hidden'] = hidden

        return outputs

    def forward_with_gt_param_type(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        param_type_targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using ground truth param_type for digit prediction.
        This provides better digit supervision during training.

        Args:
            param_type_targets: [B, L] ground truth param type indices
        """
        B, L = tokens.shape
        T_s = sensor_embeddings.size(1)
        device = tokens.device

        # Operation conditioning (with ablation support)
        if not self.no_operation_conditioning and self.operation_embed is not None:
            op_emb = self.operation_embed(operation_type)
            op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
            sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)
        else:
            sensor_with_op = sensor_embeddings

        # Sensor projection
        memory = self.sensor_projection(sensor_with_op)

        # Token embedding (with positional encoding ablation)
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)
        tgt = self.embed_dropout(tgt)
        if not self.no_positional_encoding:
            tgt = self.pos_encoding(tgt)

        # Causal mask
        tgt_mask = self._generate_causal_mask(L, device)

        # Transformer decoder - handle both decoder_layers (DropPath) and standard decoder
        if self.decoder_layers is not None:
            hidden = tgt
            if not self.no_cross_attention:
                for layer in self.decoder_layers:
                    hidden = layer(hidden, memory, tgt_mask=tgt_mask)
            else:
                for layer in self.decoder_layers:
                    hidden = layer(hidden, src_mask=tgt_mask)
        elif not self.no_cross_attention:
            hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, **kwargs)
        else:
            hidden = self.decoder(src=tgt, mask=tgt_mask)
        hidden = self.output_norm(hidden)

        # Multi-head predictions
        type_logits = self.type_head(hidden)
        command_logits = self.command_head(hidden)
        param_type_logits = self.param_type_head(hidden)

        # Digit predictions with GROUND TRUTH param_type
        digit_outputs = self.digit_value_head(
            hidden=hidden,
            operation_type=operation_type,
            param_type=param_type_targets,  # Use GT during training
        )

        # Apply sensor value prior (Phase 2)
        if self.use_sensor_prior and self.sensor_value_prior is not None:
            sensor_prior = self.sensor_value_prior(
                sensor_memory=sensor_embeddings,
                param_type=param_type_targets,  # Use GT param_type
            )
            digit_outputs['digit_logits'] = (
                digit_outputs['digit_logits'] +
                self.sensor_prior_weight * sensor_prior['digit_bias']
            )
            digit_outputs['sign_logits'] = (
                digit_outputs['sign_logits'] +
                self.sensor_prior_weight * sensor_prior['sign_bias']
            )

        # Type-constrained legacy logits
        raw_legacy_logits = self.legacy_token_head(hidden)
        if self.use_type_constraint and self.type_token_mask is not None:
            type_pred = type_logits.argmax(-1)
            type_mask = self.type_token_mask[type_pred]
            constrained_logits = raw_legacy_logits + type_mask
        else:
            constrained_logits = raw_legacy_logits

        return {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'sign_logits': digit_outputs['sign_logits'],
            'digit_logits': digit_outputs['digit_logits'],
            'aux_value': digit_outputs['aux_value'],
            'legacy_logits': constrained_logits,
            'raw_legacy_logits': raw_legacy_logits,
        }

    @torch.no_grad()
    def generate(
        self,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Autoregressive generation conditioned on sensor embeddings and operation type.

        Args:
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B]
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling

        Returns:
            generated_tokens: [B, L] token IDs
            all_outputs: Dictionary with per-position predictions
        """
        self.eval()
        B = sensor_embeddings.size(0)
        T_s = sensor_embeddings.size(1)
        device = sensor_embeddings.device

        # Operation conditioning
        op_emb = self.operation_embed(operation_type)
        op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
        sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)
        memory = self.sensor_projection(sensor_with_op)

        # Initialize with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        # Storage for predictions
        all_types = []
        all_commands = []
        all_param_types = []

        for step in range(max_length - 1):
            # Get embeddings for current sequence
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)

            # Causal mask
            tgt_mask = self._generate_causal_mask(generated.size(1), device)

            # Decode - handle both decoder_layers (DropPath) and standard decoder cases
            if self.decoder_layers is not None:
                # Use custom layers with DropPath
                hidden = tgt
                if not self.no_cross_attention:
                    for layer in self.decoder_layers:
                        hidden = layer(
                            hidden,
                            memory,
                            tgt_mask=tgt_mask,
                        )
                else:
                    for layer in self.decoder_layers:
                        hidden = layer(
                            hidden,
                            src_mask=tgt_mask,
                        )
            elif not self.no_cross_attention:
                hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            else:
                hidden = self.decoder(src=tgt, mask=tgt_mask)
            hidden = self.output_norm(hidden)

            # Get predictions for last position
            last_hidden = hidden[:, -1, :]  # [B, d_model]

            # Type prediction
            type_logits = self.type_head(last_hidden.unsqueeze(1)).squeeze(1)  # [B, n_types]
            type_pred = type_logits.argmax(-1)  # [B]
            all_types.append(type_pred)

            # Command prediction
            command_logits = self.command_head(last_hidden.unsqueeze(1)).squeeze(1)
            command_pred = command_logits.argmax(-1)
            all_commands.append(command_pred)

            # Param type prediction
            param_type_logits = self.param_type_head(last_hidden.unsqueeze(1)).squeeze(1)
            param_type_pred = param_type_logits.argmax(-1)
            all_param_types.append(param_type_pred)

            # Legacy token prediction (for actual token generation)
            token_logits = self.legacy_token_head(last_hidden)  # [B, vocab_size]

            # Apply type constraint during generation
            if self.use_type_constraint and self.type_token_mask is not None:
                type_mask = self.type_token_mask[type_pred]  # [B, vocab_size]
                token_logits = token_logits + type_mask

            # Apply temperature
            if temperature != 1.0:
                token_logits = token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(token_logits, min(top_k, token_logits.size(-1)))
                token_logits[token_logits < v[:, [-1]]] = float('-inf')

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        all_outputs = {
            'type_preds': torch.stack(all_types, dim=1) if all_types else None,
            'command_preds': torch.stack(all_commands, dim=1) if all_commands else None,
            'param_type_preds': torch.stack(all_param_types, dim=1) if all_param_types else None,
        }

        return generated, all_outputs

    def generate_with_gradients(
        self,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 32,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Autoregressive generation WITH gradients preserved for SCST/REINFORCE training.

        Unlike generate(), this method:
        - Does NOT use @torch.no_grad()
        - Returns log probabilities of sampled tokens
        - Supports backpropagation through the sampling process

        Args:
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B]
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature

        Returns:
            generated_tokens: [B, L] sampled token IDs
            log_probs: [B, L-1] log probabilities of each sampled token
            all_outputs: Dictionary with per-position predictions
        """
        B = sensor_embeddings.size(0)
        T_s = sensor_embeddings.size(1)
        device = sensor_embeddings.device

        # Operation conditioning (with gradients)
        if not self.no_operation_conditioning and self.operation_embed is not None:
            op_emb = self.operation_embed(operation_type)
            op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
            sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)
        else:
            sensor_with_op = sensor_embeddings

        memory = self.sensor_projection(sensor_with_op)

        # Initialize with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        # Storage
        all_log_probs = []
        all_types = []
        all_commands = []
        all_param_types = []

        for step in range(max_length - 1):
            # Get embeddings for current sequence
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = self.embed_dropout(tgt)
            if not self.no_positional_encoding:
                tgt = self.pos_encoding(tgt)

            # Causal mask
            tgt_mask = self._generate_causal_mask(generated.size(1), device)

            # Decode (with gradients!) - handle both decoder_layers and standard decoder
            if self.decoder_layers is not None:
                hidden = tgt
                if not self.no_cross_attention:
                    for layer in self.decoder_layers:
                        hidden = layer(hidden, memory, tgt_mask=tgt_mask)
                else:
                    for layer in self.decoder_layers:
                        hidden = layer(hidden, src_mask=tgt_mask)
            elif not self.no_cross_attention:
                hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            else:
                hidden = self.decoder(src=tgt, mask=tgt_mask)
            hidden = self.output_norm(hidden)

            # Get predictions for last position
            last_hidden = hidden[:, -1, :]  # [B, d_model]

            # Type prediction
            type_logits = self.type_head(last_hidden.unsqueeze(1)).squeeze(1)
            type_pred = type_logits.argmax(-1)
            all_types.append(type_pred)

            # Command prediction
            command_logits = self.command_head(last_hidden.unsqueeze(1)).squeeze(1)
            command_pred = command_logits.argmax(-1)
            all_commands.append(command_pred)

            # Param type prediction
            param_type_logits = self.param_type_head(last_hidden.unsqueeze(1)).squeeze(1)
            param_type_pred = param_type_logits.argmax(-1)
            all_param_types.append(param_type_pred)

            # Legacy token prediction (for actual token generation)
            token_logits = self.legacy_token_head(last_hidden)  # [B, vocab_size]

            # Apply type constraint during generation with gradients
            if self.use_type_constraint and self.type_token_mask is not None:
                type_mask = self.type_token_mask[type_pred]  # [B, vocab_size]
                token_logits = token_logits + type_mask

            # Apply temperature
            if temperature != 1.0:
                token_logits = token_logits / temperature

            # Compute probabilities
            log_probs = F.log_softmax(token_logits, dim=-1)  # [B, vocab_size]

            # Sample next token (with gradient via Gumbel-Softmax or straight-through)
            # For REINFORCE, we use regular sampling and score the log_prob
            probs = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Get log prob of sampled token
            sampled_log_prob = log_probs.gather(1, next_token).squeeze(1)  # [B]
            all_log_probs.append(sampled_log_prob)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        # Stack log probs
        log_probs_tensor = torch.stack(all_log_probs, dim=1)  # [B, L-1]

        all_outputs = {
            'type_preds': torch.stack(all_types, dim=1) if all_types else None,
            'command_preds': torch.stack(all_commands, dim=1) if all_commands else None,
            'param_type_preds': torch.stack(all_param_types, dim=1) if all_param_types else None,
        }

        return generated, log_probs_tensor, all_outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        counts = {
            'operation_embed': sum(p.numel() for p in self.operation_embed.parameters()),
            'sensor_projection': sum(p.numel() for p in self.sensor_projection.parameters()),
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'pos_encoding': sum(p.numel() for p in self.pos_encoding.parameters() if p.requires_grad),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'type_head': sum(p.numel() for p in self.type_head.parameters()),
            'command_head': sum(p.numel() for p in self.command_head.parameters()),
            'param_type_head': sum(p.numel() for p in self.param_type_head.parameters()),
            'digit_value_head': sum(p.numel() for p in self.digit_value_head.parameters()),
            'legacy_token_head': sum(p.numel() for p in self.legacy_token_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


# ============================================================================
# Complete Pipeline with Frozen Encoder
# ============================================================================

class SensorMultiHeadGenerator(nn.Module):
    """
    Complete sensor-to-token generation pipeline.

    Combines:
    1. Frozen MM-DTAE_LSTM sensor encoder (100% classification accuracy)
    2. SensorMultiHeadDecoder (to be trained)

    This model generates G-code tokens conditioned on sensor readings,
    using the encoder's operation classification to guide the decoder.
    """

    def __init__(
        self,
        sensor_encoder: nn.Module,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        n_operations: int = 9,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        """
        Args:
            sensor_encoder: Pre-trained MM-DTAE_LSTM model
            vocab_size: Token vocabulary size
            d_model: Decoder hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            n_operations: Number of operation types
            dropout: Dropout rate
            freeze_encoder: Whether to freeze the sensor encoder
        """
        super().__init__()

        self.sensor_encoder = sensor_encoder
        self.freeze_encoder = freeze_encoder

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.sensor_encoder.parameters():
                param.requires_grad = False
            self.sensor_encoder.eval()

        # Get sensor latent dimension from encoder
        sensor_dim = sensor_encoder.latent_dim

        # Multi-head decoder with operation conditioning
        self.decoder = SensorMultiHeadDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            sensor_dim=sensor_dim,
            n_operations=n_operations,
            dropout=dropout,
        )

    def encode_sensors(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sensor data and get operation classification.

        Args:
            sensor_data: Raw sensor readings [B, T_s, 155]

        Returns:
            sensor_embeddings: [B, T_s, latent_dim]
            operation_type: [B] predicted operation indices
        """
        if self.freeze_encoder:
            self.sensor_encoder.eval()

        with torch.set_grad_enabled(not self.freeze_encoder):
            # Get sensor latent representation
            latent, _ = self.sensor_encoder.encode(sensor_data)

            # Get operation classification (100% accurate)
            # Apply temporal attention for pooling
            attn_weights = self.sensor_encoder.temporal_attention(latent)  # [B, T, 1]
            pooled = (latent * attn_weights).sum(dim=1)  # [B, latent_dim]

            # Classify operation type
            op_logits = self.sensor_encoder.classification_head(pooled)  # [B, n_classes]
            operation_type = op_logits.argmax(-1)  # [B]

        return latent, operation_type

    def forward(
        self,
        sensor_data: torch.Tensor,
        tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        use_gt_param_type: bool = False,
        param_type_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            sensor_data: Raw sensor readings [B, T_s, 155]
            tokens: Input token IDs [B, L]
            tgt_key_padding_mask: Optional padding mask for tokens
            use_gt_param_type: Use ground truth param_type for digit prediction
            param_type_targets: [B, L] ground truth param types (if use_gt_param_type=True)

        Returns:
            Dictionary with all prediction logits
        """
        # Encode sensors and get operation type
        sensor_embeddings, operation_type = self.encode_sensors(sensor_data)

        # Decode tokens with multi-head outputs
        if use_gt_param_type and param_type_targets is not None:
            return self.decoder.forward_with_gt_param_type(
                tokens=tokens,
                sensor_embeddings=sensor_embeddings,
                operation_type=operation_type,
                param_type_targets=param_type_targets,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        else:
            return self.decoder(
                tokens=tokens,
                sensor_embeddings=sensor_embeddings,
                operation_type=operation_type,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

    @torch.no_grad()
    def generate(
        self,
        sensor_data: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate token sequences conditioned on sensor data."""
        self.eval()

        # Encode sensors and get operation type
        sensor_embeddings, operation_type = self.encode_sensors(sensor_data)

        # Generate tokens
        return self.decoder.generate(
            sensor_embeddings=sensor_embeddings,
            operation_type=operation_type,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    print("Testing SensorMultiHeadDecoder...")

    B, T_s, T_tok = 2, 50, 16
    sensor_dim = 128
    d_model = 192
    vocab_size = 668
    n_operations = 9

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        sensor_dim=sensor_dim,
        n_operations=n_operations,
        dropout=0.3,
    )

    # Random inputs
    tokens = torch.randint(0, vocab_size, (B, T_tok))
    sensor_emb = torch.randn(B, T_s, sensor_dim)
    operation_type = torch.randint(0, n_operations, (B,))

    # Forward pass
    outputs = decoder(tokens, sensor_emb, operation_type)

    print(f"  type_logits: {outputs['type_logits'].shape}")  # [B, T, 4]
    print(f"  command_logits: {outputs['command_logits'].shape}")  # [B, T, 6]
    print(f"  param_type_logits: {outputs['param_type_logits'].shape}")  # [B, T, 10]
    print(f"  sign_logits: {outputs['sign_logits'].shape}")  # [B, T, 3]
    print(f"  digit_logits: {outputs['digit_logits'].shape}")  # [B, T, 6, 10]
    print(f"  aux_value: {outputs['aux_value'].shape}")  # [B, T, 1]
    print(f"  legacy_logits: {outputs['legacy_logits'].shape}")  # [B, T, vocab_size]

    # Count parameters
    param_counts = decoder.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Test generation
    print("\nTesting generation...")
    generated, gen_outputs = decoder.generate(
        sensor_emb, operation_type,
        bos_token_id=1, eos_token_id=2, max_length=10
    )
    print(f"  Generated shape: {generated.shape}")

    print("\nAll tests passed!")
