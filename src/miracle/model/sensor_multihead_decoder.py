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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .digit_value_head import DigitByDigitValueHead


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

        # ============ TOKEN EMBEDDING ============
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # ============ TRANSFORMER DECODER ============
        if not no_cross_attention:
            # Standard decoder with cross-attention to sensor memory
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for better stability
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
            self.use_encoder_only = False
        else:
            # Ablation: Self-attention only (no sensor cross-attention)
            # Use TransformerEncoder layers (self-attention only)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.use_encoder_only = True

        # ============ MULTI-HEAD OUTPUTS ============
        # Type head: SPECIAL, COMMAND, PARAM_LETTER, NUMERIC
        self.type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_types),
        )

        # Command head: G0, G1, G2, G3, G53, OTHER
        self.command_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_commands),
        )

        # Parameter type head: X, Y, Z, F, R, S, I, J, K, OTHER
        self.param_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
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

        # ============ OUTPUT NORMALIZATION ============
        self.output_norm = nn.LayerNorm(d_model)

        # ============ OPTIONAL: Legacy token head for comparison ============
        self.legacy_token_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).

        Args:
            tokens: Input token IDs [B, L]
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B] (from 100% accurate encoder)
            tgt_mask: Optional causal mask [L, L]
            tgt_key_padding_mask: Optional padding mask for tokens [B, L]
            memory_key_padding_mask: Optional padding mask for sensor [B, T_s]
            return_hidden: Whether to return hidden states

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

        # ============ 3. TOKEN EMBEDDING ============
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)  # [B, L, d_model]
        tgt = self.embed_dropout(tgt)

        # Ablation: Skip positional encoding if disabled
        if not self.no_positional_encoding:
            tgt = self.pos_encoding(tgt)

        # ============ 4. CAUSAL MASK ============
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, device)

        # ============ 5. TRANSFORMER DECODER ============
        if not self.no_cross_attention:
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
        command_logits = self.command_head(hidden)  # [B, L, n_commands]
        param_type_logits = self.param_type_head(hidden)  # [B, L, n_param_types]

        # ============ 7. DIGIT PREDICTIONS ============
        # Get predicted param_type for digit conditioning
        # During training, we could use ground truth param_type (teacher forcing)
        # During inference, we use predicted param_type
        param_type_pred = param_type_logits.argmax(-1)  # [B, L]

        digit_outputs = self.digit_value_head(
            hidden=hidden,
            operation_type=operation_type,
            param_type=param_type_pred,
        )

        # ============ 8. BUILD OUTPUT DICT ============
        outputs = {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'sign_logits': digit_outputs['sign_logits'],
            'digit_logits': digit_outputs['digit_logits'],
            'aux_value': digit_outputs['aux_value'],
            'legacy_logits': self.legacy_token_head(hidden),  # For comparison/ablation
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

        # Transformer decoder (with cross-attention ablation)
        if not self.no_cross_attention:
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

        return {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'sign_logits': digit_outputs['sign_logits'],
            'digit_logits': digit_outputs['digit_logits'],
            'aux_value': digit_outputs['aux_value'],
            'legacy_logits': self.legacy_token_head(hidden),
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

            # Decode
            hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
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
