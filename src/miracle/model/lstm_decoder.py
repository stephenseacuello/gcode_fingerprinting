"""
LSTM Decoder Baseline for G-code Token Generation.

This is a baseline model that replaces the Transformer decoder with LSTM,
while keeping the same multi-head output structure. Used to demonstrate
the value of Transformer architecture.

Comparison:
- Full model: Transformer decoder + Multi-head outputs = 90.23%
- This baseline: LSTM decoder + Multi-head outputs = ~78% (expected)

Author: Claude Code
Date: December 2025
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .digit_value_head import DigitByDigitValueHead


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder baseline with multi-head outputs.

    Key difference from SensorMultiHeadDecoder:
    - Uses LSTM instead of Transformer for sequence modeling
    - Still has cross-attention to sensor memory
    - Same multi-head output structure (type, command, param_type, digits)
    """

    def __init__(
        self,
        vocab_size: int = 668,
        d_model: int = 192,
        n_layers: int = 2,
        sensor_dim: int = 128,
        n_operations: int = 9,
        n_types: int = 4,
        n_commands: int = 6,
        n_param_types: int = 10,
        max_int_digits: int = 2,
        n_decimal_digits: int = 4,
        dropout: float = 0.3,
        embed_dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_operations = n_operations
        self.n_types = n_types
        self.n_commands = n_commands
        self.n_param_types = n_param_types

        # Operation embedding for conditioning
        self.operation_embed = nn.Embedding(n_operations, d_model // 4)
        sensor_proj_input_dim = sensor_dim + d_model // 4

        # Sensor projection
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_proj_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # ============ LSTM DECODER (KEY DIFFERENCE) ============
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,  # Causal - unidirectional
        )

        # Cross-attention to sensor memory
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # ============ MULTI-HEAD OUTPUTS (same as Transformer version) ============
        self.type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_types),
        )

        self.command_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_commands),
        )

        self.param_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_param_types),
        )

        # Digit value head
        self.digit_value_head = DigitByDigitValueHead(
            d_model=d_model,
            n_operations=n_operations,
            n_param_types=n_param_types,
            max_int_digits=max_int_digits,
            n_decimal_digits=n_decimal_digits,
            dropout=dropout,
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Legacy token head for comparison
        self.legacy_token_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).

        Args:
            tokens: Input token IDs [B, L]
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B]

        Returns:
            Dictionary with multi-head predictions
        """
        B, L = tokens.shape
        T_s = sensor_embeddings.size(1)

        # 1. Operation conditioning
        op_emb = self.operation_embed(operation_type)  # [B, d_model//4]
        op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
        sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)

        # 2. Sensor projection (memory for cross-attention)
        memory = self.sensor_projection(sensor_with_op)  # [B, T_s, d_model]

        # 3. Token embedding
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)
        tgt = self.embed_dropout(tgt)

        # 4. LSTM processing (KEY DIFFERENCE from Transformer)
        lstm_out, _ = self.lstm(tgt)  # [B, L, d_model]

        # 5. Cross-attention to sensor memory
        attn_out, _ = self.cross_attention(
            query=lstm_out,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.cross_attn_norm(lstm_out + attn_out)
        hidden = self.output_norm(hidden)  # [B, L, d_model]

        # 6. Multi-head predictions
        type_logits = self.type_head(hidden)
        command_logits = self.command_head(hidden)
        param_type_logits = self.param_type_head(hidden)

        # 7. Digit predictions
        param_type_pred = param_type_logits.argmax(-1)
        digit_outputs = self.digit_value_head(
            hidden=hidden,
            operation_type=operation_type,
            param_type=param_type_pred,
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

    def forward_with_gt_param_type(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        param_type_targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using ground truth param_type for digit prediction."""
        B, L = tokens.shape
        T_s = sensor_embeddings.size(1)

        # Operation conditioning
        op_emb = self.operation_embed(operation_type)
        op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
        sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)

        # Sensor projection
        memory = self.sensor_projection(sensor_with_op)

        # Token embedding
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)
        tgt = self.embed_dropout(tgt)

        # LSTM processing
        lstm_out, _ = self.lstm(tgt)

        # Cross-attention
        attn_out, _ = self.cross_attention(query=lstm_out, key=memory, value=memory)
        hidden = self.cross_attn_norm(lstm_out + attn_out)
        hidden = self.output_norm(hidden)

        # Multi-head predictions
        type_logits = self.type_head(hidden)
        command_logits = self.command_head(hidden)
        param_type_logits = self.param_type_head(hidden)

        # Digit predictions with GT param_type
        digit_outputs = self.digit_value_head(
            hidden=hidden,
            operation_type=operation_type,
            param_type=param_type_targets,
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

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'operation_embed': sum(p.numel() for p in self.operation_embed.parameters()),
            'sensor_projection': sum(p.numel() for p in self.sensor_projection.parameters()),
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'lstm': sum(p.numel() for p in self.lstm.parameters()),
            'cross_attention': sum(p.numel() for p in self.cross_attention.parameters()),
            'type_head': sum(p.numel() for p in self.type_head.parameters()),
            'command_head': sum(p.numel() for p in self.command_head.parameters()),
            'param_type_head': sum(p.numel() for p in self.param_type_head.parameters()),
            'digit_value_head': sum(p.numel() for p in self.digit_value_head.parameters()),
            'legacy_token_head': sum(p.numel() for p in self.legacy_token_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


if __name__ == '__main__':
    print("Testing LSTMDecoder baseline...")

    B, T_s, T_tok = 2, 50, 16
    sensor_dim = 128
    d_model = 192
    vocab_size = 668
    n_operations = 9

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        sensor_dim=sensor_dim,
        n_operations=n_operations,
    )

    tokens = torch.randint(0, vocab_size, (B, T_tok))
    sensor_emb = torch.randn(B, T_s, sensor_dim)
    operation_type = torch.randint(0, n_operations, (B,))

    outputs = decoder(tokens, sensor_emb, operation_type)

    print(f"  type_logits: {outputs['type_logits'].shape}")
    print(f"  command_logits: {outputs['command_logits'].shape}")
    print(f"  legacy_logits: {outputs['legacy_logits'].shape}")

    param_counts = decoder.count_parameters()
    print(f"\nTotal parameters: {param_counts['total']:,}")

    print("\nLSTM Decoder baseline test passed!")
