"""
Flat Vocabulary Decoder Baseline for G-code Token Generation.

This is a baseline model that uses a single flat vocabulary head (668 classes)
instead of multi-head decomposition. Used to demonstrate the value of
hierarchical token decomposition.

Comparison:
- Full model: Transformer + Multi-head outputs = 90.23%
- This baseline: Transformer + Flat vocab (668 classes) = ~70% (expected)

Author: Claude Code
Date: December 2025
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FlatVocabDecoder(nn.Module):
    """
    Flat vocabulary decoder baseline.

    Key difference from SensorMultiHeadDecoder:
    - Uses single linear head → 668 vocab classes
    - No hierarchical decomposition (type, command, param_type, digits)
    - Same Transformer architecture otherwise

    This demonstrates the value of multi-head decomposition.
    """

    def __init__(
        self,
        vocab_size: int = 668,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        sensor_dim: int = 128,
        n_operations: int = 9,
        d_ff: int = None,
        dropout: float = 0.3,
        embed_dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_operations = n_operations

        if d_ff is None:
            d_ff = 4 * d_model

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
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder (same as full model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # ============ FLAT VOCABULARY HEAD (KEY DIFFERENCE) ============
        # Single head predicting all 668 token classes directly
        self.vocab_head = nn.Linear(d_model, vocab_size)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).

        Args:
            tokens: Input token IDs [B, L]
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            operation_type: Operation type indices [B]

        Returns:
            Dictionary with:
            - logits: [B, L, vocab_size] - flat vocabulary predictions
            - legacy_logits: same as logits (for compatibility)
        """
        B, L = tokens.shape
        T_s = sensor_embeddings.size(1)
        device = tokens.device

        # 1. Operation conditioning
        op_emb = self.operation_embed(operation_type)
        op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
        sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)

        # 2. Sensor projection
        memory = self.sensor_projection(sensor_with_op)

        # 3. Token embedding
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)
        tgt = self.embed_dropout(tgt)
        tgt = self.pos_encoding(tgt)

        # 4. Causal mask
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, device)

        # 5. Transformer decoder
        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.output_norm(hidden)

        # 6. FLAT VOCABULARY PREDICTION (KEY DIFFERENCE)
        logits = self.vocab_head(hidden)  # [B, L, vocab_size]

        # Return in compatible format
        # Note: No multi-head outputs - just flat vocab prediction
        return {
            'logits': logits,
            'legacy_logits': logits,  # For compatibility with eval scripts
        }

    def forward_with_gt_param_type(self, *args, **kwargs):
        """Compatibility wrapper - flat vocab doesn't use param_type."""
        return self.forward(*args, **kwargs)

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'operation_embed': sum(p.numel() for p in self.operation_embed.parameters()),
            'sensor_projection': sum(p.numel() for p in self.sensor_projection.parameters()),
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'pos_encoding': sum(p.numel() for p in self.pos_encoding.parameters() if p.requires_grad),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'vocab_head': sum(p.numel() for p in self.vocab_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts

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
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        B = sensor_embeddings.size(0)
        T_s = sensor_embeddings.size(1)
        device = sensor_embeddings.device

        # Operation conditioning
        op_emb = self.operation_embed(operation_type)
        op_emb_broadcast = op_emb.unsqueeze(1).expand(-1, T_s, -1)
        sensor_with_op = torch.cat([sensor_embeddings, op_emb_broadcast], dim=-1)
        memory = self.sensor_projection(sensor_with_op)

        # Initialize with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)
            tgt_mask = self._generate_causal_mask(generated.size(1), device)

            hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            hidden = self.output_norm(hidden)

            logits = self.vocab_head(hidden[:, -1, :])  # [B, vocab_size]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated


if __name__ == '__main__':
    print("Testing FlatVocabDecoder baseline...")

    B, T_s, T_tok = 2, 50, 16
    sensor_dim = 128
    d_model = 192
    vocab_size = 668
    n_operations = 9

    decoder = FlatVocabDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        sensor_dim=sensor_dim,
        n_operations=n_operations,
    )

    tokens = torch.randint(0, vocab_size, (B, T_tok))
    sensor_emb = torch.randn(B, T_s, sensor_dim)
    operation_type = torch.randint(0, n_operations, (B,))

    outputs = decoder(tokens, sensor_emb, operation_type)

    print(f"  logits: {outputs['logits'].shape}")  # [B, L, 668]

    param_counts = decoder.count_parameters()
    print(f"\nTotal parameters: {param_counts['total']:,}")

    # Test generation
    generated = decoder.generate(
        sensor_emb, operation_type,
        bos_token_id=1, eos_token_id=2, max_length=10
    )
    print(f"  Generated shape: {generated.shape}")

    print("\nFlat Vocab Decoder baseline test passed!")
