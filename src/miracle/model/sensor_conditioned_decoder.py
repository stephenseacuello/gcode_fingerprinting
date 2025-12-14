"""
Sensor-Conditioned Token Decoder for G-code Generation.

This module implements an encoder-decoder architecture where:
1. Sensor data is encoded by a frozen MM-DTAE_LSTM encoder (100% classification accuracy)
2. A transformer decoder generates G-code tokens conditioned on sensor embeddings via cross-attention

Architecture:
    Sensor [B, T_s, 155] → Frozen MM-DTAE_LSTM Encoder → Latent [B, T_s, 128]
                                                              ↓
    Token [B, L] → Token Decoder ← Cross-Attention ← Sensor Context
                        ↓
                  Next Token Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
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
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SensorConditionedTokenDecoder(nn.Module):
    """
    Decoder that generates G-code tokens conditioned on sensor embeddings.
    Uses cross-attention to attend to sensor context at each token position.

    Key Design Decisions:
    - Uses frozen sensor encoder (MM-DTAE_LSTM) for sensor feature extraction
    - Transformer decoder with cross-attention to sensor memory
    - Autoregressive generation with causal masking
    - Label smoothing for robust training
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        sensor_dim: int = 128,  # MM-DTAE_LSTM latent dim
        d_ff: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        embed_dropout: float = 0.0
    ):
        """
        Args:
            vocab_size: Size of token vocabulary (668 for 4-digit hybrid)
            d_model: Decoder hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            sensor_dim: Dimension of sensor encoder output (128 for MM-DTAE_LSTM)
            d_ff: Feedforward dimension (default: 4 * d_model)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        if d_ff is None:
            d_ff = 4 * d_model

        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Project sensor embeddings to decoder dimension
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Transformer decoder layers with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Scale token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            tokens: Input token IDs [B, L]
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            tgt_mask: Optional causal mask [L, L]
            tgt_key_padding_mask: Optional padding mask for tokens [B, L]
            memory_key_padding_mask: Optional padding mask for sensor [B, T_s]

        Returns:
            Token logits [B, L, vocab_size]
        """
        B, L = tokens.shape
        device = tokens.device

        # Token embeddings with positional encoding
        tgt = self.token_embedding(tokens) * math.sqrt(self.d_model)  # [B, L, d_model]
        tgt = self.embed_dropout(tgt)
        tgt = self.pos_encoding(tgt)

        # Project sensor embeddings to decoder dimension
        memory = self.sensor_projection(sensor_embeddings)  # [B, T_s, d_model]

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, device)

        # Decode with cross-attention to sensor memory
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Output projection
        output = self.output_norm(output)
        logits = self.output_head(output)

        return logits

    @torch.no_grad()
    def generate(
        self,
        sensor_embeddings: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation conditioned on sensor embeddings.

        Args:
            sensor_embeddings: Sensor encoder output [B, T_s, sensor_dim]
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling

        Returns:
            generated_tokens: [B, L] token IDs
            token_probs: [B, L] probability of each generated token
        """
        self.eval()
        B = sensor_embeddings.size(0)
        device = sensor_embeddings.device

        # Project sensor embeddings
        memory = self.sensor_projection(sensor_embeddings)

        # Initialize with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        all_probs = []

        for _ in range(max_length - 1):
            # Get embeddings for current sequence
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)

            # Causal mask
            tgt_mask = self._generate_causal_mask(generated.size(1), device)

            # Decode
            output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            output = self.output_norm(output)

            # Get logits for last position
            logits = self.output_head(output[:, -1, :])  # [B, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            token_prob = probs.gather(1, next_token)  # [B, 1]

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            all_probs.append(token_prob)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        token_probs = torch.cat(all_probs, dim=1) if all_probs else torch.zeros(B, 0, device=device)

        return generated, token_probs


class SensorConditionedGenerator(nn.Module):
    """
    Complete sensor-to-token generation pipeline.

    Combines:
    1. Frozen MM-DTAE_LSTM sensor encoder (achieves 100% classification)
    2. SensorConditionedTokenDecoder (to be trained)

    This model generates G-code tokens conditioned on sensor readings,
    leveraging the proven sensor encoder's representations.
    """

    def __init__(
        self,
        sensor_encoder: nn.Module,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = True
    ):
        """
        Args:
            sensor_encoder: Pre-trained MM-DTAE_LSTM model
            vocab_size: Token vocabulary size
            d_model: Decoder hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
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

        # Token decoder with cross-attention to sensor
        self.token_decoder = SensorConditionedTokenDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            sensor_dim=sensor_dim,
            dropout=dropout
        )

    def encode_sensors(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Encode sensor data using frozen MM-DTAE_LSTM encoder.

        Args:
            sensor_data: Raw sensor readings [B, T_s, 155]

        Returns:
            Sensor latent embeddings [B, T_s, latent_dim]
        """
        if self.freeze_encoder:
            self.sensor_encoder.eval()

        with torch.set_grad_enabled(not self.freeze_encoder):
            latent, _ = self.sensor_encoder.encode(sensor_data)

        return latent

    def forward(
        self,
        sensor_data: torch.Tensor,
        tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            sensor_data: Raw sensor readings [B, T_s, 155]
            tokens: Input token IDs [B, L]
            tgt_key_padding_mask: Optional padding mask for tokens

        Returns:
            Token logits [B, L, vocab_size]
        """
        # Encode sensors
        sensor_embeddings = self.encode_sensors(sensor_data)

        # Decode tokens with cross-attention to sensors
        logits = self.token_decoder(
            tokens=tokens,
            sensor_embeddings=sensor_embeddings,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        sensor_data: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token sequences conditioned on sensor data.

        Args:
            sensor_data: Raw sensor readings [B, T_s, 155]
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling

        Returns:
            generated_tokens: [B, L] token IDs
            token_probs: [B, L] probability of each token
        """
        self.eval()

        # Encode sensors
        sensor_embeddings = self.encode_sensors(sensor_data)

        # Generate tokens
        return self.token_decoder.generate(
            sensor_embeddings=sensor_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
