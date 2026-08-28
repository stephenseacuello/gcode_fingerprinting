"""Seq2seq baseline for the G-code decoder paper.

Addresses the R1 reviewer concern that the headline architecture
(frozen MM-DTAE encoder + grammar-constrained decoder + six-head
structured output) is a stack of off-the-shelf components and the
paper does not show how a vanilla encoder-decoder Transformer
trained end-to-end on the same data would perform.

This baseline strips every component the main paper uses for
specialization:
  - the encoder is trained end-to-end, not loaded from a frozen
    9-class operation classifier
  - the decoder is a single token-prediction head over the V8
    vocab; there is no command/axis/value structured factorization
  - there is no grammar mask / FSM at decode time
  - there is no sensor prior, no scheduled sampling, no
    auxiliary losses

Only the data layout, vocabulary, fold splits and evaluation
metrics are shared with the main paper, which makes the
comparison apples-to-apples for the architectural-novelty claim.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2SeqBaseline(nn.Module):
    """End-to-end Transformer encoder-decoder over (sensor window -> G-code tokens)."""

    def __init__(
        self,
        sensor_channels: int,
        window_size: int,
        vocab_size: int,
        d_model: int = 256,
        n_layers_enc: int = 4,
        n_layers_dec: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        max_target_len: int = 1400,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_target_len = max_target_len

        # Input projection + positional encoding over the 256-timestep window
        self.sensor_proj = nn.Linear(sensor_channels, d_model)
        self.enc_pos = SinusoidalPositionalEncoding(d_model, max_len=window_size + 4)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # Decoder token embedding + positional + Transformer decoder
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dec_pos = SinusoidalPositionalEncoding(d_model, max_len=max_target_len + 4)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)

        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, sensor: torch.Tensor) -> torch.Tensor:
        """sensor: (B, T_in, C) -> memory: (B, T_in, d_model)"""
        x = self.sensor_proj(sensor)
        x = self.enc_pos(x)
        return self.encoder(x)

    def forward(
        self,
        sensor: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward.

        sensor: (B, T_in, C)
        target_input: (B, T_tgt) -- decoder input tokens (typically [BOS, t1, t2, ...])
        Returns logits: (B, T_tgt, V)
        """
        memory = self.encode(sensor)

        y = self.token_emb(target_input) * math.sqrt(self.d_model)
        y = self.dec_pos(y)

        T = target_input.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=target_input.device, dtype=torch.bool),
            diagonal=1,
        )

        target_padding_mask = target_input.eq(self.pad_id)

        dec_out = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        return self.out_proj(dec_out)

    @torch.no_grad()
    def generate(
        self,
        sensor: torch.Tensor,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive greedy decoding. Returns token ids (B, T)."""
        max_len = max_len or self.max_target_len
        B = sensor.size(0)
        device = sensor.device

        memory = self.encode(sensor)

        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            y = self.token_emb(ys) * math.sqrt(self.d_model)
            y = self.dec_pos(y)
            T = ys.size(1)
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            dec_out = self.decoder(tgt=y, memory=memory, tgt_mask=causal_mask)
            logits = self.out_proj(dec_out[:, -1])
            next_tok = logits.argmax(-1)
            next_tok = torch.where(
                finished, torch.tensor(self.pad_id, device=device), next_tok
            )
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(self.eos_id)
            if finished.all():
                break

        return ys

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
