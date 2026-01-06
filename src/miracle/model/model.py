# ================================
# file: model.py
# ================================
"""
Multimodal DTAE + LSTM + G-code LM (+ embeddings + optional fingerprinting)

Components
- SinePositionalEncoding: sinusoidal positional encoding
- DTAE: denoising Transformer autoencoder for self-supervised robustness
- LinearModalityEncoder: per-modality MLP + LayerNorm + GELU + positional encoding
- CrossModalFusion: learned modality gates + cross-attention (+ modality dropout)
- ContextEmbeddings: sum of small categorical embeddings (TinyG state, tool, etc.)
- GCodeLMHead: causal Transformer decoder (teacher forcing + greedy generate) with weight tying
- FingerprintHead: pooled + projected, L2-normalized embedding for "G-code fingerprinting"
- MM_DTAE_LSTM: the full model; returns all heads and useful intermediates
- AdaptiveLossWeights: uncertainty-based multi-task weighting

Inputs
- mods: list of tensors [B, T, C_m] per modality
- lengths: [B] true lengths for variable-length sequences (builds PAD masks)
- gcode_in: [B, Tg] teacher-forcing tokens (optional)
- ctx_ids: dict of categorical context ids (optional) – values are [B] or [B, T]

Outputs (dict)
- recon: [B, T, D]          (DTAE reconstruction of fused latent)
- cls:   [B, 5]             (example number of classes)
- reg:   [B, 3]
- anom:  [B, 1]             (logits; apply sigmoid for prob)
- future:[B, Lf, D]
- gcode_logits: [B, Tg, V]  (if gcode_in was provided)
- fingerprint:  [B, fp_dim] (unit-norm; for "fingerprinting")
- gates, drop_mask, memory, encoded_mods: analysis artifacts
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModelConfig",
    "MM_DTAE_LSTM",
    "AdaptiveLossWeights",
    "make_pad_mask",
]

# -----------------------
# Utilities
# -----------------------

def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create boolean mask **True at PAD positions**.
    Args:
        lengths: [B] true lengths per sequence
        max_len: optional max length, otherwise max(lengths)
    Returns:
        mask: [B, T] True where index >= length (PAD)
    """
    B = lengths.numel()
    T = int(max_len if max_len is not None else int(lengths.max().item()))
    idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return idx >= lengths.unsqueeze(1)


class SinePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to inputs."""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# -----------------------
# Denoising Transformer AutoEncoder
# -----------------------
class DTAE(nn.Module):
    """Denoising Transformer AutoEncoder block.

    Given an input latent sequence x, the encoder consumes a noisy version of x;
    the decoder reconstructs x from the encoded memory.
    """
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.pos = SinePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def add_noise(x: torch.Tensor, level: float = 0.05, mask_prob: float = 0.1) -> torch.Tensor:
        """Gaussian noise + optional elementwise masking."""
        noise = torch.randn_like(x) * level
        x_noisy = x + noise
        if mask_prob > 0:
            mask = (torch.rand_like(x[..., :1]) < mask_prob).float()
            x_noisy = x_noisy * (1 - mask)
        return x_noisy

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D]
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_noisy = self.add_noise(x)
        z = self.encoder(self.pos(self.dropout(x_noisy)), src_key_padding_mask=src_key_padding_mask)
        L = int(tgt_len or T)
        tgt = torch.zeros(B, L, D, device=x.device)
        rec = self.decoder(
            self.pos(tgt), z,
            tgt_key_padding_mask=(src_key_padding_mask[:, :L] if src_key_padding_mask is not None else None),
            memory_key_padding_mask=src_key_padding_mask,
        )
        return rec, z


# -----------------------
# Per-modality encoder and fusion
# -----------------------
class LinearModalityEncoder(nn.Module):
    """Two-layer MLP + LayerNorm + GELU + Positional Encoding per modality."""
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.1)
        )
        self.pos = SinePositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        return self.pos(self.proj(x))


class CrossModalFusion(nn.Module):
    """Fuse modality sequences with learned gates + cross-attention.

    Query = mean across modalities at each timestep; Keys/Values = all modalities stacked.
    Also supports **modality dropout** during training to improve robustness.
    Flash Attention (PyTorch 2.0+) is used when available for better performance.
    """
    def __init__(self, d_model: int, n_heads: int, num_modalities: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.gates = nn.Parameter(torch.ones(num_modalities))  # learned modality importance
        self.d_model = d_model
        self.n_heads = n_heads

        # Check if Flash Attention is available (PyTorch 2.0+)
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def _flash_attention(
        self,
        q: torch.Tensor,  # [B, T, D]
        k: torch.Tensor,  # [B, T*M, D]
        v: torch.Tensor,  # [B, T*M, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T*M]
    ) -> torch.Tensor:
        """Compute attention using Flash Attention (PyTorch 2.0+).

        Flash Attention is memory-efficient and faster than standard attention.
        We need to reshape to multi-head format for scaled_dot_product_attention.
        """
        B, T_q, D = q.shape
        _, T_kv, _ = k.shape
        H = self.n_heads
        d_head = D // H

        # Reshape to [B, H, T, d_head]
        q = q.view(B, T_q, H, d_head).transpose(1, 2)
        k = k.view(B, T_kv, H, d_head).transpose(1, 2)
        v = v.view(B, T_kv, H, d_head).transpose(1, 2)

        # Convert key_padding_mask to attention mask
        # key_padding_mask: [B, T_kv] with True=ignore
        # attention_mask: needs to be broadcastable to [B, H, T_q, T_kv]
        attn_mask = None
        if key_padding_mask is not None:
            # Invert: True (pad) -> -inf, False (valid) -> 0
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_kv]
            attn_mask = attn_mask.expand(B, H, T_q, T_kv)
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))

        # Flash Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,  # No dropout in attention (can be parameterized if needed)
        )

        # Reshape back to [B, T_q, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, D)
        return attn_out

    def forward(
        self,
        mods: List[torch.Tensor],  # each [B,T,D]
        key_padding_mask: Optional[torch.Tensor],
        modality_dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = len(mods)
        B, T, D = mods[0].shape
        gates = torch.sigmoid(self.gates[:M])  # [M]

        # Optional: randomly drop entire modalities (training-time only)
        drop_mask = torch.ones(M, device=mods[0].device)
        if self.training and modality_dropout_p > 0:
            drop_mask = (torch.rand(M, device=mods[0].device) > modality_dropout_p).float()
            if drop_mask.sum() == 0:  # ensure at least one active
                keep_idx = torch.randint(0, M, (1,), device=mods[0].device)
                drop_mask[keep_idx] = 1.0
        mods = [mods[i] * drop_mask[i] for i in range(M)]

        stack = torch.stack(mods, dim=2)       # [B,T,M,D]
        q = stack.mean(dim=2)                  # [B,T,D]
        kv = stack.flatten(1, 2)               # [B,T*M,D]

        # If a key_padding_mask is provided it has shape [B,T]. Expand it
        # across modalities to match the flattened KV length [B, T*M].
        if key_padding_mask is not None:
            # key_padding_mask: True for PAD positions -> MultiheadAttention expects True to ignore
            kv_key_padding = key_padding_mask.unsqueeze(1).expand(B, M, T).transpose(1, 2).flatten(1)
        else:
            kv_key_padding = None

        # Use Flash Attention if available (PyTorch 2.0+), otherwise standard attention
        if self.use_flash:
            attn_out = self._flash_attention(q, kv, kv, kv_key_padding)
        else:
            attn_out, _ = self.attn(q, kv, kv, key_padding_mask=kv_key_padding)

        fused = self.ln(q + attn_out)
        gated_sum = sum(gates[i] * mods[i] for i in range(M)) / (gates.sum() + 1e-6)
        fused = self.ln(fused + gated_sum)
        if key_padding_mask is not None:
            fused = fused.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return fused, gates.detach(), drop_mask.detach()


# -----------------------
# Context embeddings
# -----------------------
class ContextEmbeddings(nn.Module):
    """
    Summation of multiple categorical embeddings:
    - Initialize with a dict of {"name": vocab_size}
    - Forward receives {"name": tensor} where tensor is [B] (broadcast over T) or [B,T]
    """
    def __init__(self, d_model: int, specs: Optional[Dict[str, int]] = None):
        super().__init__()
        self.d_model = d_model
        specs = specs or {}
        self.tables = nn.ModuleDict({k: nn.Embedding(v, d_model) for k, v in specs.items()})

    def forward(self, ctx_ids: Optional[Dict[str, torch.Tensor]], T: int, device) -> torch.Tensor:
        if not ctx_ids or len(self.tables) == 0:
            return torch.zeros(1, 1, self.d_model, device=device) * 0.0  # neutral
        acc = None
        for name, emb in self.tables.items():
            if name not in ctx_ids:
                continue
            ids = ctx_ids[name].to(device)
            if ids.dim() == 1:  # [B] -> broadcast over T
                x = emb(ids).unsqueeze(1).expand(-1, T, -1)         # [B,T,D]
            else:               # [B,T]
                x = emb(ids)                                       # [B,T,D]
            acc = x if acc is None else (acc + x)
        if acc is None:  # nothing matched
            return torch.zeros(1, 1, self.d_model, device=device) * 0.0
        return acc


# -----------------------
# G-code LM head (causal) with weight tying
# -----------------------
class GCodeLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.pos = SinePositionalEncoding(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (proj shares parameters with embed)
        self.proj.weight = self.embed.weight

    @staticmethod
    def causal_mask(sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forcing training.
        Args:
            memory: [B,Tm,D] (e.g., LSTM outputs)
            tgt_tokens: [B,Tg]
        Returns: logits [B,Tg,V]
        """
        x = self.pos(self.embed(tgt_tokens))
        mask = self.causal_mask(x.size(1), x.device)
        dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
        return self.proj(dec)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, max_len: int, bos_id: int = 1) -> torch.Tensor:
        """Greedy autoregressive decode. Returns token ids [B, max_len]."""
        B = memory.size(0)
        out = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        for _ in range(max_len):
            x = self.pos(self.embed(out))
            mask = self.causal_mask(x.size(1), x.device)
            dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
            logits = self.proj(dec[:, -1:])  # [B,1,V]
            nxt = torch.argmax(logits, dim=-1)
            out = torch.cat([out, nxt], dim=1)
        return out[:, 1:]


# -----------------------
# Fingerprint head (for "G-code fingerprinting")
# -----------------------
class FingerprintHead(nn.Module):
    """Fingerprint head with attention pooling.

    Uses a learnable query to attend over the sequence, producing a
    context-aware pooled representation instead of simple mean pooling.
    """
    def __init__(self, d_model: int, out_dim: int = 128, use_attention_pooling: bool = True):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.d_model = d_model

        if use_attention_pooling:
            # Learnable query for attention pooling
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            # Single-head attention for pooling
            self.attn_pool = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
            nn.init.xavier_uniform_(self.query)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, seq_latent: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq_latent: [B, T, D] sequence of latent representations
            key_padding_mask: [B, T] optional mask with True at PAD positions

        Returns:
            [B, out_dim] unit-norm fingerprint
        """
        if self.use_attention_pooling:
            # Attention pooling: learnable query attends over sequence
            B = seq_latent.size(0)
            query = self.query.expand(B, -1, -1)  # [B, 1, D]

            # Cross-attention: query attends to sequence
            pooled, _ = self.attn_pool(
                query,
                seq_latent,
                seq_latent,
                key_padding_mask=key_padding_mask,  # Ignore padding in attention
            )
            fp = pooled.squeeze(1)  # [B, D]
        else:
            # Fallback to mean pooling
            if key_padding_mask is not None:
                # Masked mean pooling (ignore padding)
                mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
                fp = (seq_latent * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                fp = seq_latent.mean(dim=1)  # [B, D]

        fp = self.proj(fp)  # [B, out_dim]
        return F.normalize(fp, dim=-1)  # unit-norm fingerprint


# -----------------------
# Full multimodal model + heads
# -----------------------
@dataclass
class ModelConfig:
    sensor_dims: List[int]
    d_model: int = 256  # Increased from 128 for better capacity
    lstm_layers: int = 2
    gcode_vocab: int = 128
    future_len: int = 8
    n_heads: int = 4
    dropout: float = 0.1  # Dropout probability for regularization
    context_specs: Optional[Dict[str, int]] = None  # {"plane":3,"units":2,"absrel":2,"wcs":6,"tool":64}
    fp_dim: int = 128                                # fingerprint dimension
    use_attention_pooling: bool = True               # use attention pooling in fingerprint head

class MM_DTAE_LSTM(nn.Module):
    """Backbone: per-modality encoders → modality+context fusion → DTAE → LSTM → heads."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList([LinearModalityEncoder(d, config.d_model) for d in config.sensor_dims])

        # Modality ID embeddings
        self.mod_emb = nn.Embedding(len(config.sensor_dims), config.d_model)

        self.fusion = CrossModalFusion(config.d_model, config.n_heads, num_modalities=len(config.sensor_dims))
        self.dtae = DTAE(config.d_model, nhead=config.n_heads, dropout=config.dropout)
        self.temporal = nn.LSTM(config.d_model, config.d_model, num_layers=config.lstm_layers,
                                batch_first=True, dropout=config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

        # Context embeddings
        self.ctx = ContextEmbeddings(config.d_model, config.context_specs or {})

        # Heads
        self.head_cls = nn.Linear(config.d_model, 5)
        self.head_reg = nn.Linear(config.d_model, 3)
        self.head_anom = nn.Linear(config.d_model, 1)  # logits
        self.head_future = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(config.d_model, config.future_len * config.d_model)
        )
        self.gcode_head = GCodeLMHead(config.d_model, config.gcode_vocab, nhead=config.n_heads)
        self.fp_head = FingerprintHead(config.d_model, out_dim=config.fp_dim, use_attention_pooling=config.use_attention_pooling)

    def forward(
        self,
        mods: List[torch.Tensor],        # list of [B,T,Cm]
        lengths: torch.Tensor,           # [B]
        gcode_in: Optional[torch.Tensor] = None,  # [B,Tg]
        modality_dropout_p: float = 0.0,
        ctx_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        pad_mask = make_pad_mask(lengths, max_len=mods[0].size(1))
        encoded_mods = []
        for i, (enc, m) in enumerate(zip(self.encoders, mods)):
            e = enc(m)  # [B,T,D]
            # add modality id embedding
            e = e + self.mod_emb.weight[i].view(1, 1, -1)
            encoded_mods.append(e)

        fused, gates, drop_mask = self.fusion(encoded_mods, key_padding_mask=pad_mask,
                                              modality_dropout_p=modality_dropout_p)

        # add context embeddings (broadcast per sequence or timestep)
        fused = fused + self.ctx(ctx_ids, T=fused.size(1), device=fused.device)

        rec, z = self.dtae(fused, src_key_padding_mask=pad_mask)
        lstm_out, _ = self.temporal(z)
        lstm_out = self.norm(lstm_out)

        # Heads from last valid step
        last_idx = (lengths.clamp(min=1) - 1).view(-1)
        last = lstm_out[torch.arange(lstm_out.size(0), device=lstm_out.device), last_idx]

        out: Dict[str, torch.Tensor] = {
            "recon": rec,
            "cls": self.head_cls(last),
            "reg": self.head_reg(last),
            "anom": self.head_anom(last),  # logits
            "future": self.head_future(last).view(last.size(0), self.config.future_len, -1),
            "gates": gates,                 # learned gates per modality
            "drop_mask": drop_mask,         # which modalities were kept this step
            "memory": lstm_out,             # LSTM outputs for LM head / analysis
            "encoded_mods": torch.stack([m.detach() for m in encoded_mods], dim=2),  # [B,T,M,D]
        }
        if gcode_in is not None:
            out["gcode_logits"] = self.gcode_head(lstm_out, gcode_in)

        # Fingerprint (unit-norm) with attention pooling
        out["fingerprint"] = self.fp_head(lstm_out, key_padding_mask=pad_mask)

        return out

    def to_config_dict(self) -> Dict:
        return asdict(self.config)

    @staticmethod
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AdaptiveLossWeights(nn.Module):
    """Uncertainty-based task weighting (Kendall & Gal)."""
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = 0.0
        contrib = {}
        for i, k in enumerate(self.task_names):
            if k not in losses:
                continue
            precision = torch.exp(-self.log_vars[i])
            term = precision * losses[k] + self.log_vars[i]
            total = total + term
            contrib[k] = float(term.detach().cpu())
        return total, contrib


# ============================================================================
# PHASE 1.6: Encoder Improvements
# ============================================================================

class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling with learned query vectors.

    Replaces simple temporal attention with multiple learned queries that
    attend to different aspects of the sequence, then projects to a fixed
    output dimension.

    This captures richer sequence-level information than single-query attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        n_queries: int = 8,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            hidden_dim: Input hidden dimension
            n_heads: Number of attention heads
            n_queries: Number of learned query vectors
            dropout: Dropout rate
            output_dim: Output dimension (default: hidden_dim)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.output_dim = output_dim or hidden_dim

        # Learned query vectors (initialized with small values)
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection: flatten queries and project to output dim
        self.out_proj = nn.Linear(hidden_dim * n_queries, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using multi-head attention.

        Args:
            x: (batch, seq_len, hidden_dim) input sequence
            mask: (batch, seq_len) padding mask (True = ignore)

        Returns:
            pooled: (batch, output_dim) pooled representation
        """
        batch_size = x.size(0)

        # Expand queries for batch: [n_queries, hidden_dim] -> [batch, n_queries, hidden_dim]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head attention: queries attend to sequence
        # queries: [B, n_queries, D], x: [B, seq_len, D]
        attn_out, attn_weights = self.attention(
            queries, x, x,
            key_padding_mask=mask
        )
        # attn_out: [B, n_queries, D]

        # Flatten queries and project
        pooled = attn_out.reshape(batch_size, -1)  # [B, n_queries * D]
        pooled = self.out_proj(pooled)  # [B, output_dim]
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        return pooled

    def forward_with_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights for visualization."""
        batch_size = x.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        attn_out, attn_weights = self.attention(
            queries, x, x,
            key_padding_mask=mask,
            average_attn_weights=False  # Keep per-head weights
        )

        pooled = attn_out.reshape(batch_size, -1)
        pooled = self.out_proj(pooled)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        return pooled, attn_weights


class MultiScaleTemporalEncoder(nn.Module):
    """
    Enhanced encoder with multi-scale temporal convolutions.

    Uses dilated convolutions at multiple scales to capture both fine-grained
    (high-frequency vibrations) and long-range (operation-level) patterns
    before feeding into LSTM for sequential modeling.

    Architecture:
    1. Multi-scale 1D convolutions with different kernel sizes and dilations
    2. Feature fusion via 1x1 convolution
    3. Bidirectional LSTM for sequence modeling
    4. Multi-head attention pooling for fixed-size output
    """

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_scales: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_multihead_pooling: bool = True,
        pooling_n_heads: int = 4,
        pooling_n_queries: int = 8,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., 155 sensor features)
            hidden_dim: Hidden dimension after convolutions
            latent_dim: Output latent dimension
            n_scales: Number of parallel conv branches
            kernel_sizes: Kernel sizes for each scale (default: [3, 5, 7, 11])
            dilations: Dilation rates for each scale (default: [1, 2, 4, 8])
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            use_multihead_pooling: Use multi-head attention pooling vs simple mean
            pooling_n_heads: Number of attention heads for pooling
            pooling_n_queries: Number of learned queries for pooling
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_scales = n_scales

        # Default kernel sizes and dilations if not provided
        kernel_sizes = kernel_sizes or [3, 5, 7, 11]
        dilations = dilations or [1, 2, 4, 8]

        # Ensure we have the right number of scales
        assert len(kernel_sizes) == n_scales, f"Expected {n_scales} kernel sizes"
        assert len(dilations) == n_scales, f"Expected {n_scales} dilations"

        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        # Multi-scale conv branches (each outputs hidden_dim // n_scales channels)
        branch_dim = hidden_dim // n_scales
        self.conv_branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            # Calculate padding to maintain temporal dimension
            padding = (k - 1) * d // 2
            branch = nn.Sequential(
                nn.Conv1d(input_dim, branch_dim, kernel_size=k, dilation=d, padding=padding),
                nn.BatchNorm1d(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.conv_branches.append(branch)

        # Fusion: combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Temporal modeling with bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Project LSTM output (bidirectional: 2*hidden_dim) to latent_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pooling to get fixed-size representation
        self.use_multihead_pooling = use_multihead_pooling
        if use_multihead_pooling:
            self.pooling = MultiHeadAttentionPooling(
                hidden_dim=latent_dim,
                n_heads=pooling_n_heads,
                n_queries=pooling_n_queries,
                dropout=dropout,
                output_dim=latent_dim,
            )
        else:
            # Simple attention pooling (single query)
            self.pooling_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
            self.pooling_attn = nn.MultiheadAttention(
                latent_dim, num_heads=1, batch_first=True
            )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale temporal encoder.

        Args:
            x: (batch, seq_len, input_dim) raw sensor features
            lengths: (batch,) actual sequence lengths for packing

        Returns:
            features: (batch, latent_dim) pooled representation
            memory: (batch, seq_len, latent_dim) sequence for cross-attention
        """
        B, T, C = x.shape

        # x: [B, T, C] -> [B, C, T] for 1D convolutions
        x_conv = x.transpose(1, 2)

        # Multi-scale convolutions
        branch_outs = [branch(x_conv) for branch in self.conv_branches]
        # Each branch output: [B, hidden_dim // n_scales, T]

        # Concatenate along channel dimension
        x_multi = torch.cat(branch_outs, dim=1)  # [B, hidden_dim, T]

        # Fuse multi-scale features
        x_fused = self.fusion(x_multi)  # [B, hidden_dim, T]

        # [B, hidden_dim, T] -> [B, T, hidden_dim] for LSTM
        x_seq = x_fused.transpose(1, 2)

        # Pack sequences for efficient LSTM processing if lengths provided
        if lengths is not None:
            # Clamp lengths to valid range
            lengths_clamped = lengths.clamp(min=1, max=T).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_seq, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(x_seq)
        # lstm_out: [B, T, hidden_dim * 2] (bidirectional)

        # Project to latent dimension
        memory = self.proj(lstm_out)  # [B, T, latent_dim]

        # Create padding mask if lengths provided
        pad_mask = None
        if lengths is not None:
            pad_mask = make_pad_mask(lengths, max_len=T)  # True at PAD positions

        # Pool to fixed-size representation
        if self.use_multihead_pooling:
            features = self.pooling(memory, mask=pad_mask)  # [B, latent_dim]
        else:
            # Simple attention pooling
            query = self.pooling_query.expand(B, -1, -1)
            features, _ = self.pooling_attn(query, memory, memory, key_padding_mask=pad_mask)
            features = features.squeeze(1)  # [B, latent_dim]

        return features, memory


class AuxiliarySupervisionHeads(nn.Module):
    """
    Auxiliary supervision heads to shape encoder representations.

    These heads predict various properties of the G-code sequence from
    the encoder's latent representation, encouraging the encoder to learn
    features that are useful for the downstream task.

    Auxiliary tasks:
    1. Motion type prediction (rapid, linear, arc_cw, arc_ccw, dwell)
    2. Sequence length prediction (regression)
    3. Movement magnitude bins (X/Y/Z movement scale)
    4. Parameter presence prediction (which params appear: X, Y, Z, R, F)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_motion_types: int = 5,
        n_magnitude_bins: int = 10,
        n_param_types: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
    ):
        """
        Args:
            latent_dim: Input latent dimension from encoder
            n_motion_types: Number of motion type classes
            n_magnitude_bins: Number of magnitude discretization bins
            n_param_types: Number of parameter types to predict presence
            hidden_dim: Hidden dimension for heads (default: latent_dim // 2)
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        hidden_dim = hidden_dim or latent_dim // 2

        # Motion type prediction (classification)
        # Predicts: rapid (G0), linear (G1), arc_cw (G2), arc_ccw (G3), dwell (G4)
        self.motion_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_motion_types),
        )

        # Sequence length prediction (regression)
        # Predicts the number of tokens in the G-code sequence
        self.length_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Movement magnitude bins (classification)
        # Predicts which magnitude bin (small, medium, large movement)
        self.magnitude_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_magnitude_bins),
        )

        # Parameter presence prediction (multi-label binary classification)
        # Predicts which parameters appear in sequence: [has_X, has_Y, has_Z, has_R, has_F]
        self.param_presence_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_param_types),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary predictions from encoder features.

        Args:
            features: (batch, latent_dim) encoder output

        Returns:
            Dict with:
                - motion_logits: (batch, n_motion_types)
                - length_pred: (batch,) predicted sequence length
                - magnitude_logits: (batch, n_magnitude_bins)
                - param_presence_logits: (batch, n_param_types)
        """
        return {
            'motion_logits': self.motion_head(features),
            'length_pred': self.length_head(features).squeeze(-1),
            'magnitude_logits': self.magnitude_head(features),
            'param_presence_logits': self.param_presence_head(features),
        }


def compute_auxiliary_loss(
    aux_outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weight: float = 0.3,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted auxiliary supervision loss.

    Args:
        aux_outputs: Dict from AuxiliarySupervisionHeads.forward()
        targets: Dict with optional keys:
            - motion_type: (batch,) motion type labels
            - seq_length: (batch,) sequence lengths
            - magnitude_bin: (batch,) magnitude bin labels
            - param_presence: (batch, n_params) binary presence labels
        weight: Overall weight for auxiliary losses
        weights: Per-task weights dict (e.g., {'motion': 1.0, 'length': 0.5})

    Returns:
        total_loss: Weighted sum of all auxiliary losses
        loss_dict: Individual loss values for logging
    """
    device = aux_outputs['motion_logits'].device
    losses = {}
    loss_values = {}

    # Default per-task weights
    task_weights = {
        'motion': 1.0,
        'length': 0.5,
        'magnitude': 0.8,
        'param_presence': 0.7,
    }
    if weights is not None:
        task_weights.update(weights)

    # Motion type loss (classification)
    if 'motion_type' in targets and targets['motion_type'] is not None:
        motion_loss = F.cross_entropy(
            aux_outputs['motion_logits'],
            targets['motion_type'].to(device)
        )
        losses['motion'] = motion_loss * task_weights['motion']
        loss_values['aux_motion_loss'] = motion_loss.item()

    # Sequence length loss (regression with smooth L1)
    if 'seq_length' in targets and targets['seq_length'] is not None:
        length_loss = F.smooth_l1_loss(
            aux_outputs['length_pred'],
            targets['seq_length'].float().to(device)
        )
        losses['length'] = length_loss * task_weights['length']
        loss_values['aux_length_loss'] = length_loss.item()

    # Magnitude bin loss (classification)
    if 'magnitude_bin' in targets and targets['magnitude_bin'] is not None:
        magnitude_loss = F.cross_entropy(
            aux_outputs['magnitude_logits'],
            targets['magnitude_bin'].to(device)
        )
        losses['magnitude'] = magnitude_loss * task_weights['magnitude']
        loss_values['aux_magnitude_loss'] = magnitude_loss.item()

    # Parameter presence loss (binary cross-entropy)
    if 'param_presence' in targets and targets['param_presence'] is not None:
        param_loss = F.binary_cross_entropy_with_logits(
            aux_outputs['param_presence_logits'],
            targets['param_presence'].float().to(device)
        )
        losses['param_presence'] = param_loss * task_weights['param_presence']
        loss_values['aux_param_presence_loss'] = param_loss.item()

    # Compute total weighted loss
    if losses:
        total_loss = sum(losses.values()) * weight
        loss_values['aux_total_loss'] = total_loss.item()
    else:
        # No auxiliary targets available - return zero loss without breaking gradients
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)
        loss_values['aux_total_loss'] = 0.0

    return total_loss, loss_values


class EnhancedEncoder(nn.Module):
    """
    Enhanced encoder combining all Phase 1.6 improvements.

    Features:
    1. Multi-scale temporal convolutions for capturing patterns at different scales
    2. Bidirectional LSTM for sequential modeling
    3. Multi-head attention pooling for rich sequence representation
    4. Optional auxiliary supervision heads
    5. Support for partial unfreezing during training

    Can be used as a drop-in replacement for the MM-DTAE-LSTM encoder.
    """

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_operations: int = 9,
        # Multi-scale conv options
        use_multiscale: bool = True,
        n_scales: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        # LSTM options
        lstm_layers: int = 2,
        # Pooling options
        use_multihead_pooling: bool = True,
        pooling_n_heads: int = 4,
        pooling_n_queries: int = 8,
        # Auxiliary heads
        use_auxiliary_heads: bool = False,
        # Regularization
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Raw sensor feature dimension
            hidden_dim: Internal hidden dimension
            latent_dim: Output latent dimension
            n_operations: Number of operation types for classification
            use_multiscale: Whether to use multi-scale conv (vs simple projection)
            n_scales: Number of conv scales
            kernel_sizes: Kernel sizes per scale
            dilations: Dilation rates per scale
            lstm_layers: Number of LSTM layers
            use_multihead_pooling: Use multi-head attention pooling
            pooling_n_heads: Number of pooling attention heads
            pooling_n_queries: Number of learned query vectors
            use_auxiliary_heads: Add auxiliary supervision heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_multiscale = use_multiscale
        self.use_auxiliary_heads = use_auxiliary_heads

        if use_multiscale:
            # Multi-scale temporal encoder
            self.temporal_encoder = MultiScaleTemporalEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                n_scales=n_scales,
                kernel_sizes=kernel_sizes,
                dilations=dilations,
                lstm_layers=lstm_layers,
                dropout=dropout,
                use_multihead_pooling=use_multihead_pooling,
                pooling_n_heads=pooling_n_heads,
                pooling_n_queries=pooling_n_queries,
            )
        else:
            # Simple encoder (for ablation)
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
            )
            if use_multihead_pooling:
                self.pooling = MultiHeadAttentionPooling(
                    latent_dim, n_heads=pooling_n_heads,
                    n_queries=pooling_n_queries, dropout=dropout
                )
            else:
                self.pooling = None

        # Operation classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_operations),
        )

        # Auxiliary supervision heads (optional)
        if use_auxiliary_heads:
            self.aux_heads = AuxiliarySupervisionHeads(
                latent_dim=latent_dim,
                dropout=dropout,
            )
        else:
            self.aux_heads = None

    def encode(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sensor features to latent representation.

        Args:
            x: (batch, seq_len, input_dim) raw sensor features
            lengths: (batch,) actual sequence lengths

        Returns:
            features: (batch, latent_dim) pooled representation
            memory: (batch, seq_len, latent_dim) sequence for cross-attention
        """
        if self.use_multiscale:
            return self.temporal_encoder(x, lengths)
        else:
            # Simple encoder path
            B, T, _ = x.shape
            x = self.input_proj(x)

            if lengths is not None:
                lengths_clamped = lengths.clamp(min=1, max=T).cpu()
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths_clamped, batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.lstm(x_packed)
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True, total_length=T
                )
            else:
                lstm_out, _ = self.lstm(x)

            memory = self.proj(lstm_out)

            # Pool
            if self.pooling is not None:
                pad_mask = make_pad_mask(lengths, max_len=T) if lengths is not None else None
                features = self.pooling(memory, mask=pad_mask)
            else:
                # Simple mean pooling
                if lengths is not None:
                    pad_mask = make_pad_mask(lengths, max_len=T)
                    mask = (~pad_mask).unsqueeze(-1).float()
                    features = (memory * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                else:
                    features = memory.mean(dim=1)

            return features, memory

    def classify(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Classify operation type from features.

        Args:
            features: (batch, latent_dim) encoder output

        Returns:
            logits: (batch, n_operations) classification logits
            aux_outputs: Optional auxiliary head outputs
        """
        logits = self.classifier(features)

        aux_outputs = None
        if self.aux_heads is not None:
            aux_outputs = self.aux_heads(features)

        return logits, aux_outputs

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (batch, seq_len, input_dim) raw sensor features
            lengths: (batch,) actual sequence lengths

        Returns:
            Dict with:
                - features: (batch, latent_dim) pooled representation
                - memory: (batch, seq_len, latent_dim) sequence
                - cls_logits: (batch, n_operations) classification logits
                - aux_outputs: Optional auxiliary predictions
        """
        features, memory = self.encode(x, lengths)
        cls_logits, aux_outputs = self.classify(features)

        return {
            'features': features,
            'memory': memory,
            'cls_logits': cls_logits,
            'aux_outputs': aux_outputs,
        }

    def get_unfrozen_params(
        self,
        n_layers: int = 0,
    ) -> List[nn.Parameter]:
        """
        Get parameters for partial unfreezing.

        Args:
            n_layers: Number of layers to unfreeze from the end
                      0 = all frozen, -1 = all unfrozen

        Returns:
            List of parameters to optimize
        """
        if n_layers == 0:
            return []

        if n_layers == -1:
            return list(self.parameters())

        # For multi-scale encoder, unfreeze LSTM and pooling
        params = []

        if self.use_multiscale:
            # Unfreeze projection and pooling (closest to output)
            params.extend(self.temporal_encoder.proj.parameters())
            if hasattr(self.temporal_encoder, 'pooling'):
                params.extend(self.temporal_encoder.pooling.parameters())

            # Optionally unfreeze LSTM
            if n_layers >= 2:
                params.extend(self.temporal_encoder.lstm.parameters())
        else:
            params.extend(self.proj.parameters())
            if self.pooling is not None:
                params.extend(self.pooling.parameters())
            if n_layers >= 2:
                params.extend(self.lstm.parameters())

        return params
