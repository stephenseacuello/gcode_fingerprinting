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
