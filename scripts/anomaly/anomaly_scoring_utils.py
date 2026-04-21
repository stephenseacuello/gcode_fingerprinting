#!/usr/bin/env python3
"""Shared utilities for all anomaly detection experiments.

Provides model loading, cached data access, NLL computation,
attack generation, detection scoring, calibration, and statistical tests.
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ============================================================
# Paths
# ============================================================

BASE_DIR = PROJECT_ROOT / "outputs" / "anomaly20260319"
CACHE_DIR = BASE_DIR / "cached_inference"
ATTACK_DIR = BASE_DIR / "attacks"
SUMMARY_FILE = BASE_DIR / "ANOMALY_SUMMARY.md"

# --- V7 paths (default, max_token_len=6) ---
DECODER_CKPT_TEMPLATE = str(
    PROJECT_ROOT / "outputs" / "decoder20260304" / "best_config_5fold"
    / "fold_{fold}" / "decoder_checkpoint" / "best_decoder.pt"
)
ENCODER_CKPT_TEMPLATE = str(
    PROJECT_ROOT / "outputs" / "experiments_2026_02_25" / "full_w256_s64_cv"
    / "fold_{fold}" / "encoder" / "checkpoint" / "best_model.pt"
)
ENCODER_MEMORY_TEMPLATE = str(
    PROJECT_ROOT / "outputs" / "decoder20260304" / "best_config_5fold"
    / "fold_{fold}" / "encoder_memory"
)
PREPROC_TEMPLATE = str(
    PROJECT_ROOT / "outputs" / "decoder20260304" / "preprocessed_v7"
    / "fold_{fold}"
)
ABLATION_2A1_TEMPLATE = str(
    PROJECT_ROOT / "outputs" / "decoder20260304" / "ablations"
    / "2A1_no_window_pos" / "fold_{fold}"
)

# --- V8 paths (multi-line decoder, max_token_len=64) ---
V8_BASE_DIR = BASE_DIR  # same top-level anomaly directory
V8_CACHE_DIR = BASE_DIR / "cached_inference_v8"
V8_ATTACK_DIR = BASE_DIR / "attacks_v8"
V8_DECODER_CKPT_TEMPLATE = str(
    BASE_DIR / "decoder_v8_multiline"
    / "fold_{fold}" / "decoder_checkpoint" / "best_decoder.pt"
)
V8_ENCODER_MEMORY_TEMPLATE = str(
    BASE_DIR / "decoder_v8_multiline"
    / "fold_{fold}" / "encoder_memory"
)
V8_PREPROC_TEMPLATE = str(
    BASE_DIR / "preprocessed_v8_multiline"
    / "fold_{fold}"
)
V8_MAX_TOKEN_LEN = 64

# Environment variable overrides: allow redirecting CACHE_DIR and ATTACK_DIR
# to V8 paths without modifying each experiment script.
# Set ANOMALY_CACHE_DIR and/or ANOMALY_ATTACK_DIR before importing this module.
if os.environ.get("ANOMALY_CACHE_DIR"):
    CACHE_DIR = Path(os.environ["ANOMALY_CACHE_DIR"])
    logger.info(f"CACHE_DIR overridden via env: {CACHE_DIR}")
if os.environ.get("ANOMALY_ATTACK_DIR"):
    ATTACK_DIR = Path(os.environ["ANOMALY_ATTACK_DIR"])
    logger.info(f"ATTACK_DIR overridden via env: {ATTACK_DIR}")

VOCAB_PATH = PROJECT_ROOT / "data" / "gcode_vocab_712.json"
AXIS_TABLE_PATH = PROJECT_ROOT / "data" / "axis_value_tables.json"

# Token type constants
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAM = 2
TYPE_NUMERIC = 3

# Special token IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

FOLDS = [1, 2, 3, 4, 5]


# ============================================================
# Vocabulary Loading
# ============================================================

_vocab_cache = None
_axis_table_cache = None


def load_vocab() -> dict:
    """Load the 712-token G-code vocabulary."""
    global _vocab_cache
    if _vocab_cache is None:
        with open(VOCAB_PATH) as f:
            _vocab_cache = json.load(f)
    return _vocab_cache


def load_axis_tables() -> dict:
    """Load axis value tables mapping tokens to numeric values."""
    global _axis_table_cache
    if _axis_table_cache is None:
        with open(AXIS_TABLE_PATH) as f:
            _axis_table_cache = json.load(f)
    return _axis_table_cache


def build_token_type_map(vocab: dict) -> Dict[int, int]:
    """Map each token ID to its type (SPECIAL/COMMAND/PARAM/NUMERIC)."""
    v = vocab["vocab"]
    special = {"PAD", "BOS", "EOS", "UNK", "MASK"}
    commands = {"G0", "G1", "G2", "G3", "G53", "M30", "NONE"}
    params = {"X", "Y", "Z", "R", "I", "F", "S", "J", "K", "A", "B", "C", "P", "Q", "E"}

    type_map = {}
    for name, idx in v.items():
        if name in special:
            type_map[idx] = TYPE_SPECIAL
        elif name in commands:
            type_map[idx] = TYPE_COMMAND
        elif name in params:
            type_map[idx] = TYPE_PARAM
        elif name.startswith("NUM_"):
            type_map[idx] = TYPE_NUMERIC
        else:
            type_map[idx] = TYPE_SPECIAL  # fallback
    return type_map


def build_id_to_name(vocab: dict) -> Dict[int, str]:
    """Reverse map: token ID -> token name."""
    return {idx: name for name, idx in vocab["vocab"].items()}


def build_name_to_id(vocab: dict) -> Dict[str, int]:
    """Forward map: token name -> token ID."""
    return dict(vocab["vocab"])


def get_axis_values(axis: str, vocab: dict) -> List[Tuple[float, int]]:
    """Get all (value, token_id) pairs for a given axis, sorted by value."""
    v = vocab["vocab"]
    prefix = f"NUM_{axis}_"
    precision = vocab["config"]["precision"].get(axis, 0.001)
    pairs = []
    for name, idx in v.items():
        if name.startswith(prefix):
            raw = name[len(prefix):]
            # Convert bucket string to float value
            value = int(raw) * precision
            pairs.append((value, idx))
    return sorted(pairs, key=lambda x: x[0])


# ============================================================
# Model Loading
# ============================================================

def load_decoder(fold: int, device: str = "cuda") -> "SensorMultiHeadDecoder":
    """Load a trained decoder checkpoint for a given fold."""
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder

    ckpt_path = DECODER_CKPT_TEMPLATE.format(fold=fold)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    vocab = load_vocab()

    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab["vocab"]),
        sensor_dim=256,
        d_model=args.get("d_model", 384),
        n_operations=9,
        n_heads=args.get("n_heads", 8),
        n_layers=args.get("n_layers", 8),
        dropout=args.get("dropout", 0.1),
        max_seq_len=args.get("max_token_len", 16),
        use_window_position=args.get("use_window_position", False),
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    decoder.set_vocab(vocab["vocab"])
    decoder.eval()
    decoder.to(device)
    return decoder


def load_decoder_v8(fold: int, device: str = "cuda") -> "SensorMultiHeadDecoder":
    """Load a trained V8 multi-line decoder checkpoint for a given fold."""
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder

    ckpt_path = V8_DECODER_CKPT_TEMPLATE.format(fold=fold)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    vocab = load_vocab()

    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab["vocab"]),
        sensor_dim=256,
        d_model=args.get("d_model", 384),
        n_operations=9,
        n_heads=args.get("n_heads", 8),
        n_layers=args.get("n_layers", 8),
        dropout=args.get("dropout", 0.1),
        max_seq_len=args.get("max_token_len", V8_MAX_TOKEN_LEN),
        use_window_position=args.get("use_window_position", False),
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    decoder.set_vocab(vocab["vocab"])
    decoder.eval()
    decoder.to(device)
    return decoder


def load_encoder(fold: int, device: str = "cuda") -> "MM_DTAE_LSTM":
    """Load a trained encoder checkpoint for a given fold."""
    from miracle.model.model import MM_DTAE_LSTM

    ckpt_path = ENCODER_CKPT_TEMPLATE.format(fold=fold)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if "config" in ckpt:
        config = ckpt["config"]
    elif "model_config" in ckpt:
        config = ckpt["model_config"]
    else:
        config = {}

    encoder = MM_DTAE_LSTM(config)
    if "model_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["model_state_dict"])
    else:
        encoder.load_state_dict(ckpt)
    encoder.eval()
    encoder.to(device)
    return encoder


# ============================================================
# Cached Data Loading (from outputs/anomaly20260319/cached_inference/)
# ============================================================

def load_cached(fold: int, filename: str) -> Union[dict, torch.Tensor]:
    """Load a cached tensor/dict from the inference cache."""
    path = CACHE_DIR / f"fold_{fold}" / filename
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_cached_logits(fold: int) -> dict:
    return load_cached(fold, "logits.pt")


def load_cached_nll(fold: int) -> dict:
    return load_cached(fold, "nll_scores.pt")


def load_cached_predictions(fold: int) -> dict:
    return load_cached(fold, "predictions.pt")


def load_cached_targets(fold: int) -> dict:
    return load_cached(fold, "targets.pt")


def load_cached_embeddings(fold: int) -> dict:
    return load_cached(fold, "embeddings.pt")


def load_cached_metadata(fold: int) -> dict:
    return load_cached(fold, "metadata.json")


def is_cache_complete() -> bool:
    """Check if inference cache has been built for all folds."""
    flag = CACHE_DIR / "cache_complete.flag"
    return flag.exists()


# ============================================================
# Raw Data Loading (encoder memory + preprocessed NPZ)
# ============================================================

def load_encoder_memory(fold: int, split: str = "test") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached encoder memory and operation predictions.

    Returns:
        memory: [N, T_s, 256] encoder hidden states
        op_pred: [N] operation type predictions
    """
    mem_dir = Path(ENCODER_MEMORY_TEMPLATE.format(fold=fold))
    memory = torch.load(mem_dir / f"{split}_memory.pt", map_location="cpu", weights_only=False)
    op_pred = torch.load(mem_dir / f"{split}_op_pred.pt", map_location="cpu", weights_only=False)
    return memory, op_pred


def load_preprocessed_npz(fold: int, split: str = "test") -> dict:
    """Load preprocessed NPZ file for a given fold and split.

    Returns dict with keys: continuous, tokens, lengths, operation_type,
    gcode_texts, operation_type_names
    """
    preproc_dir = Path(PREPROC_TEMPLATE.format(fold=fold))
    npz = np.load(preproc_dir / f"{split}_sequences.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def load_encoder_memory_v8(fold: int, split: str = "test") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached encoder memory for V8 multi-line decoder.

    Returns:
        memory: [N, T_s, 256] encoder hidden states
        op_pred: [N] operation type predictions
    """
    mem_dir = Path(V8_ENCODER_MEMORY_TEMPLATE.format(fold=fold))
    memory = torch.load(mem_dir / f"{split}_memory.pt", map_location="cpu", weights_only=False)
    op_pred = torch.load(mem_dir / f"{split}_op_pred.pt", map_location="cpu", weights_only=False)
    return memory, op_pred


def load_preprocessed_npz_v8(fold: int, split: str = "test") -> dict:
    """Load V8 preprocessed NPZ file (max_token_len=64).

    Returns dict with keys: continuous, tokens, lengths, operation_type,
    gcode_texts, operation_type_names
    """
    preproc_dir = Path(V8_PREPROC_TEMPLATE.format(fold=fold))
    npz = np.load(preproc_dir / f"{split}_sequences.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def load_cached_v8(fold: int, filename: str) -> Union[dict, torch.Tensor]:
    """Load a cached tensor/dict from the V8 inference cache."""
    path = V8_CACHE_DIR / f"fold_{fold}" / filename
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_cached_targets_v8(fold: int) -> dict:
    return load_cached_v8(fold, "targets.pt")


def build_teacher_forcing_batch(
    tokens: np.ndarray, lengths: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build teacher-forcing input/target pairs from raw token sequences.

    Args:
        tokens: [N, max_len] raw token IDs
        lengths: [N] actual lengths (excluding BOS/EOS)

    Returns:
        input_tokens: [N, max_len+1] with BOS prepended
        target_tokens: [N, max_len+1] with EOS appended
        padding_mask: [N, max_len+1] True at padded positions
    """
    N, L = tokens.shape
    out_len = L + 1  # +1 for BOS/EOS

    input_tokens = np.zeros((N, out_len), dtype=np.int64)
    target_tokens = np.zeros((N, out_len), dtype=np.int64)
    padding_mask = np.ones((N, out_len), dtype=bool)  # True = padded

    for i in range(N):
        seq_len = min(int(lengths[i]), L)
        # Input: [BOS, t1, t2, ..., tn]
        input_tokens[i, 0] = BOS_ID
        input_tokens[i, 1 : seq_len + 1] = tokens[i, :seq_len]
        # Target: [t1, t2, ..., tn, EOS]
        target_tokens[i, :seq_len] = tokens[i, :seq_len]
        target_tokens[i, seq_len] = EOS_ID
        # Mask: False for valid positions
        padding_mask[i, : seq_len + 1] = False

    return (
        torch.from_numpy(input_tokens),
        torch.from_numpy(target_tokens),
        torch.from_numpy(padding_mask),
    )


# ============================================================
# NLL Computation
# ============================================================

def compute_per_token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token negative log-likelihood.

    Args:
        logits: [N, L, V] predicted logits
        targets: [N, L] target token IDs
        padding_mask: [N, L] True at padded positions

    Returns:
        nll: [N, L] per-token NLL (0 at padded positions)
    """
    N, L, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
        ignore_index=PAD_ID,
    ).reshape(N, L)
    nll = nll.masked_fill(padding_mask, 0.0)
    return nll


def compute_window_scores(
    per_token_nll: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-token NLL into window-level scores.

    Args:
        per_token_nll: [N, L]
        lengths: [N] actual sequence lengths (used as fallback)
        targets: [N, L] optional target tokens to identify content positions
                 (excludes PAD=0 and EOS=2 from scoring)

    Returns:
        dict with S_mean, S_max, S_sum, perplexity (each [N])
    """
    # Content mask: positions with actual NLL > 0 (PAD positions have NLL=0 from ignore_index)
    # Also exclude EOS tokens which the model often mispredicts (not meaningful for anomaly detection)
    if targets is not None:
        content_mask = (targets != PAD_ID) & (targets != EOS_ID)
    else:
        # Fallback: exclude positions with NLL=0 (PAD, from ignore_index)
        # and positions with NLL > 5000 (EOS/boundary artifacts)
        content_mask = (per_token_nll > 0) & (per_token_nll < 5000)

    # Count content tokens per window
    content_lengths = content_mask.sum(dim=1).float().clamp(min=1)

    # Sum NLL only over content positions
    s_sum = (per_token_nll * content_mask.float()).sum(dim=1)
    s_mean = s_sum / content_lengths

    # Max NLL over content positions only
    masked_nll = per_token_nll.clone()
    masked_nll[~content_mask] = -float("inf")
    s_max = masked_nll.max(dim=1).values

    perplexity = torch.exp(s_mean)

    return {
        "S_mean": s_mean,
        "S_max": s_max,
        "S_sum": s_sum,
        "perplexity": perplexity,
    }


# ============================================================
# Advanced Scoring: Rank-Based and Confidence-Weighted
# ============================================================


def compute_rank_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: dict,
) -> Dict[str, torch.Tensor]:
    """Rank-based anomaly scoring for numeric tokens.

    For each numeric token, computes its rank among all same-axis
    vocabulary entries. Rank 1 = model's top prediction for that axis.
    Higher rank = more anomalous.

    This normalizes within each axis, removing calibration issues
    that plague NLL-based scoring on uncertain numeric predictions.

    Args:
        logits: [N, L, V] legacy logits
        targets: [N, L] target token IDs
        vocab: vocabulary dict (token_name -> id)

    Returns:
        dict with S_rank_mean, S_rank_max, S_rank_percentile (each [N])
    """
    id2tok = {v: k for k, v in vocab.items()}

    # Pre-build axis -> [token_ids] mapping
    axis_token_ids = {}
    for name, idx in vocab.items():
        if name.startswith("NUM_"):
            axis = name.split("_")[1]
            if axis not in axis_token_ids:
                axis_token_ids[axis] = []
            axis_token_ids[axis].append(idx)
    # Sort each axis list for consistent indexing
    for axis in axis_token_ids:
        axis_token_ids[axis] = sorted(axis_token_ids[axis])

    N, L, V = logits.shape
    rank_scores = torch.zeros(N, L)  # 0 for non-numeric positions
    is_numeric = torch.zeros(N, L, dtype=torch.bool)

    for i in range(N):
        for j in range(L):
            t = targets[i, j].item()
            if t == PAD_ID or t == EOS_ID:
                continue
            name = id2tok.get(t, "")
            if not name.startswith("NUM_"):
                continue

            axis = name.split("_")[1]
            axis_ids = axis_token_ids.get(axis, [])
            if len(axis_ids) < 2:
                continue

            # Get logits for this axis only, find rank
            axis_ids_t = torch.tensor(axis_ids, dtype=torch.long)
            axis_logits = logits[i, j, axis_ids_t]
            sorted_idx = torch.argsort(axis_logits, descending=True)

            # Find position of target in the axis-specific ranking
            target_pos_in_axis = axis_ids.index(t)
            rank = (sorted_idx == target_pos_in_axis).nonzero(as_tuple=True)[0].item() + 1

            # Normalize rank to [0, 1] (1 = worst possible)
            rank_scores[i, j] = rank / len(axis_ids)
            is_numeric[i, j] = True

    # Window-level aggregation (only over numeric positions)
    num_counts = is_numeric.sum(dim=1).float().clamp(min=1)
    s_rank_sum = rank_scores.sum(dim=1)
    s_rank_mean = s_rank_sum / num_counts

    # Max rank across numeric positions
    masked_ranks = rank_scores.clone()
    masked_ranks[~is_numeric] = -1
    s_rank_max = masked_ranks.max(dim=1).values

    return {
        "S_rank_mean": s_rank_mean,
        "S_rank_max": s_rank_max,
        "n_numeric": num_counts,
    }


def compute_confidence_weighted_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Confidence-weighted NLL scoring.

    Only includes tokens where the model assigns P > threshold to its
    top prediction. For confident positions, NLL of the claimed token
    is highly informative. For uncertain positions, NLL is noise.

    Args:
        logits: [N, L, V] legacy logits
        targets: [N, L] target token IDs
        confidence_threshold: minimum max-probability to include a token

    Returns:
        dict with S_conf_mean, S_conf_max, n_confident (each [N])
    """
    N, L, V = logits.shape
    probs = torch.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values  # [N, L]

    # Confidence mask: model is sure about something at this position
    confident = max_probs > confidence_threshold

    # Also exclude PAD and EOS
    content = (targets != PAD_ID) & (targets != EOS_ID)
    mask = confident & content

    # NLL at each position
    nll = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1),
        reduction="none", ignore_index=PAD_ID,
    ).reshape(N, L)

    n_confident = mask.sum(dim=1).float().clamp(min=1)
    s_conf_sum = (nll * mask.float()).sum(dim=1)
    s_conf_mean = s_conf_sum / n_confident

    masked_nll = nll.clone()
    masked_nll[~mask] = -float("inf")
    s_conf_max = masked_nll.max(dim=1).values
    s_conf_max[s_conf_max == -float("inf")] = 0  # no confident tokens

    return {
        "S_conf_mean": s_conf_mean,
        "S_conf_max": s_conf_max,
        "n_confident": n_confident,
    }


def compute_hybrid_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: dict,
    confidence_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Hybrid scoring: confidence-weighted NLL + rank-based for numerics.

    Uses NLL for confident tokens (commands, params where model is sure)
    and rank-based scoring for numeric tokens (where NLL is unreliable).

    Args:
        logits: [N, L, V] legacy logits
        targets: [N, L] target token IDs
        vocab: vocabulary dict
        confidence_threshold: for confidence-weighted component

    Returns:
        dict with S_hybrid, S_nll_component, S_rank_component
    """
    conf_scores = compute_confidence_weighted_scores(
        logits, targets, confidence_threshold
    )
    rank_scores = compute_rank_scores(logits, targets, vocab)

    # Hybrid: weighted combination
    # NLL component handles commands/params, rank handles numerics
    alpha = 0.5
    s_hybrid = (
        alpha * conf_scores["S_conf_mean"]
        + (1 - alpha) * rank_scores["S_rank_mean"]
    )

    return {
        "S_hybrid": s_hybrid,
        "S_conf_mean": conf_scores["S_conf_mean"],
        "S_rank_mean": rank_scores["S_rank_mean"],
        "S_rank_max": rank_scores["S_rank_max"],
        "n_confident": conf_scores["n_confident"],
        "n_numeric": rank_scores["n_numeric"],
    }


# ============================================================
# Attack Generation
# ============================================================

def inject_command_swap(
    tokens: torch.Tensor,
    vocab: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A1: Swap command tokens (G0<->G1, G1->G2/G3).

    Returns:
        attacked_tokens: [N_attacks, L]
        attack_info: [N_attacks] indices into original batch
    """
    name2id = build_name_to_id(vocab)
    id2name = build_id_to_name(vocab)
    type_map = build_token_type_map(vocab)

    swap_pairs = {
        name2id["G0"]: name2id["G1"],
        name2id["G1"]: name2id["G0"],
    }

    attacked = []
    orig_indices = []

    N, L = tokens.shape
    for i in range(N):
        for j in range(L):
            tid = tokens[i, j].item()
            if tid in swap_pairs:
                new_tokens = tokens[i].clone()
                new_tokens[j] = swap_pairs[tid]
                attacked.append(new_tokens)
                orig_indices.append(i)

    if not attacked:
        return torch.empty(0, L, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.stack(attacked), torch.tensor(orig_indices, dtype=torch.long)


def inject_coordinate_shift(
    tokens: torch.Tensor,
    axis: str,
    delta_mm: float,
    vocab: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A2: Shift coordinate values by delta_mm for a given axis.

    Returns:
        attacked_tokens: [N_attacks, L]
        orig_indices: [N_attacks]
        actual_deltas: [N_attacks] the actual delta applied (may differ from requested)
    """
    axis_values = get_axis_values(axis, vocab)
    if not axis_values:
        return (
            torch.empty(0, tokens.shape[1], dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )

    values = np.array([v for v, _ in axis_values])
    ids = np.array([i for _, i in axis_values])

    prefix = f"NUM_{axis}_"
    name2id = build_name_to_id(vocab)
    id2name = build_id_to_name(vocab)

    # Build token_id -> value lookup
    tid_to_val = {}
    for val, tid in axis_values:
        tid_to_val[tid] = val

    attacked = []
    orig_indices = []
    actual_deltas = []

    N, L = tokens.shape
    for i in range(N):
        for j in range(L):
            tid = tokens[i, j].item()
            if tid in tid_to_val:
                orig_val = tid_to_val[tid]
                target_val = orig_val + delta_mm
                # Find nearest vocab value
                nearest_idx = np.argmin(np.abs(values - target_val))
                nearest_tid = int(ids[nearest_idx])
                if nearest_tid != tid:  # Only if actually different
                    new_tokens = tokens[i].clone()
                    new_tokens[j] = nearest_tid
                    attacked.append(new_tokens)
                    orig_indices.append(i)
                    actual_deltas.append(values[nearest_idx] - orig_val)

    if not attacked:
        return (
            torch.empty(0, L, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )
    return (
        torch.stack(attacked),
        torch.tensor(orig_indices, dtype=torch.long),
        torch.tensor(actual_deltas, dtype=torch.float),
    )


def inject_param_swap(
    tokens: torch.Tensor,
    vocab: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A3: Swap parameter letters (X<->Y)."""
    name2id = build_name_to_id(vocab)
    swap_pairs = {}
    if "X" in name2id and "Y" in name2id:
        swap_pairs[name2id["X"]] = name2id["Y"]
        swap_pairs[name2id["Y"]] = name2id["X"]

    attacked = []
    orig_indices = []

    N, L = tokens.shape
    for i in range(N):
        for j in range(L):
            tid = tokens[i, j].item()
            if tid in swap_pairs:
                new_tokens = tokens[i].clone()
                new_tokens[j] = swap_pairs[tid]
                attacked.append(new_tokens)
                orig_indices.append(i)

    if not attacked:
        return torch.empty(0, L, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.stack(attacked), torch.tensor(orig_indices, dtype=torch.long)


# ============================================================
# Detection Scoring
# ============================================================

def score_with_cached_logits(
    logits: torch.Tensor,
    original_targets: torch.Tensor,
    attacked_targets: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
    orig_indices: torch.Tensor,
    scoring: str = "S_mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Score normal vs attacked sequences using cached logits.

    The logits were computed with teacher-forcing on ORIGINAL tokens.
    For detection, we compare:
    - Normal score: NLL of original target under the predicted distribution
    - Attack score: NLL of attacked target under the SAME predicted distribution

    Returns:
        normal_scores: [N_normal]
        attack_scores: [N_attacks]
    """
    # Normal scores
    nll_normal = compute_per_token_nll(logits, original_targets, padding_mask)
    normal_window = compute_window_scores(nll_normal, lengths)

    # Attack scores: same logits, different targets
    attack_logits = logits[orig_indices]
    attack_mask = padding_mask[orig_indices]
    attack_lengths = lengths[orig_indices]
    nll_attack = compute_per_token_nll(attack_logits, attacked_targets, attack_mask)
    attack_window = compute_window_scores(nll_attack, attack_lengths)

    return (
        normal_window[scoring].numpy(),
        attack_window[scoring].numpy(),
    )


# ============================================================
# AUROC / AUPRC with Bootstrap CIs
# ============================================================

def compute_auroc(normal_scores: np.ndarray, attack_scores: np.ndarray) -> float:
    """Compute AUROC. Higher attack scores = positive class."""
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
    scores = np.concatenate([normal_scores, attack_scores])
    return roc_auc_score(labels, scores)


def compute_auprc(normal_scores: np.ndarray, attack_scores: np.ndarray) -> float:
    """Compute AUPRC."""
    from sklearn.metrics import average_precision_score

    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
    scores = np.concatenate([normal_scores, attack_scores])
    return average_precision_score(labels, scores)


def compute_detection_at_fpr(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Detection rate (TPR) at a given false positive rate."""
    threshold = np.percentile(normal_scores, 100 * (1 - target_fpr))
    return float(np.mean(attack_scores > threshold))


def bootstrap_auroc_ci(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Stratified bootstrap CI for AUROC."""
    rng = np.random.RandomState(seed)
    aurocs = []
    n_normal = len(normal_scores)
    n_attack = len(attack_scores)

    for _ in range(n_bootstrap):
        idx_n = rng.choice(n_normal, n_normal, replace=True)
        idx_a = rng.choice(n_attack, n_attack, replace=True)
        try:
            aurocs.append(compute_auroc(normal_scores[idx_n], attack_scores[idx_a]))
        except ValueError:
            continue

    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    return {
        "auroc": float(compute_auroc(normal_scores, attack_scores)),
        "ci_lower": float(np.percentile(aurocs, 100 * alpha)),
        "ci_upper": float(np.percentile(aurocs, 100 * (1 - alpha))),
        "std": float(np.std(aurocs)),
    }


# ============================================================
# Calibration
# ============================================================

def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_mce(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Find optimal temperature T for calibration via L-BFGS.

    Args:
        logits: [N, L, V]
        targets: [N, L]
        mask: [N, L] True at padded positions

    Returns:
        optimal temperature T
    """
    # Flatten valid positions
    valid = ~mask
    flat_logits = logits[valid]  # [M, V]
    flat_targets = targets[valid]  # [M]

    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled = flat_logits / temperature
        loss = F.cross_entropy(scaled, flat_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


# ============================================================
# Statistical Tests
# ============================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d with Hedges' g correction for small samples."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    # Hedges' g correction
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * correction)


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test comparing means. Returns p-value."""
    rng = np.random.RandomState(seed)
    observed = abs(np.mean(group1) - np.mean(group2))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if perm_diff >= observed:
            count += 1
    return count / n_perm


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def delong_test(
    y_true: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
) -> float:
    """DeLong test for comparing two AUROCs. Returns p-value.

    Simplified implementation using bootstrap approximation.
    """
    from sklearn.metrics import roc_auc_score

    auc1 = roc_auc_score(y_true, scores1)
    auc2 = roc_auc_score(y_true, scores2)
    diff = auc1 - auc2

    # Bootstrap the difference
    rng = np.random.RandomState(42)
    n = len(y_true)
    diffs = []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        try:
            d = roc_auc_score(y_true[idx], scores1[idx]) - roc_auc_score(y_true[idx], scores2[idx])
            diffs.append(d)
        except ValueError:
            continue

    diffs = np.array(diffs)
    se = np.std(diffs)
    if se == 0:
        return 1.0
    z = diff / se
    from scipy.stats import norm
    return float(2 * norm.sf(abs(z)))


# ============================================================
# Summary File Management
# ============================================================

def append_to_summary(
    exp_id: int,
    name: str,
    status: str,
    auroc: str = "--",
    finding: str = "--",
    output_dir: Optional[Path] = None,
):
    """Append/update an experiment's status in ANOMALY_SUMMARY.md."""
    summary_path = output_dir / "ANOMALY_SUMMARY.md" if output_dir else SUMMARY_FILE
    if not summary_path.exists():
        logger.warning(f"Summary file not found at {summary_path}")
        return

    content = summary_path.read_text()

    # Update timestamp
    content = content.replace(
        content.split("_Last updated:")[1].split("_")[0] if "_Last updated:" in content else "",
        f" {datetime.now().strftime('%Y-%m-%d %H:%M')} ",
    )

    # Update status in dashboard table
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if f"| {exp_id}" in line and f"| {exp_id} " in line:
            parts = line.split("|")
            if len(parts) >= 7:
                parts[3] = f" {status:7s} "
                parts[5] = f" {auroc:20s} "
                parts[6] = f" {finding} "
                lines[i] = "|".join(parts)
                break

    summary_path.write_text("\n".join(lines))


# ============================================================
# I/O Utilities
# ============================================================

def save_results(results: dict, output_dir: Path, filename: str = "aggregate_results.json"):
    """Save results dict as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    # Convert numpy/torch types
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    logger.info(f"Saved results to {path}")


def save_figure(fig, output_dir: Path, name: str):
    """Save a matplotlib figure as PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved figure to {path}")


def setup_logging(exp_name: str):
    """Configure logging for an experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{exp_name}] %(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# Experiment Script Boilerplate
# ============================================================

def get_default_parser(exp_name: str, exp_id: int):
    """Create a standard argument parser for experiment scripts."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Experiment {exp_id}: {exp_name}")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / f"exp{exp_id:02d}_{exp_name.lower().replace(' ', '_')}",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    return parser
