#!/usr/bin/env python3
"""
measure_kl.py
=============

Measure the KL divergence between the model's grammar-masked and unmasked
type-head distributions on held-out G-code lines. This is the cheap empirical
defense against the Park et al. critique that grammar-constrained decoding
distorts the underlying distribution.

Setup mirrors scripts/evaluation/eval_constrained_decoding.py exactly so we
reuse the existing checkpoints and data pipeline. The only addition is, at
each non-pad position, we compute:
    p_unmasked = softmax(type_logits)                      [4-class]
    p_masked   = softmax(type_logits + grammar_mask)       [4-class]
    KL(p_masked || p_unmasked)
where the grammar mask is the type-level transition mask from the DFA in
Section 4.5 of the paper:

    SPECIAL (BOS) -> {COMMAND, PARAM, SPECIAL(EOS)}
    COMMAND       -> {PARAM, SPECIAL(EOS)}
    PARAM         -> {NUMERIC}             (mandatory)
    NUMERIC       -> {PARAM, SPECIAL(EOS)}

The "previous type" is taken from the teacher-forced target sequence so the
DFA state is unambiguous. Pad positions are skipped.

Output: prints aggregated mean/median/p99 KL across all valid positions, per
fold and overall. Saves a JSON with the per-fold breakdown for the paper.

Usage:
    python3 scripts/experiments/grammar_design_space/measure_kl.py \
        --out outputs/decoder20260303/grammarpaper/design_space_results/kl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "evaluation"))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig  # noqa: E402,F401
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder  # noqa: E402
from miracle.utilities.gcode_tokenizer import GCodeTokenizer  # noqa: E402

from run_decoder_quick_test import (  # noqa: E402
    PAD, BOS, EOS,
    TYPE_SPECIAL, TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC,
    build_modality_indices,
    DecoderQuickTestDataset, CachedDecoderDataset,
    load_frozen_encoder, cache_encoder_memory, decoder_collate_fn,
)


# ---------------------------------------------------------------------------
# Type-level grammar mask: previous type -> set of allowed next types
# ---------------------------------------------------------------------------
# Encoded as a [4, 4] boolean matrix: VALID[prev, next] = True
VALID = torch.zeros(4, 4, dtype=torch.bool)
# After SPECIAL (BOS or EOS): allow COMMAND, PARAM, SPECIAL (the EOS path)
VALID[TYPE_SPECIAL, TYPE_SPECIAL] = True
VALID[TYPE_SPECIAL, TYPE_COMMAND] = True
VALID[TYPE_SPECIAL, TYPE_PARAM] = True
# After COMMAND: allow PARAM or SPECIAL (EOS)
VALID[TYPE_COMMAND, TYPE_PARAM] = True
VALID[TYPE_COMMAND, TYPE_SPECIAL] = True
# After PARAM: allow NUMERIC only (mandatory)
VALID[TYPE_PARAM, TYPE_NUMERIC] = True
# After NUMERIC: allow PARAM or SPECIAL (EOS)
VALID[TYPE_NUMERIC, TYPE_PARAM] = True
VALID[TYPE_NUMERIC, TYPE_SPECIAL] = True


def kl_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(P || Q) elementwise, summed over the last dimension."""
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


@torch.no_grad()
def measure_fold(decoder, loader, device) -> dict:
    decoder.eval()
    valid_global = VALID.to(device)

    kls_pmask_qunmask: list[float] = []
    kls_punmask_qmask: list[float] = []
    n_positions = 0
    n_mass_blocked = 0.0  # cumulative prob mass the unmasked dist puts on disallowed types

    for batch in loader:
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        memory = batch["memory"].to(device)
        op_pred = batch["op_pred"].to(device)
        type_targets = batch["type_targets"].to(device)
        padding_mask = (input_tokens == PAD)

        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op_pred,
            tgt_key_padding_mask=padding_mask,
        )
        type_logits = outputs["type_logits"]  # [B, L, 4]
        B, L, C = type_logits.shape

        # Determine the previous type at each position to derive the mask.
        # Use teacher-forced target_tokens' type. For position t, previous is type at t-1.
        # We don't have a per-token type for input_tokens, but type_targets has it.
        # type_targets[t] is the type *at position t*. So prev_type[t] = type_targets[t-1].
        # For t=0 we set prev = SPECIAL (BOS).
        prev_type = torch.full_like(type_targets, TYPE_SPECIAL)
        prev_type[:, 1:] = type_targets[:, :-1]
        # Where the previous target was -100 (ignore index), we still leave it as SPECIAL.

        # Build the per-position allowed-types mask: [B, L, 4]
        # valid_global has shape [4, 4]. Index by prev_type clipped to valid range.
        prev_clipped = prev_type.clamp(min=0, max=3)
        allowed = valid_global[prev_clipped]  # [B, L, 4]

        # Mask out invalid types in the logits for the masked distribution
        masked_logits = type_logits.masked_fill(~allowed, float("-inf"))

        p_unmasked = F.softmax(type_logits, dim=-1)
        p_masked = F.softmax(masked_logits, dim=-1)

        # Probability mass that the unmasked dist puts on now-disallowed types
        blocked_mass = (p_unmasked * (~allowed)).sum(dim=-1)  # [B, L]

        kl1 = kl_pq(p_masked, p_unmasked)         # KL(masked || unmasked)
        kl2 = kl_pq(p_unmasked, p_masked)         # KL(unmasked || masked) — diverges if mass on blocked

        # Mask out positions where the target is pad / ignore-index (-100) or where
        # we have no valid prev type information.
        valid_mask = (type_targets >= 0) & (target_tokens != PAD)

        kl1_v = kl1[valid_mask]
        kl2_v = kl2[valid_mask]
        bm_v = blocked_mass[valid_mask]

        kls_pmask_qunmask.extend(kl1_v.cpu().tolist())
        kls_punmask_qmask.extend(kl2_v.cpu().tolist())
        n_mass_blocked += bm_v.sum().item()
        n_positions += int(valid_mask.sum().item())

    arr1 = np.asarray(kls_pmask_qunmask, dtype=np.float64)
    arr2 = np.asarray(kls_punmask_qmask, dtype=np.float64)
    return {
        "n_positions": n_positions,
        "kl_masked_unmasked": {
            "mean": float(arr1.mean()) if arr1.size else float("nan"),
            "median": float(np.median(arr1)) if arr1.size else float("nan"),
            "p99": float(np.percentile(arr1, 99)) if arr1.size else float("nan"),
            "max": float(arr1.max()) if arr1.size else float("nan"),
        },
        "kl_unmasked_masked": {
            "mean": float(arr2[np.isfinite(arr2)].mean()) if arr2.size else float("nan"),
            "median": float(np.median(arr2[np.isfinite(arr2)])) if arr2.size else float("nan"),
            "p99": float(np.percentile(arr2[np.isfinite(arr2)], 99)) if arr2.size else float("nan"),
            "n_inf": int(np.isinf(arr2).sum()),
        },
        "mean_blocked_mass": n_mass_blocked / max(n_positions, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", default="data/gcode_vocab_712.json")
    p.add_argument("--encoder-base", default="outputs/experiments_2026_02_25/full_w128_s32_cv")
    p.add_argument("--decoder-base", default="outputs/decoder20260303/feat110")
    p.add_argument("--out", default="outputs/decoder20260303/grammarpaper/design_space_results/kl.json")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = GCodeTokenizer.load(Path(args.vocab))
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab: {vocab_size} tokens")

    enc_base = Path(args.encoder_base)
    dec_base = Path(args.decoder_base)

    per_fold = {}
    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")
        data_dir = enc_base / f"fold_{fold}" / "preprocessed"
        ckpt_path = enc_base / f"fold_{fold}" / "encoder" / "checkpoint" / "best_model.pt"
        decoder_ckpt_path = dec_base / f"fold_{fold}" / "decoder_checkpoint" / "best_decoder.pt"

        encoder, enc_config, _ = load_frozen_encoder(str(ckpt_path), device)
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        columns = metadata["continuous_columns"]
        _, group_indices, _ = build_modality_indices(columns)

        test_ds = DecoderQuickTestDataset(data_dir / "test_sequences.npz", tokenizer, max_token_len=16)
        memory, op_pred, _ = cache_encoder_memory(encoder, test_ds, group_indices, device, batch_size=32)
        cached_ds = CachedDecoderDataset(test_ds, memory, op_pred)
        loader = DataLoader(cached_ds, batch_size=32, shuffle=False, collate_fn=decoder_collate_fn)

        decoder = SensorMultiHeadDecoder(
            vocab_size=vocab_size, d_model=192, sensor_dim=enc_config.d_model,
            n_operations=9, n_heads=8, n_layers=4,
            max_int_digits=2, n_decimal_digits=4, dropout=0.3, max_seq_len=16,
        )
        decoder.set_vocab(tokenizer.vocab)
        decoder = decoder.to(device)
        ckpt = torch.load(decoder_ckpt_path, map_location=device, weights_only=False)
        # strict=False because newer model versions may have added buffers
        # (e.g., grammar_mask) that were not present at training time;
        # those are initialized in __init__ and don't need loading.
        missing, unexpected = decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
        if missing:
            print(f"  [load] missing keys (initialized fresh): {missing}")
        if unexpected:
            print(f"  [load] unexpected keys: {unexpected}")

        result = measure_fold(decoder, loader, device)
        per_fold[fold] = result
        print(f"  positions:                 {result['n_positions']}")
        print(f"  KL(masked||unmasked) mean:  {result['kl_masked_unmasked']['mean']:.6f}")
        print(f"  KL(masked||unmasked) p99:   {result['kl_masked_unmasked']['p99']:.6f}")
        print(f"  KL(unmasked||masked) mean:  {result['kl_unmasked_masked']['mean']:.6f}  "
              f"(n_inf={result['kl_unmasked_masked']['n_inf']})")
        print(f"  mean blocked prob mass:     {result['mean_blocked_mass']:.6f}")

    # Aggregate across folds
    means_mu = [r["kl_masked_unmasked"]["mean"] for r in per_fold.values()]
    means_mu_um = [r["kl_unmasked_masked"]["mean"] for r in per_fold.values()]
    blocked = [r["mean_blocked_mass"] for r in per_fold.values()]
    summary = {
        "per_fold": per_fold,
        "aggregate": {
            "kl_masked_unmasked_mean": float(np.mean(means_mu)),
            "kl_masked_unmasked_std": float(np.std(means_mu)),
            "kl_unmasked_masked_mean": float(np.mean(means_mu_um)),
            "mean_blocked_mass": float(np.mean(blocked)),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nAggregate (5-fold mean):")
    print(f"  KL(masked||unmasked):  {summary['aggregate']['kl_masked_unmasked_mean']:.6f} "
          f"+- {summary['aggregate']['kl_masked_unmasked_std']:.6f}")
    print(f"  Mean blocked prob mass: {summary['aggregate']['mean_blocked_mass']:.6f}")


if __name__ == "__main__":
    main()
