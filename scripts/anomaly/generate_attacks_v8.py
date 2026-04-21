#!/usr/bin/env python3
"""Phase 0b (V8): Pre-generate all attack token sequences for V8 multi-line decoder.

Same attack types (A1-A4) as generate_attacks.py but uses V8 cached targets
with longer token sequences (up to 64 tokens per window instead of 6).

With 64 tokens per window there are MORE attack positions per window,
so attack counts will be significantly higher than V7.

Saves to outputs/anomaly20260319/attacks_v8/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    V8_ATTACK_DIR,
    V8_CACHE_DIR,
    V8_MAX_TOKEN_LEN,
    FOLDS,
    load_vocab,
    load_axis_tables,
    load_cached_targets_v8,
    build_teacher_forcing_batch,
    build_token_type_map,
    build_name_to_id,
    build_id_to_name,
    get_axis_values,
    inject_command_swap,
    inject_coordinate_shift,
    inject_param_swap,
    setup_logging,
    TYPE_COMMAND,
    TYPE_PARAM,
    TYPE_NUMERIC,
    PAD_ID,
)

import logging

logger = logging.getLogger(__name__)


def generate_vocab_analysis(vocab: dict, output_dir: Path):
    """Analyze vocabulary and compute per-axis statistics."""
    analysis = {}
    for axis in ["X", "Y", "Z", "R", "F"]:
        pairs = get_axis_values(axis, vocab)
        if not pairs:
            continue
        values = [v for v, _ in pairs]
        token_ids = [i for _, i in pairs]

        # Minimum resolvable delta
        sorted_vals = sorted(values)
        if len(sorted_vals) > 1:
            deltas = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            min_delta = min(deltas)
            mean_delta = np.mean(deltas)
            max_delta = max(deltas)
        else:
            min_delta = mean_delta = max_delta = 0.0

        analysis[axis] = {
            "n_values": len(pairs),
            "min_value": min(values),
            "max_value": max(values),
            "min_resolvable_delta_mm": round(float(min_delta), 6),
            "mean_delta_mm": round(float(mean_delta), 6),
            "max_delta_mm": round(float(max_delta), 6),
            "precision": vocab["config"]["precision"].get(axis, 0.001),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "vocab_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Vocab analysis saved. Axes: {list(analysis.keys())}")
    for ax, info in analysis.items():
        logger.info(
            f"  {ax}: {info['n_values']} values, "
            f"range=[{info['min_value']:.4f}, {info['max_value']:.4f}], "
            f"min_delta={info['min_resolvable_delta_mm']:.6f}mm"
        )
    return analysis


def generate_a1_attacks(all_targets: dict, vocab: dict, output_dir: Path):
    """A1: Command swap (G0<->G1)."""
    results = {"fold_attacks": {}, "total_attacks": 0}
    for fold, targets in all_targets.items():
        attacked, orig_idx = inject_command_swap(targets, vocab)
        results["fold_attacks"][fold] = {
            "attacked_tokens": attacked,
            "orig_indices": orig_idx,
            "n_attacks": len(attacked),
        }
        results["total_attacks"] += len(attacked)

    torch.save(results, output_dir / "attack_A1_command_swap.pt")
    logger.info(f"A1 Command Swap: {results['total_attacks']} total attacks "
                f"(V8: more positions per window with {V8_MAX_TOKEN_LEN}-token sequences)")


def generate_a2_attacks(all_targets: dict, vocab: dict, output_dir: Path):
    """A2: Coordinate tampering at graded severity levels."""
    deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    axes = ["X", "Y", "Z"]

    results = {"deltas": deltas, "axes": axes, "attacks_by_delta_axis": {}}

    for delta in deltas:
        for axis in axes:
            key = f"{axis}_delta_{delta}"
            fold_data = {}
            total = 0
            for fold, targets in all_targets.items():
                attacked, orig_idx, actual_deltas = inject_coordinate_shift(
                    targets, axis, delta, vocab
                )
                fold_data[fold] = {
                    "attacked_tokens": attacked,
                    "orig_indices": orig_idx,
                    "actual_deltas": actual_deltas,
                    "n_attacks": len(attacked),
                }
                total += len(attacked)
            results["attacks_by_delta_axis"][key] = {
                "fold_attacks": fold_data,
                "total_attacks": total,
            }
            if total > 0:
                logger.info(f"A2 {axis} delta={delta}mm: {total} attacks")

    torch.save(results, output_dir / "attack_A2_coord_graded.pt")


def generate_a3_attacks(all_targets: dict, vocab: dict, output_dir: Path):
    """A3: Parameter substitution (X<->Y)."""
    results = {"fold_attacks": {}, "total_attacks": 0}
    for fold, targets in all_targets.items():
        attacked, orig_idx = inject_param_swap(targets, vocab)
        results["fold_attacks"][fold] = {
            "attacked_tokens": attacked,
            "orig_indices": orig_idx,
            "n_attacks": len(attacked),
        }
        results["total_attacks"] += len(attacked)

    torch.save(results, output_dir / "attack_A3_param_swap.pt")
    logger.info(f"A3 Param Swap: {results['total_attacks']} total attacks")


def generate_a4_attacks(all_targets: dict, all_op_types: dict, output_dir: Path):
    """A4: Cross-operation swap -- replace tokens with those from a different class."""
    results = {"fold_attacks": {}, "total_attacks": 0}

    for fold, targets in all_targets.items():
        op_types = all_op_types[fold]

        attacked_list = []
        orig_indices_list = []
        N = targets.shape[0]

        for i in range(N):
            # Find a window from a different operation type
            other_mask = op_types != op_types[i]
            if other_mask.sum() == 0:
                continue
            other_indices = np.where(other_mask)[0]
            # Pick a random other window
            other_idx = other_indices[i % len(other_indices)]
            attacked_list.append(targets[other_idx].clone())
            orig_indices_list.append(i)

        if attacked_list:
            attacked = torch.stack(attacked_list)
            orig_idx = torch.tensor(orig_indices_list, dtype=torch.long)
        else:
            attacked = torch.empty(0, targets.shape[1], dtype=torch.long)
            orig_idx = torch.empty(0, dtype=torch.long)

        results["fold_attacks"][fold] = {
            "attacked_tokens": attacked,
            "orig_indices": orig_idx,
            "n_attacks": len(attacked),
        }
        results["total_attacks"] += len(attacked)

    torch.save(results, output_dir / "attack_A4_cross_op.pt")
    logger.info(f"A4 Cross-Op Swap: {results['total_attacks']} total attacks")


def main():
    parser = argparse.ArgumentParser(description="Generate V8 attack token sequences")
    parser.add_argument("--output-dir", type=Path, default=V8_ATTACK_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    args = parser.parse_args()

    setup_logging("generate_attacks_v8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"V8 attack generation (max_token_len={V8_MAX_TOKEN_LEN})")
    logger.info(f"Loading V8 cached targets from: {V8_CACHE_DIR}")

    vocab = load_vocab()

    # Load test targets for all folds from V8 cache
    logger.info("Loading V8 test targets for all folds...")
    all_targets = {}
    all_op_types = {}
    for fold in args.folds:
        cached = load_cached_targets_v8(fold)
        target_tokens = cached["target_tokens"]
        all_targets[fold] = target_tokens
        all_op_types[fold] = cached["operation_type"].numpy()
        logger.info(f"  Fold {fold}: {target_tokens.shape[0]} windows, "
                     f"seq_len={target_tokens.shape[1]} (V8)")

    # Vocab analysis
    generate_vocab_analysis(vocab, args.output_dir)

    # Generate attacks
    t0 = time.time()
    generate_a1_attacks(all_targets, vocab, args.output_dir)
    generate_a2_attacks(all_targets, vocab, args.output_dir)
    generate_a3_attacks(all_targets, vocab, args.output_dir)
    generate_a4_attacks(all_targets, all_op_types, args.output_dir)

    elapsed = time.time() - t0
    logger.info(f"All V8 attacks generated in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
