#!/usr/bin/env python3
"""Per-operation-class accuracy breakdown.

Buckets the 5-fold AR predictions by operation_type_names (9 cutting
classes) and reports token / numeric / command accuracy per class.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"


def collect(sweep_root: Path, prep_root: Path):
    """Returns {op_class: {'token': [..], 'numeric': [..]}}."""
    out = defaultdict(lambda: {"token": [], "numeric": []})
    for F in range(1, 6):
        pred_files = list(sweep_root.glob(f"fold_{F}/*/results/beam_1_all_predictions.json"))
        pred_files = [p for p in pred_files if "_fsm" not in str(p)]
        if not pred_files:
            continue
        pred_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        samples = json.loads(pred_files[0].read_text())

        npz_path = prep_root / f"fold_{F}" / "test_sequences.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        op_names = d["operation_type_names"]
        if len(op_names) != len(samples):
            print(f"  fold {F}: mismatch n_samples={len(samples)} vs n_op_names={len(op_names)}, skipping")
            continue

        for s, op in zip(samples, op_names):
            t = s.get("true", "").split()
            p = s.get("pred", "").split()
            if not t:
                continue
            n = min(len(t), len(p))
            if n == 0:
                continue
            tok_acc = sum(1 for i in range(n) if t[i] == p[i]) / len(t)
            num_tok_idx = [i for i, x in enumerate(t) if x.startswith("NUM_")]
            num_tok_idx = [i for i in num_tok_idx if i < len(p)]
            if num_tok_idx:
                num_acc = sum(1 for i in num_tok_idx if t[i] == p[i]) / len(num_tok_idx)
            else:
                num_acc = float("nan")
            out[str(op)]["token"].append(tok_acc)
            if not np.isnan(num_acc):
                out[str(op)]["numeric"].append(num_acc)
    return out


def plot_op_breakdown(data, ax, metric, label):
    classes = sorted(data.keys())
    means = [np.mean(data[c][metric]) if data[c][metric] else float("nan") for c in classes]
    stds = [np.std(data[c][metric]) if data[c][metric] else 0 for c in classes]
    ns = [len(data[c][metric]) for c in classes]
    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, capsize=4, color="#3866b3", alpha=0.8, edgecolor="black")
    for i, n in enumerate(ns):
        ax.text(i, max(means[i] + stds[i] + 0.02, 0.05), f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"{label} accuracy (AR)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Per-operation-class {label} accuracy (autoregressive, 5-fold pooled)")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base = collect(
        REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold",
        REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window",
    )
    print(f"Per-class summary ({len(base)} classes detected):")
    for c in sorted(base):
        n = len(base[c]["token"])
        if n > 0:
            print(f"  {c:25s} n={n:3d} tok={np.mean(base[c]['token']):.3f} num={np.mean(base[c]['numeric']) if base[c]['numeric'] else float('nan'):.3f}")

    fig, (ax_t, ax_n) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plot_op_breakdown(base, ax_t, "token", "Token")
    plot_op_breakdown(base, ax_n, "numeric", "Numeric")
    plt.tight_layout()
    out = FIG_DIR / "per_op_class_breakdown.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
