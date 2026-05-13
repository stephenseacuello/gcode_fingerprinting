#!/usr/bin/env python3
"""Phase-D figure: confusion matrices for command / type / param_type.

Aggregates confusion matrices across the 5-fold sweep, then renders
sklearn-style heatmaps. Output: PDF + PNG per head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
SWEEP = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold"
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"


def _find_fold_metrics(fold_dir: Path) -> Path | None:
    """Handle wandb-subdir layout: prefer metrics.json over beam_0_metrics.json."""
    for cand in [fold_dir / "results" / "metrics.json",
                 fold_dir / "results" / "beam_0_metrics.json"]:
        if cand.exists():
            return cand
    nested = list(fold_dir.glob("*/results/metrics.json"))
    nested.extend(fold_dir.glob("*/results/beam_0_metrics.json"))
    if nested:
        nested.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return nested[0]
    return None


def _aggregate_confusion(head: str) -> tuple[np.ndarray, list[str]]:
    """Sum confusion matrices across 5 folds, returning agg + labels."""
    agg = None
    label_order = None
    for F in [1, 2, 3, 4, 5]:
        bm_path = _find_fold_metrics(SWEEP / f"fold_{F}")
        if bm_path is None or not bm_path.exists():
            continue
        bm = json.loads(bm_path.read_text())
        head_data = bm.get("test_metrics", {}).get("per_class", {}).get(head, {})
        cm = head_data.get("confusion_matrix")
        labels = head_data.get("confusion_matrix_labels")
        if cm is None or labels is None:
            continue
        cm = np.array(cm)
        if agg is None:
            agg = cm
            label_order = labels
        else:
            # Reconcile label sets — only add matrices with matching labels
            if labels == label_order:
                agg = agg + cm
    return agg, label_order


def plot_cm(cm: np.ndarray, labels: list[str], title: str, out_stem: Path,
            normalize: bool = True, cmap: str = "Blues"):
    if normalize:
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels) + 3),
                                     max(5, 0.5 * len(labels) + 2)))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None,
                   aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            if val > 0.005:
                ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                        ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for head, title in [
        ("type", "Type head — confusion matrix (5-fold aggregate)"),
        ("command", "Command head — confusion matrix (5-fold aggregate)"),
        ("param_type", "Param-type head — confusion matrix (5-fold aggregate)"),
    ]:
        cm, labels = _aggregate_confusion(head)
        if cm is None:
            print(f"  skip {head}: no confusion matrices found")
            continue
        plot_cm(cm, labels, title, args.out_dir / f"confusion_matrix_{head}_normalized", normalize=True)
        plot_cm(cm, labels, title.replace("5-fold aggregate", "5-fold aggregate counts"),
                args.out_dir / f"confusion_matrix_{head}_counts", normalize=False)
        print(f"  wrote {head} confusion matrix ({len(labels)} classes, sum={cm.sum()})")

    print(f"\nFigures saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
