#!/usr/bin/env python3
"""Mode-collapse heatmap.

For one fold's AR predictions, compute pairwise token-overlap similarity
between every pair of predicted sequences. If the model is mode-collapsed,
the matrix is mostly bright (high similarity off-diagonal); if predictions
are sample-specific, the matrix has structure that follows operation
class.

Two side-by-side panels:
  (a) baseline AR fold 2 — strong mode collapse expected
  (b) with_shortcuts AR fold 2 — weaker mode collapse expected
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"


def load_fold_predictions(sweep_root: Path, fold: int):
    cands = list(sweep_root.glob(f"fold_{fold}/*/results/beam_1_all_predictions.json"))
    cands = [c for c in cands if "_fsm" not in str(c)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads(cands[0].read_text())


def load_op_names(prep_root: Path, fold: int):
    npz = prep_root / f"fold_{fold}" / "test_sequences.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    return [str(x) for x in d["operation_type_names"]]


def pairwise_similarity(samples, max_len=200):
    """Compute pairwise token-overlap similarity (first max_len tokens) for all
    samples. Returns (N, N) matrix in [0, 1]."""
    N = len(samples)
    truncated = [s.get("pred", "").split()[:max_len] for s in samples]
    M = np.zeros((N, N))
    for i in range(N):
        ti = truncated[i]
        for j in range(N):
            tj = truncated[j]
            n = min(len(ti), len(tj))
            if n == 0:
                continue
            M[i, j] = sum(1 for k in range(n) if ti[k] == tj[k]) / n
    return M


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FOLD = 2

    base_root = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold"
    sc_root = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold_with_shortcuts"
    prep_root = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window"

    op_names = load_op_names(prep_root, FOLD)
    base = load_fold_predictions(base_root, FOLD)
    sc = load_fold_predictions(sc_root, FOLD)
    print(f"fold {FOLD}: baseline n={len(base)}, with_shortcuts n={len(sc)}, op_names n={len(op_names) if op_names else 0}")

    # Sort samples by operation class for visual coherence
    if op_names:
        order = sorted(range(len(base)), key=lambda i: op_names[i])
        base = [base[i] for i in order]
        sc = [sc[i] for i in order]
        op_sorted = [op_names[i] for i in order]
    else:
        op_sorted = None

    M_base = pairwise_similarity(base)
    M_sc = pairwise_similarity(sc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, M, title, mean in [(ax1, M_base, f"Baseline AR (fold {FOLD})", M_base.mean()),
                                (ax2, M_sc, f"+ positional metadata AR (fold {FOLD})", M_sc.mean())]:
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("test sample index (sorted by op-class)")
        ax.set_ylabel("test sample index")
        # Annotate off-diagonal mean (excluding self-similarity)
        N = M.shape[0]
        off_diag = (M.sum() - np.trace(M)) / (N * N - N)
        ax.set_title(f"{title}\noff-diagonal mean similarity = {off_diag:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="token-overlap fraction")

    fig.suptitle("Pairwise token-overlap of AR predictions across test samples.\n"
                 "Bright matrix = same prediction for everything (mode collapse). "
                 "Sample order matches operation-class grouping.",
                 fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "mode_collapse_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  baseline off-diag mean: {(M_base.sum() - np.trace(M_base)) / (M_base.shape[0]**2 - M_base.shape[0]):.4f}")
    print(f"  shortcuts off-diag mean: {(M_sc.sum() - np.trace(M_sc)) / (M_sc.shape[0]**2 - M_sc.shape[0]):.4f}")


if __name__ == "__main__":
    main()
