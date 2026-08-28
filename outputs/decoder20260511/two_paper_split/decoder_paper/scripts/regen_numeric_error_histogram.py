#!/usr/bin/env python3
"""Regenerate figures/numeric_error_histogram.pdf (and .png) for Paper A from the
PATCHED full-window 5-fold checkpoints, so the per-axis |pred-target| distribution
is consistent with the patched per-axis MAE reported in Table 5
(X 0.248 / Y 0.205 / Z 0.009 / R 0.021 inch).

Self-contained: reads only
  <DEC>/checkpoints/full_window_5fold_patched/fold_*/results/predictions.npz
reconstructs per-axis signed numeric values in inches from the 6-slot digit heads
(pt_t -> axis, sign_t/sign_p, digit_t/digit_p), computes |pred - target| in inches,
and renders a per-axis log-x histogram with the leftmost-bin (exact-match) fraction
annotated per axis. Writes the PDF/PNG into this paper's figures/ directory only.

Slot grid (V8 tokenizer): slot 0 = 10^1 ... slot 5 = 10^-4 inch; PAD = 10.
Axis map (pt_t): 0->X 1->Y 2->Z 3->F 4->R.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIG_DIR = PAPER / "figures"
DEC = Path("/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260511")
PATCHED = DEC / "checkpoints/full_window_5fold_patched"

DIGIT_PAD = 10
TYPE_NUMERIC = 3
PT_TO_AXIS = {0: "X", 1: "Y", 2: "Z", 3: "F", 4: "R"}
SLOTW = np.array([10.0, 1.0, 0.1, 0.01, 0.001, 0.0001])  # slot 0..5 weights (inch)
AXES_PLOT = ["X", "Y", "Z", "R"]  # F has thin/units-only support; omit from the 4-panel grid
MAX_PER_AXIS = 5000  # subsample for figure-file size (matches original convention)
RNG = np.random.RandomState(42)
SMALL = 1e-5  # floor for log-axis placement of exact-match (zero-error) mass


def reconstruct():
    """Pool |pred-target| (inch) per axis across the 5 patched folds, plus
    per-fold MAEs (mean-of-fold-means, to match the audit JSON / Table 5)."""
    pooled_err = {a: [] for a in PT_TO_AXIS.values()}
    fold_mae = {a: [] for a in PT_TO_AXIS.values()}
    exact = {a: [0, 0] for a in PT_TO_AXIS.values()}  # [n_exact, n_total]
    for fold in range(1, 6):
        path = PATCHED / f"fold_{fold}" / "results" / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        z = np.load(path)
        dp, dt = z["digit_p"], z["digit_t"]
        pt, tt = z["pt_t"], z["type_t"]
        sp, st = z["sign_p"], z["sign_t"]
        for pv, ax in PT_TO_AXIS.items():
            m = (tt == TYPE_NUMERIC) & (pt == pv) & (dt[..., 0] != DIGIT_PAD)
            if int(m.sum()) == 0:
                continue
            td = dt[m].astype(float)
            pd = dp[m].astype(float)
            # PAD digits contribute nothing to the composed value.
            tdv = np.where(td == DIGIT_PAD, 0.0, td)
            pdv = np.where(pd == DIGIT_PAD, 0.0, pd)
            tval = (tdv * SLOTW).sum(axis=1)
            pval = (pdv * SLOTW).sum(axis=1)
            tval = np.where(st[m] == 1, -tval, tval)
            pval = np.where(sp[m] == 1, -pval, pval)
            err = np.abs(pval - tval)
            pooled_err[ax].extend(err.tolist())
            fold_mae[ax].append(float(err.mean()))
            exact[ax][0] += int((err == 0).sum())
            exact[ax][1] += int(len(err))
    return pooled_err, fold_mae, exact


def main():
    pooled_err, fold_mae, exact = reconstruct()

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    axes = axes.ravel()
    # log-spaced bins over the 10^-5 .. 10^1 inch precision span.
    bins = np.logspace(-5, 1, 49)
    tier_lines = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    colors = {"X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c", "R": "#d62728"}

    for ax_obj, axis in zip(axes, AXES_PLOT):
        err = np.asarray(pooled_err[axis], dtype=float)
        n_tot = exact[axis][1]
        exact_frac = exact[axis][0] / n_tot if n_tot else 0.0
        mae = float(np.mean(fold_mae[axis])) if fold_mae[axis] else float("nan")
        # Subsample for the rendered histogram only (annotations use full pool).
        if len(err) > MAX_PER_AXIS:
            err = err[RNG.choice(len(err), MAX_PER_AXIS, replace=False)]
        # Place exact matches (err == 0) into the leftmost (10^-5) bin for the log axis.
        err_plot = np.where(err <= 0.0, SMALL, err)
        ax_obj.hist(err_plot, bins=bins, color=colors[axis], alpha=0.85, edgecolor="none")
        ax_obj.set_xscale("log")
        for t in tier_lines:
            ax_obj.axvline(t, color="0.6", lw=0.7, ls="--", zorder=0)
        ax_obj.set_xlim(bins[0], bins[-1])
        ax_obj.set_title(f"{axis}-axis", fontsize=11)
        ax_obj.set_xlabel(r"$|\hat{v}-v|$ (inch, log scale)", fontsize=9)
        ax_obj.set_ylabel("count", fontsize=9)
        ax_obj.tick_params(labelsize=8)
        ax_obj.text(
            0.97, 0.95,
            f"exact-match: {exact_frac:.1%}\nMAE: {mae:.3f} in\n$n$={n_tot:,}",
            transform=ax_obj.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )

    fig.suptitle(
        "Per-axis numeric-value error magnitude (patched 5-fold retrain, teacher-forced)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf = FIG_DIR / "numeric_error_histogram.pdf"
    png = FIG_DIR / "numeric_error_histogram.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf, "and", png)
    for axis in AXES_PLOT:
        n_tot = exact[axis][1]
        print(
            f"  {axis}: MAE(mean-of-folds)={np.mean(fold_mae[axis]):.4f} "
            f"exact-match={exact[axis][0] / n_tot:.4f} n={n_tot}"
        )


if __name__ == "__main__":
    main()
