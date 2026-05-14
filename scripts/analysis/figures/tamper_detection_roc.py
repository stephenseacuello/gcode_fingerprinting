#!/usr/bin/env python3
"""Tamper-detection ROC curve.

Sweeps the detection threshold for each attack class and plots TPR vs FPR.
For command-swap, the threshold is "minimum number of mismatched command
tokens between claimed and predicted G-code"; for sign-flip, "minimum
number of axis-sign disagreements"; for feed-edit, "minimum relative F
deviation."

Uses the existing AR (beam_1) predictions from baseline and with_shortcuts
5-fold runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"

ALL_GCODES = re.compile(r"\b(G\d+|M\d+)\b")
AXES_SIGNED = ("X", "Y", "Z")


def _detok(s):
    return s.replace("NUM_X_", "X").replace("NUM_Y_", "Y").replace("NUM_Z_", "Z")


def commands_in(text):
    return set(ALL_GCODES.findall(text))


def sign_pattern(text):
    out = {}
    for ax in AXES_SIGNED:
        m = re.search(rf"\b{ax}(-?)\d", text)
        if m:
            out[ax] = m.group(1) == "-"
    return out


def feed_rate(text):
    m = re.search(r"\bF(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def tamper_command(text):
    cmds = ALL_GCODES.findall(text)
    if not cmds:
        return None
    swap = {"G0": "G3", "G1": "G2", "G2": "G1", "G3": "G0", "G53": "G0", "M30": "G1"}
    new = swap.get(cmds[0], cmds[0])
    return text.replace(cmds[0], new, 1)


def tamper_sign(text):
    for ax in AXES_SIGNED:
        m = re.search(rf"\b{ax}(-?\d+\.?\d*)", text)
        if m:
            val = m.group(1)
            new_val = val.lstrip("-") if val.startswith("-") else "-" + val
            return text[: m.start(1)] + new_val + text[m.end(1) :]
    return None


def tamper_feed(text):
    m = re.search(r"\bF(\d+\.?\d*)", text)
    if not m:
        return None
    val = float(m.group(1))
    return text[: m.start(1)] + str(val * 10) + text[m.end(1) :]


def detect_command_disagreement(claim, pred):
    """Count how many G-commands in claim disagree with pred (positionally
    first command). Returns 0 if same set or both empty."""
    c = ALL_GCODES.findall(claim)
    p = ALL_GCODES.findall(pred)
    if not c and not p:
        return 0
    if not c or not p:
        return 1
    return 0 if c[0] == p[0] else 1


def detect_sign_disagreement(claim, pred):
    c = sign_pattern(claim)
    p = sign_pattern(pred)
    return sum(1 for ax in c if ax in p and c[ax] != p[ax])


def detect_feed_relative(claim, pred):
    a = feed_rate(claim)
    b = feed_rate(pred)
    if a is None or b is None:
        return 0.0
    return abs(a - b) / max(abs(a), 1e-6)


def load_predictions(sweep_root: Path):
    out = []
    for F in range(1, 6):
        cands = list(sweep_root.glob(f"fold_{F}/*/results/beam_1_all_predictions.json"))
        cands = [c for c in cands if "_fsm" not in str(c)]
        if not cands:
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        samples = json.loads(cands[0].read_text())
        for s in samples:
            out.append((s["true"], s["pred"]))
    return out


def sweep_roc(pairs, tamper_fn, detect_fn, thresholds):
    """For each threshold, compute (FPR, TPR). For each pair:
       - honest: score = detect_fn(true, pred). Flagged if score >= threshold.
       - tampered: score = detect_fn(tamper_fn(true), pred). Flagged similarly.
    """
    honest_scores = []
    tampered_scores = []
    for true, pred in pairs:
        honest_scores.append(detect_fn(true, pred))
        t = tamper_fn(true)
        if t is None or t == true:
            continue
        tampered_scores.append(detect_fn(t, pred))
    honest_scores = np.array(honest_scores, dtype=float)
    tampered_scores = np.array(tampered_scores, dtype=float)
    fpr = []
    tpr = []
    for th in thresholds:
        fp = (honest_scores >= th).mean()
        tp = (tampered_scores >= th).mean()
        fpr.append(fp); tpr.append(tp)
    return np.array(fpr), np.array(tpr)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base = load_predictions(REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold")
    sc = load_predictions(REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold_with_shortcuts")
    print(f"Loaded baseline n={len(base)}, with_shortcuts n={len(sc)}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    attacks = [
        ("Command-swap", tamper_command, detect_command_disagreement, np.linspace(0, 1.01, 50)),
        ("Sign-flip", tamper_sign, detect_sign_disagreement, np.linspace(0, 3, 60)),
        ("Feed-rate $\\pm 10\\times$", tamper_feed, detect_feed_relative, np.linspace(0, 5, 100)),
    ]

    for ax, (name, tfn, dfn, ths) in zip(axes, attacks):
        for label, pairs, color in [("baseline", base, "#b33838"),
                                     ("+ shortcuts", sc, "#3866b3")]:
            fpr, tpr = sweep_roc(pairs, tfn, dfn, ths)
            # AUC via trapezoidal
            order = np.argsort(fpr)
            auc = np.trapz(tpr[order], fpr[order])
            ax.plot(fpr, tpr, "o-", color=color, alpha=0.85, markersize=4,
                    label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8, label="chance")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("FPR (false alarm on honest pair)")
        ax.set_ylabel("TPR (true detection)")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    fig.suptitle("Tamper-detection ROC curves (autoregressive predictions, 5-fold pooled)", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "tamper_detection_roc.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
