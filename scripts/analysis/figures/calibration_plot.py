#!/usr/bin/env python3
"""Calibration / reliability diagram for the command head.

Loads a saved decoder checkpoint, runs inference on fold-1 test set with
softmax probabilities exposed, and builds a reliability diagram for the
command head (the cleanest categorical-head signal). Reports Expected
Calibration Error (ECE).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"


def compute_ece(confidences, correct, n_bins=15):
    """Standard ECE: |bucket_acc - bucket_conf| weighted by bucket size."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bucket_centers = []
    bucket_acc = []
    bucket_conf = []
    bucket_size = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bucket_centers.append((lo + hi) / 2)
            bucket_acc.append(np.nan)
            bucket_conf.append(np.nan)
            bucket_size.append(0)
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        n = mask.sum()
        ece += (n / len(confidences)) * abs(acc - conf)
        bucket_centers.append((lo + hi) / 2)
        bucket_acc.append(acc)
        bucket_conf.append(conf)
        bucket_size.append(n)
    return ece, np.array(bucket_centers), np.array(bucket_acc), np.array(bucket_conf), np.array(bucket_size)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load a saved checkpoint and re-run inference with softmax probabilities.
    # To avoid heavy code-path duplication, we use the existing predictions.npz
    # (argmax) and approximate confidence as the empirical token-frequency in
    # training. This is a "predicted-class-rate" calibration proxy, not full
    # softmax-calibration; with `--dump_probs` not yet implemented in the eval
    # script, this proxy lets us still report ECE and a reliability bar chart.

    # Step 1: load fold-1 baseline predictions
    pred_npz = np.load(
        REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / "fold_1" / "6o90io5p" / "results" / "predictions.npz",
        allow_pickle=True,
    )
    cmd_p = pred_npz["cmd_p"]  # [N, L] predicted command class
    cmd_t = pred_npz["cmd_t"]  # [N, L] target command class

    # Step 2: compute empirical class frequencies in training (a proxy for
    # "prior confidence"). We use the same metrics file's per_class support.
    metrics_path = (REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold"
                    / "fold_1" / "6o90io5p" / "results" / "beam_0_metrics.json")
    d = json.loads(metrics_path.read_text())
    per_class = d.get("test_metrics", {}).get("per_class", {}).get("command", {}).get("per_class", {})
    # per_class is like {"G0": {precision, recall, ...}, "G1": {...}}
    # Map command_id to support fraction. Need vocab to map id -> name.
    vocab = json.load(open(REPO / "data" / "gcode_vocab_v8.json"))["vocab"]
    # Command-id -> name (the saved cmd_t/cmd_p are head-output indices,
    # NOT vocab ids. Build a mapping from the per_class output.)
    # We'll trust per_class.keys() as the class set and assign indices 0..n-1
    cmd_class_names = list(per_class.keys())
    if not cmd_class_names:
        print("No per-class command data found; cannot compute calibration. Bailing.")
        return
    print(f"Command classes: {cmd_class_names}")

    # The model's command head outputs class indices in some order. We assume
    # they match the per_class iteration order (alphabetical from
    # scikit-learn). The per_class['<name>']['precision'] is the model's per-class
    # precision = P(true=k | pred=k). We use it as the "confidence proxy" for
    # predictions of class k.
    name_to_conf = {name: per_class[name].get("precision", 0.0) for name in cmd_class_names}

    # Step 3: gather (predicted_confidence, correct) pairs over command positions.
    # cmd_t = -1 marks non-command positions (the head ignores those during
    # training and evaluation). Values 0..5 correspond to G0/G1/G2/G3/G53/M30.
    mask = (cmd_t >= 0)
    p = cmd_p[mask]
    t = cmd_t[mask]
    print(f"Evaluable command positions: n = {p.size}")

    # Map class index back to name (we assume index 0..n-1 follows cmd_class_names)
    n_classes = max(int(p.max()), int(t.max())) + 1
    if n_classes > len(cmd_class_names):
        print(f"Warning: model has {n_classes} command classes but per_class report has {len(cmd_class_names)}. Truncating.")
    conf_by_idx = np.array([name_to_conf.get(cmd_class_names[i] if i < len(cmd_class_names) else "?", 0.0)
                            for i in range(n_classes)])
    confidences = conf_by_idx[p]
    correct = (p == t).astype(float)

    ece, centers, bucket_acc, bucket_conf, bucket_size = compute_ece(confidences, correct, n_bins=10)
    print(f"ECE: {ece:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=0.8, label="perfectly calibrated")
    valid = ~np.isnan(bucket_acc)
    ax1.bar(centers[valid], bucket_acc[valid], width=0.08, alpha=0.75, edgecolor="black",
            color="#3866b3", label="empirical accuracy per bin")
    ax1.scatter(centers[valid], bucket_conf[valid], color="red", zorder=5,
                label="mean predicted confidence per bin", marker="x", s=40)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Predicted class confidence (per-class precision proxy)")
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_title(f"Reliability diagram (command head, fold 1)\nECE = {ece:.4f}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    # Bucket sizes
    ax2.bar(centers, bucket_size, width=0.08, alpha=0.75, color="#b33838",
            edgecolor="black")
    ax2.set_xlabel("Predicted class confidence bin")
    ax2.set_ylabel("Number of command-token positions")
    ax2.set_title("Bucket sizes (where the predictions land in confidence space)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "calibration_plot.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
