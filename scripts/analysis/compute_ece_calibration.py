#!/usr/bin/env python3
"""Expected Calibration Error (ECE) for each categorical head.

Loads fold-1 checkpoint, runs decoder teacher-forced, dumps softmax probabilities
for the command / type / param_type / sign heads, computes ECE with 10 bins +
maximum calibration error (MCE) + per-bin accuracy/confidence/count.

Outputs:
  audit/calibration_fold1.json     -- ECE/MCE per head + per-bin breakdown
  decoder_paper_v2/figures/reliability_diagram.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

N_BINS = 10


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Standard ECE: weighted average of |accuracy - confidence| per bin."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    correct = (predictions == labels).astype(np.float64)
    ece = 0.0
    mce = 0.0
    per_bin = []
    n_total = len(confidences)
    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            per_bin.append({"bin_low": float(lo), "bin_high": float(hi),
                            "count": 0, "accuracy": None, "confidence": None,
                            "gap": None})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (n_bin / n_total) * gap
        mce = max(mce, gap)
        per_bin.append({"bin_low": float(lo), "bin_high": float(hi),
                        "count": n_bin, "accuracy": bin_acc, "confidence": bin_conf,
                        "gap": gap})
    return ece, mce, per_bin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-json", type=Path,
                   default=REPO / "outputs" / "decoder20260511" / "audit" / "calibration_fold1.json")
    p.add_argument("--output-fig", type=Path,
                   default=REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures" / "reliability_diagram.pdf")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base_dir = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / "fold_1" / "6o90io5p"
    memory = torch.load(base_dir / "encoder_memory" / "test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(base_dir / "encoder_memory" / "test_op_pred.pt", map_location=device, weights_only=False)
    data_dir = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window" / "fold_1"
    d = np.load(data_dir / "test_sequences.npz", allow_pickle=True)
    tokens = torch.from_numpy(d["tokens"]).long().to(device)
    n_test, T = tokens.shape
    print(f"n_test={n_test}, T={T}, memory={memory.shape}, op_pred={op_pred.shape}")

    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    dec_ckpt_path = base_dir / "decoder_checkpoint" / "best_decoder.pt"
    blob = torch.load(dec_ckpt_path, map_location=device, weights_only=False)
    a = blob["args"]
    vocab = json.load(open(REPO / "data" / "gcode_vocab_v8.json"))["vocab"]
    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab),
        sensor_dim=memory.shape[-1],
        d_model=a.get("d_model", 384),
        n_layers=a.get("n_layers", 8),
        n_heads=a.get("n_heads", 12),
        dropout=0.0,
        max_seq_len=a.get("max_token_len", 1400),
        n_operations=9,
    ).to(device)
    decoder.set_vocab(vocab)
    sd = blob["decoder_state_dict"]
    ms = decoder.state_dict()
    decoder.load_state_dict({k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}, strict=False)
    decoder.eval()

    pred_npz = np.load(base_dir / "results" / "predictions.npz", allow_pickle=True)
    cmd_t = torch.from_numpy(pred_npz["cmd_t"]).long().to(device)
    type_t = torch.from_numpy(pred_npz["type_t"]).long().to(device)
    pt_t = torch.from_numpy(pred_npz["pt_t"]).long().to(device)

    PAD = 0
    head_confs = {"command": [], "type": [], "param_type": []}
    head_preds = {"command": [], "type": [], "param_type": []}
    head_labels = {"command": [], "type": [], "param_type": []}

    BATCH = 4
    with torch.no_grad():
        for i in range(0, n_test, BATCH):
            end = min(i + BATCH, n_test)
            in_tok = tokens[i:end, :-1]
            padding = (in_tok == PAD)
            out = decoder(
                tokens=in_tok,
                sensor_embeddings=memory[i:end],
                operation_type=op_pred[i:end],
                tgt_key_padding_mask=padding,
            )
            for head_name, logit_key, target_t in [
                ("command", "command_logits", cmd_t[i:end, 1:]),
                ("type", "type_logits", type_t[i:end, 1:]),
                ("param_type", "param_type_logits", pt_t[i:end, 1:]),
            ]:
                logits = out[logit_key]
                probs = F.softmax(logits, dim=-1)
                top_p, top_idx = probs.max(dim=-1)
                L = min(top_p.shape[1], target_t.shape[1])
                mask = target_t[:, :L] >= 0
                head_confs[head_name].append(top_p[:, :L][mask].cpu().numpy())
                head_preds[head_name].append(top_idx[:, :L][mask].cpu().numpy())
                head_labels[head_name].append(target_t[:, :L][mask].cpu().numpy())

    results = {}
    for h in head_confs:
        c = np.concatenate(head_confs[h])
        p_ = np.concatenate(head_preds[h])
        l_ = np.concatenate(head_labels[h])
        ece, mce, per_bin = compute_ece(c, p_, l_)
        acc = float((p_ == l_).mean())
        results[h] = {"ece": float(ece), "mce": float(mce), "accuracy": acc,
                      "n": int(len(c)), "mean_confidence": float(c.mean()),
                      "per_bin": per_bin}
        print(f"  {h:12s}: acc={acc:.4f}  ECE={ece:.4f}  MCE={mce:.4f}  "
              f"mean_conf={c.mean():.4f}  n={len(c)}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output_json}")

    # Reliability diagram
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = {"command": "#3866b3", "type": "#b33838", "param_type": "#38b376"}
    for ax, h in zip(axes, ["command", "type", "param_type"]):
        bins = results[h]["per_bin"]
        xs = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins if b["accuracy"] is not None]
        ys_acc = [b["accuracy"] for b in bins if b["accuracy"] is not None]
        ys_conf = [b["confidence"] for b in bins if b["confidence"] is not None]
        counts = [b["count"] for b in bins if b["accuracy"] is not None]
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect")
        widths = 0.09
        ax.bar(xs, ys_acc, width=widths, color=colors[h], alpha=0.7, label="accuracy")
        ax.scatter(xs, ys_conf, color="black", marker="x", s=40, label="mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{h} head\nECE={results[h]['ece']:.3f}, MCE={results[h]['mce']:.3f}")
        ax.grid(alpha=0.3)
        if h == "command":
            ax.legend(loc="upper left")
    plt.tight_layout()
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_fig, bbox_inches="tight")
    plt.savefig(args.output_fig.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {args.output_fig}")


if __name__ == "__main__":
    main()
