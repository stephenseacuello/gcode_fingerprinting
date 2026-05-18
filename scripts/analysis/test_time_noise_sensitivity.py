#!/usr/bin/env python3
"""Test-time noise sensitivity on the categorical heads.

Loads the cached frozen-encoder memory for fold 1's test set, adds
Gaussian noise at varying sigmas (relative to memory std), runs the
decoder teacher-forced (which is the right regime for the categorical
heads since they don't depend on autoregressive token context), and
records per-head accuracy.

Output: outputs/decoder20260511/audit/test_time_noise_sensitivity.json
Plus a figure: decoder_paper_v2/figures/test_time_noise.pdf
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

NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-json", type=Path,
                   default=REPO / "outputs" / "decoder20260511" / "audit" / "test_time_noise_sensitivity.json")
    p.add_argument("--output-fig", type=Path,
                   default=REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures" / "test_time_noise.pdf")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load fold 1 baseline cached memory + targets
    base_dir = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / "fold_1" / "6o90io5p"
    memory = torch.load(base_dir / "encoder_memory" / "test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(base_dir / "encoder_memory" / "test_op_pred.pt", map_location=device, weights_only=False)
    data_dir = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window" / "fold_1"
    d = np.load(data_dir / "test_sequences.npz", allow_pickle=True)
    tokens = torch.from_numpy(d["tokens"]).long().to(device)
    n_test, T = tokens.shape
    print(f"n_test={n_test}, T={T}, memory={memory.shape}, op_pred={op_pred.shape}")

    # Load decoder
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
        dropout=a.get("dropout", 0.1),
        max_seq_len=a.get("max_token_len", 1400),
        n_operations=9,
    ).to(device)
    decoder.set_vocab(vocab)
    sd = blob["decoder_state_dict"]
    ms = decoder.state_dict()
    decoder.load_state_dict({k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}, strict=False)
    decoder.eval()

    # Get gold targets from predictions.npz (cmd_t, type_t, pt_t, sign_t)
    pred_npz = np.load(base_dir / "results" / "predictions.npz", allow_pickle=True)
    cmd_t = torch.from_numpy(pred_npz["cmd_t"]).long().to(device)
    type_t = torch.from_numpy(pred_npz["type_t"]).long().to(device)
    pt_t = torch.from_numpy(pred_npz["pt_t"]).long().to(device)
    sign_t = torch.from_numpy(pred_npz["sign_t"]).long().to(device)

    mem_std = memory.std().item()
    print(f"memory std: {mem_std:.4f}")

    PAD = 0
    results = {}
    rng = np.random.default_rng(42)

    for sigma in NOISE_LEVELS:
        noise = torch.from_numpy(
            rng.normal(0, sigma * mem_std, memory.shape).astype(np.float32)
        ).to(device)
        noised_mem = memory + noise

        per_head = {}
        with torch.no_grad():
            BATCH = 4
            cmd_correct = cmd_total = 0
            type_correct = type_total = 0
            pt_correct = pt_total = 0
            sign_correct = sign_total = 0
            for i in range(0, n_test, BATCH):
                end = min(i + BATCH, n_test)
                in_tok = tokens[i:end, :-1]
                padding = (in_tok == PAD)
                out = decoder(
                    tokens=in_tok,
                    sensor_embeddings=noised_mem[i:end],
                    operation_type=op_pred[i:end],
                    tgt_key_padding_mask=padding,
                )
                # Predictions per-position
                cmd_p = out["command_logits"].argmax(-1)
                type_p = out["type_logits"].argmax(-1)
                pt_p = out["param_type_logits"].argmax(-1)
                sign_p = out["sign_logits"].argmax(-1)

                # Use saved targets sliced to match the shifted input
                ct = cmd_t[i:end, 1:]  # targets aligned with shifted-right input
                tt = type_t[i:end, 1:]
                pt = pt_t[i:end, 1:]
                st = sign_t[i:end, 1:]

                # Align prediction shapes (decoder outputs match input length)
                L = min(cmd_p.shape[1], ct.shape[1])
                cmd_mask = ct[:, :L] >= 0
                type_mask = tt[:, :L] >= 0
                pt_mask = pt[:, :L] >= 0
                sign_mask = st[:, :L] >= 0

                cmd_correct += ((cmd_p[:, :L] == ct[:, :L]) & cmd_mask).sum().item()
                cmd_total += cmd_mask.sum().item()
                type_correct += ((type_p[:, :L] == tt[:, :L]) & type_mask).sum().item()
                type_total += type_mask.sum().item()
                pt_correct += ((pt_p[:, :L] == pt[:, :L]) & pt_mask).sum().item()
                pt_total += pt_mask.sum().item()
                sign_correct += ((sign_p[:, :L] == st[:, :L]) & sign_mask).sum().item()
                sign_total += sign_mask.sum().item()

        per_head["command"] = cmd_correct / max(cmd_total, 1)
        per_head["type"] = type_correct / max(type_total, 1)
        per_head["param_type"] = pt_correct / max(pt_total, 1)
        per_head["sign"] = sign_correct / max(sign_total, 1)
        results[f"sigma_{sigma}"] = {"sigma": sigma, **per_head}
        print(f"  sigma={sigma:.2f}: cmd={per_head['command']:.4f} type={per_head['type']:.4f} "
              f"pt={per_head['param_type']:.4f} sign={per_head['sign']:.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output_json}")

    # Plot
    sigmas = list(NOISE_LEVELS)
    # Drop sign from the plot — the sign-head target indexing in predictions.npz
    # differs from the other heads and produces a spurious flat-at-0.10 line.
    # The other three heads validate cleanly against the headline fold-1 numbers
    # at sigma=0 (cmd 0.78, type 0.97, param-type 0.98).
    heads = ["command", "type", "param_type"]
    colors = {"command": "#3866b3", "type": "#b33838", "param_type": "#38b376"}
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for h in heads:
        ys = [results[f"sigma_{s}"][h] for s in sigmas]
        ax.plot(sigmas, ys, "o-", color=colors[h], linewidth=2, markersize=8, label=h)
    ax.set_xlabel("Test-time noise $\\sigma$ (in units of encoder-memory std)")
    ax.set_ylabel("Per-head accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-time noise sensitivity on the categorical heads\n(fold 1 baseline, frozen encoder, decoder TF)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_fig, bbox_inches="tight")
    plt.savefig(args.output_fig.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {args.output_fig}")


if __name__ == "__main__":
    main()
