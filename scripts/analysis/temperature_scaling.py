#!/usr/bin/env python3
"""Post-hoc temperature scaling for the categorical heads (Guo et al. 2017).

Pipeline:
  1. Run decoder teacher-forced on fold-1 test set and cache logits per head.
  2. Split the cached positions into a 50/50 calibration / evaluation set
     (random with seed=42).
  3. Per head: fit T > 0 by minimising NLL on the calibration half via LBFGS.
  4. Apply the fit T to logits on the evaluation half and recompute ECE.

Output:
  audit/calibration_temp_scaled.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS = 10
SEED = 42


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    correct = (predictions == labels).astype(np.float64)
    ece = 0.0
    mce = 0.0
    n_total = len(confidences)
    if n_total == 0:
        return 0.0, 0.0
    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (n_bin / n_total) * gap
        mce = max(mce, gap)
    return ece, mce


def fit_temperature(logits_cal: torch.Tensor, labels_cal: torch.Tensor) -> float:
    """Fit a single scalar T > 0 minimising NLL on calibration set."""
    T = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits_cal / T.clamp(min=1e-3), labels_cal)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-json", type=Path,
                   default=REPO / "outputs" / "decoder20260511" / "audit" / "calibration_temp_scaled.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base_dir = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / "fold_1" / "6o90io5p"
    memory = torch.load(base_dir / "encoder_memory" / "test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(base_dir / "encoder_memory" / "test_op_pred.pt", map_location=device, weights_only=False)
    data_dir = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window" / "fold_1"
    d = np.load(data_dir / "test_sequences.npz", allow_pickle=True)
    tokens = torch.from_numpy(d["tokens"]).long().to(device)
    n_test, T_len = tokens.shape
    print(f"n_test={n_test}, T={T_len}, memory={memory.shape}, op_pred={op_pred.shape}")

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
    # Collect full logits + labels per head, masked
    head_logits = {"command": [], "type": [], "param_type": []}
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
                L = min(logits.shape[1], target_t.shape[1])
                mask = target_t[:, :L] >= 0
                # Flatten (batch, position) into one dim, keeping (vocab) dim
                flat_logits = logits[:, :L].reshape(-1, logits.shape[-1])
                flat_labels = target_t[:, :L].reshape(-1)
                flat_mask = mask.reshape(-1)
                head_logits[head_name].append(flat_logits[flat_mask].cpu())
                head_labels[head_name].append(flat_labels[flat_mask].cpu())

    results = {}
    rng = np.random.default_rng(SEED)
    for h in head_logits:
        L = torch.cat(head_logits[h], dim=0)
        Y = torch.cat(head_labels[h], dim=0)
        N = L.shape[0]
        idx = rng.permutation(N)
        half = N // 2
        cal_idx = torch.from_numpy(idx[:half]).long()
        eval_idx = torch.from_numpy(idx[half:]).long()
        L_cal = L[cal_idx]
        Y_cal = Y[cal_idx]
        L_eval = L[eval_idx]
        Y_eval = Y[eval_idx]

        # Uncalibrated ECE on eval half
        probs_eval = F.softmax(L_eval, dim=-1)
        conf_eval = probs_eval.max(dim=-1).values.numpy()
        pred_eval = probs_eval.argmax(dim=-1).numpy()
        ece_pre, mce_pre = compute_ece(conf_eval, pred_eval, Y_eval.numpy())
        acc_eval = float((pred_eval == Y_eval.numpy()).mean())

        # Fit T on calibration half
        T_fit = fit_temperature(L_cal, Y_cal)

        # Apply T to eval half
        probs_post = F.softmax(L_eval / T_fit, dim=-1)
        conf_post = probs_post.max(dim=-1).values.numpy()
        pred_post = probs_post.argmax(dim=-1).numpy()
        ece_post, mce_post = compute_ece(conf_post, pred_post, Y_eval.numpy())

        results[h] = {
            "n_total": int(N),
            "n_cal": int(half),
            "n_eval": int(N - half),
            "accuracy_eval": acc_eval,
            "ece_pre": float(ece_pre),
            "mce_pre": float(mce_pre),
            "mean_conf_pre": float(conf_eval.mean()),
            "T_fit": float(T_fit),
            "ece_post": float(ece_post),
            "mce_post": float(mce_post),
            "mean_conf_post": float(conf_post.mean()),
            "ece_reduction_pp": float(100 * (ece_pre - ece_post)),
            "mce_reduction_pp": float(100 * (mce_pre - mce_post)),
        }
        print(f"  {h:12s}: acc={acc_eval:.4f}  T={T_fit:.3f}")
        print(f"               pre:  ECE={ece_pre:.4f}  MCE={mce_pre:.4f}  mean_conf={conf_eval.mean():.4f}")
        print(f"               post: ECE={ece_post:.4f}  MCE={mce_post:.4f}  mean_conf={conf_post.mean():.4f}")
        print(f"               Δ:    ECE {ece_pre - ece_post:+.4f}  MCE {mce_pre - mce_post:+.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
