#!/usr/bin/env python3
"""Cross-fold post-hoc temperature scaling.

Refits T on each fold's calibration half independently, evaluates ECE
on the held-out half, and reports mean ± std of T_fit, ECE_pre, ECE_post
across the 5 folds. This is the cross-fold version of compute_ece_calibration.py
+ temperature_scaling.py.

Output: audit/calibration_cross_fold.json
"""
from __future__ import annotations

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


def compute_ece(confidences, predictions, labels, n_bins=N_BINS):
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


def fit_temperature(logits_cal, labels_cal):
    T = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_cal / T.clamp(min=1e-3), labels_cal)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def process_fold(fold: int):
    device = torch.device("cpu")
    base = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / f"fold_{fold}"
    # Find canonical fold dir (not _fsm)
    dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.endswith("_fsm")
            and (d / "decoder_checkpoint" / "best_decoder.pt").exists()]
    if not dirs:
        return None
    base = dirs[0]

    memory = torch.load(base / "encoder_memory" / "test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(base / "encoder_memory" / "test_op_pred.pt", map_location=device, weights_only=False)
    data_dir = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window" / f"fold_{fold}"
    d = np.load(data_dir / "test_sequences.npz", allow_pickle=True)
    tokens = torch.from_numpy(d["tokens"]).long().to(device)
    n_test, T_len = tokens.shape

    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    blob = torch.load(base / "decoder_checkpoint" / "best_decoder.pt", map_location=device, weights_only=False)
    a = blob["args"]
    vocab = json.load(open(REPO / "data" / "gcode_vocab_v8.json"))["vocab"]
    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab), sensor_dim=memory.shape[-1],
        d_model=a.get("d_model", 384), n_layers=a.get("n_layers", 8),
        n_heads=a.get("n_heads", 12), dropout=0.0,
        max_seq_len=a.get("max_token_len", 1400), n_operations=9,
    ).to(device)
    decoder.set_vocab(vocab)
    sd = blob["decoder_state_dict"]
    ms = decoder.state_dict()
    decoder.load_state_dict({k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}, strict=False)
    decoder.eval()

    pred_npz = np.load(base / "results" / "predictions.npz", allow_pickle=True)
    cmd_t = torch.from_numpy(pred_npz["cmd_t"]).long().to(device)
    type_t = torch.from_numpy(pred_npz["type_t"]).long().to(device)
    pt_t = torch.from_numpy(pred_npz["pt_t"]).long().to(device)

    PAD = 0
    head_logits = {"command": [], "type": [], "param_type": []}
    head_labels = {"command": [], "type": [], "param_type": []}

    BATCH = 4
    with torch.no_grad():
        for i in range(0, n_test, BATCH):
            end = min(i + BATCH, n_test)
            in_tok = tokens[i:end, :-1]
            padding = (in_tok == PAD)
            out = decoder(tokens=in_tok, sensor_embeddings=memory[i:end],
                          operation_type=op_pred[i:end], tgt_key_padding_mask=padding)
            for head_name, logit_key, target_t in [
                ("command", "command_logits", cmd_t[i:end, 1:]),
                ("type", "type_logits", type_t[i:end, 1:]),
                ("param_type", "param_type_logits", pt_t[i:end, 1:]),
            ]:
                logits = out[logit_key]
                L = min(logits.shape[1], target_t.shape[1])
                mask = target_t[:, :L] >= 0
                flat_logits = logits[:, :L].reshape(-1, logits.shape[-1])
                flat_labels = target_t[:, :L].reshape(-1)
                flat_mask = mask.reshape(-1)
                head_logits[head_name].append(flat_logits[flat_mask].cpu())
                head_labels[head_name].append(flat_labels[flat_mask].cpu())

    rng = np.random.default_rng(SEED)
    out = {"fold": fold, "n_test_windows": int(n_test)}
    for h in head_logits:
        Lg = torch.cat(head_logits[h], dim=0)
        Yl = torch.cat(head_labels[h], dim=0)
        N = Lg.shape[0]
        idx = rng.permutation(N)
        half = N // 2
        cal_idx = torch.from_numpy(idx[:half]).long()
        eval_idx = torch.from_numpy(idx[half:]).long()

        probs_pre = F.softmax(Lg[eval_idx], dim=-1)
        conf_pre = probs_pre.max(dim=-1).values.numpy()
        pred_pre = probs_pre.argmax(dim=-1).numpy()
        ece_pre, mce_pre = compute_ece(conf_pre, pred_pre, Yl[eval_idx].numpy())

        T_fit = fit_temperature(Lg[cal_idx], Yl[cal_idx])

        probs_post = F.softmax(Lg[eval_idx] / T_fit, dim=-1)
        conf_post = probs_post.max(dim=-1).values.numpy()
        pred_post = probs_post.argmax(dim=-1).numpy()
        ece_post, mce_post = compute_ece(conf_post, pred_post, Yl[eval_idx].numpy())

        out[h] = {
            "n_eval": int(N - half),
            "T_fit": float(T_fit),
            "ece_pre": float(ece_pre),
            "mce_pre": float(mce_pre),
            "ece_post": float(ece_post),
            "mce_post": float(mce_post),
            "accuracy_eval": float((pred_pre == Yl[eval_idx].numpy()).mean()),
        }
    return out


def main():
    all_folds = []
    for f in range(1, 6):
        print(f"=== fold {f} ===")
        r = process_fold(f)
        if r is None:
            print(f"  no checkpoint found, skip")
            continue
        all_folds.append(r)
        for h in ["command", "type", "param_type"]:
            v = r[h]
            print(f"  {h:12s} T={v['T_fit']:.3f}  "
                  f"ECE {v['ece_pre']:.4f}→{v['ece_post']:.4f}  "
                  f"MCE {v['mce_pre']:.4f}→{v['mce_post']:.4f}")

    # Aggregate
    agg = {}
    for h in ["command", "type", "param_type"]:
        Ts = np.array([r[h]["T_fit"] for r in all_folds])
        eces_pre = np.array([r[h]["ece_pre"] for r in all_folds])
        eces_post = np.array([r[h]["ece_post"] for r in all_folds])
        mces_pre = np.array([r[h]["mce_pre"] for r in all_folds])
        mces_post = np.array([r[h]["mce_post"] for r in all_folds])
        agg[h] = {
            "n_folds": len(all_folds),
            "T_mean": float(Ts.mean()),
            "T_std": float(Ts.std()),
            "ece_pre_mean": float(eces_pre.mean()),
            "ece_pre_std": float(eces_pre.std()),
            "ece_post_mean": float(eces_post.mean()),
            "ece_post_std": float(eces_post.std()),
            "mce_pre_mean": float(mces_pre.mean()),
            "mce_post_mean": float(mces_post.mean()),
            "ece_reduction_mean_pp": float(100 * (eces_pre - eces_post).mean()),
        }

    print("\n=== AGGREGATE (5-fold) ===")
    for h, a in agg.items():
        print(f"  {h:12s} T={a['T_mean']:.3f}±{a['T_std']:.3f}  "
              f"ECE pre {a['ece_pre_mean']:.4f}±{a['ece_pre_std']:.4f}  "
              f"ECE post {a['ece_post_mean']:.4f}±{a['ece_post_std']:.4f}  "
              f"Δ={a['ece_reduction_mean_pp']:+.2f}pp")

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "calibration_cross_fold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_fold": all_folds, "aggregate": agg}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
