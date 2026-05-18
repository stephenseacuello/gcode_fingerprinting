#!/usr/bin/env python3
"""Per-source-file performance variance.

We can't fire 120-way leave-one-file-out (~60 GPU-hours), but we already
have the 5-fold test predictions and the source_file metadata. This
script slices the existing 5-fold test predictions by source file and
reports per-file accuracy + cross-file variance for command + token +
numeric heads. Lets us answer: "is the model's accuracy roughly uniform
across files, or does it concentrate on a small set of memorisable files?"

Reads predictions.npz from full_window 5-fold + per_row 5-fold and
joins to source_file labels from the corresponding NPZ.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def score_file_level(mode: str = "full_window"):
    out_rows = []
    sweep_root = REPO / "outputs" / "decoder20260511" / "checkpoints" / f"{mode}_5fold"
    for F in range(1, 6):
        # Find the canonical baseline fold dir (not _fsm)
        cands = [c for c in sweep_root.glob(f"fold_{F}/*/results/predictions.npz")
                 if "_fsm" not in str(c)]
        if not cands:
            print(f"  fold {F}: no predictions.npz, skip")
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        pred_npz = np.load(cands[0], allow_pickle=True)

        data_dir = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / mode / f"fold_{F}"
        d = np.load(data_dir / "test_sequences.npz", allow_pickle=True)
        source_files = d["source_file"]
        N = len(source_files)

        # Per-sample command accuracy: at COMMAND positions
        cmd_p = pred_npz["cmd_p"]  # (N, T)
        cmd_t = pred_npz["cmd_t"]  # (N, T)
        type_t = pred_npz["type_t"]  # to find COMMAND positions
        pt_p = pred_npz["pt_p"]
        pt_t = pred_npz["pt_t"]
        # token-level accuracy from pred_tokens / target_tokens
        tok_p = pred_npz["pred_tokens"]
        tok_t = pred_npz["target_tokens"]

        L = min(cmd_p.shape[1], cmd_t.shape[1], type_t.shape[1])
        # per-sample command accuracy: correct command predictions / total command positions
        per_sample = []
        for i in range(N):
            cmd_mask = (cmd_t[i, :L] >= 0)  # valid command positions
            if cmd_mask.sum() > 0:
                cmd_acc = float(((cmd_p[i, :L] == cmd_t[i, :L]) & cmd_mask).sum() / cmd_mask.sum())
            else:
                cmd_acc = float("nan")
            tok_mask = (tok_t[i, :L] >= 0)
            tok_acc = float(((tok_p[i, :L] == tok_t[i, :L]) & tok_mask).sum() / max(tok_mask.sum(), 1))
            pt_mask = (pt_t[i, :L] >= 0)
            if pt_mask.sum() > 0:
                pt_acc = float(((pt_p[i, :L] == pt_t[i, :L]) & pt_mask).sum() / pt_mask.sum())
            else:
                pt_acc = float("nan")
            per_sample.append({
                "fold": F,
                "source_file": str(source_files[i]),
                "command_acc": cmd_acc,
                "token_acc": tok_acc,
                "param_type_acc": pt_acc,
            })
        out_rows.extend(per_sample)

    # Aggregate by source file
    by_file = defaultdict(list)
    for r in out_rows:
        by_file[r["source_file"]].append(r)

    per_file_stats = []
    for fn, rows in by_file.items():
        cmd_accs = np.array([r["command_acc"] for r in rows if not np.isnan(r["command_acc"])])
        tok_accs = np.array([r["token_acc"] for r in rows if not np.isnan(r["token_acc"])])
        pt_accs = np.array([r["param_type_acc"] for r in rows if not np.isnan(r["param_type_acc"])])
        per_file_stats.append({
            "source_file": fn,
            "n_windows": int(len(rows)),
            "command_acc_mean": float(cmd_accs.mean()) if cmd_accs.size > 0 else float("nan"),
            "token_acc_mean": float(tok_accs.mean()) if tok_accs.size > 0 else float("nan"),
            "param_type_acc_mean": float(pt_accs.mean()) if pt_accs.size > 0 else float("nan"),
        })

    # Cross-file aggregate
    cmd_means = np.array([r["command_acc_mean"] for r in per_file_stats if not np.isnan(r["command_acc_mean"])])
    tok_means = np.array([r["token_acc_mean"] for r in per_file_stats if not np.isnan(r["token_acc_mean"])])
    pt_means = np.array([r["param_type_acc_mean"] for r in per_file_stats if not np.isnan(r["param_type_acc_mean"])])

    summary = {
        "mode": mode,
        "n_files": len(per_file_stats),
        "command_acc_mean_across_files": float(cmd_means.mean()),
        "command_acc_std_across_files": float(cmd_means.std()),
        "command_acc_min_across_files": float(cmd_means.min()),
        "command_acc_max_across_files": float(cmd_means.max()),
        "command_acc_p10_p90": [float(np.percentile(cmd_means, 10)), float(np.percentile(cmd_means, 90))],
        "token_acc_mean_across_files": float(tok_means.mean()),
        "token_acc_std_across_files": float(tok_means.std()),
        "token_acc_min_across_files": float(tok_means.min()),
        "token_acc_max_across_files": float(tok_means.max()),
        "param_type_acc_mean_across_files": float(pt_means.mean()),
        "param_type_acc_std_across_files": float(pt_means.std()),
    }
    # Top-5 best and worst files for command
    sorted_files = sorted(per_file_stats, key=lambda r: r["command_acc_mean"] if not np.isnan(r["command_acc_mean"]) else 0)
    summary["top5_worst_command"] = sorted_files[:5]
    summary["top5_best_command"] = sorted_files[-5:][::-1]

    return summary, per_file_stats


def main():
    out = {}
    for mode in ["full_window", "per_row"]:
        try:
            summary, per_file = score_file_level(mode)
            out[mode] = {"summary": summary, "per_file": per_file}
            print(f"\n=== {mode} (n_files={summary['n_files']}) ===")
            print(f"  command_acc across files: {summary['command_acc_mean_across_files']:.4f} ± "
                  f"{summary['command_acc_std_across_files']:.4f}, "
                  f"range [{summary['command_acc_min_across_files']:.3f}, "
                  f"{summary['command_acc_max_across_files']:.3f}], "
                  f"P10/P90 = [{summary['command_acc_p10_p90'][0]:.3f}, "
                  f"{summary['command_acc_p10_p90'][1]:.3f}]")
            print(f"  token_acc across files: {summary['token_acc_mean_across_files']:.4f} ± "
                  f"{summary['token_acc_std_across_files']:.4f}, "
                  f"range [{summary['token_acc_min_across_files']:.3f}, "
                  f"{summary['token_acc_max_across_files']:.3f}]")
            print(f"  param_type_acc across files: {summary['param_type_acc_mean_across_files']:.4f}")
        except FileNotFoundError as e:
            print(f"  {mode}: SKIP ({e})")

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "per_file_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
