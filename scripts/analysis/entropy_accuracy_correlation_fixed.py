#!/usr/bin/env python3
"""Recompute the entropy-vs-accuracy correlation using LIVE per-(head, op-class)
teacher-forcing accuracies (audit/per_op_class_v8_fixed.json) against the
per-(head, op-class) source entropy (audit/per_head_per_opclass_entropy.json).

Replaces the hardcoded headline-TF constants that generate_entropy_accuracy_scatter.py
fell back to. Reports Pearson r and Spearman rho for several cell sets:

  * 54-cell : all 6 heads x 9 op-classes
  * 45-cell : drop the composite `token` head (5 heads x 9)
  * 36-cell : drop `token` and the near-constant `type` head (4 heads x 9)
  * 30-cell : the 5 non-token heads x the 6 best-supported op-classes
              (face, face150025, pocket150025, adaptive150025, adaptive, damagepocket)
              -- a defensible-n subset that drops the smallest op-classes.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/seacuello/Documents/gcode_fingerprinting")
ENT = ROOT / "outputs/decoder20260511/audit/per_head_per_opclass_entropy.json"
ACC = ROOT / "outputs/decoder20260511/audit/per_op_class_v8_fixed.json"

ent = json.loads(ENT.read_text())
acc = json.loads(ACC.read_text())

heads = ent["heads"]                 # token,type,command,param_type,sign,digit
op_classes = ent["op_classes"]       # 9, sorted
ent_mat = ent["entropy_matrix"]      # [6][9]
plot = acc["aggregate_head_opclass"] # plot[head][op_class] -> pooled acc

# n_windows per op-class (support) for picking the best-supported subset.
nwin = {op: acc["aggregate"][op]["n_windows_total"] for op in op_classes
        if op in acc["aggregate"]}

# Build the full cell table.
cells = []  # (head, op, entropy, accuracy, n_windows)
for hi, h in enumerate(heads):
    for oi, op in enumerate(op_classes):
        a = plot.get(h, {}).get(op)
        if a is None or not (a == a):  # skip nan/missing
            continue
        cells.append((h, op, ent_mat[hi][oi], a, nwin.get(op, 0)))


def corr(subset, label):
    xs = np.array([c[2] for c in subset], dtype=float)
    ys = np.array([c[3] for c in subset], dtype=float)
    if len(xs) < 3:
        print(f"{label}: n={len(xs)} too few")
        return
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    print(f"{label:<58} n={len(xs):>2}  Pearson r={pr:+.3f} (p={pp:.2g})  "
          f"Spearman rho={sr:+.3f} (p={sp:.2g})")
    return {"n": len(xs), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp)}


results = {}
print("=" * 100)
print("LIVE entropy-vs-accuracy correlation (per-op-class TF accuracy, full_window_5fold)")
print("=" * 100)

results["all_54"] = corr(cells, "54-cell: all 6 heads x 9 op-classes")
results["no_token_45"] = corr([c for c in cells if c[0] != "token"],
                              "45-cell: drop composite token head (5 heads x 9)")
results["no_token_type_36"] = corr(
    [c for c in cells if c[0] not in ("token", "type")],
    "36-cell: drop token + near-constant type (4 heads x 9)")

# 30-cell: 5 non-token heads x 6 best-supported op-classes.
best6 = sorted(nwin, key=lambda o: -nwin[o])[:6]
sub30 = [c for c in cells if c[0] != "token" and c[1] in best6]
results["best6_30"] = corr(
    sub30, f"30-cell: 5 non-token heads x 6 best-supported op-classes")
print(f"          (best-6 op-classes by window support: {best6})")

# Also: numeric-only heads (sign+digit) x 9 = 18, and the 'recoverability' heads
results["numeric_18"] = corr(
    [c for c in cells if c[0] in ("sign", "digit")],
    "18-cell: numeric heads (sign, digit) x 9 op-classes")

# Per-head within-head correlation across op-classes (does entropy predict
# accuracy WITHIN each head as op-class varies?).
print("\nWithin-head (entropy vs accuracy across the 9 op-classes):")
within = {}
for h in heads:
    sub = [c for c in cells if c[0] == h]
    xs = np.array([c[2] for c in sub]); ys = np.array([c[3] for c in sub])
    if len(xs) >= 3 and xs.std() > 1e-9 and ys.std() > 1e-9:
        pr, pp = stats.pearsonr(xs, ys)
        sr, sp = stats.spearmanr(xs, ys)
        print(f"  {h:<12} n={len(xs)}  Pearson r={pr:+.3f} (p={pp:.2g})  "
              f"Spearman rho={sr:+.3f} (p={sp:.2g})")
        within[h] = {"pearson_r": float(pr), "spearman_rho": float(sr),
                     "n": len(xs)}
    else:
        print(f"  {h:<12} n={len(xs)}  (degenerate: entropy or acc ~constant)")
        within[h] = {"note": "degenerate variance", "n": len(xs)}
results["within_head"] = within

out = ROOT / "outputs/decoder20260511/audit/entropy_accuracy_correlation_fixed.json"
out.write_text(json.dumps({"results": results,
                           "cells": [{"head": c[0], "op_class": c[1],
                                      "entropy_bits": c[2], "accuracy": c[3],
                                      "n_windows": c[4]} for c in cells]},
                          indent=2))
print(f"\nwrote {out}")
