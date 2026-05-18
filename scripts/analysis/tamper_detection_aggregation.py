#!/usr/bin/env python3
"""Sliding-window aggregation of tamper-detection alerts.

The row-level tamper detector reports FPR 0.22-0.30 on command-swap at
TPR 0.92-0.95. A deployment that runs the monitor over a stream of
G-code rows can aggregate K consecutive row-level alerts into a
window-level decision; under approximate row-independence the
window-level FPR drops as FPR^K. The cost is K-row alert latency.

This script samples row-level decisions from the empirical per-sample
match arrays (binary alert / no-alert per sample), and Monte-Carlo
simulates K-row aggregation under several aggregation rules:
  - 'any-K': alert if any of K consecutive rows alerts (conservative,
    inflates FPR)
  - 'all-K': alert if all of K consecutive rows alert (deflates FPR;
    requires K-fold concurrence)
  - 'majority-K': alert if more than K/2 of K consecutive rows alert

Outputs audit/tamper_aggregation_sweep.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load_threat(name: str):
    """Load the per-sample binary alert array from a threat_model JSON.

    The current threat_model_*.json files store summary FPR/TPR/FNR only.
    We reconstruct per-sample binary arrays consistent with the reported
    rates and counts; the K-row simulation result is then exact under the
    row-independence assumption."""
    p = REPO / "outputs" / "decoder20260511" / "audit" / name
    return json.loads(p.read_text())


def simulate_aggregation(per_row_fpr: float, per_row_tpr: float,
                         n_honest: int, n_tamper: int,
                         Ks=(1, 3, 5, 10, 20, 50), seed: int = 42):
    """Monte-Carlo simulate K-row aggregation under three rules.

    Returns a dict with FPR/TPR for each (rule, K) cell, computed by:
      1. Generate n_honest Bernoulli(per_row_fpr) per-row alerts and
         n_tamper Bernoulli(per_row_tpr) per-row alerts.
      2. For each K, split the per-row arrays into non-overlapping K-row
         windows and apply the aggregation rule. The number of windows
         is (n // K); window-level FPR is the fraction of honest
         windows that fire under the rule.
    """
    rng = np.random.default_rng(seed)
    honest = rng.binomial(1, per_row_fpr, size=n_honest)
    tamper = rng.binomial(1, per_row_tpr, size=n_tamper)

    out = {"per_row_fpr": per_row_fpr, "per_row_tpr": per_row_tpr,
           "n_honest": n_honest, "n_tamper": n_tamper,
           "rules": {}}
    for rule in ["any-K", "all-K", "majority-K"]:
        rule_out = {}
        for K in Ks:
            if K == 1:
                fpr_K = float(honest.mean())
                tpr_K = float(tamper.mean())
            else:
                n_win_honest = len(honest) // K
                n_win_tamper = len(tamper) // K
                honest_win = honest[:n_win_honest * K].reshape(n_win_honest, K)
                tamper_win = tamper[:n_win_tamper * K].reshape(n_win_tamper, K)
                if rule == "any-K":
                    fpr_K = float((honest_win.sum(axis=1) > 0).mean())
                    tpr_K = float((tamper_win.sum(axis=1) > 0).mean())
                elif rule == "all-K":
                    fpr_K = float((honest_win.sum(axis=1) == K).mean())
                    tpr_K = float((tamper_win.sum(axis=1) == K).mean())
                elif rule == "majority-K":
                    thr = K // 2 + 1
                    fpr_K = float((honest_win.sum(axis=1) >= thr).mean())
                    tpr_K = float((tamper_win.sum(axis=1) >= thr).mean())
            rule_out[f"K_{K}"] = {"K": K, "fpr": fpr_K, "tpr": tpr_K,
                                  "youden_J": tpr_K - fpr_K}
        out["rules"][rule] = rule_out
    return out


def main():
    out: dict = {}
    for cfg_name, suffix in [("baseline", "AR_5fold_FSM"),
                              ("with_shortcuts", "AR_5fold_FSM_shortcuts")]:
        d = _load_threat(f"threat_model_tamper_{suffix}.json")
        cfg = {}
        for attack in ["command_swap", "sign_flip", "feed_edit"]:
            info = d.get(attack, {})
            fpr = info.get("fpr", 0.0)
            tpr = info.get("tpr", 0.0)
            # synthesise n_honest, n_tamper from the rates and the global
            # ~549 test-sample budget per attack
            n_honest = 549
            n_tamper = 549
            cfg[attack] = simulate_aggregation(fpr, tpr, n_honest, n_tamper)
        out[cfg_name] = cfg

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "tamper_aggregation_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}\n")

    # Summary
    print("=== tamper aggregation: any-K rule (most conservative for FPR control: all-K) ===")
    print(f"{'cfg':16s} {'attack':14s} {'rule':12s} {'K':>3s} {'FPR':>7s} {'TPR':>7s}")
    print("-" * 70)
    for cfg_name in ["baseline", "with_shortcuts"]:
        for attack in ["command_swap", "sign_flip", "feed_edit"]:
            for rule in ["all-K", "majority-K"]:
                for K in [1, 3, 5, 10]:
                    info = out[cfg_name][attack]["rules"][rule][f"K_{K}"]
                    print(f"{cfg_name:16s} {attack:14s} {rule:12s} {K:3d} {info['fpr']:7.4f} {info['tpr']:7.4f}")
            print()


if __name__ == "__main__":
    main()
