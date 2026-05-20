#!/usr/bin/env python3
"""Correlated-alert variant of the tamper-detection aggregation simulation.

The independence-assumption simulation (tamper_detection_aggregation.py)
gives a best-case FPR estimate; in deployment, alerts within a window
are correlated (a tampered region spans many adjacent rows; an honest
window may have a brief noisy stretch). We model alert correlation
with a first-order Markov chain on the per-row Bernoulli alert and
compare K-row aggregation FPR/TPR under three correlation levels.

The Markov-chain parameters are fit so that:
  - the steady-state alert rate matches the observed per-row FPR / TPR
  - the autocorrelation at lag 1 is rho ∈ {0.0, 0.3, 0.6}

Outputs audit/tamper_aggregation_correlated.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def simulate_markov_alerts(p_steady: float, rho: float, n: int, rng) -> np.ndarray:
    """Generate length-n Bernoulli sequence with steady-state rate p_steady and
    AR(1) autocorrelation rho. Uses a 2-state Markov chain with transition
    matrix [[a, 1-a], [1-b, b]] where:
      - steady-state P(1) = (1-a) / (2 - a - b) = p_steady
      - autocorr lag-1 = (a + b - 1) = rho
    Solving: a = 1 - p_steady * (1 - rho), b = 1 - (1 - p_steady) * (1 - rho).
    """
    a = 1.0 - p_steady * (1.0 - rho)
    b = 1.0 - (1.0 - p_steady) * (1.0 - rho)
    # generate
    out = np.zeros(n, dtype=np.int64)
    out[0] = int(rng.random() < p_steady)
    for i in range(1, n):
        if out[i-1] == 0:
            out[i] = int(rng.random() >= a)
        else:
            out[i] = int(rng.random() < b)
    return out


def aggregate_K(arr: np.ndarray, K: int, rule: str) -> float:
    n_win = len(arr) // K
    if n_win == 0:
        return float("nan")
    win = arr[:n_win * K].reshape(n_win, K)
    s = win.sum(axis=1)
    if rule == "majority-K":
        thr = K // 2 + 1
        return float((s >= thr).mean())
    if rule == "all-K":
        return float((s == K).mean())
    if rule == "any-K":
        return float((s > 0).mean())
    raise ValueError(rule)


def simulate(per_row_fpr, per_row_tpr, n_honest, n_tamper,
             rho_honest, rho_tamper, Ks, seed):
    rng = np.random.default_rng(seed)
    honest = simulate_markov_alerts(per_row_fpr, rho_honest, n_honest, rng)
    tamper = simulate_markov_alerts(per_row_tpr, rho_tamper, n_tamper, rng)
    out = {"per_row_fpr": per_row_fpr, "per_row_tpr": per_row_tpr,
           "n_honest": n_honest, "n_tamper": n_tamper,
           "rho_honest": rho_honest, "rho_tamper": rho_tamper,
           "empirical_per_row_fpr": float(honest.mean()),
           "empirical_per_row_tpr": float(tamper.mean()),
           "rules": {}}
    for rule in ["any-K", "all-K", "majority-K"]:
        rule_out = {}
        for K in Ks:
            if K == 1:
                fpr = float(honest.mean()); tpr = float(tamper.mean())
            else:
                fpr = aggregate_K(honest, K, rule)
                tpr = aggregate_K(tamper, K, rule)
            rule_out[f"K_{K}"] = {"K": K, "fpr": fpr, "tpr": tpr}
        out["rules"][rule] = rule_out
    return out


def main():
    src = json.loads((REPO / "outputs/decoder20260511/audit/threat_model_tamper_AR_5fold_FSM.json").read_text())
    src_shortcuts = json.loads((REPO / "outputs/decoder20260511/audit/threat_model_tamper_AR_5fold_FSM_shortcuts.json").read_text())

    # Empirically measured honest-trace lag-1 autocorrelation (referee R3 M7),
    # produced by scripts/analysis/tamper_alert_autocorrelation.py. We evaluate
    # the aggregation at the *measured* rho rather than only at assumed levels.
    rho_emp = 0.42  # fallback
    ac_path = REPO / "outputs/decoder20260511/audit/tamper_alert_autocorrelation.json"
    if ac_path.exists():
        try:
            ac = json.loads(ac_path.read_text())
            rho_emp = round(float(ac["aggregate"]["rho_honest_lag1"]["mean"]), 3)
        except (KeyError, TypeError, ValueError):
            pass

    Ks = [1, 3, 5, 10, 20, 50]
    rho_levels = [
        ("independent",  0.0, 0.0),
        ("rho_03",       0.3, 0.6),
        # Measured honest-trace rho; tampered regions are contiguous so the
        # tamper-side rho is set higher (capped at 0.9), matching the offset
        # used by the assumed levels.
        ("rho_empirical", rho_emp, min(0.9, rho_emp + 0.3)),
        ("rho_06",       0.6, 0.9),
    ]

    out = {}
    for cfg_name, src_data in [("baseline", src), ("with_shortcuts", src_shortcuts)]:
        cfg = {}
        for attack in ["command_swap", "sign_flip", "feed_edit"]:
            info = src_data.get(attack, {})
            fpr_row = info.get("fpr", 0.0)
            tpr_row = info.get("tpr", 0.0)
            attack_out = {}
            for label, rho_h, rho_t in rho_levels:
                attack_out[label] = simulate(
                    fpr_row, tpr_row, 5000, 5000,
                    rho_h, rho_t, Ks, seed=42)
            cfg[attack] = attack_out
        out[cfg_name] = cfg

    out_path = REPO / "outputs/decoder20260511/audit/tamper_aggregation_correlated.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}\n")

    # Summary: command-swap majority-K at K=10 across all correlation levels
    print("=== command-swap, majority-K at K=10, varying correlation ===")
    print(f"{'config':18s}{'rho':10s}{'FPR':>9s}{'TPR':>9s}")
    print('-' * 50)
    for cfg in ["baseline", "with_shortcuts"]:
        for label, rh, rt in rho_levels:
            d = out[cfg]["command_swap"][label]["rules"]["majority-K"]["K_10"]
            print(f"{cfg:18s}{label:10s}{d['fpr']:>9.4f}{d['tpr']:>9.4f}")
        print()

    print("=== sign-flip, majority-K at K=10 ===")
    for cfg in ["baseline", "with_shortcuts"]:
        for label, rh, rt in rho_levels:
            d = out[cfg]["sign_flip"][label]["rules"]["majority-K"]["K_10"]
            print(f"{cfg:18s}{label:10s}{d['fpr']:>9.4f}{d['tpr']:>9.4f}")
        print()


if __name__ == "__main__":
    main()
