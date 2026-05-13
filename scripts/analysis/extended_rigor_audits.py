#!/usr/bin/env python3
"""Three rigor audits requested as recommendations #6, #7, #8.

  6. Power analysis on per-class long-tail F1.
     Given fold-level support per class, what minimum effect size is
     detectable at alpha=0.05, beta=0.20 (power=0.80) for a per-class
     accuracy comparison? Surfaces when "not significant" actually
     means "underpowered."

  7. Train-vs-test covariate-shift audit.
     For each fold, compare the train-split's per-field positive-support
     frequency against the test-split's. A 3-5 pp shift on, say, has-Z
     changes the interpretation of has-Z accuracy as a recoverability
     metric. Reports per-fold delta + a flag for any field >3pp shifted.

  8. Token-position failure analysis.
     For per_row predictions, compute accuracy at each OUTPUT-sequence
     position (position 0, 1, 2, ...). At which position does the
     decoder fail most? Complements the per-digit-position breakdown
     inside NUM tokens.

Output: outputs/decoder20260511/audit/extended_rigor_audits.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRED_NPZ = REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage2/scheduled_sampling_0.5/fold_1/results/predictions.npz"
DATA_ROOT = REPO / "outputs/decoder20260511/preprocessed_f98/per_row"
OUT_JSON = REPO / "outputs/decoder20260511/audit/extended_rigor_audits.json"


# ------------------- #6 Power analysis -------------------

def power_minimum_detectable_effect(n: int, baseline: float,
                                    alpha: float = 0.05, power: float = 0.80) -> float:
    """Two-proportion z-test minimum detectable effect (one-tailed).

    Returns the smallest delta (in accuracy percentage points) such that
    a two-proportion z-test between a baseline of rate p0 and the
    proposed rate p1 = p0 + delta is significant at alpha with given
    statistical power.
    """
    # z-quantiles
    try:
        from scipy.stats import norm
        z_alpha = float(norm.ppf(1 - alpha))      # one-tailed
        z_beta = float(norm.ppf(power))
    except Exception:
        # Approximations
        z_alpha = 1.6449 if alpha == 0.05 else 1.96
        z_beta = 0.8416 if power == 0.80 else 1.2816

    # Pooled-variance approximation. Solve for delta s.t.
    #   delta = (z_alpha * sqrt(2 * p_bar * (1 - p_bar))
    #            + z_beta * sqrt(p0(1-p0) + p1(1-p1))) / sqrt(n)
    # Iterate to converge p_bar = (p0 + p1)/2.
    p0 = baseline
    p1 = baseline
    for _ in range(20):
        p_bar = (p0 + p1) / 2.0
        se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        delta = z_alpha * se_null + z_beta * se_alt
        p1_new = min(1.0, p0 + delta)
        if abs(p1_new - p1) < 1e-6:
            break
        p1 = p1_new
    return float(delta)


def power_audit_per_class() -> dict:
    """For each per-class command/param-type test support, report MDE.

    Per-class supports from V8 per_row fold 1 test split (n=7937 total):
    G0=86, G1=1095, G2=475, G3=395, none=5886 (cmd).
    Per-class param-type supports from the same: X≈7000, Y≈6800, Z≈2500, R≈1370.
    """
    classes = {
        # Each entry: (n_support, baseline_estimate)
        "command_G0":  (86,   0.40),
        "command_G1":  (1095, 0.94),
        "command_G2":  (475,  0.50),
        "command_G3":  (395,  0.30),
        "command_none":(5886, 0.95),
        "paramtype_X": (7000, 0.94),
        "paramtype_Y": (6800, 0.87),
        "paramtype_Z": (2500, 1.00),
        "paramtype_R": (1370, 1.00),
    }
    out = {}
    for k, (n, base) in classes.items():
        mde = power_minimum_detectable_effect(n, base, alpha=0.05, power=0.80)
        out[k] = {
            "n_support": n,
            "baseline_acc": base,
            "minimum_detectable_delta_pp": round(mde * 100, 2),
            "interpretation": (
                f"With n={n} test samples at p_0={base:.2f}, the smallest "
                f"accuracy difference detectable at α=0.05, β=0.20 is "
                f"{mde*100:.1f} pp. Differences below this size cannot be "
                f"statistically distinguished from sampling noise."
            ),
        }
    return out


# ------------------- #7 Covariate-shift audit -------------------

_FIELD_RE = re.compile(r'([XYZFSRIJ])(-?\d+\.?\d*)')


def field_freq(texts) -> dict:
    n = len(texts)
    counts = {f: sum(1 for t in texts if re.search(rf'(?<![A-Za-z_]){f}-?\d', str(t)))
              for f in 'XYZFSRIJ'}
    return {f: c / n if n else 0.0 for f, c in counts.items()}


def covariate_shift_audit(mode: str = "per_row") -> dict:
    """Per-fold train-vs-test covariate shift on per-field positive support.

    Reports the absolute delta between train and test field-frequency for
    each fold and flags any field shifted by >3 pp.
    """
    results = {}
    for fold in range(1, 6):
        try:
            train_d = np.load(REPO / f"outputs/decoder20260511/preprocessed_f98/{mode}/fold_{fold}/train_sequences.npz", allow_pickle=True)
            test_d  = np.load(REPO / f"outputs/decoder20260511/preprocessed_f98/{mode}/fold_{fold}/test_sequences.npz",  allow_pickle=True)
        except FileNotFoundError:
            results[f"fold_{fold}"] = None
            continue
        f_train = field_freq(train_d['gcode_texts'])
        f_test  = field_freq(test_d['gcode_texts'])
        deltas = {f: f_test[f] - f_train[f] for f in 'XYZFSRIJ'}
        flagged = {f: round(d * 100, 2) for f, d in deltas.items() if abs(d) > 0.03}
        results[f"fold_{fold}"] = {
            "train_frequency": {f: round(v * 100, 2) for f, v in f_train.items()},
            "test_frequency":  {f: round(v * 100, 2) for f, v in f_test.items()},
            "delta_pp":        {f: round(d * 100, 2) for f, d in deltas.items()},
            "flagged_fields_above_3pp": flagged,
        }
    return results


# ------------------- #8 Token-position failure analysis -------------------

def token_position_failure(pred_npz: Path) -> dict | None:
    """For per_row predictions, compute accuracy at each output-sequence position.

    Loads predictions.npz from a trained decoder cell. Returns dict mapping
    output_position → {accuracy, n_observed}.
    """
    if not pred_npz.exists():
        return None
    d = np.load(pred_npz, allow_pickle=True)
    # Field names differ across runs: prefer pred_tokens/target_tokens (newer
    # format) and fall back to tok_p/tok_t.
    if "pred_tokens" in d.files and "target_tokens" in d.files:
        tok_p, tok_t = d["pred_tokens"], d["target_tokens"]
    elif "tok_p" in d.files and "tok_t" in d.files:
        tok_p, tok_t = d["tok_p"], d["tok_t"]
    else:
        return None
    PAD = 0
    N, L = tok_p.shape
    per_position = []
    for pos in range(L):
        target = tok_t[:, pos]
        mask = target != PAD
        if mask.sum() == 0:
            per_position.append(None)
            continue
        acc = float((tok_p[mask, pos] == target[mask]).mean())
        per_position.append({
            "position": pos,
            "n_observed": int(mask.sum()),
            "accuracy": round(acc, 4),
        })
    return {
        "n_samples": int(N),
        "max_seq_len": int(L),
        "per_position": per_position,
    }


# ------------------- Main -------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-npz", type=Path, default=PRED_NPZ)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    args = p.parse_args()

    print("=== #6 Power analysis ===")
    power = power_audit_per_class()
    for k, v in power.items():
        print(f"  {k:<18} n={v['n_support']:>5}  base={v['baseline_acc']:.2f}  MDE={v['minimum_detectable_delta_pp']:.1f} pp")

    print()
    print("=== #7 Covariate shift (per_row) ===")
    cov_per_row = covariate_shift_audit("per_row")
    for fold, payload in cov_per_row.items():
        if payload is None:
            continue
        flagged = payload['flagged_fields_above_3pp']
        if flagged:
            print(f"  {fold}: FLAGGED fields > 3 pp shift: {flagged}")
        else:
            print(f"  {fold}: all fields within 3 pp")

    print()
    print("=== #7 Covariate shift (full_window) ===")
    cov_full = covariate_shift_audit("full_window")
    for fold, payload in cov_full.items():
        if payload is None:
            continue
        flagged = payload['flagged_fields_above_3pp']
        if flagged:
            print(f"  {fold}: FLAGGED fields > 3 pp shift: {flagged}")
        else:
            print(f"  {fold}: all fields within 3 pp")

    print()
    print("=== #8 Token-position failure analysis ===")
    tok_pos = token_position_failure(args.pred_npz)
    if tok_pos is None:
        print(f"  ! {args.pred_npz} not loadable; skipping")
    else:
        print(f"  N={tok_pos['n_samples']} samples, L={tok_pos['max_seq_len']} max positions")
        for entry in tok_pos['per_position']:
            if entry is None:
                continue
            print(f"  position {entry['position']:>2}: n={entry['n_observed']:>5} accuracy={entry['accuracy']:.4f}")

    payload = {
        "rec6_power_analysis": power,
        "rec7_covariate_shift_per_row": cov_per_row,
        "rec7_covariate_shift_full_window": cov_full,
        "rec8_token_position_failure": tok_pos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
