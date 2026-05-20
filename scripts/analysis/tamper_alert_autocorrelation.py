#!/usr/bin/env python3
"""Empirical honest-trace alert autocorrelation + sensor-vs-prior control.

Addresses referee R3 M7 and R3 M1 with no new training:

  (A) Empirical lag-1 autocorrelation of the honest-trace command-swap
      alert series, measured on the real 5-fold AR+FSM predictions in
      source-file / window-index order (within-file adjacent pairs only).
      The correlated-alert aggregation simulation
      (tamper_aggregation_correlated.py) previously assumed rho in
      {0.0, 0.3, 0.6}; this measures the real value so the deployable-
      operating-point claim is evaluated at the empirical rho rather than
      an assumed ceiling.

  (B) Modal-prediction substitution control: replace each window's decoder
      prediction with the corpus-modal predicted string and recompute the
      command-swap detector's TPR/FPR. If the detector performs comparably
      with the modal (sensor-independent) prediction substituted, the
      row-level alert is driven by the closed-vocabulary class prior rather
      than by sensor-conditioned recovery. This is the no-compute analogue
      of an encoder-memory-shuffle control.

Outputs audit/tamper_alert_autocorrelation.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analysis.threat_model_tamper_injection import (
    DETECT_MODE,
    TAMPERS,
    _detok,
    disagrees,
)

REPO = Path(__file__).resolve().parents[2]
FSM_SUB = {
    1: "fold_1/6o90io5p_fsm",
    2: "fold_2/wf2ulnx7_fsm",
    3: "fold_3/ua4ht85b_fsm",
    4: "fold_4/4rrcx1qd_fsm",
    5: "fold_5/gmtf655s_fsm",
}
SWEEP = REPO / "outputs/decoder20260511/checkpoints/full_window_5fold"
SRC = REPO / "outputs/decoder20260511/preprocessed_f98/full_window"
ATTACK = "command_swap"
MODE = DETECT_MODE[ATTACK]


def lag1_autocorr_within_file(flags: np.ndarray, file_ids: np.ndarray) -> float:
    """Lag-1 autocorrelation over adjacent rows *within the same source file*."""
    pairs_x, pairs_y = [], []
    for fid in np.unique(file_ids):
        seq = flags[file_ids == fid].astype(float)
        if len(seq) >= 2:
            pairs_x.append(seq[:-1])
            pairs_y.append(seq[1:])
    if not pairs_x:
        return float("nan")
    x = np.concatenate(pairs_x)
    y = np.concatenate(pairs_y)
    if x.std() == 0 or y.std() == 0:
        # Degenerate (all-alert or no-alert): autocorrelation undefined ->
        # report 1.0 (perfectly correlated constant series) which is the
        # conservative (worst-case FPR) reading for the aggregation claim.
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fold_analysis(fold: int) -> dict | None:
    pj = SWEEP / FSM_SUB[fold] / "results" / "beam_1_all_predictions.json"
    npz = SRC / f"fold_{fold}" / "test_sequences.npz"
    if not pj.exists() or not npz.exists():
        print(f"  skip fold {fold}: missing artifact")
        return None
    recs = json.loads(pj.read_text())
    d = np.load(npz, allow_pickle=True)
    sf = np.array([str(s) for s in d["source_file"]])
    wi = np.array(d["window_index"], dtype=int)
    if len(recs) != len(sf):
        print(f"  skip fold {fold}: len {len(recs)} != {len(sf)}")
        return None

    order = np.lexsort((wi, sf))  # by source_file, then window_index
    true_txt = [_detok(recs[i]["true"]) for i in order]
    pred_txt = [_detok(recs[i]["pred"]) for i in order]
    file_ids = sf[order]

    # (A) honest-trace alert series + empirical lag-1 autocorrelation
    honest_alert = np.array(
        [disagrees(t, p, mode=MODE) for t, p in zip(true_txt, pred_txt)], dtype=int
    )
    rho_honest = lag1_autocorr_within_file(honest_alert, file_ids)

    # (B) modal-prediction substitution control
    modal_pred = Counter(pred_txt).most_common(1)[0][0]
    fp = sum(disagrees(t, p, mode=MODE) for t, p in zip(true_txt, pred_txt))
    fp_modal = sum(disagrees(t, modal_pred, mode=MODE) for t in true_txt)
    tamper_fn = TAMPERS[ATTACK]
    tp = tp_modal = n_app = 0
    for t, p in zip(true_txt, pred_txt):
        tam = tamper_fn(t)
        if tam is None or tam == t:
            continue
        n_app += 1
        tp += int(disagrees(tam, p, mode=MODE))
        tp_modal += int(disagrees(tam, modal_pred, mode=MODE))
    n = len(true_txt)
    return {
        "n_samples": n,
        "rho_honest_lag1": rho_honest,
        "honest_alert_rate": float(honest_alert.mean()),
        "real": {"fpr": fp / max(n, 1), "tpr": tp / max(n_app, 1)},
        "modal_substituted": {
            "fpr": fp_modal / max(n, 1),
            "tpr": tp_modal / max(n_app, 1),
            "modal_pred_share": Counter(pred_txt).most_common(1)[0][1] / max(n, 1),
        },
    }


def main() -> int:
    per_fold = {}
    for f in range(1, 6):
        r = fold_analysis(f)
        if r:
            per_fold[f] = r

    def agg(path):
        vals = []
        for r in per_fold.values():
            cur = r
            for k in path.split("."):
                cur = cur[k]
            if isinstance(cur, (int, float)) and not np.isnan(cur):
                vals.append(cur)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    summary = {
        "attack": ATTACK,
        "per_fold": per_fold,
        "aggregate": {
            "rho_honest_lag1": agg("rho_honest_lag1"),
            "real_fpr": agg("real.fpr"),
            "real_tpr": agg("real.tpr"),
            "modal_fpr": agg("modal_substituted.fpr"),
            "modal_tpr": agg("modal_substituted.tpr"),
        },
        "interpretation": (
            "rho_honest_lag1 is the empirical within-file lag-1 "
            "autocorrelation of the honest command-swap alert; feed it into "
            "tamper_aggregation_correlated.py instead of the assumed 0.6 "
            "ceiling. If modal_substituted TPR/FPR approx real TPR/FPR, the "
            "row-level alert is closed-vocabulary/prior-driven rather than "
            "sensor-conditioned (referee R3 M1)."
        ),
    }
    out = REPO / "outputs/decoder20260511/audit/tamper_alert_autocorrelation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    a = summary["aggregate"]
    print("\n=== Empirical alert autocorrelation + sensor-vs-prior control ===\n")
    for k, v in a.items():
        if v:
            print(f"{k:<22} {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
