#!/usr/bin/env python3
"""Per-axis (X / Y / Z / F / S) recoverability for V8 decoder predictions.

Round-2 Phase A. Consumes the saved predictions from `--eval_only` runs
(`beam_0_all_predictions.json`) which contain TRUE / PRED token strings
per test sample. For each axis we report:

  - has_<axis>: classification accuracy + precision + recall + F1
  - <axis>_sign: classification metrics (positive / negative / absent)
  - <axis>_val: presence-recall + MAE on the regression value

Output: `audit/v8_per_field.json` aggregating across whichever fold dirs
the user passes via `--checkpoint-dirs`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from score_recoverability import parse_fields  # type: ignore  # noqa: E402
from v7_per_field_eval import tokens_to_canonical_gcode  # type: ignore  # noqa: E402

try:
    from miracle.training.per_class_metrics import (  # noqa: E402
        compute_full_classification_metrics,
        regression_metrics,
    )
except ImportError as e:
    print(f"failed to import per_class_metrics: {e}")
    raise

# Axes the manuscript reports on. Add/remove here to expand.
AXES = ["x", "y", "z", "f", "s", "r", "i", "j"]


def _detok(prediction_or_true_str: str) -> str:
    """The samples save TRUE/PRED as space-joined token strings.
    Convert to canonical G-code so parse_fields can consume it."""
    toks = prediction_or_true_str.split()
    return tokens_to_canonical_gcode(toks)


def score_single_fold(beam_predictions_json: Path) -> dict[str, Any]:
    """Score one fold's beam_0_all_predictions.json."""
    samples = json.loads(beam_predictions_json.read_text())
    n = len(samples)

    true_fields = []
    pred_fields = []
    for s in samples:
        true_text = _detok(s["true"])
        pred_text = _detok(s["pred"])
        true_fields.append(parse_fields(true_text))
        pred_fields.append(parse_fields(pred_text))

    report: dict[str, Any] = {"path": str(beam_predictions_json), "n": n, "axes": {}}

    # Command field
    y_true = np.array([d["command"] for d in true_fields])
    y_pred = np.array([d["command"] for d in pred_fields])
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    to_int = {c: i for i, c in enumerate(classes)}
    yi_true = np.array([to_int[c] for c in y_true])
    yi_pred = np.array([to_int[c] for c in y_pred])
    label_names = {v: k for k, v in to_int.items()}
    report["command"] = compute_full_classification_metrics(
        yi_true, yi_pred, label_names=label_names
    )

    for axis in AXES:
        # has_<axis>
        has_true = np.array([int(d.get(f"has_{axis}", 0)) for d in true_fields])
        has_pred = np.array([int(d.get(f"has_{axis}", 0)) for d in pred_fields])
        has_metrics = compute_full_classification_metrics(
            has_true, has_pred, label_names={0: "absent", 1: "present"}
        )

        # <axis>_sign — only meaningful when both true and pred have the axis
        sign_true = np.array([int(d.get(f"{axis}_sign", 0)) for d in true_fields])
        sign_pred = np.array([int(d.get(f"{axis}_sign", 0)) for d in pred_fields])
        # Map -1/0/+1 to 0/1/2 for sklearn
        signmap = {-1: 0, 0: 1, 1: 2}
        sign_true_m = np.array([signmap[int(v)] for v in sign_true])
        sign_pred_m = np.array([signmap[int(v)] for v in sign_pred])
        sign_metrics = compute_full_classification_metrics(
            sign_true_m, sign_pred_m, label_names={0: "negative", 1: "absent", 2: "positive"}
        )

        # <axis>_val regression
        val_true = np.array([d.get(f"{axis}_val", np.nan) for d in true_fields], dtype=np.float64)
        val_pred = np.array([d.get(f"{axis}_val", np.nan) for d in pred_fields], dtype=np.float64)
        val_metrics = regression_metrics(val_true, val_pred)

        report["axes"][axis.upper()] = {
            "has_axis": has_metrics,
            "sign": sign_metrics,
            "value_regression": val_metrics,
        }

    return report


def aggregate(fold_reports: list[dict]) -> dict[str, Any]:
    """Aggregate per-fold reports to mean ± std for each metric."""
    if not fold_reports:
        return {}
    agg: dict[str, Any] = {"n_folds": len(fold_reports), "command": {}, "axes": {}}

    # Command aggregate
    cmd_acc = [r["command"]["accuracy"] for r in fold_reports]
    cmd_f1 = [r["command"]["macro_f1"] for r in fold_reports]
    agg["command"] = {
        "accuracy_mean": float(np.mean(cmd_acc)),
        "accuracy_std": float(np.std(cmd_acc)),
        "macro_f1_mean": float(np.mean(cmd_f1)),
        "macro_f1_std": float(np.std(cmd_f1)),
    }

    # Axes aggregate
    axes_seen = set()
    for r in fold_reports:
        axes_seen.update(r["axes"].keys())
    for axis in sorted(axes_seen):
        has_acc = [r["axes"][axis]["has_axis"]["accuracy"] for r in fold_reports if axis in r["axes"]]
        has_f1 = [r["axes"][axis]["has_axis"]["macro_f1"] for r in fold_reports if axis in r["axes"]]
        sign_acc = [r["axes"][axis]["sign"]["accuracy"] for r in fold_reports if axis in r["axes"]]
        sign_f1 = [r["axes"][axis]["sign"]["macro_f1"] for r in fold_reports if axis in r["axes"]]
        mae = [r["axes"][axis]["value_regression"]["mae_when_both_present"] for r in fold_reports if axis in r["axes"]]
        mae_clean = [v for v in mae if not np.isnan(v)]
        presence_recall = [r["axes"][axis]["value_regression"]["presence_recall"] for r in fold_reports if axis in r["axes"]]
        agg["axes"][axis] = {
            "has_axis_accuracy_mean": float(np.mean(has_acc)),
            "has_axis_accuracy_std": float(np.std(has_acc)),
            "has_axis_macro_f1_mean": float(np.mean(has_f1)),
            "has_axis_macro_f1_std": float(np.std(has_f1)),
            "sign_accuracy_mean": float(np.mean(sign_acc)),
            "sign_macro_f1_mean": float(np.mean(sign_f1)),
            "value_mae_mean": float(np.mean(mae_clean)) if mae_clean else float("nan"),
            "value_mae_std": float(np.std(mae_clean)) if mae_clean else float("nan"),
            "presence_recall_mean": float(np.mean(presence_recall)),
        }

    return agg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dirs", nargs="+", type=Path, required=True,
                   help="Directories containing results/beam_0_all_predictions.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="V8 per_row no_shortcuts",
                   help="Human label for the aggregate report")
    args = p.parse_args()

    fold_reports = []
    for d in args.checkpoint_dirs:
        beam_json = d / "results" / "beam_0_all_predictions.json"
        if not beam_json.exists():
            print(f"  skip {d}: no beam_0_all_predictions.json")
            continue
        rep = score_single_fold(beam_json)
        fold_reports.append(rep)

    agg = aggregate(fold_reports)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "label": args.label,
        "n_folds": len(fold_reports),
        "per_fold": fold_reports,
        "aggregate": agg,
    }, indent=2))

    # Print summary
    print(f"\n=== {args.label} ({len(fold_reports)} folds) ===\n")
    if agg.get("command"):
        print(f"Command:  acc {agg['command']['accuracy_mean']:.4f} ± {agg['command']['accuracy_std']:.4f}, "
              f"macroF1 {agg['command']['macro_f1_mean']:.4f}")
    print()
    print(f"{'Axis':<5}{'has_axis_acc':>14}{'has_axis_F1':>14}{'sign_acc':>11}{'value_MAE':>12}{'presence_R':>12}")
    print("-" * 70)
    for axis, a in agg.get("axes", {}).items():
        print(f"{axis:<5}{a['has_axis_accuracy_mean']:>14.4f}{a['has_axis_macro_f1_mean']:>14.4f}"
              f"{a['sign_accuracy_mean']:>11.4f}"
              f"{a['value_mae_mean']:>12.4f}{a['presence_recall_mean']:>12.4f}")

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
