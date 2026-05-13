"""Per-class accuracy / precision / recall / F1 helpers.

Round-2 (decoder20260511) addition. The existing
`compute_classification_metrics` in `metrics.py` returns macro averages only.
For the manuscript we need per-class precision/recall/F1 with sklearn's
standard reporting plus confusion matrices.

This module is the single source of truth for per-class reporting across all
training/eval scripts. The downstream consumers (`aggregate_v8_results.py`,
the paper-table generators, the per-axis breakdown) all import from here.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def compute_full_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: Optional[dict[int, str]] = None,
    ignore_labels: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Compute accuracy + macro/per-class precision/recall/F1 + confusion matrix.

    Args:
        y_true: 1D array of integer class labels.
        y_pred: 1D array of predicted integer class labels.
        label_names: Optional mapping int -> human-readable label string. Used
            in the per-class dict and confusion matrix labels.
        ignore_labels: Optional set of integer labels to exclude from the
            computation entirely (e.g., padding tokens marked -1 or PAD=0).

    Returns:
        Dict shaped:
        {
            "n": <count after ignore>,
            "accuracy": float,
            "macro_precision": float,
            "macro_recall": float,
            "macro_f1": float,
            "weighted_precision": float,
            "weighted_recall": float,
            "weighted_f1": float,
            "per_class": {
                "<label_name>": {"precision": .., "recall": .., "f1": .., "support": ..},
                ...
            },
            "confusion_matrix": [[...]],
            "confusion_matrix_labels": [<label_name in row/col order>, ...],
        }
    """
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn not available; install scikit-learn")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"

    if ignore_labels:
        mask = ~np.isin(y_true, list(ignore_labels))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_precision": float("nan"),
            "weighted_recall": float("nan"),
            "weighted_f1": float("nan"),
            "per_class": {},
            "confusion_matrix": [],
            "confusion_matrix_labels": [],
        }

    # Universe of labels = union of true and predicted (sklearn idiom).
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    name_for = (lambda i: (label_names or {}).get(int(i), str(int(i))))

    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name_for(lbl): {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i]),
        }
        for i, lbl in enumerate(labels)
    }

    return {
        "n": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [name_for(l) for l in labels],
    }


def sequence_level_accuracy(
    y_true_seqs: list[list[int]],
    y_pred_seqs: list[list[int]],
    *,
    ignore_labels: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Exact-match sequence accuracy.

    A predicted sequence matches if, after stripping `ignore_labels` from both,
    the remaining tokens are identical in length and value.
    """
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(f"seq count mismatch: {len(y_true_seqs)} vs {len(y_pred_seqs)}")

    def strip(seq):
        if ignore_labels is None:
            return list(seq)
        return [t for t in seq if t not in ignore_labels]

    correct = 0
    total = 0
    for t, p in zip(y_true_seqs, y_pred_seqs):
        ts = strip(t)
        ps = strip(p)
        total += 1
        if ts == ps:
            correct += 1
    return {
        "n": total,
        "accuracy": float(correct / max(total, 1)),
        "correct": correct,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nan_in_pred_counts_as_miss: bool = True,
) -> dict[str, Any]:
    """MAE + presence recall + false-positive count for a regression field
    that may be NaN (absent in the ground truth).

    A "present" prediction means non-NaN. A "true present" means truth is
    non-NaN. We report:
      - mae_when_both_present: |t - p| averaged over indices where both are non-NaN.
      - presence_recall: fraction of true-present indices where we also predicted non-NaN.
      - false_positive: count of indices where pred is non-NaN but truth is NaN.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask_true = ~np.isnan(y_true)
    mask_pred = ~np.isnan(y_pred)
    both = mask_true & mask_pred
    n_true_present = int(mask_true.sum())
    n_both = int(both.sum())
    mae = float(np.abs(y_true[both] - y_pred[both]).mean()) if n_both else float("nan")
    return {
        "n_true_present": n_true_present,
        "n_pred_present_when_true": n_both,
        "presence_recall": float(n_both / max(n_true_present, 1)),
        "false_positive_count": int((mask_pred & ~mask_true).sum()),
        "mae_when_both_present": mae,
    }
