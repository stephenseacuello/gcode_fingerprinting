"""Runtime diagnostics for the decoder preprocessing pipeline.

Phase-3 addition (decoder20260511). Emits a structured JSON report each
time it is invoked so that future regressions of the V7 silent-failure
class (lengths-is-sensor-length, all-windows-collapsed-to-one-line) are
immediately visible in CI logs / output directories.

Usage as a library:

    from miracle.dataset.preprocessing_diagnostics import report_preprocessed_dir
    report_preprocessed_dir(Path("outputs/decoder20260511/preprocessed/full_window/fold_1"),
                            out_json=Path("outputs/decoder20260511/diagnostics_run.json"))

Usage as CLI:

    python -m miracle.dataset.preprocessing_diagnostics \\
        --input-dir outputs/decoder20260511/preprocessed \\
        --output outputs/decoder20260511/audit/diagnostics_run.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PAD, BOS, EOS = 0, 1, 2

# Hard-fail thresholds. These are designed so any future preprocessing
# regression that re-introduces the V7 failures will trip them.
INVARIANTS = {
    "lengths_must_equal_token_length",
    "token_length_must_equal_derived_content_length",
    "window_length_must_equal_continuous_T",
    "full_window_must_have_multiline_targets",
    "per_row_must_have_singleline_targets",
}


def probe_single_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    files = list(d.files)
    report: Dict[str, Any] = {"path": str(path), "fields": files, "issues": []}
    if "tokens" not in files:
        report["issues"].append("missing_tokens_field")
        return report

    tokens = d["tokens"]
    content_mask = (tokens != PAD) & (tokens != BOS) & (tokens != EOS)
    derived = content_mask.sum(axis=1)

    n_samples = int(tokens.shape[0])
    max_token_dim = int(tokens.shape[1]) if tokens.ndim == 2 else 0

    report.update({
        "n_samples": n_samples,
        "tokens_max_dim": max_token_dim,
        "content_token_length": {
            "min": int(derived.min()) if n_samples else 0,
            "max": int(derived.max()) if n_samples else 0,
            "mean": float(derived.mean()) if n_samples else 0.0,
            "median": float(np.median(derived)) if n_samples else 0.0,
        },
    })

    # Field equivalence checks
    if "token_length" in files:
        tl = d["token_length"]
        if not np.array_equal(tl, derived):
            report["issues"].append("token_length_disagrees_with_derived")
    else:
        report["issues"].append("missing_token_length_field")

    if "lengths" in files and "token_length" in files:
        if not np.array_equal(d["lengths"], d["token_length"]):
            report["issues"].append("lengths_disagrees_with_token_length")

    if "window_length" in files and "continuous" in files:
        T_s = int(d["continuous"].shape[1])
        if not np.all(d["window_length"] == T_s):
            report["issues"].append("window_length_disagrees_with_continuous_T")

    # Label-mode-aware checks
    label_mode = str(d.get("label_mode", "unknown"))
    report["label_mode"] = label_mode

    gcode_texts = [str(t) for t in d.get("gcode_texts", [])]
    multi_count = sum(1 for t in gcode_texts if "\n" in t and t.strip())
    distinct = len(set(gcode_texts))
    report["gcode_texts_distinct"] = distinct
    report["gcode_texts_multiline_count"] = multi_count
    report["gcode_texts_multiline_fraction"] = (
        float(multi_count / max(len(gcode_texts), 1))
    )

    if "full_window" in str(path):
        if multi_count == 0:
            report["issues"].append("full_window_collapsed_to_singleline")
        if report["content_token_length"]["max"] <= 6:
            report["issues"].append("full_window_token_max_le_6_like_v7")
    elif "per_row" in str(path):
        if multi_count > 0:
            report["issues"].append("per_row_has_multiline_target")

    return report


def report_preprocessed_dir(input_dir: Path, out_json: Path,
                            *, fail_on_issues: bool = True) -> Dict[str, Any]:
    """Walk `input_dir` for *_sequences.npz and write a combined report."""
    reports: List[Dict[str, Any]] = []
    for npz in sorted(input_dir.rglob("*_sequences.npz")):
        try:
            reports.append(probe_single_npz(npz))
        except Exception as exc:
            reports.append({"path": str(npz), "error": repr(exc)})

    summary = {
        "n_npz": len(reports),
        "n_with_issues": sum(1 for r in reports if r.get("issues")),
        "all_issues": sorted({i for r in reports for i in r.get("issues", [])}),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "reports": reports}, indent=2))

    if fail_on_issues and summary["n_with_issues"] > 0:
        first_bad = next(r for r in reports if r.get("issues"))
        raise SystemExit(
            f"[preprocessing_diagnostics] {summary['n_with_issues']} NPZ files "
            f"have issues. First: {first_bad['path']}: {first_bad['issues']}"
        )
    return summary


def _cli() -> int:
    p = argparse.ArgumentParser(description="Decoder preprocessing diagnostics report")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-fail", action="store_true",
                   help="Write the report but don't exit non-zero on issues.")
    args = p.parse_args()
    summary = report_preprocessed_dir(args.input_dir, args.output, fail_on_issues=not args.no_fail)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
