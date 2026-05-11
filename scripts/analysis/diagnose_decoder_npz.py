#!/usr/bin/env python3
"""Structural probe of decoder preprocessed NPZ files.

Phase-1 verification artifact for `outputs/decoder20260511/`. Read-only.

For every NPZ file under `--input-dir`, emit a structural fingerprint:
- tokens shape and dtype
- `lengths` field distribution and what it actually represents
- content-token length distribution (counting non-PAD/BOS/EOS)
- `gcode_texts` line-count distribution and distinct count
- presence of position-leakage fields (`window_index`, `total_windows`, `source_file`)
- truncation indicators relative to a target `--max-token-len`

Writes a single JSON to `--output`. Optionally also writes a markdown summary
suitable for inclusion in `AUDIT_REPORT.md`.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

# Special token IDs aligned with src/miracle/dataset/decoder_dataset.py
PAD = 0
BOS = 1
EOS = 2


def _summarise_int_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "unique_values": int(np.unique(arr).size),
    }


def probe_npz(npz_path: Path, max_token_len: int = 16) -> dict[str, Any]:
    """Probe a single NPZ file and return a structural report."""
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.files)
    report: dict[str, Any] = {
        "path": str(npz_path),
        "keys": keys,
        "issues": [],
    }

    # Shapes / dtypes
    shapes = {}
    for k in keys:
        v = d[k]
        shapes[k] = {"dtype": str(v.dtype), "shape": list(v.shape)}
    report["fields"] = shapes

    # Tokens
    if "tokens" not in keys:
        report["issues"].append("missing tokens field")
        return report
    tokens = d["tokens"]
    n_samples, max_token_dim = tokens.shape
    report["n_samples"] = int(n_samples)
    report["tokens_shape"] = [int(n_samples), int(max_token_dim)]

    # Content-token length: count non-PAD/BOS/EOS per row
    content_mask = (tokens != PAD) & (tokens != BOS) & (tokens != EOS)
    content_lengths = content_mask.sum(axis=1)
    report["content_token_length"] = _summarise_int_array(content_lengths)

    # Token-length distribution histogram (capped at 32)
    bins = list(range(0, 33))
    hist, _ = np.histogram(np.minimum(content_lengths, 32), bins=bins + [33])
    report["content_token_length_hist"] = {
        f"{b}": int(c) for b, c in zip(bins, hist)
    }
    report["content_token_length_hist_note"] = "key 32 includes any length >= 32"

    # NPZ-stored `lengths` field vs derived content length
    if "lengths" in keys:
        lengths = d["lengths"].astype(np.int64)
        report["npz_lengths"] = _summarise_int_array(lengths)
        # Diagnose whether `lengths` matches window size or content length
        equals_window_size = bool(np.all(lengths == lengths[0])) and lengths.size > 0
        if equals_window_size and lengths[0] not in set(content_lengths.tolist()):
            report["npz_lengths_interpretation"] = (
                f"CONSTANT at {int(lengths[0])}; does NOT match content_token_length range "
                f"({int(content_lengths.min())}-{int(content_lengths.max())}). "
                "Likely SENSOR window length, not token length. BUG."
            )
            report["issues"].append("lengths_field_is_sensor_not_token_length")
        elif np.array_equal(lengths, content_lengths):
            report["npz_lengths_interpretation"] = "matches derived content_token_length exactly"
        else:
            report["npz_lengths_interpretation"] = "varies but does not match derived content_token_length"

    # gcode_texts: line-count distribution and uniques
    if "gcode_texts" in keys:
        gt = d["gcode_texts"]
        texts = [str(t) if t is not None else "" for t in gt]
        line_counts = collections.Counter(
            (t.count("\n") + 1 if t.strip() else 0) for t in texts
        )
        report["gcode_texts_line_count_distribution"] = dict(line_counts)
        report["gcode_texts_distinct"] = int(len(set(texts)))
        if line_counts and max(line_counts.keys()) <= 1:
            report["issues"].append("gcode_texts_single_line_only")
        examples = []
        for t in texts[:3]:
            examples.append(t[:120])
        report["gcode_texts_examples"] = examples

    # Position leakage fields
    leakage_fields = ["window_index", "total_windows", "source_file", "operation_type"]
    leakage_present = {k: (k in keys) for k in leakage_fields}
    report["position_leakage_fields_present"] = leakage_present
    if "window_index" in keys:
        report["window_index_unique"] = int(np.unique(d["window_index"]).size)
    if "source_file" in keys:
        sf = d["source_file"]
        try:
            report["source_file_unique"] = int(len(set(str(x) for x in sf)))
        except Exception:
            report["source_file_unique"] = -1

    # Truncation indicators vs target max_token_len
    truncated_at_max = int((content_lengths > max_token_len - 2).sum())
    truncated_at_npz_dim = int((content_lengths >= max_token_dim).sum())
    report["truncation_check"] = {
        "max_token_len_target": max_token_len,
        "samples_exceeding_target_minus_2": truncated_at_max,
        "samples_at_npz_max_dim": truncated_at_npz_dim,
        "pct_exceeding_target": float(100 * truncated_at_max / max(n_samples, 1)),
    }
    if truncated_at_max > 0:
        report["issues"].append(
            f"{truncated_at_max} samples ({100*truncated_at_max/n_samples:.1f}%) "
            f"exceed max_token_len={max_token_len} content cap"
        )

    return report


def walk_and_probe(input_dir: Path, max_token_len: int, pattern: str = "*.npz") -> list[dict[str, Any]]:
    reports = []
    for npz in sorted(input_dir.rglob(pattern)):
        try:
            reports.append(probe_npz(npz, max_token_len=max_token_len))
        except Exception as exc:
            reports.append({"path": str(npz), "error": repr(exc)})
    return reports


def write_markdown(reports: list[dict[str, Any]], out_path: Path) -> None:
    """Render a human-readable summary for AUDIT_REPORT.md inclusion."""
    lines = ["## NPZ structural diagnostics", ""]
    lines.append("| Path | N | tokens.shape | content_len min/med/max | distinct gcode | lengths | leakage fields | issues |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        if "error" in r:
            lines.append(f"| `{r['path']}` | ERROR | | | | | | {r['error']} |")
            continue
        ctl = r.get("content_token_length", {})
        npzl = r.get("npz_lengths", {})
        leak = r.get("position_leakage_fields_present", {})
        leak_str = ",".join(k for k, v in leak.items() if v) or "—"
        issues = "; ".join(r.get("issues", [])) or "—"
        lines.append(
            f"| `{r['path']}` "
            f"| {r.get('n_samples','?')} "
            f"| {r.get('tokens_shape','?')} "
            f"| {ctl.get('min','?')}/{ctl.get('median','?')}/{ctl.get('max','?')} "
            f"| {r.get('gcode_texts_distinct','?')} "
            f"| {npzl.get('min','?')}-{npzl.get('max','?')} "
            f"| {leak_str} "
            f"| {issues} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Structural NPZ probe for decoder preprocessing")
    p.add_argument("--input-dir", required=True, type=Path, help="Directory tree containing NPZ files")
    p.add_argument("--output", required=True, type=Path, help="Output JSON path")
    p.add_argument("--markdown", type=Path, default=None, help="Optional markdown summary path")
    p.add_argument("--max-token-len", type=int, default=16, help="Target max_token_len to assess truncation against")
    p.add_argument("--pattern", default="*.npz")
    args = p.parse_args()

    reports = walk_and_probe(args.input_dir, args.max_token_len, args.pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reports": reports, "max_token_len": args.max_token_len}, indent=2))
    if args.markdown:
        write_markdown(reports, args.markdown)
    print(f"wrote {args.output} ({len(reports)} reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
