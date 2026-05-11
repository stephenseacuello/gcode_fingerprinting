#!/usr/bin/env python3
"""Static truncation-impact analysis for V7 decoder preprocessing.

Phase-1 verification artifact. Read-only.

Rather than running a full model evaluation (which requires hours of GPU
inference and would not exercise the bug because V7's `gcode_texts` are
single-line), this script measures the truncation that *would* occur under
each candidate `max_token_len` cap, given the actual NPZ contents.

Two analyses:

A. **Direct**: For each V7 NPZ, walk `gcode_texts`, re-tokenize each entry
   with the production tokenizer, and report:
   - token count distribution after re-tokenization
   - count of samples that would lose tokens at caps of 8 / 14 / 16 / 32 / 64
   - percent of total tokens lost at each cap

B. **Hypothetical**: For each NPZ, simulate the multi-line target that
   *would* result if preprocessing kept all unique G-code lines per window.
   Currently the V7 NPZ only stores one line per window, so this simulation
   pools across all single-line windows from the same `source_file` to estimate
   what a multi-line target would look like (per-file unique-line concatenation,
   capped at the window count). This is a *bound*, not a measurement, but it
   demonstrates the scale of the latent bug.

Outputs JSON: `outputs/decoder20260511/audit/truncation_impact.json`.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make src importable
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from miracle.utilities.gcode_tokenizer import GCodeTokenizer  # noqa: E402

CAPS = [8, 14, 16, 32, 64, 128]


def _token_count_with_specials(n_content: int) -> int:
    """Match the DecoderQuickTestDataset accounting: BOS + tokens + EOS."""
    return n_content + 2


def analyse_npz(npz_path: Path, tokenizer: GCodeTokenizer) -> dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    gt = d["gcode_texts"]
    source_file = d.get("source_file", None)

    # Direct analysis: re-tokenize each window's text as-stored
    per_window_lens: list[int] = []
    for txt in gt:
        text_str = str(txt)
        lines = [l for l in text_str.split("\n") if l.strip()]
        canon = tokenizer.canonicalize(lines)
        toks = tokenizer.tokenize_canonical(canon)
        per_window_lens.append(len(toks))
    per_window_lens_arr = np.asarray(per_window_lens, dtype=np.int64)

    cap_stats_direct = {}
    for cap in CAPS:
        keep_per_sample = np.minimum(per_window_lens_arr, cap - 1)  # cap reserves 1 for EOS
        lost_tokens = (per_window_lens_arr - keep_per_sample).sum()
        n_truncated = int((per_window_lens_arr > cap - 1).sum())
        cap_stats_direct[str(cap)] = {
            "samples_truncated": n_truncated,
            "pct_samples_truncated": float(100 * n_truncated / max(len(per_window_lens_arr), 1)),
            "tokens_lost": int(lost_tokens),
            "pct_tokens_lost": float(100 * lost_tokens / max(per_window_lens_arr.sum(), 1)),
        }

    direct = {
        "n_samples": int(len(per_window_lens_arr)),
        "content_token_count_min": int(per_window_lens_arr.min()),
        "content_token_count_max": int(per_window_lens_arr.max()),
        "content_token_count_median": float(np.median(per_window_lens_arr)),
        "content_token_count_mean": float(per_window_lens_arr.mean()),
        "total_content_tokens": int(per_window_lens_arr.sum()),
        "cap_impact": cap_stats_direct,
    }

    # Hypothetical analysis: bound on what a multi-line target would be.
    # Group windows by source_file (proxy for an experimental run), pool the unique
    # G-code lines that appeared across all windows for that file, and tokenise the
    # concatenated unique set. This is an OVERestimate of multi-line targets per
    # window (since not every line appears in every window) but a useful upper bound.
    hypothetical = None
    if source_file is not None:
        groups: dict[str, set[str]] = collections.defaultdict(set)
        for txt, sf in zip(gt, source_file):
            t = str(txt).strip()
            if t:
                groups[str(sf)].add(t)

        per_file_token_counts = []
        per_file_line_counts = []
        for sf, line_set in groups.items():
            ordered = sorted(line_set)
            tokens_all = tokenizer.encode(ordered, add_bos_eos=False)
            per_file_token_counts.append(len(tokens_all))
            per_file_line_counts.append(len(ordered))

        arr_tok = np.asarray(per_file_token_counts, dtype=np.int64)
        arr_lin = np.asarray(per_file_line_counts, dtype=np.int64)
        cap_stats_hypo = {}
        for cap in CAPS:
            keep = np.minimum(arr_tok, cap - 1)
            lost = (arr_tok - keep).sum()
            n_truncated = int((arr_tok > cap - 1).sum())
            cap_stats_hypo[str(cap)] = {
                "source_files_truncated": n_truncated,
                "pct_files_truncated": float(100 * n_truncated / max(arr_tok.size, 1)),
                "tokens_lost": int(lost),
                "pct_tokens_lost": float(100 * lost / max(arr_tok.sum(), 1)),
            }

        hypothetical = {
            "interpretation": "Per source_file: concatenate all unique G-code lines that appear across all windows of that file, tokenise, then test caps. Overestimate of per-window multi-line target token count.",
            "n_source_files": int(arr_tok.size),
            "per_file_lines_min": int(arr_lin.min()),
            "per_file_lines_max": int(arr_lin.max()),
            "per_file_lines_median": float(np.median(arr_lin)),
            "per_file_tokens_min": int(arr_tok.min()),
            "per_file_tokens_max": int(arr_tok.max()),
            "per_file_tokens_median": float(np.median(arr_tok)),
            "per_file_tokens_mean": float(arr_tok.mean()),
            "cap_impact": cap_stats_hypo,
        }

    return {
        "path": str(npz_path),
        "direct_truncation": direct,
        "hypothetical_multiline_truncation": hypothetical,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Static truncation impact analyser")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pattern", default="*.npz")
    args = p.parse_args()

    tokenizer = GCodeTokenizer.load(args.vocab)

    reports = []
    for npz in sorted(args.input_dir.rglob(args.pattern)):
        try:
            reports.append(analyse_npz(npz, tokenizer))
        except Exception as exc:
            reports.append({"path": str(npz), "error": repr(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"caps_tested": CAPS, "reports": reports}, indent=2))
    print(f"wrote {args.output} ({len(reports)} reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
