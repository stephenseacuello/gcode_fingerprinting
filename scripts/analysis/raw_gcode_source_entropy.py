#!/usr/bin/env python3
"""Decoder-free RAW-corpus source-entropy apparatus for Paper B.

Computes per-(axis, decimal-slot) Shannon entropy of CNC toolpath coordinates
DIRECTLY from the raw CAM-authored G-code text files, with NO model, NO decoder
target-encoding, and NO dependency on any preprocessed NPZ / tokenizer. This is
the "salami-defeating" control: it proves the entropy structure reported in
audit/digit_entropy.json (the per-decimal-place bathtub, the three op-class
regimes, the slot-5 face concentration, the adaptive ~3.3-bit uniform ceiling,
and the R discrete-vocabulary low entropy) is a property of the SOURCE DATA, not
an artifact of the model's digit_t targets or the preprocessing pipeline.

Methodology is matched line-for-line to
scripts/analysis/digit_entropy_analysis.py so the numbers are directly
comparable:
  * the SAME parameter regex PARAM_RE = ([XYZIJKRF])(-?\\d+\\.?\\d*)
  * the SAME fixed 6-slot grid [10^1, 10^0, 10^-1, 10^-2, 10^-3, 10^-4]
  * the SAME model_encode() integer/decimal-truncation digit encoding
    (encode_values_to_digits semantics from digit_value_head.py)
  * the SAME Shannon-entropy estimator (bits) and the SAME per-slot CNC verdict
    classifier (classify_slot).

The ONLY differences vs the reference are by design:
  1. Input is the raw .gcode source text files (data_clean/*.gcode and
     data/vocab_corpus.gcode), not preprocessed NPZ windows.
  2. Each source file's basename is taken verbatim as its operation class.
  3. G-code line comments -- both "(...)" blocks and ";..." trailers -- are
     stripped before regex extraction so that vendor/tool metadata such as
     "(T2 D=0.25 CR=0. - ZMIN=-0.25 ...)" cannot leak coordinate-like tokens.
     (The reference PARAM_RE already happens not to match those tokens, because
     they are written as "D=0.25"/"ZMIN=-0.25" rather than "Dxxx"; stripping is
     belt-and-braces and keeps the raw-vs-NPZ comparison honest.)

Outputs three entropy views, all decoder-free:
  (a) per-(axis, slot) digit-value Shannon entropy + CNC verdict   -> "per_axis"
  (b) per-(axis, slot, operation-class) entropy                    -> "per_op_class_slots"
  (c) per-axis sign entropy (pos/neg/zero)                         -> "per_axis_sign"

Writes a NEW file:
  outputs/decoder20260511/audit/raw_gcode_source_entropy.json
and prints a side-by-side comparison against audit/digit_entropy.json. It does
NOT overwrite digit_entropy.json and modifies no existing file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# --- Reused VERBATIM from digit_entropy_analysis.py (which reuses it from
#     score_recoverability.py) so extraction is identical to the reference. ---
PARAM_RE = re.compile(r"([XYZIJKRF])(-?\d+\.?\d*)", re.IGNORECASE)

# Strip G-code comments: parenthesized "(...)" blocks and ";..." trailers.
PAREN_COMMENT_RE = re.compile(r"\([^)]*\)")
SEMI_COMMENT_RE = re.compile(r";.*$")

# Fixed 6-slot grid the model uses for every axis (max over PARAM_VALUE_CONFIG).
MAX_INT_DIGITS = 2
MAX_DEC_DIGITS = 4
N_SLOTS = MAX_INT_DIGITS + MAX_DEC_DIGITS  # 6

SLOT_LABELS = {
    0: "10^1 (tens)",
    1: "10^0 (units)",
    2: "10^-1 (tenths)",
    3: "10^-2 (hundredths)",
    4: "10^-3 (thousandths)",
    5: "10^-4 (ten-thousandths)",
}

# Match the reference axis set (I/J/K kept for parity; absent in this corpus).
AXES = ["X", "Y", "Z", "R", "F", "I", "J"]

# Raw source files. Basename (sans .gcode) is the operation class, per the task.
DEFAULT_SOURCES = [
    "data_clean/face.gcode",
    "data_clean/face075.gcode",
    "data_clean/face150.gcode",
    "data_clean/pocket.gcode",
    "data_clean/pocket075.gcode",
    "data_clean/pocket150.gcode",
    "data_clean/adaptive.gcode",
    "data_clean/adaptive075.gcode",
    "data_clean/adaptive150.gcode",
    # NOTE: data/vocab_corpus.gcode (a synthetic vocabulary-coverage file, NOT a real
    # CAM toolpath) is intentionally EXCLUDED so the source-entropy estimate reflects
    # only authored toolpaths. Pass it explicitly via --extra if a vocab-coverage view
    # is wanted; it is never part of the reported per-axis/op-class entropy.
]

# Coarse op-class family (face / pocket / adaptive) for the three-regime rollup,
# obtained by stripping the scale suffix from the basename.
def op_family(op_class: str) -> str:
    for fam in ("face", "pocket", "adaptive"):
        if op_class.startswith(fam):
            return fam
    return op_class


def model_encode(val: float, precision: float | None = None) -> list[int]:
    """Round-to-scaled-integer digit decomposition (matches the PATCHED
    target_utils.decompose_value_to_digits / digit_value_head.encode_values_to_digits).
    The prior iterative ``dec_part *= 10`` chain leaked float-epsilon into the fine
    slots (terminal zeros -> 9), inflating fine-digit entropy.

    If ``precision`` (the per-axis tokenizer step, e.g. 0.001 for X/Y/Z, 0.0001 for R)
    is given, the value is first quantized to that grid -- this yields the
    TOKENIZED-TARGET decomposition the model is actually trained against (X/Y/Z lose
    the 10^-4 slot to the 0.001 grid; R keeps it). With precision=None it returns the
    raw 4-decimal SOURCE decomposition.
    Slots: [10^1, 10^0, 10^-1, 10^-2, 10^-3, 10^-4].
    """
    abs_val = abs(float(val))
    if precision:
        abs_val = round(abs_val / precision) * precision
    total = MAX_INT_DIGITS + MAX_DEC_DIGITS
    scale = 10 ** MAX_DEC_DIGITS
    scaled = min(int(round(abs_val * scale)), 10 ** total - 1)
    return [(scaled // (10 ** (total - 1 - pos))) % 10 for pos in range(total)]


def shannon_entropy(counts: Counter | dict[int, int]) -> float:
    """Shannon entropy in bits. Identical to the reference."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        h -= p * np.log2(p)
    return float(h)


def classify_slot(counter: Counter, slot: int) -> dict:
    """Per-slot CNC verdict + top-3 modal digits. Identical to the reference."""
    total = sum(counter.values())
    if total == 0:
        return {
            "verdict": "no_data",
            "top3": [],
            "entropy_bits": 0.0,
            "n": 0,
            "dist": [0.0] * 10,
            "max_deviation_from_uniform": 0.0,
        }
    dist = [counter.get(d, 0) / total for d in range(10)]
    h = shannon_entropy(counter)
    order = sorted(range(10), key=lambda d: -dist[d])
    top3 = [{"digit": d, "frac": dist[d]} for d in order[:3]]
    max_deviation = max(abs(p - 0.1) for p in dist)  # vs uniform 1/10

    if slot in (0, 1):
        if top3[0]["frac"] >= 0.50:
            verdict = "low_entropy_physically_constrained"
        elif h >= 3.0:
            verdict = "high_entropy_unexpected"
        else:
            verdict = "intermediate"
    else:
        if h >= 3.0 and max_deviation <= 0.05:
            verdict = "high_entropy_quasi_uniform"
        elif h < 2.5 or top3[0]["frac"] >= 0.30:
            verdict = "low_entropy_cam_quantized"
        else:
            verdict = "intermediate"

    return {
        "verdict": verdict,
        "top3": top3,
        "entropy_bits": h,
        "n": total,
        "dist": dist,
        "max_deviation_from_uniform": max_deviation,
    }


def strip_comments(line: str) -> str:
    """Remove G-code "(...)" and ";..." comments from a raw source line."""
    line = PAREN_COMMENT_RE.sub(" ", line)
    line = SEMI_COMMENT_RE.sub("", line)
    return line


def load_raw_pairs(sources: list[Path]) -> list[tuple[str, str]]:
    """Read raw .gcode files; return (clean_line, op_class) for every code line.

    op_class is the file basename without the .gcode extension, verbatim.
    Comment-only / blank lines are kept (they simply yield no PARAM_RE matches),
    so the extraction logic is the sole arbiter of what counts as a coordinate.
    """
    pairs: list[tuple[str, str]] = []
    for path in sources:
        op_class = path.stem  # basename sans .gcode
        text = path.read_text(errors="replace")
        for raw_line in text.splitlines():
            clean = strip_comments(raw_line)
            pairs.append((clean, op_class))
    return pairs


def per_axis_distribution(lines: list[str]) -> dict:
    """(a) Per-(axis, slot) digit-value histograms + entropy + verdicts + sign.

    Mirrors digit_entropy_analysis.per_axis_distribution exactly.
    """
    counters: dict[str, list[Counter]] = {
        a: [Counter() for _ in range(N_SLOTS)] for a in AXES
    }
    sign_counters: dict[str, Counter] = {a: Counter() for a in AXES}
    for text in lines:
        for axis, val_str in PARAM_RE.findall(text.upper()):
            axis = axis.upper()
            if axis not in counters:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            sign_counters[axis][
                "neg" if val < 0 else ("zero" if val == 0 else "pos")
            ] += 1
            for s, d in enumerate(model_encode(val)):
                counters[axis][s][d] += 1
    out = {}
    for axis in AXES:
        slot_data = {}
        for s in range(N_SLOTS):
            slot_data[str(s)] = {
                **classify_slot(counters[axis][s], s),
                "slot_label": SLOT_LABELS[s],
            }
        total_sign = sum(sign_counters[axis].values())
        out[axis] = {
            "slots": slot_data,
            "sign": {
                "pos_frac": sign_counters[axis]["pos"] / total_sign if total_sign else 0.0,
                "neg_frac": sign_counters[axis]["neg"] / total_sign if total_sign else 0.0,
                "zero_frac": sign_counters[axis]["zero"] / total_sign if total_sign else 0.0,
                "n": total_sign,
            },
        }
    return out


def per_axis_sign_entropy(per_axis: dict) -> dict:
    """(c) Per-axis sign Shannon entropy in bits over {pos, neg, zero}.

    Reports the explicit sign-channel entropy that the reference only stores as
    fractions; computed here directly from the same pos/neg/zero counts.
    """
    out: dict = {}
    for axis in AXES:
        sign = per_axis[axis]["sign"]
        n = sign["n"]
        if n == 0:
            out[axis] = {"sign_entropy_bits": 0.0, "n": 0,
                         "pos_frac": 0.0, "neg_frac": 0.0, "zero_frac": 0.0}
            continue
        counts = {
            "pos": int(round(sign["pos_frac"] * n)),
            "neg": int(round(sign["neg_frac"] * n)),
            "zero": int(round(sign["zero_frac"] * n)),
        }
        out[axis] = {
            "sign_entropy_bits": shannon_entropy(counts),
            "n": n,
            "pos_frac": sign["pos_frac"],
            "neg_frac": sign["neg_frac"],
            "zero_frac": sign["zero_frac"],
        }
    return out


def per_op_class_slot_distribution(
    pairs: list[tuple[str, str]], slots_of_interest: list[int] = [2, 3, 4, 5]
) -> dict:
    """(b) Per-(op_class, axis, slot) digit-value histograms + entropy.

    Mirrors digit_entropy_analysis.per_op_class_slot_distribution exactly, but
    keyed by raw-file basenames as op classes.
    """
    counters: dict[str, dict[str, dict[int, Counter]]] = {}
    for text, op in pairs:
        if op not in counters:
            counters[op] = {a: {s: Counter() for s in slots_of_interest} for a in AXES}
        for axis, val_str in PARAM_RE.findall(text.upper()):
            axis = axis.upper()
            if axis not in counters[op]:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            digits = model_encode(val)
            for s in slots_of_interest:
                counters[op][axis][s][digits[s]] += 1
    out: dict = {}
    for op, ax_data in counters.items():
        out[op] = {}
        for axis, slot_data in ax_data.items():
            out[op][axis] = {}
            for s, ctr in slot_data.items():
                total = sum(ctr.values())
                if total == 0:
                    out[op][axis][str(s)] = {"n": 0, "entropy_bits": 0.0, "dist": [0.0] * 10}
                    continue
                dist = [ctr.get(d, 0) / total for d in range(10)]
                out[op][axis][str(s)] = {
                    "n": total,
                    "entropy_bits": shannon_entropy(ctr),
                    "dist": dist,
                    "modal_digit": max(range(10), key=lambda d: dist[d]),
                    "modal_frac": max(dist),
                }
    return out


def per_op_family_slot_distribution(pairs: list[tuple[str, str]]) -> dict:
    """Three-regime rollup: collapse face*/pocket*/adaptive* scale variants into
    a single family per the task's 'three op-class regimes' framing."""
    fam_pairs = [(text, op_family(op)) for text, op in pairs]
    return per_op_class_slot_distribution(fam_pairs, slots_of_interest=[2, 3, 4, 5])


def summarize_cnc_verdict(per_axis: dict) -> dict:
    """Roll up per-axis verdicts at slot 5 (10^-4) and slot 0 (tens).

    Mirrors digit_entropy_analysis.summarize_cnc_verdict.
    """
    lowest = {a: per_axis[a]["slots"]["5"]["verdict"] for a in AXES}
    high_order = {a: per_axis[a]["slots"]["0"]["verdict"] for a in AXES}
    bits_5 = {a: per_axis[a]["slots"]["5"]["entropy_bits"] for a in AXES}
    bits_0 = {a: per_axis[a]["slots"]["0"]["entropy_bits"] for a in AXES}
    visible = [a for a in ("X", "Y", "Z") if per_axis[a]["slots"]["5"]["n"] > 0]
    sample = ", ".join(
        f"{a}: {bits_5[a]:.2f} bits ({lowest[a]})" for a in visible
    )
    summary = (
        f"High-order integer digits are near-deterministic across X/Y/Z "
        f"(slot-0 entropy {min(bits_0[a] for a in visible):.2f}-"
        f"{max(bits_0[a] for a in visible):.2f} bits, mass on '0'/'1'); "
        f"the lowest decimal (10^-4) entropy per axis: {sample}."
    )
    return {
        "lowest_decimal": lowest,
        "high_order": high_order,
        "lowest_decimal_entropy_bits": bits_5,
        "high_order_entropy_bits": bits_0,
        "summary": summary,
    }


def compare_to_reference(per_axis: dict, ref_path: Path) -> dict:
    """Side-by-side raw-vs-NPZ comparison of the headline entropy structure."""
    if not ref_path.exists():
        return {"available": False, "reason": f"{ref_path} not found"}
    ref = json.loads(ref_path.read_text())
    ref_axis = ref.get("per_axis", {})
    rows = {}
    for axis in ("X", "Y", "Z", "R", "F"):
        raw_slots = per_axis[axis]["slots"]
        ref_slots = ref_axis.get(axis, {}).get("slots", {})
        per_slot = {}
        for s in range(N_SLOTS):
            raw_h = raw_slots[str(s)]["entropy_bits"]
            ref_h = ref_slots.get(str(s), {}).get("entropy_bits")
            per_slot[str(s)] = {
                "raw_source": raw_h,
                "reference_npz": ref_h,
                "delta": (raw_h - ref_h) if ref_h is not None else None,
            }
        rows[axis] = {
            "slots": per_slot,
            "raw_slot5_verdict": raw_slots["5"]["verdict"],
            "reference_slot5_verdict": ref_slots.get("5", {}).get("verdict"),
            "raw_slot0_entropy": raw_slots["0"]["entropy_bits"],
            "reference_slot0_entropy": ref_slots.get("0", {}).get("entropy_bits"),
        }
    return {"available": True, "reference_path": str(ref_path), "per_axis": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[REPO / s for s in DEFAULT_SOURCES],
        help="Raw .gcode source files (basename = operation class).",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=REPO / "outputs/decoder20260511/audit/digit_entropy.json",
        help="Existing NPZ-derived audit file to compare against (read-only).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs/decoder20260511/audit/raw_gcode_source_entropy.json",
    )
    args = ap.parse_args()

    sources = [p for p in args.sources if p.exists()]
    missing = [str(p) for p in args.sources if not p.exists()]
    if missing:
        print(f"[raw_entropy] WARNING missing sources skipped: {missing}")
    if not sources:
        raise FileNotFoundError("no raw .gcode source files found")

    pairs = load_raw_pairs(sources)
    lines = [t for t, _ in pairs]
    op_classes = sorted({op for _, op in pairs})
    print(
        f"[raw_entropy] read {len(sources)} raw .gcode files, "
        f"{len(lines):,} code lines; op_classes={op_classes}"
    )

    # (a) per-(axis, slot) + sign counts
    per_axis = per_axis_distribution(lines)
    n_extracted = {a: per_axis[a]["slots"]["0"]["n"] for a in AXES}
    print(f"[raw_entropy] (a) per-axis tokens extracted: "
          f"{ {a: n_extracted[a] for a in AXES if n_extracted[a] > 0} }")

    # (b) per-(axis, slot, op-class) + (b') three-regime family rollup
    per_op = per_op_class_slot_distribution(pairs, slots_of_interest=[2, 3, 4, 5])
    per_fam = per_op_family_slot_distribution(pairs)
    print(f"[raw_entropy] (b) per-op-class: {sorted(per_op.keys())}")
    print(f"[raw_entropy] (b') op-family rollup: {sorted(per_fam.keys())}")

    # (c) per-axis sign entropy
    sign = per_axis_sign_entropy(per_axis)
    print("[raw_entropy] (c) sign entropy (bits): "
          + ", ".join(f"{a}={sign[a]['sign_entropy_bits']:.3f}"
                      for a in ("X", "Y", "Z", "R", "F") if sign[a]["n"] > 0))

    cnc = summarize_cnc_verdict(per_axis)
    print(f"[raw_entropy] cnc verdict: {cnc['summary']}")

    cmp = compare_to_reference(per_axis, args.reference)
    if cmp["available"]:
        print("[raw_entropy] RAW vs NPZ slot-5 entropy (bits):")
        for axis in ("X", "Y", "Z", "R", "F"):
            row = cmp["per_axis"][axis]["slots"]["5"]
            print(f"    {axis}: raw={row['raw_source']:.3f}  "
                  f"npz={row['reference_npz']}  "
                  f"delta={row['delta']:+.3f}" if row["reference_npz"] is not None
                  else f"    {axis}: raw={row['raw_source']:.3f}  npz=None")

    out = {
        "meta": {
            "description": (
                "Decoder-free per-(axis, slot[, op-class]) Shannon entropy of "
                "CNC toolpath coordinates computed DIRECTLY from raw CAM G-code "
                "source files. No model, no decoder target-encoding, no NPZ. "
                "Methodology (regex, 6-slot grid, model_encode, entropy, "
                "classify_slot) matched to "
                "scripts/analysis/digit_entropy_analysis.py for comparability."
            ),
            "sources": [str(p.relative_to(REPO)) for p in sources],
            "op_class_definition": "file basename without .gcode extension",
            "op_family_rollup": "face*/pocket*/adaptive* collapsed by prefix",
            "n_code_lines": len(lines),
            "slot_layout": SLOT_LABELS,
            "axes": AXES,
            "n_slots": N_SLOTS,
            "param_regex": PARAM_RE.pattern,
            "comment_handling": "(...) and ;... stripped before extraction",
            "decoupled_from": "NPZ / tokenizer / digit_t model targets",
        },
        "per_axis": per_axis,
        "per_axis_sign": sign,
        "cnc_verdict": cnc,
        "per_op_class_slots": per_op,
        "per_op_family_slots": per_fam,
        "comparison_to_reference_npz": cmp,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[raw_entropy] wrote {args.output}")


if __name__ == "__main__":
    main()
