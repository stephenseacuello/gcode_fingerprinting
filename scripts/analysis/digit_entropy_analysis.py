#!/usr/bin/env python3
"""Per-decimal-place digit-value distributions and Shannon entropy for the V8
G-code corpus, to explain the per-digit-position accuracy "bathtub" via source
entropy and the CAM toolpath-generation hypothesis.

Hypothesis under test:
  - High-order integer digits of coordinates are physically constrained
    (mostly 0/1 for the present small-part corpus) -> low entropy -> easy.
  - The lowest decimal (10^-4 in, "ten-thousandths") is shaped by CAM
    interpolation/postprocessing; the analysis empirically classifies each
    axis as either `high_entropy_quasi_uniform` (information-theoretically
    near-unrecoverable from sensors) or `low_entropy_cam_quantized`
    (postprocessor grid; recoverable).

Outputs:
  - audit/digit_entropy.json (per-axis source distributions, pooled-digit_t
    distributions, model accuracy attached, CNC verdicts, Pearson/Spearman r).
  - (optional, with --emit-table) tables/per_digit_position.tex regenerated
    with an Entropy (bits) column and corrected provenance footnote.

Slot layout reproduces src/miracle/model/digit_value_head.py
encode_values_to_digits exactly:
    slots [0,1] = 10^1, 10^0 (integer tens, units)
    slots [2,3,4,5] = 10^-1, 10^-2, 10^-3, 10^-4 (decimals)
PAD=10 in digit_t marks unsupervised positions; we mask identically.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Reused verbatim from scripts/analysis/score_recoverability.py:44.
PARAM_RE = re.compile(r"([XYZIJKRF])(-?\d+\.?\d*)", re.IGNORECASE)

# Fixed 6-slot grid the model uses for every axis (max over PARAM_VALUE_CONFIG).
MAX_INT_DIGITS = 2
MAX_DEC_DIGITS = 4
N_SLOTS = MAX_INT_DIGITS + MAX_DEC_DIGITS  # 6
DIGIT_PAD = 10

SLOT_LABELS = {
    0: "10^1 (tens)",
    1: "10^0 (units)",
    2: "10^-1 (tenths)",
    3: "10^-2 (hundredths)",
    4: "10^-3 (thousandths)",
    5: "10^-4 (ten-thousandths)",
}

AXES = ["X", "Y", "Z", "R", "F", "I", "J"]


def model_encode(val: float) -> list[int]:
    """Replicate digit_value_head.encode_values_to_digits semantics in Python.

    Returns 6 digit values 0-9 for the slots
    [10^1, 10^0, 10^-1, 10^-2, 10^-3, 10^-4].
    """
    abs_val = abs(float(val))
    int_part = int(abs_val)
    slots: list[int] = []
    for pos in range(MAX_INT_DIGITS):
        power = MAX_INT_DIGITS - 1 - pos
        divisor = 10 ** power
        slots.append((int_part // divisor) % 10)
    dec_part = abs_val - int_part
    for _ in range(MAX_DEC_DIGITS):
        dec_part *= 10
        slots.append(int(dec_part) % 10)
    return slots


def shannon_entropy(counts: Counter | dict[int, int]) -> float:
    """Shannon entropy in bits."""
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
    """Per-slot CNC verdict + top-3 modal digits."""
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


def load_gcode_texts(preproc_root: Path, split: str, folds: list[int]) -> list[str]:
    texts: list[str] = []
    for fold in folds:
        path = preproc_root / f"fold_{fold}" / f"{split}_sequences.npz"
        z = np.load(path, allow_pickle=True)
        texts.extend(z["gcode_texts"].tolist())
    return texts


def load_gcode_with_op_class(
    preproc_root: Path, split: str, folds: list[int]
) -> list[tuple[str, str]]:
    """Return list of (gcode_text, op_class_name) pairs across folds."""
    pairs: list[tuple[str, str]] = []
    for fold in folds:
        path = preproc_root / f"fold_{fold}" / f"{split}_sequences.npz"
        z = np.load(path, allow_pickle=True)
        for t, op in zip(z["gcode_texts"].tolist(), z["operation_type_names"].tolist()):
            pairs.append((t, op))
    return pairs


def per_axis_distribution(texts: list[str]) -> dict:
    """Per-(axis, slot) digit-value histograms + entropy + verdicts."""
    counters: dict[str, list[Counter]] = {
        a: [Counter() for _ in range(N_SLOTS)] for a in AXES
    }
    sign_counters: dict[str, Counter] = {a: Counter() for a in AXES}
    for text in texts:
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


def per_op_class_slot_distribution(
    pairs: list[tuple[str, str]], slots_of_interest: list[int] = [2, 3, 4, 5]
) -> dict:
    """Per-(op_class, axis, slot) digit-value histograms + entropy.

    Used to test whether quantization concentrates in specific operation classes
    (face, pocket = round-grid CAM) and washes out in adaptive HSM toolpaths.
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


def float_epsilon_audit(texts: list[str]) -> dict:
    """Audit how often a CAM-authored slot-5 = '0' (terminal-zero string) becomes
    slot-5 = '9' after float-conversion + iterative int(dec*10)%10 encoding.

    This quantifies the {0, 9}-bimodal artifact in pooled digit_t at slot 5.
    """
    # For each (axis, val_str), classify the value by its *string* trailing
    # decimal vs the model_encode result.
    counts = Counter()  # (axis, str_d5, encoded_d5)
    axis_specific: dict[str, Counter] = {a: Counter() for a in AXES}
    n_axis: dict[str, int] = {a: 0 for a in AXES}
    for text in texts:
        for axis, val_str in PARAM_RE.findall(text.upper()):
            axis = axis.upper()
            if axis not in AXES:
                continue
            # String slot-5: zero-pad decimal to 4, take 4th char (index 3) of decimal
            v = val_str.lstrip("-")
            if "." in v:
                int_part, dec_part = v.split(".", 1)
            else:
                int_part, dec_part = v, ""
            dec_padded = (dec_part + "0000")[:4]
            str_d5 = int(dec_padded[3])
            try:
                fval = float(val_str)
            except ValueError:
                continue
            enc_d5 = model_encode(fval)[5]
            counts[(axis, str_d5, enc_d5)] += 1
            axis_specific[axis][(str_d5, enc_d5)] += 1
            n_axis[axis] += 1
    # Headline stats per axis
    out: dict = {}
    total_zero_to_nine = 0
    total_zero = 0
    total_match = 0
    grand_total = 0
    for axis in AXES:
        if n_axis[axis] == 0:
            continue
        c = axis_specific[axis]
        n = n_axis[axis]
        str_zero = sum(v for (s, _), v in c.items() if s == 0)
        zero_to_nine = c.get((0, 9), 0)
        zero_to_zero = c.get((0, 0), 0)
        match = sum(v for (s, e), v in c.items() if s == e)
        out[axis] = {
            "n": n,
            "str_slot5_zero_frac": str_zero / n,
            "str0_enc0_frac": zero_to_zero / n,
            "str0_enc9_frac": zero_to_nine / n,
            "exact_match_frac": match / n,
            "str0_enc9_share_of_str0": zero_to_nine / str_zero if str_zero else 0.0,
        }
        total_zero += str_zero
        total_zero_to_nine += zero_to_nine
        total_match += match
        grand_total += n
    out["_pooled"] = {
        "n": grand_total,
        "str_slot5_zero_frac": total_zero / grand_total if grand_total else 0.0,
        "str0_enc9_frac": total_zero_to_nine / grand_total if grand_total else 0.0,
        "exact_match_frac": total_match / grand_total if grand_total else 0.0,
        "str0_enc9_share_of_str0": total_zero_to_nine / total_zero if total_zero else 0.0,
        "interpretation": (
            "A CAM-authored '.X0' terminal-zero value should encode to slot5=0. "
            "It encodes to slot5=9 instead whenever float(val_str) yields a "
            "value slightly less than val_str (e.g. '1.5680' -> 1.567999...), "
            "and the model's int(dec_part*10)%10 truncation hits 9. "
            "str0_enc9_share_of_str0 is the fraction of CAM-authored terminal "
            "zeros that the model's targets corrupt to 9."
        ),
    }
    return out


def pooled_distribution_from_digit_t(fw_sweep_root: Path) -> tuple[dict, list[int]]:
    """Pooled per-slot histograms from the model's digit_t targets across folds.

    This is the overlay-aligned denominator: same population as
    diagnose_numeric_accuracy.py uses for accuracy.
    """
    counters = [Counter() for _ in range(N_SLOTS)]
    npz_paths = sorted(fw_sweep_root.glob("fold_*/*/results/predictions.npz"))
    if not npz_paths:
        raise FileNotFoundError(
            f"no predictions.npz under {fw_sweep_root}/fold_*/*/results/"
        )
    folds_seen: list[int] = []
    for path in npz_paths:
        fold = int(path.parent.parent.parent.name.split("_")[1])
        folds_seen.append(fold)
        z = np.load(path)
        dt = z["digit_t"]
        for s in range(N_SLOTS):
            slot = dt[..., s].ravel()
            slot = slot[slot != DIGIT_PAD]
            vals, cnts = np.unique(slot, return_counts=True)
            for d, c in zip(vals.tolist(), cnts.tolist()):
                counters[s][int(d)] += int(c)
    per_slot = {}
    for s in range(N_SLOTS):
        c = counters[s]
        total = sum(c.values())
        dist = [c.get(d, 0) / total if total else 0.0 for d in range(10)]
        per_slot[str(s)] = {
            "dist_from_digit_t": dist,
            "entropy_bits": shannon_entropy(c),
            "n": total,
            "slot_label": SLOT_LABELS[s],
        }
    return per_slot, sorted(set(folds_seen))


def attach_accuracy(per_slot: dict, accuracy_json_path: Path) -> None:
    """Add model_accuracy_mean/std from numeric_accuracy_diagnosis.json."""
    j = json.loads(accuracy_json_path.read_text())
    n_pos = len(j["per_fold"][0]["per_position"])
    for s in range(n_pos):
        accs = [f["per_position"][s]["accuracy"] for f in j["per_fold"]]
        per_slot[str(s)]["model_accuracy_mean"] = float(np.mean(accs))
        per_slot[str(s)]["model_accuracy_std"] = float(np.std(accs, ddof=1))


def correlate(per_slot: dict) -> dict:
    """Pearson + Spearman r between pooled entropy and model accuracy."""
    from scipy import stats

    entropies = [per_slot[str(s)]["entropy_bits"] for s in range(N_SLOTS)]
    accs = [per_slot[str(s)]["model_accuracy_mean"] for s in range(N_SLOTS)]
    pr = stats.pearsonr(entropies, accs)
    sr = stats.spearmanr(entropies, accs)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "n_positions": N_SLOTS,
        "entropies_bits": entropies,
        "accuracies": accs,
    }


def summarize_cnc_verdict(per_axis: dict) -> dict:
    """Roll up per-axis verdicts at slot 5 (10^-4) and slot 0 (tens)."""
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


def emit_latex_table(output_path: Path, per_slot: dict) -> None:
    """Regenerate tables/per_digit_position.tex with an Entropy column."""
    rows = []
    for s in range(N_SLOTS):
        acc_m = per_slot[str(s)]["model_accuracy_mean"]
        acc_s = per_slot[str(s)]["model_accuracy_std"]
        h = per_slot[str(s)]["entropy_bits"]
        n = per_slot[str(s)]["n"]
        pos_str = SLOT_LABELS[s].replace("^", r"$^{") + "}$"
        # Above produces e.g. "10$^{1} (tens)}$" — rewrite cleanly:
        pos_str = {
            0: r"$10^{1}$ (tens)",
            1: r"$10^{0}$ (units)",
            2: r"$10^{-1}$ (tenths)",
            3: r"$10^{-2}$ (hundredths)",
            4: r"$10^{-3}$ (thousandths)",
            5: r"$10^{-4}$ (ten-thousandths)",
        }[s]
        rows.append(
            f"{s} & {pos_str} & ${acc_m:.4f} \\pm {acc_s:.4f}$ "
            f"& ${h:.3f}$ & ${n:,}$ \\\\"
        )
    body = "\n".join(rows)
    content = (
        "% V8 FULL_WINDOW 5-FOLD --- per-digit-position breakdown.\n"
        "% Regenerated by scripts/analysis/digit_entropy_analysis.py\n"
        "%   (use --emit-table to rewrite this file).\n"
        "% Note: scripts/analysis/diagnose_numeric_accuracy.py's --sweep-root\n"
        "% default points at a non-existent per_row_5fold path; the accuracy\n"
        "% column is in fact computed on the full_window predictions globbed\n"
        "% at checkpoints/full_window_5fold/fold_*/*/results/predictions.npz\n"
        "% (audit/numeric_accuracy_diagnosis.json), confirmed by matching the\n"
        "% per-fold n_samples [132, 110, 108, 108, 93].\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Per-digit-position accuracy and source entropy on V8 "
        "full\\_window (5-fold mean $\\pm$ std). Slot 0 is the most-significant "
        "magnitude digit (tens place); slot 5 is the least-significant decimal "
        "(ten-thousandths). Entropy is Shannon entropy in bits of the empirical "
        "digit-value distribution across all supervised positions, pooled across "
        "axes (matching the accuracy denominator). The bathtub-shaped accuracy "
        "curve tracks per-slot source entropy: positions where the empirical "
        "digit distribution is highly non-uniform are recovered accurately; "
        "positions with near-uniform distributions are not. See "
        "Figure~\\ref{fig:entropy_accuracy_overlay} for the overlay and "
        "Figure~\\ref{fig:digit_value_histograms} for the per-axis decomposition.}\n"
        "\\label{tab:per_digit_position}\n"
        "\\begin{tabular}{c l r r r}\n"
        "\\toprule\n"
        "Slot & Place & Accuracy (5-fold) & Entropy (bits) & $n$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\par\\smallskip\n"
        "\\footnotesize\\textit{Per-axis decomposition and CNC verdicts (lowest "
        "decimal classified as either \\emph{high-entropy quasi-uniform} --- "
        "information-theoretically near-unrecoverable from sensors --- or "
        "\\emph{low-entropy CAM-quantized}, postprocessor grid) are released in "
        "\\texttt{audit/digit\\_entropy.json}.}\n"
        "\\end{table}\n"
    )
    output_path.write_text(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preproc-root",
        type=Path,
        default=REPO / "outputs/decoder20260511/preprocessed/per_row",
    )
    ap.add_argument(
        "--fw-sweep-root",
        type=Path,
        default=REPO / "outputs/decoder20260511/checkpoints/full_window_5fold",
    )
    ap.add_argument(
        "--accuracy-json",
        type=Path,
        default=REPO
        / "outputs/decoder20260511/audit/numeric_accuracy_diagnosis.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs/decoder20260511/audit/digit_entropy.json",
    )
    ap.add_argument(
        "--emit-table",
        type=Path,
        default=None,
        help="If set, regenerate the per_digit_position.tex table here.",
    )
    ap.add_argument(
        "--split", default="test", choices=["test", "train", "train+test"]
    )
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    splits = args.split.split("+")
    texts: list[str] = []
    pairs: list[tuple[str, str]] = []
    for sp in splits:
        texts.extend(load_gcode_texts(args.preproc_root, sp, args.folds))
        pairs.extend(load_gcode_with_op_class(args.preproc_root, sp, args.folds))
    print(
        f"[digit_entropy] loaded {len(texts):,} gcode_text rows "
        f"(split={args.split}, folds={args.folds})"
    )

    per_axis = per_axis_distribution(texts)
    print(f"[digit_entropy] per-axis distributions over {len(AXES)} axes done.")

    per_op = per_op_class_slot_distribution(pairs, slots_of_interest=[2, 3, 4, 5])
    print(
        f"[digit_entropy] per-op-class distribution: "
        f"{sorted(per_op.keys())}"
    )

    epsilon = float_epsilon_audit(texts)
    print(
        f"[digit_entropy] float-epsilon audit: pooled str0_enc9 "
        f"= {epsilon['_pooled']['str0_enc9_frac']:.2%} of all slot5 positions, "
        f"or {epsilon['_pooled']['str0_enc9_share_of_str0']:.2%} of CAM-authored "
        f"terminal-zero values"
    )

    pooled, folds_seen = pooled_distribution_from_digit_t(args.fw_sweep_root)
    print(
        f"[digit_entropy] pooled digit_t distributions; folds={folds_seen}; "
        f"n_per_slot={[pooled[str(s)]['n'] for s in range(N_SLOTS)]}"
    )

    attach_accuracy(pooled, args.accuracy_json)
    corr = correlate(pooled)
    print(
        f"[digit_entropy] pearson r(entropy, accuracy) = "
        f"{corr['pearson_r']:+.3f} (p={corr['pearson_p']:.2g})"
    )
    print(
        f"[digit_entropy] spearman r              = "
        f"{corr['spearman_r']:+.3f} (p={corr['spearman_p']:.2g})"
    )

    cnc = summarize_cnc_verdict(per_axis)
    print(f"[digit_entropy] cnc verdict: {cnc['summary']}")

    out = {
        "meta": {
            "split": args.split,
            "folds": folds_seen,
            "slot_layout": SLOT_LABELS,
            "pad_value": DIGIT_PAD,
            "source": {
                "per_axis": "gcode_texts (per_row preprocessed NPZ)",
                "pooled": "full_window digit_t (model targets)",
                "accuracy": str(args.accuracy_json.relative_to(REPO)),
            },
            "n_axes": len(AXES),
            "n_slots": N_SLOTS,
        },
        "per_axis": per_axis,
        "pooled_position": pooled,
        "cnc_verdict": cnc,
        "entropy_vs_accuracy_correlation": corr,
        "per_op_class_slots": per_op,
        "float_epsilon_audit": epsilon,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[digit_entropy] wrote {args.output}")

    if args.emit_table is not None:
        emit_latex_table(args.emit_table, pooled)
        print(f"[digit_entropy] wrote LaTeX table {args.emit_table}")


if __name__ == "__main__":
    main()
