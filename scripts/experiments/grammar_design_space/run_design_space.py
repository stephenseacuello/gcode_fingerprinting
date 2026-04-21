"""
run_design_space.py
===================

Run experiments A, B, D for the grammar paper's design-space defense.

Experiment A (vocabulary growth):
    Walk the 120 aligned CSV files in canonical order. After each file,
    record the cumulative number of distinct token ids each tokenizer has
    emitted so far. Output a curve per tokenizer.

Experiment B (precision-bounded reconstruction):
    Extract every (parameter, value) pair from the canonicalized corpus.
    Encode then decode each value through every tokenizer. Measure the
    absolute reconstruction error in the parameter's native units (mm for
    X/Y/Z/R, mm/min for F). Report mean and worst-case per parameter.

Experiment D (cross-geometry OOV):
    Split files by operation type (face / pocket / adaptive). For each fold,
    "train" the tokenizer on two operation types, then evaluate the OOV rate
    on the held-out operation. For pretrained / structural tokenizers there
    is no training step but the active-vocab tracking gives an honest read
    on which tokens were ever seen.

Outputs:
    results.json    -- raw numbers
    table_a.tex     -- vocabulary growth table snippet
    table_b.tex     -- reconstruction error table snippet
    table_d.tex     -- OOV table snippet
    growth.csv      -- raw growth curve data for plotting

Usage:
    python scripts/experiments/grammar_design_space/run_design_space.py \
        --data-dir data_clean \
        --out-dir outputs/decoder20260303/grammarpaper/design_space_results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer_zoo import (  # noqa: E402
    all_tokenizers,
    HierarchicalTokenizer,
    GPT4BPETokenizer,
    GPT2BPETokenizer,
    DomainBPETokenizer,
    WordPieceTokenizer,
    CharLevelTokenizer,
    FlatPerValueTokenizer,
    ITokenizer,
)

# Pull GCodeTokenizer for canonicalization-only use in value extraction
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig  # noqa: E402

# Word pattern for extracting (address, value) pairs from canonicalized lines
WORD_RE = re.compile(r"^([A-Z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?$")
ADDR_NUMERIC = set("XYZIJKFRSPQEABC")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus(data_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return [(file_name, [gcode_line, ...]), ...] in sorted order."""
    csv_files = sorted(data_dir.glob("*_aligned.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_aligned.csv files in {data_dir}")
    out: List[Tuple[str, List[str]]] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        if "gcode_string" not in df.columns:
            continue
        lines = [str(x).strip() for x in df["gcode_string"].dropna().tolist() if str(x).strip()]
        if lines:
            out.append((csv_file.name, lines))
    return out


def operation_of(file_name: str) -> str:
    """Map a CSV filename to its operation type (face / pocket / adaptive)."""
    name = file_name.lower()
    if name.startswith("face"):
        return "face"
    if name.startswith("pocket"):
        return "pocket"
    if name.startswith("adaptive") or name.startswith("damage"):
        return "adaptive"
    return "other"


# ---------------------------------------------------------------------------
# Experiment A: vocabulary growth
# ---------------------------------------------------------------------------
def exp_a_vocab_growth(
    tokenizers: List[ITokenizer],
    corpus: List[Tuple[str, List[str]]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Return {tokenizer_name: [(file_idx, cumulative_distinct_ids), ...]}."""
    print("\n=== Experiment A: vocabulary growth ===")
    results: Dict[str, List[Tuple[int, int]]] = {}
    for tok in tokenizers:
        tok.reset_active_vocab()
        curve: List[Tuple[int, int]] = []
        for idx, (fname, lines) in enumerate(corpus, start=1):
            for ln in lines:
                tok.encode(ln)
            curve.append((idx, len(tok.active_vocab())))
        results[tok.name] = curve
        final = curve[-1][1]
        cap = tok.vocab_size()
        print(f"  {tok.name:30s}  active={final:6d}  capacity={cap:6d}  bounded={tok.bounded_vocab}")
    return results


# ---------------------------------------------------------------------------
# Experiment B: precision-bounded reconstruction error
# ---------------------------------------------------------------------------
def extract_value_pairs(corpus: List[Tuple[str, List[str]]]) -> List[Tuple[str, float]]:
    """Pull every (address, value) pair from canonicalized lines."""
    cfg = TokenizerConfig(mode="hybrid", min_freq=1, vocab_size=200000)
    gtok = GCodeTokenizer(cfg)
    pairs: List[Tuple[str, float]] = []
    seen = set()
    for _fname, lines in corpus:
        for ln in lines:
            canon = gtok.canonicalize_line(ln)
            if canon is None:
                continue
            for word in canon.split(" "):
                m = WORD_RE.match(word)
                if not m:
                    continue
                addr, val = m.group(1), m.group(2)
                if addr not in ADDR_NUMERIC or val is None:
                    continue
                try:
                    fv = float(val)
                except ValueError:
                    continue
                key = (addr, fv)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
    print(f"  extracted {len(pairs)} unique (addr,value) pairs")
    return pairs


def _decode_to_value(addr: str, decoded: str) -> float:
    """Best-effort: parse a decoded string back to a float for `addr`."""
    s = decoded.replace(" ", "")
    # Strip common BOS/EOS markers
    for marker in ["[BOS]", "[EOS]", "BOS", "EOS", "PAD", "[PAD]"]:
        s = s.replace(marker, "")
    # Tokens of the form NUM_<addr>_<bucket>
    m = re.search(rf"NUM_{addr}_(-?\d+)", s)
    if m:
        bucket = int(m.group(1))
        # Map bucket back to value via precision
        precision = {"X": 1e-3, "Y": 1e-3, "Z": 1e-3, "I": 1e-4, "J": 1e-4,
                     "K": 1e-4, "F": 1.0, "S": 10.0, "R": 1e-4,
                     "P": 1e-3, "Q": 1e-3, "E": 1e-4,
                     "A": 1e-3, "B": 1e-3, "C": 1e-3}.get(addr, 1e-3)
        return bucket * precision
    # Otherwise look for a literal number adjacent to the address letter
    m = re.search(rf"{addr}\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")
    # Last resort: any float in the decoded string
    m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")
    return float("nan")


def exp_b_reconstruction(
    tokenizers: List[ITokenizer],
    pairs: List[Tuple[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each tokenizer x address, compute mean and max reconstruction error."""
    print("\n=== Experiment B: reconstruction error ===")
    # Group pairs by address
    by_addr: Dict[str, List[float]] = defaultdict(list)
    for addr, val in pairs:
        by_addr[addr].append(val)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tok in tokenizers:
        per_addr: Dict[str, Dict[str, float]] = {}
        for addr, vals in by_addr.items():
            errs: List[float] = []
            unparseable = 0
            for v in vals:
                # Format the value as a typical G-code word
                if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
                    word = f"{addr}{int(round(v))}"
                else:
                    word = f"{addr}{v:.6f}".rstrip("0").rstrip(".")
                try:
                    ids = tok.encode(word)
                    decoded = tok.decode(ids)
                    rec = _decode_to_value(addr, decoded)
                    if rec != rec:  # NaN
                        unparseable += 1
                        continue
                    errs.append(abs(rec - v))
                except Exception:
                    unparseable += 1
            if errs:
                mean = sum(errs) / len(errs)
                worst = max(errs)
            else:
                mean = float("nan")
                worst = float("nan")
            per_addr[addr] = {
                "n": len(vals),
                "n_parsed": len(errs),
                "n_unparseable": unparseable,
                "mean_err": mean,
                "max_err": worst,
            }
        out[tok.name] = per_addr
        # Compact summary
        total_n = sum(d["n"] for d in per_addr.values())
        total_unp = sum(d["n_unparseable"] for d in per_addr.values())
        ranked = sorted(per_addr.items())
        print(f"  {tok.name}: {total_n} values, unparseable={total_unp}")
        for a, d in ranked:
            if d["mean_err"] == d["mean_err"]:
                print(f"    {a}: mean={d['mean_err']:.6g}  max={d['max_err']:.6g}  "
                      f"unparsed={d['n_unparseable']}/{d['n']}")
            else:
                print(f"    {a}: all unparseable ({d['n_unparseable']}/{d['n']})")
    return out


# ---------------------------------------------------------------------------
# Experiment D: cross-geometry OOV
# ---------------------------------------------------------------------------
def exp_d_oov(
    tokenizer_factories: List[callable],
    corpus: List[Tuple[str, List[str]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each held-out operation type, train on the others, eval OOV on held-out.

    tokenizer_factories must be callables returning fresh tokenizer instances.
    """
    print("\n=== Experiment D: cross-geometry OOV ===")
    # Bucket files by operation
    by_op: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for fname, lines in corpus:
        by_op[operation_of(fname)].append((fname, lines))
    ops = sorted(by_op.keys())
    print(f"  operations found: {ops}")

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for factory in tokenizer_factories:
        # Inspect class to get name without instantiating yet
        sample = factory()
        name = sample.name
        out[name] = {}
        for held_out in ops:
            train_files = [f for op, files in by_op.items() if op != held_out for f in files]
            test_files = by_op[held_out]
            train_lines = [ln for _, lines in train_files for ln in lines]

            tok = factory()
            try:
                tok.train(train_lines)
            except Exception as e:
                print(f"    [{name} | held-out {held_out}] train failed: {e}")
                continue

            # "Train" the active vocab by encoding the training corpus once
            tok.reset_active_vocab()
            for ln in train_lines:
                tok.encode(ln)
            train_vocab = set(tok.active_vocab())

            # Evaluate on held-out
            test_total = 0
            test_oov = 0
            for _, lines in test_files:
                for ln in lines:
                    ids = tok.encode(ln)
                    for i in ids:
                        test_total += 1
                        if i not in train_vocab:
                            test_oov += 1

            rate = test_oov / test_total if test_total else 0.0
            out[name][held_out] = {
                "test_tokens": test_total,
                "test_oov": test_oov,
                "oov_rate": rate,
                "train_vocab_size": len(train_vocab),
            }
            print(f"    {name:30s} | held-out {held_out:8s} | "
                  f"OOV {test_oov:7d}/{test_total:7d} = {rate*100:6.3f}%  "
                  f"(train vocab {len(train_vocab)})")
    return out


# ---------------------------------------------------------------------------
# LaTeX emitters
# ---------------------------------------------------------------------------
def latex_table_a(growth: Dict[str, List[Tuple[int, int]]], n_files: int) -> str:
    sample_idxs = [1, max(2, n_files // 6), max(2, n_files // 2), n_files]
    sample_idxs = sorted(set(sample_idxs))
    header = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiment~A: Cumulative Distinct Token Types Observed (Vocabulary Growth)}\n"
        "\\label{tab:exp_a_growth}\n\\scriptsize\n"
        "\\begin{tabular}{@{}l" + "r" * len(sample_idxs) + "@{}}\n\\toprule\n"
    )
    cols = " & ".join(f"\\textbf{{{i}}}" for i in sample_idxs)
    header += f"\\textbf{{Tokenizer}} & {cols} \\\\\n\\midrule\n"
    rows = []
    for name, curve in growth.items():
        lookup = {i: v for i, v in curve}
        cells = []
        for s in sample_idxs:
            v = lookup.get(s, lookup.get(min(lookup.keys(), key=lambda k: abs(k - s))))
            cells.append(f"{v:,}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    footer = (
        "\n\\bottomrule\n"
        f"\\multicolumn{{{len(sample_idxs)+1}}}{{@{{}}l}}{{\\scriptsize "
        f"Columns are file count; cells are cumulative distinct token ids emitted.}}\n"
        "\\end{tabular}\n\\normalsize\n\\end{table}\n"
    )
    return header + body + footer


def latex_table_b(rec: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    addrs = ["X", "Y", "Z", "R", "F"]
    header = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiment~B: Mean Coordinate Reconstruction Error (mm or mm/min for F)}\n"
        "\\label{tab:exp_b_recon}\n\\scriptsize\n"
        "\\begin{tabular}{@{}l" + "r" * len(addrs) + "@{}}\n\\toprule\n"
        "\\textbf{Tokenizer} & " + " & ".join(f"\\textbf{{{a}}}" for a in addrs) + " \\\\\n\\midrule\n"
    )
    rows = []
    for name, per_addr in rec.items():
        cells = []
        for a in addrs:
            d = per_addr.get(a, {})
            me = d.get("mean_err", float("nan"))
            if me != me:
                cells.append("---")
            elif me == 0:
                cells.append("0")
            elif me < 1e-4:
                cells.append(f"{me:.1e}")
            else:
                cells.append(f"{me:.4f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n\\normalsize\n\\end{table}\n"
    )
    return header + body + footer


def latex_table_d(oov: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    ops = sorted({op for d in oov.values() for op in d.keys()})
    header = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiment~D: Held-Out Operation OOV Rate (\\%)}\n"
        "\\label{tab:exp_d_oov}\n\\scriptsize\n"
        "\\begin{tabular}{@{}l" + "r" * len(ops) + "@{}}\n\\toprule\n"
        "\\textbf{Tokenizer} & " + " & ".join(f"\\textbf{{{o}}}" for o in ops) + " \\\\\n\\midrule\n"
    )
    rows = []
    for name, per_op in oov.items():
        cells = []
        for o in ops:
            d = per_op.get(o, {})
            r = d.get("oov_rate", float("nan"))
            if r != r:
                cells.append("---")
            else:
                cells.append(f"{r*100:.3f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    footer = "\n\\bottomrule\n\\end{tabular}\n\\normalsize\n\\end{table}\n"
    return header + body + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data_clean")
    p.add_argument("--out-dir", default="outputs/decoder20260303/grammarpaper/design_space_results")
    p.add_argument("--skip-trained", action="store_true",
                   help="Skip BPE/WordPiece training (faster smoke test)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {data_dir}...")
    corpus = load_corpus(data_dir)
    print(f"  {len(corpus)} files loaded")
    total_lines = sum(len(l) for _, l in corpus)
    print(f"  {total_lines} total lines")

    train_corpus = [ln for _, lines in corpus for ln in lines]

    # Build the full tokenizer zoo for A and B
    if args.skip_trained:
        toks = all_tokenizers(train_corpus=None)
    else:
        toks = all_tokenizers(train_corpus=train_corpus)

    # Experiment A
    growth = exp_a_vocab_growth(toks, corpus)

    # Experiment B (re-encode values; reset active vocab to avoid pollution)
    for t in toks:
        t.reset_active_vocab()
    pairs = extract_value_pairs(corpus)
    rec = exp_b_reconstruction(toks, pairs)

    # Experiment D needs *factories* so each fold gets a fresh instance
    factories = [
        HierarchicalTokenizer,
        GPT4BPETokenizer,
        GPT2BPETokenizer,
        CharLevelTokenizer,
        FlatPerValueTokenizer,
    ]
    if not args.skip_trained:
        factories.append(lambda: DomainBPETokenizer(vocab_size=1000))
        factories.append(lambda: WordPieceTokenizer(vocab_size=1000))
    oov = exp_d_oov(factories, corpus)

    # Persist
    results = {
        "n_files": len(corpus),
        "total_lines": total_lines,
        "experiment_a": {k: v for k, v in growth.items()},
        "experiment_b": rec,
        "experiment_d": oov,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "table_a.tex").write_text(latex_table_a(growth, len(corpus)))
    (out_dir / "table_b.tex").write_text(latex_table_b(rec))
    (out_dir / "table_d.tex").write_text(latex_table_d(oov))

    # Growth CSV for plotting
    with (out_dir / "growth.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "n_files", "active_vocab"])
        for name, curve in growth.items():
            for i, v in curve:
                w.writerow([name, i, v])

    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
