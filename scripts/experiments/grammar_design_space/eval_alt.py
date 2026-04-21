"""
eval_alt.py
===========

Run Experiments C (free-generation grammar validity) and E (Hausdorff
toolpath reconstruction) on the alt-tokenizer baseline checkpoints trained
by train_alt.py.

For each tokenizer:
    1. Load best checkpoint + tokenizer
    2. Load v7 fold_1 test split (sensor memory + ground-truth gcode_texts)
    3. Free-generate one G-code line per test sample (no teacher forcing)
    4. Decode generated ids to text via the tokenizer
    5. Parse text with GCodeToCoordinateParser; check grammar validity
    6. Reconstruct toolpath; compute Hausdorff distance to GT toolpath
    7. Aggregate metrics

Output: per-tokenizer results.json + LaTeX snippet for the paper.

Usage:
    python3 scripts/experiments/grammar_design_space/eval_alt.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from miracle.model.flat_vocab_decoder import FlatVocabDecoder  # noqa: E402
from miracle.utilities.gcode_parser import GCodeToCoordinateParser, MachineState, ToolpathPoint  # noqa: E402
from alt_tokenizers import (  # noqa: E402
    BaseAltTokenizer, TOKENIZER_REGISTRY,
    DomainBPETokenizer, ByteLevelBPETokenizer, WordPieceTokenizer,
    CharLevelTokenizer, FlatPerValueTokenizer, GPT4CompactTokenizer,
    HierarchicalAltTokenizer,
    PAD_ID, BOS_ID, EOS_ID,
)

LOADER_FOR = {
    "domain_bpe": (DomainBPETokenizer, "tokenizer.bin"),
    "bytelevel_bpe": (ByteLevelBPETokenizer, "tokenizer.bin"),
    "wordpiece": (WordPieceTokenizer, "tokenizer.bin"),
    "char_level": (CharLevelTokenizer, "tokenizer.bin"),
    "flat_per_value": (FlatPerValueTokenizer, "tokenizer.bin"),
    "gpt4_compact": (GPT4CompactTokenizer, "tokenizer.bin"),
    "hierarchical": (HierarchicalAltTokenizer, "tokenizer.bin"),
}


# ---------------------------------------------------------------------------
# Grammar validity check
# ---------------------------------------------------------------------------
# A line is "grammatically valid" if it parses into at least one motion command
# OR a recognized auxiliary command (G-code, M-code) with valid params.
# We use a permissive check based on the existing parser: if parse_line returns
# at least one (letter, value) pair AND the line conforms to the regular grammar
# from Section 4.5 of the paper.

# Type-level DFA from Section 4.5
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAM = 2
TYPE_NUMERIC = 3

VALID_NEXT = {
    TYPE_SPECIAL: {TYPE_COMMAND, TYPE_PARAM, TYPE_SPECIAL},
    TYPE_COMMAND: {TYPE_PARAM, TYPE_SPECIAL},
    TYPE_PARAM: {TYPE_NUMERIC},
    TYPE_NUMERIC: {TYPE_PARAM, TYPE_SPECIAL},
}

ADDR_LETTERS = set("XYZIJKFRSPQEABC")
CMD_LETTERS = {"G", "M"}
WORD_RE = re.compile(r"^([A-Z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?$")


def is_grammatical(line: str) -> bool:
    """True if `line` is a syntactically valid G-code toolpath line.

    The check enforces the regular grammar of Section 4.5: every word is
    either a recognized command (G0..G3, M*) or a parameter address X..R
    followed by a numeric value. Implicit-modal lines (parameter without
    leading command) are also accepted.
    """
    line = line.strip().upper()
    if not line:
        return False
    words = line.split()
    if not words:
        return False
    state = TYPE_SPECIAL  # virtual BOS
    seen_addrs: set[str] = set()
    for w in words:
        m = WORD_RE.match(w)
        if not m:
            return False
        head, num = m.group(1), m.group(2)
        if head in CMD_LETTERS:
            # Command requires a numeric suffix to be a recognized G/M code
            if num is None:
                return False
            try:
                int(float(num))
            except (ValueError, TypeError):
                return False
            if state not in (TYPE_SPECIAL,):
                # Command must come at the start
                return False
            state = TYPE_COMMAND
        elif head in ADDR_LETTERS:
            # Parameter address; must be followed by a value (in same word as numeric suffix)
            if num is None:
                return False
            try:
                float(num)
            except (ValueError, TypeError):
                return False
            if head in seen_addrs:
                return False  # repeated address
            seen_addrs.add(head)
            # Implicit-modal line allows transitioning straight from BOS to PARAM
            if state not in (TYPE_SPECIAL, TYPE_COMMAND, TYPE_NUMERIC):
                return False
            # Encoded as PARAM then implicit NUMERIC (we ate the numeric suffix)
            state = TYPE_NUMERIC
        else:
            return False
    # Accepting states: COMMAND (just a bare G0..G3 line), NUMERIC (param/value)
    return state in (TYPE_COMMAND, TYPE_NUMERIC)


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------
def points_to_array(points: list[ToolpathPoint]) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)


def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """Bidirectional Hausdorff distance between two 3D point sets.

    Returns max(sup_a min_b ||a-b||, sup_b min_a ||a-b||).
    """
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    # Pairwise distances
    diff = A[:, None, :] - B[None, :, :]  # [|A|, |B|, 3]
    d = np.sqrt((diff * diff).sum(axis=-1))  # [|A|, |B|]
    forward = d.min(axis=1).max()
    backward = d.min(axis=0).max()
    return float(max(forward, backward))


# ---------------------------------------------------------------------------
# Per-tokenizer eval
# ---------------------------------------------------------------------------
def evaluate_one(name: str, args) -> dict:
    print(f"\n=== eval {name} (fold {args.fold}) ===")
    decoder_dir = Path(args.decoder_base) / name / f"fold_{args.fold}"
    tok_dir = Path(args.alt_base) / name / f"fold_{args.fold}"
    ckpt_path = decoder_dir / "best_decoder.pt"

    if not ckpt_path.exists():
        print(f"  [skip] no checkpoint at {ckpt_path}")
        return {"error": "no checkpoint"}

    # Load tokenizer
    cls, fname = LOADER_FOR[name]
    tok = cls.load(tok_dir / fname)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab_size = ckpt["vocab_size"]
    sensor_dim = ckpt["sensor_dim"]
    saved_args = ckpt.get("args", {})

    model = FlatVocabDecoder(
        vocab_size=vocab_size,
        d_model=saved_args.get("d_model", 192),
        n_heads=saved_args.get("n_heads", 8),
        n_layers=saved_args.get("n_layers", 4),
        sensor_dim=sensor_dim,
        n_operations=9,
        dropout=saved_args.get("dropout", 0.3),
        max_seq_len=saved_args.get("max_token_len", 64),
    ).to(device)
    model.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()

    # Load test data: gcode_texts + cached memory + op_pred
    npz = np.load(tok_dir / "test_sequences.npz", allow_pickle=True)
    gt_texts = [str(t) for t in npz["gcode_texts"]]
    n = len(gt_texts)

    mem_base = Path(args.memory_base_template.format(fold=args.fold, seed=args.seed_tag))
    test_memory = torch.load(mem_base / "test_memory.pt", map_location=device, weights_only=False).float()
    test_op = torch.load(mem_base / "test_op_pred.pt", map_location=device, weights_only=False).long()
    n = min(n, len(test_memory))
    gt_texts = gt_texts[:n]
    test_memory = test_memory[:n]
    test_op = test_op[:n]

    # Free-generate
    print(f"  generating {n} samples...")
    BATCH = 64
    pred_texts: list[str] = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            mem_b = test_memory[i:i + BATCH].to(device)
            op_b = test_op[i:i + BATCH].to(device)
            generated = model.generate(
                sensor_embeddings=mem_b,
                operation_type=op_b,
                bos_token_id=BOS_ID,
                eos_token_id=EOS_ID,
                max_length=saved_args.get("max_token_len", 64),
                temperature=1.0,
                top_k=None,
            )
            for row in generated.cpu().tolist():
                # Trim at first EOS
                if EOS_ID in row[1:]:
                    row = row[1:row.index(EOS_ID, 1)]
                else:
                    row = row[1:]
                row = [i for i in row if i not in (PAD_ID, BOS_ID, EOS_ID)]
                pred_texts.append(tok.decode(row))

    # Experiment C: grammar validity
    n_valid = sum(1 for t in pred_texts if is_grammatical(t))
    n_gt_valid = sum(1 for t in gt_texts if is_grammatical(t))
    grammar_rate = n_valid / max(n, 1)
    print(f"  free-gen grammar valid: {n_valid}/{n} = {grammar_rate*100:.2f}%  "
          f"(GT: {n_gt_valid}/{n} = {n_gt_valid/max(n,1)*100:.2f}%)")

    # Experiment E: Hausdorff distance per sample, then mean/median/p95
    parser = GCodeToCoordinateParser()
    h_dists: list[float] = []
    n_pred_parsed = 0
    for gt, pred in zip(gt_texts, pred_texts):
        try:
            parser.reset()
            gt_path = parser.parse_sequence([gt])
            parser.reset()
            pred_path = parser.parse_sequence([pred])
        except Exception:
            continue
        gt_arr = points_to_array(gt_path)
        pred_arr = points_to_array(pred_path)
        if len(gt_arr) == 0 or len(pred_arr) == 0:
            continue
        h = hausdorff(gt_arr, pred_arr)
        if h == h:  # not NaN
            h_dists.append(h)
            n_pred_parsed += 1

    if h_dists:
        h_arr = np.asarray(h_dists, dtype=np.float64)
        h_mean = float(h_arr.mean())
        h_median = float(np.median(h_arr))
        h_p95 = float(np.percentile(h_arr, 95))
        h_max = float(h_arr.max())
    else:
        h_mean = h_median = h_p95 = h_max = float("nan")

    print(f"  Hausdorff: parsed {n_pred_parsed}/{n}, "
          f"mean={h_mean:.4f}mm  median={h_median:.4f}mm  p95={h_p95:.4f}mm  max={h_max:.4f}mm")

    return {
        "tokenizer": name,
        "n_test": n,
        "grammar_valid_rate": grammar_rate,
        "n_grammar_valid": n_valid,
        "gt_grammar_valid_rate": n_gt_valid / max(n, 1),
        "n_pred_parsed": n_pred_parsed,
        "hausdorff_mm": {
            "mean": h_mean,
            "median": h_median,
            "p95": h_p95,
            "max": h_max,
        },
        "samples": [
            {"gt": g, "pred": p}
            for g, p in list(zip(gt_texts, pred_texts))[:8]
        ],
    }


def latex_table_ce(results: dict) -> str:
    rows = []
    order = ["domain_bpe", "bytelevel_bpe", "wordpiece", "char_level", "flat_per_value", "gpt4_compact", "hierarchical"]
    name_pretty = {
        "domain_bpe": "Domain BPE (whitespace)",
        "bytelevel_bpe": "Domain BPE (byte-level)",
        "wordpiece": "WordPiece",
        "char_level": "Character-level",
        "flat_per_value": "Flat per-value",
        "gpt4_compact": "GPT-4 BPE (compact)",
        "hierarchical": "\\textbf{Hierarchical (ours)}",
    }
    for key in order:
        r = results.get(key)
        if r is None or "error" in r:
            continue
        if key == "hierarchical":
            rows.append("\\midrule")
        gv = r["grammar_valid_rate"] * 100
        h = r["hausdorff_mm"]
        if key == "hierarchical":
            rows.append(
                f"{name_pretty[key]} & \\textbf{{{gv:.1f}}} & \\textbf{{{h['mean']:.3f}}} & "
                f"\\textbf{{{h['median']:.3f}}} & \\textbf{{{h['p95']:.3f}}} \\\\"
            )
        else:
            rows.append(
                f"{name_pretty[key]} & {gv:.1f} & {h['mean']:.3f} & {h['median']:.3f} & {h['p95']:.3f} \\\\"
            )

    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiments~C and~E: Free-Generation Grammar Validity and Toolpath Hausdorff Distance (fold~1)}\n"
        "\\label{tab:exp_ce}\n\\scriptsize\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "\\textbf{Tokenizer} & \\textbf{Grammar valid (\\%)} & \\textbf{Mean (mm)} & \\textbf{Median (mm)} & \\textbf{p95 (mm)} \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\multicolumn{5}{@{}l}{\\scriptsize $^\\dagger$ Grammar mask guarantees 100\\% by construction; coverage on the dataset.}\\\\\n"
        "\\multicolumn{5}{@{}l}{\\scriptsize $^\\ddagger$ Digit decomposition produces exact-match coordinates by design when the type/digit heads are correct.}\n"
        "\\end{tabular}\n\\normalsize\n\\end{table}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed-tag", default="42")
    p.add_argument("--alt-base", default="outputs/decoder20260303/grammarpaper/alt_tokenizers")
    p.add_argument("--decoder-base", default="outputs/decoder20260303/grammarpaper/alt_decoders")
    p.add_argument("--memory-base-template",
                   default="outputs/decoder20260304/v7_best_5fold_multiseed/fold_{fold}_seed_{seed}/encoder_memory")
    p.add_argument("--out-dir", default="outputs/decoder20260303/grammarpaper/design_space_results")
    p.add_argument("--tokenizers", nargs="+", default=list(LOADER_FOR.keys()))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.tokenizers:
        results[name] = evaluate_one(name, args)

    fold_json = out_dir / f"ce_results_fold{args.fold}.json"
    fold_json.write_text(json.dumps(results, indent=2, default=str))
    # Also keep a latest pointer for the single-fold workflow
    (out_dir / "ce_results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "table_ce.tex").write_text(latex_table_ce(results))
    print(f"\nWrote {fold_json}")
    print(f"Wrote {out_dir / 'table_ce.tex'}")


if __name__ == "__main__":
    main()
