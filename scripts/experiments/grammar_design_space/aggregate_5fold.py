"""
aggregate_5fold.py
==================

Aggregate per-fold ce_results_fold{1..5}.json into a single mean+std summary
and emit updated LaTeX tables for the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/decoder20260303/grammarpaper/design_space_results")
ORDER = ["domain_bpe", "bytelevel_bpe", "wordpiece", "char_level", "flat_per_value", "gpt4_compact", "hierarchical"]
PRETTY = {
    "domain_bpe": "Domain BPE (whitespace)",
    "bytelevel_bpe": "Domain BPE (byte-level)",
    "wordpiece": "WordPiece",
    "char_level": "Character-level",
    "flat_per_value": "Flat per-value",
    "gpt4_compact": "GPT-4 BPE (compact)",
    "hierarchical": "\\textbf{Hierarchical (ours)}",
}


def main() -> None:
    fold_data = {}
    for fold in range(1, 6):
        fp = OUT_DIR / f"ce_results_fold{fold}.json"
        if not fp.exists():
            print(f"  [warn] {fp} missing")
            continue
        fold_data[fold] = json.loads(fp.read_text())

    if not fold_data:
        raise SystemExit("No fold results found")

    print(f"Loaded {len(fold_data)} folds")

    # Aggregate
    agg = {}
    for tok in ORDER:
        gv_rates = []
        hmean = []
        hmed = []
        hp95 = []
        n_test_total = 0
        for fold, data in fold_data.items():
            r = data.get(tok)
            if r is None or "error" in r:
                continue
            gv_rates.append(r["grammar_valid_rate"])
            hmean.append(r["hausdorff_mm"]["mean"])
            hmed.append(r["hausdorff_mm"]["median"])
            hp95.append(r["hausdorff_mm"]["p95"])
            n_test_total += r["n_test"]
        if not gv_rates:
            continue
        agg[tok] = {
            "n_folds": len(gv_rates),
            "n_test_total": n_test_total,
            "grammar_valid": {
                "mean": float(np.mean(gv_rates)),
                "std": float(np.std(gv_rates, ddof=1)) if len(gv_rates) > 1 else 0.0,
                "values": gv_rates,
            },
            "hausdorff_mm": {
                "mean": {"mean": float(np.mean(hmean)),
                         "std": float(np.std(hmean, ddof=1)) if len(hmean) > 1 else 0.0},
                "median": {"mean": float(np.mean(hmed)),
                           "std": float(np.std(hmed, ddof=1)) if len(hmed) > 1 else 0.0},
                "p95": {"mean": float(np.mean(hp95)),
                        "std": float(np.std(hp95, ddof=1)) if len(hp95) > 1 else 0.0},
            },
        }

    (OUT_DIR / "ce_5fold_summary.json").write_text(json.dumps(agg, indent=2))
    print(f"Wrote {OUT_DIR / 'ce_5fold_summary.json'}")

    # Print summary
    print("\n=== 5-fold mean ± std ===")
    print(f"{'tokenizer':25s}  {'grammar(%)':>14s}  {'H_mean(mm)':>14s}  {'H_med(mm)':>14s}  {'H_p95(mm)':>14s}")
    for tok in ORDER:
        if tok not in agg:
            continue
        a = agg[tok]
        gv_m = a["grammar_valid"]["mean"] * 100
        gv_s = a["grammar_valid"]["std"] * 100
        hm = a["hausdorff_mm"]["mean"]
        hd = a["hausdorff_mm"]["median"]
        hp = a["hausdorff_mm"]["p95"]
        print(
            f"{tok:25s}  "
            f"{gv_m:7.2f} ± {gv_s:5.2f}  "
            f"{hm['mean']:6.3f} ± {hm['std']:5.3f}  "
            f"{hd['mean']:6.3f} ± {hd['std']:5.3f}  "
            f"{hp['mean']:6.3f} ± {hp['std']:5.3f}"
        )

    # Emit Experiment C table (grammar validity, 5-fold)
    rows = []
    for tok in ORDER:
        if tok not in agg:
            continue
        a = agg[tok]
        gv_m = a["grammar_valid"]["mean"] * 100
        gv_s = a["grammar_valid"]["std"] * 100
        if tok == "hierarchical":
            rows.append("\\midrule")
        # Bold the cell text only when the value is striking; standard formatting otherwise
        rows.append(
            f"{PRETTY[tok]} & {gv_m:.1f} $\\pm$ {gv_s:.1f} \\\\"
        )
    table_c = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiment~C: Free-Generation Grammar Validity (5-fold mean $\\pm$ std)}\n"
        "\\label{tab:exp_c}\n\\scriptsize\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\n"
        "\\textbf{Tokenizer} & \\textbf{Grammar valid (\\%)} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\multicolumn{2}{@{}p{0.8\\columnwidth}}{\\scriptsize 5-fold cross-validation, "
        "$\\sim$130 test samples per fold. Ground-truth grammar coverage 96.2\\% (some "
        "auxiliary lines fall outside our regular grammar fragment).}\n"
        "\\end{tabular}\n\\normalsize\n\\end{table}\n"
    )
    (OUT_DIR / "table_c_5fold.tex").write_text(table_c)
    print(f"\nWrote {OUT_DIR / 'table_c_5fold.tex'}")

    # Emit Experiment E table (Hausdorff, 5-fold)
    rows = []
    for tok in ORDER:
        if tok not in agg:
            continue
        a = agg[tok]
        hm = a["hausdorff_mm"]["mean"]
        hd = a["hausdorff_mm"]["median"]
        hp = a["hausdorff_mm"]["p95"]
        if tok == "hierarchical":
            rows.append("\\midrule")
        rows.append(
            f"{PRETTY[tok]} & "
            f"{hm['mean']:.3f} $\\pm$ {hm['std']:.3f} & "
            f"{hd['mean']:.3f} $\\pm$ {hd['std']:.3f} & "
            f"{hp['mean']:.3f} $\\pm$ {hp['std']:.3f} \\\\"
        )
    table_e = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Experiment~E: Toolpath Hausdorff Distance (mm, 5-fold mean $\\pm$ std)}\n"
        "\\label{tab:exp_e}\n\\scriptsize\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "\\textbf{Tokenizer} & \\textbf{Mean} & \\textbf{Median} & \\textbf{p95} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\multicolumn{4}{@{}p{0.95\\columnwidth}}{\\scriptsize p95 saturates near "
        "the dataset's geometric extent because samples whose generation collapses "
        "default to that bound. Median is the more discriminative statistic.}\n"
        "\\end{tabular}\n\\normalsize\n\\end{table}\n"
    )
    (OUT_DIR / "table_e_5fold.tex").write_text(table_e)
    print(f"Wrote {OUT_DIR / 'table_e_5fold.tex'}")


if __name__ == "__main__":
    main()
