#!/usr/bin/env python3
"""Post-analysis: Update ANOMALY_SUMMARY.md with results from all experiments.

Reads exp*/aggregate_results.json files, ranks scoring methods, cross-references
between experiments, and writes a comprehensive summary.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import BASE_DIR, SUMMARY_FILE, save_results


# ============================================================
# Experiment metadata: maps exp_id -> (short name, group)
# ============================================================
EXPERIMENT_META = {
    1:  ("NLL Scoring (CORE)",            "A"),
    2:  ("Calibration",                   "A"),
    3:  ("Grammar Violation Partition",   "A"),
    4:  ("Multi-Head Disagreement",       "A"),
    5:  ("Sensor Modality Ablation",      "C"),
    6:  ("Partial Sensor Failure",        "C"),
    7:  ("Cross-Condition Transfer",      "B"),
    8:  ("Sliding Window Ensemble",       "B"),
    9:  ("Error Taxonomy",                "D"),
    10: ("Leave-One-Class-Out (LOCO)",    "C"),
    11: ("Embedding Distance",            "A"),
    12: ("Graded Synthetic Injection",    "B"),
    13: ("NLL vs Argmax Head-to-Head",    "A"),
    14: ("Encoder DTAE vs Decoder NLL",   "A"),
    15: ("Window Position Confound",      "C"),
    16: ("Feature Config (f110/f98/f56)", "C"),
    17: ("Cost-Benefit Deployment",       "D"),
    18: ("CUSUM Temporal Detection",      "B"),
}


def discover_experiment_dirs() -> dict:
    """Find all exp*/ directories that contain aggregate_results.json.

    Returns:
        dict mapping exp_id (int) -> dict of loaded results
    """
    results = {}
    if not BASE_DIR.exists():
        return results

    for exp_dir in sorted(BASE_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue
        agg_file = exp_dir / "aggregate_results.json"
        if not agg_file.exists():
            continue

        # Parse experiment ID from directory name like "exp01_nll_scoring"
        try:
            exp_id = int(exp_dir.name[3:5])
        except (ValueError, IndexError):
            continue

        with open(agg_file) as f:
            data = json.load(f)
        data["_dir"] = str(exp_dir)
        data["_dir_name"] = exp_dir.name
        results[exp_id] = data

    return results


def extract_auroc(data: dict) -> str:
    """Extract a human-readable AUROC string from experiment results.

    Looks for common keys that experiments use to store AUROC values.
    Returns formatted string like '0.953 +/- 0.012' or '--' if not found.
    """
    # Try common patterns found in aggregate_results.json
    for key in ["auroc_mean", "mean_auroc", "auroc"]:
        if key in data:
            val = data[key]
            if isinstance(val, dict):
                mean = val.get("mean", val.get("auroc", None))
                std = val.get("std", val.get("ci_std", None))
                if mean is not None:
                    if std is not None:
                        return f"{mean:.3f} +/- {std:.3f}"
                    return f"{mean:.3f}"
            elif isinstance(val, (int, float)):
                std_key = key.replace("auroc", "auroc_std").replace("mean_auroc", "auroc_std")
                std = data.get(std_key, data.get("auroc_std", None))
                if std is not None:
                    return f"{val:.3f} +/- {std:.3f}"
                return f"{val:.3f}"

    # Try nested: best_scoring -> auroc
    if "best_scoring" in data:
        bs = data["best_scoring"]
        if isinstance(bs, dict):
            auroc = bs.get("auroc", bs.get("mean_auroc", None))
            if auroc is not None:
                return f"{auroc:.3f}"

    # Try per-attack best AUROC
    if "per_attack" in data and isinstance(data["per_attack"], dict):
        aurocs = []
        for attack_name, attack_data in data["per_attack"].items():
            if isinstance(attack_data, dict):
                for key in ["auroc", "best_auroc", "auroc_mean"]:
                    if key in attack_data:
                        val = attack_data[key]
                        if isinstance(val, (int, float)):
                            aurocs.append(val)
                            break
        if aurocs:
            mean_a = sum(aurocs) / len(aurocs)
            return f"{mean_a:.3f} (avg)"

    # Calibration experiment uses ECE
    for key in ["ece", "mean_ece", "ece_mean"]:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return f"ECE={val:.4f}"

    return "--"


def extract_finding(data: dict) -> str:
    """Extract a one-line key finding from experiment results."""
    # Look for explicit finding/summary fields
    for key in ["key_finding", "finding", "summary", "conclusion"]:
        if key in data and isinstance(data[key], str) and data[key]:
            finding = data[key]
            # Truncate to keep table readable
            if len(finding) > 60:
                finding = finding[:57] + "..."
            return finding

    return "--"


def rank_scoring_methods(all_results: dict) -> list:
    """Rank scoring methods by AUROC across experiments.

    Returns list of (method_name, mean_auroc, experiments_tested_in) sorted descending.
    """
    method_aurocs = defaultdict(list)

    for exp_id, data in all_results.items():
        # Look for per-scoring-method AUROC breakdowns
        for key in ["scoring_comparison", "per_scoring", "scoring_methods"]:
            if key in data and isinstance(data[key], dict):
                for method, mdata in data[key].items():
                    auroc = None
                    if isinstance(mdata, dict):
                        auroc = mdata.get("auroc", mdata.get("mean_auroc", None))
                    elif isinstance(mdata, (int, float)):
                        auroc = mdata
                    if auroc is not None and isinstance(auroc, (int, float)):
                        method_aurocs[method].append((exp_id, auroc))

        # Also check for S_mean / S_max / S_sum / perplexity keys directly
        for method in ["S_mean", "S_max", "S_sum", "perplexity"]:
            if method in data and isinstance(data[method], dict):
                auroc = data[method].get("auroc", data[method].get("mean_auroc", None))
                if auroc is not None and isinstance(auroc, (int, float)):
                    method_aurocs[method].append((exp_id, auroc))

    ranked = []
    for method, entries in method_aurocs.items():
        aurocs = [a for _, a in entries]
        mean_auroc = sum(aurocs) / len(aurocs)
        exp_ids = sorted(set(e for e, _ in entries))
        ranked.append((method, mean_auroc, exp_ids))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def build_cross_references(all_results: dict) -> list:
    """Identify cross-experiment references and comparisons.

    Returns list of (description, exp_ids_involved) tuples.
    """
    cross_refs = []

    # Check if exp14 compares DTAE vs NLL
    if 14 in all_results and 1 in all_results:
        cross_refs.append((
            "Exp14 compares encoder DTAE anomaly score vs decoder NLL from Exp1",
            [1, 14],
        ))

    # Check if exp7 references exp1 thresholds
    if 7 in all_results and 1 in all_results:
        cross_refs.append((
            "Exp7 tests cross-condition transfer of Exp1 NLL thresholds",
            [1, 7],
        ))

    # Check if exp8 builds on exp1 window scores
    if 8 in all_results and 1 in all_results:
        cross_refs.append((
            "Exp8 ensembles sliding window scores from Exp1 scoring functions",
            [1, 8],
        ))

    # Check if exp12 uses graded versions of exp1 attacks
    if 12 in all_results:
        cross_refs.append((
            "Exp12 tests detection sensitivity at graded attack magnitudes",
            [1, 12],
        ))

    # Check if exp13 compares NLL (exp1) vs argmax
    if 13 in all_results:
        cross_refs.append((
            "Exp13 compares continuous NLL vs discrete argmax for detection",
            [1, 13],
        ))

    # GPU experiments that share encoder
    gpu_exps = [e for e in [5, 6, 10, 15, 16] if e in all_results]
    if len(gpu_exps) >= 2:
        cross_refs.append((
            f"GPU experiments {gpu_exps} share encoder re-inference pipeline",
            gpu_exps,
        ))

    # Post-analysis dependencies
    if 9 in all_results:
        cross_refs.append((
            "Exp9 error taxonomy aggregates misdetection patterns from all experiments",
            sorted(all_results.keys()),
        ))

    if 17 in all_results:
        cross_refs.append((
            "Exp17 cost-benefit analysis uses detection rates from all experiments",
            sorted(all_results.keys()),
        ))

    # CUSUM uses exp1 NLL scores
    if 18 in all_results and 1 in all_results:
        cross_refs.append((
            "Exp18 applies CUSUM change-point detection to Exp1 NLL time series",
            [1, 18],
        ))

    return cross_refs


def generate_key_findings(all_results: dict) -> list:
    """Generate a list of key findings from all experiments."""
    findings = []

    for exp_id in sorted(all_results.keys()):
        data = all_results[exp_id]
        name = EXPERIMENT_META.get(exp_id, (f"Experiment {exp_id}", "?"))[0]
        finding = extract_finding(data)
        auroc = extract_auroc(data)
        if finding != "--" or auroc != "--":
            findings.append(f"- **Exp {exp_id} ({name})**: AUROC={auroc}. {finding}")

    return findings


def build_summary_markdown(all_results: dict) -> str:
    """Build the complete ANOMALY_SUMMARY.md content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    completed = set(all_results.keys())
    all_exp_ids = set(EXPERIMENT_META.keys())
    pending = all_exp_ids - completed

    # -- Header --
    lines = [
        "# CNC Anomaly Detection -- Results Summary",
        "",
        f"_Last updated: {now}_",
        "",
    ]

    # -- Status Dashboard --
    lines.append("## Status Dashboard")
    lines.append("")
    lines.append("| Exp | Name                          | Status  | Group | AUROC (mean +/- std) | Key Finding |")
    lines.append("|-----|-------------------------------|---------|-------|----------------------|-------------|")

    for exp_id in sorted(EXPERIMENT_META.keys()):
        name, group = EXPERIMENT_META[exp_id]
        if exp_id in all_results:
            status = "DONE"
            auroc = extract_auroc(all_results[exp_id])
            finding = extract_finding(all_results[exp_id])
        else:
            status = "queued"
            auroc = "--"
            finding = "--"

        lines.append(
            f"| {exp_id:<3d} | {name:<29s} | {status:<7s} | {group:<5s} | {auroc:<20s} | {finding} |"
        )

    lines.append("")

    # -- Execution Progress --
    lines.append("## Execution Progress")
    lines.append("")

    group_exps = defaultdict(list)
    for eid, (_, g) in EXPERIMENT_META.items():
        group_exps[g].append(eid)

    phase0_done = (BASE_DIR / "cached_inference" / "cache_complete.flag").exists()

    lines.append(f"- Phase 0 (Cache + Attacks): {'done' if phase0_done else 'not started'}")

    for group_name, label in [
        ("A", "Group A (CPU, no deps)"),
        ("B", "Group B (CPU, depends on Exp 1)"),
        ("C", "Group C (GPU experiments)"),
        ("D", "Group D (post-analysis)"),
    ]:
        exps_in_group = group_exps[group_name]
        done_in_group = [e for e in exps_in_group if e in completed]
        total = len(exps_in_group)
        n_done = len(done_in_group)
        if n_done == total:
            status = "done"
        elif n_done > 0:
            status = f"{n_done}/{total} done"
        else:
            status = "not started"
        lines.append(f"- {label}: {status}")

    all_main_done = len(pending - {9, 17}) == 0
    lines.append(f"- Paper generation: {'done' if all_main_done else 'not started'}")
    lines.append("")

    # -- Scoring Method Rankings --
    ranked = rank_scoring_methods(all_results)
    if ranked:
        lines.append("## Scoring Method Rankings (by mean AUROC)")
        lines.append("")
        lines.append("| Rank | Method     | Mean AUROC | Tested In Experiments |")
        lines.append("|------|------------|------------|----------------------|")
        for i, (method, mean_auroc, exp_ids) in enumerate(ranked, 1):
            exp_str = ", ".join(str(e) for e in exp_ids)
            lines.append(f"| {i:<4d} | {method:<10s} | {mean_auroc:.4f}     | {exp_str} |")
        lines.append("")

    # -- Key Findings --
    findings = generate_key_findings(all_results)
    lines.append("## Key Findings")
    lines.append("")
    if findings:
        lines.extend(findings)
    else:
        lines.append("(No experiments completed yet)")
    lines.append("")

    # -- Cross-Experiment Comparisons --
    cross_refs = build_cross_references(all_results)
    lines.append("## Cross-Experiment Comparisons")
    lines.append("")
    if cross_refs:
        for desc, exp_ids in cross_refs:
            exp_str = ", ".join(str(e) for e in exp_ids)
            lines.append(f"- {desc} (Exps: {exp_str})")
    else:
        lines.append("(Filled in by update_summary.py after experiments complete)")
    lines.append("")

    # -- Surprises and Unexpected Results --
    lines.append("## Surprises and Unexpected Results")
    lines.append("")
    surprises = []
    for exp_id, data in sorted(all_results.items()):
        for key in ["surprise", "unexpected", "anomaly_in_results"]:
            if key in data and isinstance(data[key], str) and data[key]:
                name = EXPERIMENT_META.get(exp_id, (f"Exp {exp_id}", "?"))[0]
                surprises.append(f"- **Exp {exp_id} ({name})**: {data[key]}")
    if surprises:
        lines.extend(surprises)
    else:
        lines.append("(Updated as they arise)")
    lines.append("")

    # -- Per-experiment detail summaries --
    if all_results:
        lines.append("## Per-Experiment Detail")
        lines.append("")
        for exp_id in sorted(all_results.keys()):
            data = all_results[exp_id]
            name = EXPERIMENT_META.get(exp_id, (f"Experiment {exp_id}", "?"))[0]
            lines.append(f"### Exp {exp_id}: {name}")
            lines.append("")
            lines.append(f"- **Directory**: `{data.get('_dir_name', 'N/A')}`")
            lines.append(f"- **AUROC**: {extract_auroc(data)}")
            finding = extract_finding(data)
            if finding != "--":
                lines.append(f"- **Finding**: {finding}")

            # Include fold-level results if present
            if "per_fold" in data and isinstance(data["per_fold"], dict):
                lines.append("- **Per-fold AUROC**:")
                for fold_key in sorted(data["per_fold"].keys()):
                    fold_data = data["per_fold"][fold_key]
                    if isinstance(fold_data, dict):
                        fold_auroc = fold_data.get("auroc", fold_data.get("mean_auroc", "?"))
                        lines.append(f"  - Fold {fold_key}: {fold_auroc}")

            # Include attack-level breakdown if present
            if "per_attack" in data and isinstance(data["per_attack"], dict):
                lines.append("- **Per-attack breakdown**:")
                for attack_name in sorted(data["per_attack"].keys()):
                    attack_data = data["per_attack"][attack_name]
                    if isinstance(attack_data, dict):
                        a_auroc = attack_data.get("auroc", attack_data.get("best_auroc", "?"))
                        lines.append(f"  - {attack_name}: AUROC={a_auroc}")

            lines.append("")

    # -- Implications for Paper Narrative --
    lines.append("## Implications for Paper Narrative")
    lines.append("")
    if len(completed) >= 10:
        lines.append("- Sufficient experiments completed for full paper draft.")
        if ranked:
            best_method = ranked[0][0]
            lines.append(f"- Best scoring method: **{best_method}** (lead with this in Section 5).")
        if 14 in completed:
            lines.append("- Encoder vs decoder comparison (Exp 14) informs architecture discussion.")
        if 7 in completed:
            lines.append("- Cross-condition transfer (Exp 7) supports generalization claims.")
        if 12 in completed:
            lines.append("- Graded injection (Exp 12) enables sensitivity analysis figure.")
    else:
        lines.append("(Updated as findings accumulate)")
    lines.append("")

    # -- Open Questions --
    lines.append("## Open Questions")
    lines.append("")
    open_qs = []
    if 5 in completed and 6 in completed:
        open_qs.append("- How much does sensor dropout degrade detection in production?")
    if 10 in completed:
        open_qs.append("- Which operation classes are hardest to protect with LOCO?")
    if 15 in completed:
        open_qs.append("- Does window position bias affect real-world deployment?")
    if not open_qs:
        open_qs.append("(Updated during analysis)")
    lines.extend(open_qs)
    lines.append("")

    # -- Final Recommendations --
    lines.append("## Final Recommendations")
    lines.append("")
    if len(completed) == len(all_exp_ids):
        lines.append("All experiments completed. Recommendations:")
        lines.append("")
        if ranked:
            lines.append(f"1. **Primary scoring method**: {ranked[0][0]} (AUROC={ranked[0][1]:.3f})")
            if len(ranked) > 1:
                lines.append(f"2. **Runner-up**: {ranked[1][0]} (AUROC={ranked[1][1]:.3f})")
        lines.append(f"3. Completed {len(completed)}/{len(all_exp_ids)} experiments successfully.")
    else:
        remaining = sorted(pending)
        lines.append(f"(Waiting for {len(pending)} experiments: {remaining})")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"[update_summary] Scanning {BASE_DIR} for experiment results...")

    all_results = discover_experiment_dirs()
    print(f"[update_summary] Found {len(all_results)} completed experiments: "
          f"{sorted(all_results.keys())}")

    if not all_results:
        print("[update_summary] No completed experiments found. Nothing to update.")
        return

    # Build and write the summary
    summary_content = build_summary_markdown(all_results)

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(summary_content)
    print(f"[update_summary] Wrote {SUMMARY_FILE}")

    # Also save a machine-readable consolidated results file
    consolidated = {}
    for exp_id, data in sorted(all_results.items()):
        name = EXPERIMENT_META.get(exp_id, (f"exp{exp_id}", "?"))[0]
        consolidated[f"exp{exp_id:02d}"] = {
            "name": name,
            "auroc": extract_auroc(data),
            "finding": extract_finding(data),
            "status": "DONE",
        }

    save_results(consolidated, BASE_DIR, filename="consolidated_results.json")
    print(f"[update_summary] Wrote {BASE_DIR / 'consolidated_results.json'}")

    # Print a quick summary to stdout
    print()
    print("=" * 60)
    print("  ANOMALY DETECTION EXPERIMENT SUMMARY")
    print("=" * 60)
    for exp_id in sorted(all_results.keys()):
        name = EXPERIMENT_META.get(exp_id, (f"Exp {exp_id}", "?"))[0]
        auroc = extract_auroc(all_results[exp_id])
        print(f"  Exp {exp_id:2d}: {name:35s} AUROC={auroc}")
    print("=" * 60)
    n_total = len(EXPERIMENT_META)
    n_done = len(all_results)
    print(f"  {n_done}/{n_total} experiments complete.")
    if n_done < n_total:
        pending = sorted(set(EXPERIMENT_META.keys()) - set(all_results.keys()))
        print(f"  Pending: {pending}")
    print()


if __name__ == "__main__":
    main()
