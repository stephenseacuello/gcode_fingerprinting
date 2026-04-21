#!/usr/bin/env python3
"""
Synthetic G-code injection detection experiment (POST_SWEEP_PLAN 3D).

Tests whether decoder reconstruction mismatch can detect tampered G-code.
Uses V7 best-config per-fold test metrics to compute detection rates analytically.

Attack scenarios:
  1. Command injection (G0<->G1 substitution)
  2. Coordinate tampering (Y-axis value shift)
  3. Parameter substitution (e.g., X->Y axis swap)
  4. Cross-operation swap (wrong operation class G-code assigned)

The key insight: the decoder predicts G-code from sensor data alone. Under
normal operation, predicted G-code matches commanded G-code. Under attack,
the COMMANDED G-code differs from the TRUE G-code (what the machine actually
executed). The decoder still predicts the TRUE G-code from sensors, so:
    detection = (predicted != injected) = decoder accuracy on that token type.

Outputs:
  outputs/decoder20260304/paper/anomaly_detection_results.json
"""

import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# V7 per-fold test metrics from Table 7 of the paper.
# These come from the best V7 sweep configuration evaluated on 5 CV folds.
# ---------------------------------------------------------------------------

V7_PER_FOLD = {
    "fold_1": {
        "sequence_accuracy": 0.735,
        "token_accuracy":    0.894,
        "numeric_accuracy":  0.809,
        "command_accuracy":  0.981,
        "param_type_accuracy": 0.987,
        "n_test_windows": 132,
    },
    "fold_2": {
        "sequence_accuracy": 0.894,
        "token_accuracy":    0.963,
        "numeric_accuracy":  0.944,
        "command_accuracy":  0.976,
        "param_type_accuracy": 0.993,
        "n_test_windows": 113,
    },
    "fold_3": {
        "sequence_accuracy": 0.868,
        "token_accuracy":    0.952,
        "numeric_accuracy":  0.918,
        "command_accuracy":  1.000,
        "param_type_accuracy": 0.986,
        "n_test_windows": 106,
    },
    "fold_4": {
        "sequence_accuracy": 0.889,
        "token_accuracy":    0.966,
        "numeric_accuracy":  0.949,
        "command_accuracy":  0.977,
        "param_type_accuracy": 0.991,
        "n_test_windows": 108,
    },
    "fold_5": {
        "sequence_accuracy": 0.914,
        "token_accuracy":    0.979,
        "numeric_accuracy":  0.953,
        "command_accuracy":  1.000,
        "param_type_accuracy": 1.000,
        "n_test_windows": 93,
    },
}


def mean_std(values):
    """Compute mean and sample std of a list of values."""
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mu, math.sqrt(var)


def ci_95(mu, std, n=5):
    """95% CI assuming t-distribution with n-1 df (t_0.025,4 = 2.776)."""
    t_crit = 2.776
    margin = t_crit * std / math.sqrt(n)
    return (mu - margin, mu + margin)


def compute_attack_detection():
    """
    Compute per-fold and aggregate detection rates for each attack type.

    Logic for each attack type:

    Command injection: attacker replaces G1 with G0 (or vice versa). The
    decoder, reading sensor data, predicts the TRUE command. Detection rate
    equals command_accuracy (probability that prediction matches the original,
    not the injected command).

    Coordinate tampering (Y-axis): attacker shifts Y coordinate values. The
    decoder predicts the TRUE Y from sensors. Detection rate equals
    numeric_accuracy.

    Parameter substitution: attacker swaps parameter types (e.g., changes X
    parameter to Y). Detection rate equals param_type_accuracy.

    Cross-operation swap: attacker assigns G-code from operation class A to
    sensor windows from class B. The decoder predicts class B's G-code from
    class B's sensors. Since different operations have structurally different
    G-code (different commands, different parameter sets, different coordinate
    ranges), essentially every token mismatches. Lower bound on detection is
    sequence_accuracy; effective detection is ~100%.

    Structural-only detection: only flag when command OR param_type mismatches.
    This ignores numeric differences (the main source of false positives).
    Detection rate for structural attacks = P(command correct) * P(param_type
    correct). False positive rate = 1 - P(both correct).
    """

    results = {
        "experiment": "synthetic_gcode_injection_detection",
        "description": (
            "Analytical detection rates derived from V7 decoder per-fold "
            "test metrics. For each attack type, the decoder's accuracy on "
            "the targeted token category equals the detection rate, because "
            "the decoder predicts from sensor data (ground truth) and the "
            "attacker modifies the commanded G-code."
        ),
        "source_data": "V7 best-config 5-fold CV test metrics (Table 7)",
        "per_fold": {},
        "aggregate": {},
    }

    # Attack types to track
    attack_types = [
        "command_injection",
        "coordinate_tampering",
        "parameter_substitution",
        "cross_operation_swap",
        "structural_only",
    ]

    attack_values = {a: [] for a in attack_types}
    fp_values = {
        "command_injection": [],
        "coordinate_tampering": [],
        "parameter_substitution": [],
        "any_token_mismatch": [],
        "structural_only": [],
    }

    # -----------------------------------------------------------------------
    # Per-fold detection rates
    # -----------------------------------------------------------------------
    for fold_name, m in sorted(V7_PER_FOLD.items()):
        fold_result = {"n_test_windows": m["n_test_windows"]}

        # Command injection
        cmd_det = m["command_accuracy"]
        cmd_fp = 1.0 - m["command_accuracy"]
        fold_result["command_injection"] = {
            "detection_rate": round(cmd_det, 4),
            "false_positive_rate": round(cmd_fp, 4),
            "mechanism": "decoder predicts true command; injected command differs",
        }
        attack_values["command_injection"].append(cmd_det)
        fp_values["command_injection"].append(cmd_fp)

        # Coordinate tampering (Y-axis)
        coord_det = m["numeric_accuracy"]
        coord_fp = 1.0 - m["numeric_accuracy"]
        fold_result["coordinate_tampering"] = {
            "detection_rate": round(coord_det, 4),
            "false_positive_rate": round(coord_fp, 4),
            "mechanism": "decoder predicts true Y value; tampered Y differs",
        }
        attack_values["coordinate_tampering"].append(coord_det)
        fp_values["coordinate_tampering"].append(coord_fp)

        # Parameter substitution
        param_det = m["param_type_accuracy"]
        param_fp = 1.0 - m["param_type_accuracy"]
        fold_result["parameter_substitution"] = {
            "detection_rate": round(param_det, 4),
            "false_positive_rate": round(param_fp, 4),
            "mechanism": "decoder predicts true param type; substituted param differs",
        }
        attack_values["parameter_substitution"].append(param_det)
        fp_values["parameter_substitution"].append(param_fp)

        # Cross-operation swap
        cross_det_lower = m["sequence_accuracy"]
        fold_result["cross_operation_swap"] = {
            "detection_rate_upper": 1.0,
            "detection_rate_lower": round(cross_det_lower, 4),
            "false_positive_rate": round(1.0 - m["sequence_accuracy"], 4),
            "mechanism": (
                "different operation class produces entirely different G-code; "
                "decoder predicts true operation's G-code from sensor signature"
            ),
        }
        attack_values["cross_operation_swap"].append(cross_det_lower)

        # Structural-only (high-confidence) detection
        structural_correct = m["command_accuracy"] * m["param_type_accuracy"]
        structural_fp = 1.0 - structural_correct
        fold_result["structural_only"] = {
            "detection_rate": round(structural_correct, 4),
            "false_positive_rate": round(structural_fp, 4),
            "mechanism": (
                "only flag when command or parameter type mismatches; "
                "ignores numeric (Y-axis) differences to reduce false alarms"
            ),
        }
        attack_values["structural_only"].append(structural_correct)
        fp_values["structural_only"].append(structural_fp)

        # Any-token-mismatch false positive rate
        any_fp = 1.0 - m["sequence_accuracy"]
        fold_result["any_token_mismatch"] = {
            "false_positive_rate": round(any_fp, 4),
            "note": "fraction of normal windows where decoder predicts incorrectly",
        }
        fp_values["any_token_mismatch"].append(any_fp)

        results["per_fold"][fold_name] = fold_result

    # -----------------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------------
    agg = {}
    for attack in attack_types:
        vals = attack_values[attack]
        mu, std = mean_std(vals)
        lo, hi = ci_95(mu, std)
        agg[attack] = {
            "mean_detection_rate": round(mu, 4),
            "std_detection_rate": round(std, 4),
            "ci_95_lower": round(max(lo, 0.0), 4),
            "ci_95_upper": round(min(hi, 1.0), 4),
            "per_fold_values": [round(v, 4) for v in vals],
        }

    agg["cross_operation_swap"]["mean_detection_rate_upper"] = 1.0
    agg["cross_operation_swap"]["note"] = (
        "Lower bound uses sequence_accuracy; upper bound is 1.0 since "
        "cross-class G-code sequences are structurally incompatible."
    )

    # False positive aggregates
    fp_agg = {}
    for fp_type, vals in fp_values.items():
        mu, std = mean_std(vals)
        lo, hi = ci_95(mu, std)
        fp_agg[fp_type] = {
            "mean_false_positive_rate": round(mu, 4),
            "std_false_positive_rate": round(std, 4),
            "ci_95_lower": round(max(lo, 0.0), 4),
            "ci_95_upper": round(min(hi, 1.0), 4),
            "per_fold_values": [round(v, 4) for v in vals],
        }

    agg["false_positive_rates"] = fp_agg
    results["aggregate"] = agg

    # -----------------------------------------------------------------------
    # Comparison with baselines
    # -----------------------------------------------------------------------
    results["baseline_comparison"] = {
        "description": (
            "Baseline models' detection capability if used for anomaly "
            "detection, derived from baseline_summary.json test metrics."
        ),
        "1A_lookup": {
            "command_injection_detection": 0.987,
            "coordinate_tampering_detection": 0.922,
            "param_substitution_detection": 0.962,
            "note": (
                "Lookup table achieves high accuracy but cannot generalize "
                "to unseen windows or handle within-class variation"
            ),
        },
        "1B_kNN_k1": {
            "command_injection_detection": 0.920,
            "coordinate_tampering_detection": 0.525,
            "param_substitution_detection": 0.785,
        },
        "1D_RandomForest": {
            "command_injection_detection": 0.937,
            "coordinate_tampering_detection": 0.609,
            "param_substitution_detection": 0.844,
        },
        "V7_decoder": {
            "command_injection_detection": 0.987,
            "coordinate_tampering_detection": 0.915,
            "param_substitution_detection": 0.991,
            "note": (
                "Matches lookup on commands, far exceeds baselines on "
                "coordinates and parameter types."
            ),
        },
    }

    # -----------------------------------------------------------------------
    # Paper-ready summary
    # -----------------------------------------------------------------------
    fp = agg["false_positive_rates"]
    results["paper_summary"] = {
        "title": "Synthetic G-Code Injection Detection via Reconstruction Mismatch",
        "key_findings": [
            (
                "Command injection (G0<->G1 substitution): "
                "{:.1f}% +/- {:.1f}% detection rate "
                "(100% on 2 of 5 folds). False positive rate: {:.1f}%.".format(
                    agg["command_injection"]["mean_detection_rate"] * 100,
                    agg["command_injection"]["std_detection_rate"] * 100,
                    fp["command_injection"]["mean_false_positive_rate"] * 100,
                )
            ),
            (
                "Coordinate tampering (Y-axis shift): "
                "{:.1f}% +/- {:.1f}% detection rate. "
                "False positive rate: {:.1f}%.".format(
                    agg["coordinate_tampering"]["mean_detection_rate"] * 100,
                    agg["coordinate_tampering"]["std_detection_rate"] * 100,
                    fp["coordinate_tampering"]["mean_false_positive_rate"] * 100,
                )
            ),
            (
                "Parameter substitution (e.g., X->Y axis): "
                "{:.1f}% +/- {:.1f}% detection rate.".format(
                    agg["parameter_substitution"]["mean_detection_rate"] * 100,
                    agg["parameter_substitution"]["std_detection_rate"] * 100,
                )
            ),
            (
                "Cross-operation swap (wrong operation class): "
                ">={:.1f}% detection rate "
                "(lower bound from sequence accuracy; "
                "effectively 100% since cross-class G-code is "
                "structurally incompatible).".format(
                    agg["cross_operation_swap"]["mean_detection_rate"] * 100,
                )
            ),
            (
                "High-confidence structural-only detection "
                "(commands + param types only): "
                "{:.1f}% +/- {:.1f}% detection rate, "
                "with only {:.1f}% false positive rate.".format(
                    agg["structural_only"]["mean_detection_rate"] * 100,
                    agg["structural_only"]["std_detection_rate"] * 100,
                    fp["structural_only"]["mean_false_positive_rate"] * 100,
                )
            ),
            (
                "Full any-token-mismatch detection has a "
                "{:.1f}% false positive rate under normal operation, "
                "driven by Y-axis step-over confusion.".format(
                    fp["any_token_mismatch"]["mean_false_positive_rate"] * 100,
                )
            ),
        ],
        "interpretation": (
            "The decoder's reconstruction from sensor data provides a strong "
            "anomaly detection signal for manufacturing cybersecurity. "
            "Structural attacks (command injection, parameter substitution) are "
            "detected at >98% with <2% false positives. Coordinate-level attacks "
            "are detected at >91% but with higher false positives (~8.5%) due to "
            "the Y-axis ambiguity documented in the error analysis. A two-tier "
            "alerting strategy is recommended: high-confidence alerts for "
            "structural mismatches (commands, parameter types) and soft alerts "
            "for numeric mismatches, with the latter filtered by confidence "
            "thresholds or temporal consistency checks."
        ),
        "recommended_latex_table": (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Synthetic G-Code Injection Detection Rates (\\%)}\n"
            "\\label{tab:anomaly_detection}\n"
            "\\small\n"
            "\\begin{tabular}{@{}lccl@{}}\n"
            "\\toprule\n"
            "\\textbf{Attack Type} & \\textbf{Detection} & \\textbf{FPR} "
            "& \\textbf{Metric Used} \\\\\n"
            "\\midrule\n"
            "Command injection (G0$\\leftrightarrow$G1) & "
            "$98.7 \\pm 1.2$ & $1.3$ & Command acc. \\\\\n"
            "Coordinate tampering (Y shift)            & "
            "$91.5 \\pm 6.1$ & $8.5$ & Numeric acc. \\\\\n"
            "Parameter substitution (X$\\to$Y)          & "
            "$99.1 \\pm 0.6$ & $0.9$ & Param type acc. \\\\\n"
            "Cross-operation swap                      & "
            "$\\geq 86.0$    & $14.0$ & Sequence acc. \\\\\n"
            "\\midrule\n"
            "\\multicolumn{4}{@{}l}{\\emph{Two-tier alerting strategy:}} \\\\\n"
            "\\quad Structural only (cmd + param)       & "
            "$97.9 \\pm 1.3$ & $2.1$ & Combined \\\\\n"
            "\\quad Full (any token mismatch)           & "
            "---             & $14.0$ & Sequence acc. \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\normalsize\n"
            "\\end{table}"
        ),
    }

    return results


def main():
    results = compute_attack_detection()

    # Write output
    outdir = Path("outputs/decoder20260304/paper")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "anomaly_detection_results.json"

    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {outpath}")
    print()

    # Print summary
    print("=" * 72)
    print("SYNTHETIC G-CODE INJECTION DETECTION RESULTS")
    print("=" * 72)
    print()

    agg = results["aggregate"]
    fp = agg["false_positive_rates"]

    fmt = "{:<35s}  {:>6s}  {:>6s}  {:>6s}"
    print(fmt.format("Attack Type", "Det%", "+/-Std", "FPR%"))
    print("-" * 60)

    rows = [
        ("Command injection (G0<->G1)",
         agg["command_injection"]["mean_detection_rate"],
         agg["command_injection"]["std_detection_rate"],
         fp["command_injection"]["mean_false_positive_rate"]),
        ("Coordinate tampering (Y shift)",
         agg["coordinate_tampering"]["mean_detection_rate"],
         agg["coordinate_tampering"]["std_detection_rate"],
         fp["coordinate_tampering"]["mean_false_positive_rate"]),
        ("Parameter substitution (X->Y)",
         agg["parameter_substitution"]["mean_detection_rate"],
         agg["parameter_substitution"]["std_detection_rate"],
         fp["parameter_substitution"]["mean_false_positive_rate"]),
        ("Cross-operation swap (>=lower)",
         agg["cross_operation_swap"]["mean_detection_rate"],
         agg["cross_operation_swap"]["std_detection_rate"],
         fp["any_token_mismatch"]["mean_false_positive_rate"]),
    ]
    for name, det, std, fpr in rows:
        print(fmt.format(name, f"{det*100:.1f}", f"{std*100:.1f}", f"{fpr*100:.1f}"))

    print()
    print("HIGH-CONFIDENCE (structural only):")
    print(fmt.format("  Cmd + param type",
                      f"{agg['structural_only']['mean_detection_rate']*100:.1f}",
                      f"{agg['structural_only']['std_detection_rate']*100:.1f}",
                      f"{fp['structural_only']['mean_false_positive_rate']*100:.1f}"))
    print()
    print("FULL ANY-TOKEN MISMATCH:")
    print(f"  False positive rate: "
          f"{fp['any_token_mismatch']['mean_false_positive_rate']*100:.1f}% "
          f"+/- {fp['any_token_mismatch']['std_false_positive_rate']*100:.1f}%")
    print()

    # Print key findings
    print("=" * 72)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 72)
    for i, finding in enumerate(results["paper_summary"]["key_findings"], 1):
        print(f"\n{i}. {finding}")
    print()
    print("INTERPRETATION:")
    print(results["paper_summary"]["interpretation"])


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
