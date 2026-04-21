#!/usr/bin/env python3
"""Experiment 6: Sensor Board Failure Robustness

Simulates single-board (k=1) and dual-board (k=2) sensor failures by
zeroing the memory dimensions corresponding to each board. For each of
C(6,1)=6 single and C(6,2)=15 dual failure scenarios, runs decoder
inference and computes NLL/AUROC on A1 attacks. Reports worst-case AUROC
per failure cardinality k.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    ATTACK_DIR,
    FOLDS,
    load_decoder,
    load_encoder_memory,
    load_preprocessed_npz,
    build_teacher_forcing_batch,
    compute_per_token_nll,
    compute_window_scores,
    compute_auroc,
    bootstrap_auroc_ci,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

N_BOARDS = 6
# Each board contributes a block of memory dimensions.
# Memory is [N, T_s, 256]; partition into 6 boards of ~42 dims each,
# with the remaining 4 dims unassigned (treated as shared).
BOARD_DIMS = {}
dims_per_board = 256 // N_BOARDS  # 42
for b in range(N_BOARDS):
    start = b * dims_per_board
    end = start + dims_per_board
    BOARD_DIMS[b] = list(range(start, end))


def _run_failure_scenario(
    fold: int,
    failed_boards: tuple,
    device: str,
) -> dict:
    """Run decoder inference with specified boards zeroed out."""
    memory, op_pred = load_encoder_memory(fold, split="test")
    test_data = load_preprocessed_npz(fold, split="test")
    input_tokens, target_tokens, padding_mask = build_teacher_forcing_batch(
        test_data["tokens"], test_data["lengths"]
    )
    lengths = torch.from_numpy(test_data["lengths"].copy())

    # Truncate to match sizes (V7 NPZ may have slightly more windows than encoder memory)
    N = min(len(input_tokens), len(memory))
    input_tokens = input_tokens[:N]
    target_tokens = target_tokens[:N]
    padding_mask = padding_mask[:N]
    memory = memory[:N]
    op_pred = op_pred[:N]
    lengths = lengths[:N]

    # Zero the memory dimensions for the failed boards
    ablated_memory = memory.clone()
    for board_id in failed_boards:
        dims = BOARD_DIMS[board_id]
        ablated_memory[:, :, dims] = 0.0

    # Run decoder inference
    decoder = load_decoder(fold, device=device)
    N = input_tokens.shape[0]
    batch_size = 64
    all_logits = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            outputs = decoder(
                tokens=input_tokens[start:end].to(device),
                sensor_embeddings=ablated_memory[start:end].to(device),
                operation_type=op_pred[start:end].to(device),
                tgt_key_padding_mask=padding_mask[start:end].to(device),
            )
            all_logits.append(outputs["legacy_logits"].cpu())

    logits = torch.cat(all_logits, dim=0)
    del decoder
    torch.cuda.empty_cache()

    # Compute normal NLL
    nll = compute_per_token_nll(logits, target_tokens, padding_mask)
    scores = compute_window_scores(nll, lengths)

    # Score A1 attacks
    a1_path = ATTACK_DIR / "attack_A1_command_swap.pt"
    auroc = 0.5
    auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}

    if a1_path.exists():
        a1 = torch.load(a1_path, weights_only=False)
        fold_data = a1["fold_attacks"].get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            orig_idx = fold_data["orig_indices"]
            attacked = fold_data["attacked_tokens"]

            attack_logits = logits[orig_idx]
            attack_mask = padding_mask[orig_idx]
            attack_lengths = lengths[orig_idx]

            nll_attack = compute_per_token_nll(attack_logits, attacked, attack_mask)
            attack_scores = compute_window_scores(nll_attack, attack_lengths)

            normal_s = scores["S_mean"][orig_idx].numpy()
            attack_s = attack_scores["S_mean"].numpy()

            try:
                auroc_info = bootstrap_auroc_ci(normal_s, attack_s, n_bootstrap=5000)
                auroc = auroc_info["auroc"]
            except ValueError:
                pass

    return {
        "failed_boards": list(failed_boards),
        "k": len(failed_boards),
        "fold": fold,
        "auroc": auroc,
        "auroc_ci": [auroc_info.get("ci_lower", 0.0), auroc_info.get("ci_upper", 1.0)],
        "normal_s_mean": float(scores["S_mean"].mean()),
    }


def run_fold(fold: int, output_dir: Path, device: str) -> dict:
    """Run all failure scenarios for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    scenarios = []

    # Single-board failures: C(6,1) = 6
    for board in range(N_BOARDS):
        label = f"board_{board}"
        logger.info(f"  k=1 failure: {label}")
        result = _run_failure_scenario(fold, (board,), device)
        result["label"] = label
        scenarios.append(result)
        logger.info(f"    AUROC = {result['auroc']:.3f}")

    # Dual-board failures: C(6,2) = 15
    for b1, b2 in combinations(range(N_BOARDS), 2):
        label = f"board_{b1}+{b2}"
        logger.info(f"  k=2 failure: {label}")
        result = _run_failure_scenario(fold, (b1, b2), device)
        result["label"] = label
        scenarios.append(result)
        logger.info(f"    AUROC = {result['auroc']:.3f}")

    save_results({"fold": fold, "scenarios": scenarios}, fold_dir, "fold_results.json")
    return {"fold": fold, "scenarios": scenarios}


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Aggregate failure scenarios across folds."""
    # Group by scenario label
    by_label = {}
    for fr in fold_results:
        for sc in fr["scenarios"]:
            label = sc["label"]
            if label not in by_label:
                by_label[label] = {"k": sc["k"], "aurocs": []}
            by_label[label]["aurocs"].append(sc["auroc"])

    scenario_agg = {}
    for label, data in by_label.items():
        aurocs = np.array(data["aurocs"])
        scenario_agg[label] = {
            "k": data["k"],
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "min_auroc": float(np.min(aurocs)),
            "per_fold": aurocs.tolist(),
        }

    # Worst-case per k
    worst_case = {}
    for k in [1, 2]:
        k_scenarios = {l: d for l, d in scenario_agg.items() if d["k"] == k}
        if k_scenarios:
            worst_label = min(k_scenarios, key=lambda l: k_scenarios[l]["mean_auroc"])
            worst_case[f"k={k}"] = {
                "worst_scenario": worst_label,
                "worst_mean_auroc": k_scenarios[worst_label]["mean_auroc"],
                "best_scenario": max(k_scenarios, key=lambda l: k_scenarios[l]["mean_auroc"]),
                "best_mean_auroc": k_scenarios[max(k_scenarios, key=lambda l: k_scenarios[l]["mean_auroc"])]["mean_auroc"],
                "all_mean_aurocs": {l: d["mean_auroc"] for l, d in k_scenarios.items()},
            }

    results = {
        "per_scenario": scenario_agg,
        "worst_case": worst_case,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate failure robustness figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scenarios = agg_results.get("per_scenario", {})
    if not scenarios:
        return

    # Separate k=1 and k=2
    k1 = {l: d for l, d in scenarios.items() if d["k"] == 1}
    k2 = {l: d for l, d in scenarios.items() if d["k"] == 2}

    # Figure: AUROC bar chart grouped by k
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, k_data, title in [(axes[0], k1, "Single Board Failure (k=1)"),
                                (axes[1], k2, "Dual Board Failure (k=2)")]:
        if not k_data:
            ax.set_visible(False)
            continue
        labels = sorted(k_data.keys())
        means = [k_data[l]["mean_auroc"] for l in labels]
        stds = [k_data[l]["std_auroc"] for l in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color="#4C72B0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Detection Robustness Under Sensor Board Failures")
    fig.tight_layout()
    save_figure(fig, fig_dir, "sensor_failure_auroc.pdf")
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Sensor Failure", 6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging("exp06_sensor_failure")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir, args.device)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Summary
    worst = agg.get("worst_case", {})
    findings = []
    for k_key in ["k=1", "k=2"]:
        wc = worst.get(k_key, {})
        if wc:
            findings.append(f"{k_key} worst={wc['worst_mean_auroc']:.3f} ({wc['worst_scenario']})")
    finding = "; ".join(findings) if findings else "No results"

    append_to_summary(
        exp_id=6,
        name="Sensor Failure",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 6 complete.")


if __name__ == "__main__":
    main()
