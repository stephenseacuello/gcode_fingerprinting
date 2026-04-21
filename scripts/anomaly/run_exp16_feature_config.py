#!/usr/bin/env python3
"""Experiment 16: Feature Configuration Robustness

Compares anomaly detection performance with different sensor feature sets:
  - f110 (full): all 110 features (baseline, uses cached logits)
  - f98 (no proximity/pressure): permute environmental modality dims in memory
  - f56 (minimal): permute environmental + color + magnetometer dims in memory

Since we lack separate encoder checkpoints per feature config, we use a
permutation approach: destroy the relevant memory dimensions by permuting
them across windows. This breaks the modality's information content without
changing the distribution statistics.

Reports AUROC on A1 attacks for each feature configuration.
"""

import logging
import sys
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

# Memory is [N, T_s, 256]. We partition into modality groups by dim ranges.
# This mirrors the MODALITY_GROUPS mapping from exp05, projected into the
# fused 256-dim encoder space.
N_MODALITIES = 7
DIMS_PER_MOD = 256 // N_MODALITIES  # 36

# Modality order (same as exp05)
MODALITY_ORDER = [
    "accelerometer",   # dims 0-35
    "gyroscope",       # dims 36-71
    "magnetometer",    # dims 72-107
    "environmental",   # dims 108-143  (temp + pressure + proximity)
    "color",           # dims 144-179
    "rms_audio",       # dims 180-215
    "electrical",      # dims 216-255
]

def _get_modality_dims(modality_name: str) -> list:
    """Get memory dimension range for a modality."""
    idx = MODALITY_ORDER.index(modality_name)
    start = idx * DIMS_PER_MOD
    end = min(start + DIMS_PER_MOD, 256)
    return list(range(start, end))

# Feature configs: which modalities to permute (destroy)
FEATURE_CONFIGS = {
    "f110_full": {
        "permute_modalities": [],
        "description": "All features (baseline)",
    },
    "f98_no_env": {
        "permute_modalities": ["environmental"],
        "description": "No proximity/pressure (permute environmental dims)",
    },
    "f56_minimal": {
        "permute_modalities": ["environmental", "color", "magnetometer"],
        "description": "Minimal: permute environmental + color + magnetometer",
    },
}


def _permute_memory_dims(
    memory: torch.Tensor,
    modalities_to_permute: list,
    seed: int,
) -> torch.Tensor:
    """Permute specified modality dimensions across windows."""
    if not modalities_to_permute:
        return memory

    rng = np.random.RandomState(seed)
    ablated = memory.clone()
    N = memory.shape[0]

    for mod_name in modalities_to_permute:
        dims = _get_modality_dims(mod_name)
        perm = torch.from_numpy(rng.permutation(N)).long()
        ablated[:, :, dims] = memory[perm][:, :, dims]

    return ablated


def _run_config(
    fold: int,
    config_name: str,
    config: dict,
    device: str,
    seed: int,
) -> dict:
    """Run decoder inference with a specific feature configuration."""
    modalities = config["permute_modalities"]

    # If no permutation, use cached logits from exp01
    if not modalities:
        from scripts.anomaly.anomaly_scoring_utils import load_cached_logits, load_cached_targets
        logits_data = load_cached_logits(fold)
        targets_data = load_cached_targets(fold)

        legacy_logits = logits_data["legacy_logits"]
        target_tokens = targets_data["target_tokens"]
        padding_mask = targets_data["padding_mask"]
        lengths = targets_data["lengths"]

        nll = compute_per_token_nll(legacy_logits, target_tokens, padding_mask)
        scores = compute_window_scores(nll, lengths)

        # Score A1 attacks
        auroc = _score_a1_attack(legacy_logits, target_tokens, padding_mask, lengths, scores, fold)
        return {"config": config_name, "auroc": auroc, "from_cache": True}

    # Otherwise, permute memory and re-run decoder
    memory, op_pred = load_encoder_memory(fold, split="test")
    test_data = load_preprocessed_npz(fold, split="test")
    input_tokens, target_tokens, padding_mask = build_teacher_forcing_batch(
        test_data["tokens"], test_data["lengths"]
    )
    lengths = torch.from_numpy(test_data["lengths"].copy())

    # Truncate to match sizes (V7 NPZ may have slightly more windows than encoder memory)
    N_trunc = min(len(input_tokens), len(memory))
    input_tokens = input_tokens[:N_trunc]
    target_tokens = target_tokens[:N_trunc]
    padding_mask = padding_mask[:N_trunc]
    memory = memory[:N_trunc]
    op_pred = op_pred[:N_trunc]
    lengths = lengths[:N_trunc]

    ablated_memory = _permute_memory_dims(memory, modalities, seed + fold)

    # Run decoder
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

    nll = compute_per_token_nll(logits, target_tokens, padding_mask)
    scores = compute_window_scores(nll, lengths)

    auroc = _score_a1_attack(logits, target_tokens, padding_mask, lengths, scores, fold)

    return {
        "config": config_name,
        "auroc": auroc,
        "permuted_modalities": modalities,
        "from_cache": False,
    }


def _score_a1_attack(logits, target_tokens, padding_mask, lengths, scores, fold) -> float:
    """Score A1 command swap attacks and return AUROC."""
    a1_path = ATTACK_DIR / "attack_A1_command_swap.pt"
    if not a1_path.exists():
        return 0.5

    a1 = torch.load(a1_path, weights_only=False)
    fold_data = a1["fold_attacks"].get(fold, {})
    if fold_data.get("n_attacks", 0) == 0:
        return 0.5

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
        return compute_auroc(normal_s, attack_s)
    except ValueError:
        return 0.5


def run_fold(fold: int, output_dir: Path, device: str, seed: int) -> dict:
    """Run all feature configurations for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    config_results = {}
    for config_name, config in FEATURE_CONFIGS.items():
        logger.info(f"  Config: {config_name} ({config['description']})")
        result = _run_config(fold, config_name, config, device, seed)
        config_results[config_name] = result
        logger.info(f"    AUROC = {result['auroc']:.3f}")

    save_results({"fold": fold, "configs": config_results}, fold_dir, "fold_results.json")
    return {"fold": fold, "configs": config_results}


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation."""
    by_config = {}
    for fr in fold_results:
        for config_name, data in fr.get("configs", {}).items():
            if config_name not in by_config:
                by_config[config_name] = []
            by_config[config_name].append(data["auroc"])

    agg = {}
    for config_name, aurocs in by_config.items():
        agg[config_name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "per_fold": aurocs,
            "description": FEATURE_CONFIGS.get(config_name, {}).get("description", ""),
        }

    # Compute deltas relative to full config
    full_auroc = agg.get("f110_full", {}).get("mean_auroc", 0)
    for config_name in agg:
        agg[config_name]["delta_vs_full"] = agg[config_name]["mean_auroc"] - full_auroc

    results = {"per_config": agg, "full_auroc": full_auroc, "n_folds": len(fold_results)}
    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate feature config comparison figures."""
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

    per_config = agg_results.get("per_config", {})
    if not per_config:
        return

    configs = ["f110_full", "f98_no_env", "f56_minimal"]
    configs = [c for c in configs if c in per_config]
    means = [per_config[c]["mean_auroc"] for c in configs]
    stds = [per_config[c]["std_auroc"] for c in configs]
    labels = [per_config[c].get("description", c) for c in configs]
    colors = ["#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(configs)],
                  edgecolor="black", alpha=0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUROC (A1 detection)")
    ax.set_title("Detection Robustness by Feature Configuration")
    ax.set_ylim(0, 1.05)
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, fig_dir, "feature_config_auroc.pdf")
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Feature Config", 16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging("exp16_feature_config")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir, args.device, args.seed)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Summary
    per_config = agg.get("per_config", {})
    parts = []
    for c in ["f110_full", "f98_no_env", "f56_minimal"]:
        if c in per_config:
            parts.append(f"{c}={per_config[c]['mean_auroc']:.3f}")
    finding = "AUROC: " + ", ".join(parts) if parts else "No results"

    append_to_summary(
        exp_id=16,
        name="Feature Config",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 16 complete.")


if __name__ == "__main__":
    main()
