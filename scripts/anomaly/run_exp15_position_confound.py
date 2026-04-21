#!/usr/bin/env python3
"""Experiment 15: Window Position Confound Analysis

Compare detection AUROC between the full model (with window position) and
the 2A1 ablation (no window position). If detection drops significantly
without position, detection is partly artifactual.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    ATTACK_DIR,
    ABLATION_2A1_TEMPLATE,
    FOLDS,
    load_vocab,
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

import logging

logger = logging.getLogger(__name__)


def load_ablation_decoder(fold: int, device: str = "cuda"):
    """Load the 2A1 (no window position) ablation decoder."""
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder

    ablation_dir = Path(ABLATION_2A1_TEMPLATE.format(fold=fold))
    ckpt_path = ablation_dir / "decoder_checkpoint" / "best_decoder.pt"
    if not ckpt_path.exists():
        # Try alternative paths
        for name in ["best_decoder.pt", "final_decoder.pt"]:
            alt = ablation_dir / name
            if alt.exists():
                ckpt_path = alt
                break
        else:
            logger.error(f"No ablation checkpoint found in {ablation_dir}")
            return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    vocab = load_vocab()

    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab["vocab"]),
        sensor_dim=256,
        d_model=args.get("d_model", 384),
        n_operations=9,
        n_heads=args.get("n_heads", 8),
        n_layers=args.get("n_layers", 8),
        dropout=args.get("dropout", 0.1),
        max_seq_len=args.get("max_token_len", 16),
        use_window_position=args.get("use_window_position", False),
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    decoder.set_vocab(vocab["vocab"])
    decoder.eval()
    decoder.to(device)
    return decoder


def run_fold(fold: int, output_dir: Path, device: str = "cuda") -> dict:
    """Compare full vs no-position detection for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_data = load_preprocessed_npz(fold, split="test")
    memory, op_pred = load_encoder_memory(fold, split="test")
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

    # Load ablation decoder
    decoder = load_ablation_decoder(fold, device)
    if decoder is None:
        logger.warning(f"Fold {fold}: ablation decoder not found, skipping")
        return {"fold": fold, "status": "skipped"}

    # Run inference
    N = input_tokens.shape[0]
    batch_size = 64
    all_logits = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            outputs = decoder(
                tokens=input_tokens[start:end].to(device),
                sensor_embeddings=memory[start:end].to(device),
                operation_type=op_pred[start:end].to(device),
                tgt_key_padding_mask=padding_mask[start:end].to(device),
            )
            all_logits.append(outputs["legacy_logits"].cpu())

    ablation_logits = torch.cat(all_logits, dim=0)

    # NLL scores
    nll = compute_per_token_nll(ablation_logits, target_tokens, padding_mask)
    ablation_scores = compute_window_scores(nll, lengths)

    # Load full model scores from exp01
    from scripts.anomaly.anomaly_scoring_utils import load_cached_logits, load_cached_targets
    full_logits = load_cached_logits(fold)["legacy_logits"]
    full_nll = compute_per_token_nll(full_logits, target_tokens, padding_mask)
    full_scores = compute_window_scores(full_nll, lengths)

    # Score attacks (A1 command swap)
    a1_path = ATTACK_DIR / "attack_A1_command_swap.pt"
    results = {"fold": fold}

    if a1_path.exists():
        a1 = torch.load(a1_path, weights_only=False)
        fold_data = a1["fold_attacks"].get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            orig_idx = fold_data["orig_indices"]
            attacked = fold_data["attacked_tokens"]

            for model_name, model_logits, model_scores in [
                ("full_model", full_logits, full_scores),
                ("no_position", ablation_logits, ablation_scores),
            ]:
                attack_logits = model_logits[orig_idx]
                attack_mask = padding_mask[orig_idx]
                attack_lengths = lengths[orig_idx]
                nll_a = compute_per_token_nll(attack_logits, attacked, attack_mask)
                attack_scores = compute_window_scores(nll_a, attack_lengths)

                normal_s = model_scores["S_mean"][orig_idx].numpy()
                attack_s = attack_scores["S_mean"].numpy()

                try:
                    auroc_info = bootstrap_auroc_ci(normal_s, attack_s, n_bootstrap=5000)
                except ValueError:
                    auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0}

                results[model_name] = {
                    "auroc": auroc_info["auroc"],
                    "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
                }

            results["delta_auroc"] = results.get("full_model", {}).get("auroc", 0) - \
                                     results.get("no_position", {}).get("auroc", 0)

    del decoder
    torch.cuda.empty_cache()
    save_results(results, fold_dir)
    return results


def main():
    parser = get_default_parser("Position Confound", 15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging("exp15_position")

    fold_results = [run_fold(f, args.output_dir, args.device) for f in args.folds]

    # Aggregate
    full_aurocs = [r.get("full_model", {}).get("auroc", 0) for r in fold_results if "full_model" in r]
    ablation_aurocs = [r.get("no_position", {}).get("auroc", 0) for r in fold_results if "no_position" in r]
    deltas = [r.get("delta_auroc", 0) for r in fold_results if "delta_auroc" in r]

    agg = {
        "full_model_auroc": {"mean": float(np.mean(full_aurocs)), "std": float(np.std(full_aurocs))} if full_aurocs else {},
        "no_position_auroc": {"mean": float(np.mean(ablation_aurocs)), "std": float(np.std(ablation_aurocs))} if ablation_aurocs else {},
        "delta_auroc": {"mean": float(np.mean(deltas)), "std": float(np.std(deltas))} if deltas else {},
    }
    save_results(agg, args.output_dir)

    delta = agg.get("delta_auroc", {}).get("mean", 0)
    finding = f"Position contributes delta={delta:.3f} AUROC"
    append_to_summary(15, "Position Confound", "done", "--", finding, BASE_DIR)
    logger.info("Experiment 15 complete.")


if __name__ == "__main__":
    main()
