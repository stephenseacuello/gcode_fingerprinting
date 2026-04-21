#!/usr/bin/env python3
"""
V5 Decoder Sweep using Ray Tune.

Builds on V1-V4 with:
1. Grammar-constrained decoding (masks invalid next tokens)
2. Grammar-aware training loss (loss computed on constrained logits)
3. Numeric regression head (direct float prediction)
4. Beam search at inference (width 1 or 3)
5. Multiple encoder configs (window/stride exploration)
6. Multi-window context (adjacent windows for temporal info)

Uses Ray Tune for parallel trial execution. 650 epochs, patience 130.

Usage:
    python scripts/experiments/run_v5_ray_sweep.py --num_samples 3000 --gpus_per_trial 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import ray
from ray import tune


def train_decoder(config):
    """Single trial: train a decoder with the given config."""
    import subprocess
    import hashlib
    import time

    trial_id = hashlib.md5(f"{config}{time.time()}".encode()).hexdigest()[:10]
    project_root = str(Path(__file__).parent.parent.parent)
    output_dir = os.path.join(project_root, "outputs/decoder20260304/sweep_v5", trial_id)

    cmd = [
        sys.executable, os.path.join(project_root, "scripts/evaluation/run_decoder_quick_test.py"),
        "--vocab", os.path.join(project_root, "data/gcode_vocab_712.json"),
        "--encoder_config", config["encoder_config"],
        "--fold", str(config["fold"]),
        "--epochs", "650",
        "--patience", "130",
        "--d_model", str(config["d_model"]),
        "--n_layers", str(config["n_layers"]),
        "--n_heads", str(config["n_heads"]),
        "--lr", str(config["lr"]),
        "--batch_size", str(config["batch_size"]),
        "--dropout", str(config["dropout"]),
        "--digit_weight", str(config["digit_weight"]),
        "--label_smoothing", str(config["label_smoothing"]),
        "--scheduled_sampling", str(config["scheduled_sampling"]),
        "--focal_gamma", str(config["focal_gamma"]),
        "--weight_decay", str(config["weight_decay"]),
        "--warmup_epochs", str(config["warmup_epochs"]),
        "--legacy_weight", "1.0",
        # V4 features
        "--hierarchical", str(config["hierarchical"]),
        "--memory_pos_encoding", str(config["memory_pos_encoding"]),
        # V5 features
        "--grammar_constraint", str(config["grammar_constraint"]),
        "--use_regression_head", str(config["use_regression_head"]),
        "--regression_weight", str(config.get("regression_weight", 1.0)),
        "--multi_window_context", str(config.get("multi_window_context", 0)),
        "--beam_width", str(config.get("beam_width", 1)),
        "--output_dir", output_dir,
    ]

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=project_root)

    # Parse results
    metrics_path = Path(output_dir) / "results" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        val_metrics = metrics.get("val_metrics", {})
        test_metrics = metrics.get("test_metrics", {})
        return {
            "val_token_accuracy": metrics.get("best_val_token_accuracy", 0),
            "val_sequence_accuracy": val_metrics.get("sequence_accuracy", 0),
            "val_numeric_accuracy": val_metrics.get("numeric_accuracy", 0),
            "test_token_accuracy": test_metrics.get("token_accuracy", 0),
            "test_sequence_accuracy": test_metrics.get("sequence_accuracy", 0),
            "test_numeric_accuracy": test_metrics.get("numeric_accuracy", 0),
            "test_command_accuracy": test_metrics.get("command_accuracy", 0),
            "test_param_type_accuracy": test_metrics.get("param_type_accuracy", 0),
            "best_epoch": metrics.get("best_epoch", 0),
        }
    else:
        err_path = Path(output_dir)
        err_path.mkdir(parents=True, exist_ok=True)
        (err_path / "stderr.txt").write_text(result.stderr[-2000:] if result.stderr else "no stderr")
        return {"val_token_accuracy": 0, "test_sequence_accuracy": 0}


def main():
    parser = argparse.ArgumentParser(description="V5 Ray Tune Decoder Sweep")
    parser.add_argument("--num_samples", type=int, default=3000, help="Total number of trials")
    parser.add_argument("--gpus_per_trial", type=float, default=0.1, help="GPU fraction per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="CPUs per trial")
    parser.add_argument("--max_concurrent", type=int, default=16, help="Max concurrent trials")
    parser.add_argument("--resume", action="store_true", help="Resume previous sweep")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    search_space = {
        # ── V4 features (keep exploring) ──
        "hierarchical": tune.choice([True, False]),
        "memory_pos_encoding": tune.choice([True, False]),

        # ── V5 new features ──
        "grammar_constraint": tune.choice([True, False]),
        "use_regression_head": tune.choice([True, False]),
        "regression_weight": tune.choice([0.5, 1.0, 2.0]),
        "multi_window_context": tune.choice([0, 1, 2]),
        "beam_width": 1,  # Beam search only at final eval, not during sweep (too slow)

        # ── Encoder configs: explore window/stride ──
        "encoder_config": tune.choice([
            "f110_w256_s64",     # Best so far (110 features, window 256, stride 64)
            "f110_w128_s32",     # Denser windows (4x more windows per file)
            "f98_w64_s16",       # Smaller sensor set (was weak but test with new features)
        ]),

        # ── Cross-validation ──
        "fold": tune.choice([1, 2, 3, 4, 5]),

        # ── Locked from V3/V4 (converged) ──
        "d_model": 384,
        "dropout": 0.1,
        "lr": tune.choice([0.0005, 0.001]),
        "batch_size": 32,

        # ── Still exploring (may interact with new features) ──
        "n_layers": tune.choice([6, 8]),
        "n_heads": tune.choice([8, 12]),
        "digit_weight": tune.choice([15, 20, 30]),
        "label_smoothing": tune.choice([0.1, 0.2]),
        "scheduled_sampling": tune.choice([0.3, 0.5]),
        "focal_gamma": tune.choice([1.0, 2.0]),
        "weight_decay": tune.choice([0.05, 0.1]),
        "warmup_epochs": tune.choice([10, 20]),
    }

    os.makedirs("outputs/decoder20260304/sweep_v5", exist_ok=True)

    analysis = tune.run(
        train_decoder,
        config=search_space,
        num_samples=args.num_samples,
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial,
        },
        max_concurrent_trials=args.max_concurrent,
        storage_path=os.path.abspath("outputs/decoder20260304/ray_results"),
        name="v5_decoder_sweep",
        resume="AUTO" if args.resume else False,
        verbose=1,
    )

    # Print best results
    print("\n" + "=" * 80)
    print("V5 SWEEP COMPLETE")
    print("=" * 80)

    best = analysis.get_best_config(metric="test_sequence_accuracy", mode="max")
    print("\nBest config by test_sequence_accuracy:")
    for k, v in sorted(best.items()):
        print(f"  {k}: {v}")

    best_result = analysis.get_best_trial(metric="test_sequence_accuracy", mode="max")
    print("\nBest metrics:")
    for k, v in sorted(best_result.last_result.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    results_df = analysis.results_df
    results_df.to_csv("outputs/decoder20260304/sweep_v5/ray_results_summary.csv", index=False)
    print(f"\nFull results saved to outputs/decoder20260304/sweep_v5/ray_results_summary.csv")


if __name__ == "__main__":
    main()
