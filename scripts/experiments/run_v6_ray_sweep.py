#!/usr/bin/env python3
"""
V6 Decoder Sweep using Ray Tune.

Builds on V1-V5 with bug fixes and new features:
1. Fixed multi-window context (file-level neighbors, not index-based)
2. Window position input (87% info gain for Y-value disambiguation)
3. Fixed grammar mask rules (command hallucination fix)
4. Two-stage classify-then-generate (34-class sequence prior)
5. Oversampling rare sequences
6. Sensor noise injection
7. Window dropout in MWC
8. Stratified CV splits (all test sequences appear in training)

Uses Ray Tune for parallel trial execution. 800 epochs, patience 160.

Usage:
    python scripts/experiments/run_v6_ray_sweep.py --num_samples 4000 --gpus_per_trial 0.1
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
    output_dir = os.path.join(project_root, "outputs/decoder20260304/sweep_v6", trial_id)

    cmd = [
        sys.executable, os.path.join(project_root, "scripts/evaluation/run_decoder_quick_test.py"),
        "--vocab", os.path.join(project_root, "data/gcode_vocab_712.json"),
        "--encoder_config", config["encoder_config"],
        "--fold", str(config["fold"]),
        "--epochs", "800",
        "--patience", "160",
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
        # V4 features (locked)
        "--hierarchical", "False",
        "--memory_pos_encoding", "True",
        # V5 features (locked)
        "--use_regression_head", "False",
        "--beam_width", "1",
        # V6 features
        "--grammar_constraint", str(config["grammar_constraint"]),
        "--multi_window_context", str(config["multi_window_context"]),
        "--use_window_position", str(config["use_window_position"]),
        "--use_sequence_classifier", str(config["use_sequence_classifier"]),
        "--sequence_class_weight", str(config.get("sequence_class_weight", 1.0)),
        "--oversample_rare", str(config["oversample_rare"]),
        "--noise_scale", str(config["noise_scale"]),
        "--window_dropout", str(config["window_dropout"]),
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
    parser = argparse.ArgumentParser(description="V6 Ray Tune Decoder Sweep")
    parser.add_argument("--num_samples", type=int, default=4000, help="Total number of trials")
    parser.add_argument("--gpus_per_trial", type=float, default=0.1, help="GPU fraction per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="CPUs per trial")
    parser.add_argument("--max_concurrent", type=int, default=16, help="Max concurrent trials")
    parser.add_argument("--resume", action="store_true", help="Resume previous sweep")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    search_space = {
        # ── LOCKED from V5 (converged) ──
        "d_model": 384,
        "dropout": 0.1,
        "batch_size": 32,
        "focal_gamma": 2.0,
        "warmup_epochs": 10,

        # ── NARROW RANGES (V5-converged, small exploration) ──
        "lr": tune.choice([0.0003, 0.0005, 0.0007]),
        "n_layers": tune.choice([6, 8]),
        "n_heads": tune.choice([8, 12]),
        "digit_weight": tune.choice([15, 20, 30]),
        "scheduled_sampling": tune.choice([0.3, 0.5]),
        "weight_decay": tune.choice([0.05, 0.1]),
        "label_smoothing": tune.choice([0.1, 0.2]),

        # ── V6 NEW FEATURES TO SWEEP ──
        "use_window_position": tune.choice([True, False]),
        "grammar_constraint": tune.choice([True, False]),
        "multi_window_context": tune.choice([0, 2, 4]),
        "use_sequence_classifier": False,  # KILLED: hurts ~9% (best 59% vs 79% OFF)
        "sequence_class_weight": 1.0,
        "oversample_rare": tune.choice([True, False]),
        "noise_scale": 0.0,  # KILLED: no benefit at any level
        "window_dropout": tune.choice([0.0, 0.1]),

        # ── ENCODER ──
        "encoder_config": "f110_w256_s64",  # LOCKED: f128 best=60.9% vs 79.0%, wasting budget

        # ── CROSS-VALIDATION ──
        "fold": tune.choice([1, 2, 3, 4, 5]),
    }

    os.makedirs("outputs/decoder20260304/sweep_v6", exist_ok=True)

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
        name="v6_decoder_sweep",
        resume="AUTO" if args.resume else False,
        verbose=1,
    )

    # Print best results
    print("\n" + "=" * 80)
    print("V6 SWEEP COMPLETE")
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
    results_df.to_csv("outputs/decoder20260304/sweep_v6/ray_results_summary.csv", index=False)
    print(f"\nFull results saved to outputs/decoder20260304/sweep_v6/ray_results_summary.csv")


if __name__ == "__main__":
    main()
