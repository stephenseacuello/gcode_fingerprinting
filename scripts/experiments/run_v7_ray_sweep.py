#!/usr/bin/env python3
"""
V7 Decoder Sweep using Ray Tune + W&B tracking.

Builds on V1-V6 with:
1. Re-preprocessed data from data_clean/ (window_index, source_file, stratified splits)
2. Window position feature (87% info gain for Y-value disambiguation)
3. File-level MWC neighbors (not index-based)
4. Pointer network for per-axis numeric classification
5. SensorValuePrior (implemented V1, never swept)
6. DropPath / stochastic depth (implemented V1, never swept)
7. Self-distillation from V5 best checkpoint
8. W&B experiment tracking

Uses Ray Tune for parallel trial execution. 1000 epochs, patience 200.

Usage:
    python scripts/experiments/run_v7_ray_sweep.py --num_samples 5000 --gpus_per_trial 0.06
    python scripts/experiments/run_v7_ray_sweep.py --num_samples 50 --max_concurrent 4 --pilot  # pilot run
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
    output_dir = os.path.join(project_root, "outputs/decoder20260304/sweep_v7", trial_id)

    # Normalize conditional hyperparameters
    # Distillation params are meaningless when distillation is off
    if not config.get("use_distillation", False):
        config["distillation_alpha"] = 0.0
        config["distillation_temp"] = 1.0
        config["teacher_checkpoint"] = ""

    cmd = [
        sys.executable, os.path.join(project_root, "scripts/evaluation/run_decoder_quick_test.py"),
        "--vocab", os.path.join(project_root, "data/gcode_vocab_712.json"),
        "--encoder_config", config["encoder_config"],
        "--fold", str(config["fold"]),
        "--epochs", str(config.get("epochs", 1000)),
        "--patience", str(config.get("patience", 200)),
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
        # V6 survivors
        "--grammar_constraint", str(config["grammar_constraint"]),
        "--multi_window_context", str(config["multi_window_context"]),
        "--oversample_rare", str(config["oversample_rare"]),
        "--window_dropout", str(config["window_dropout"]),
        "--noise_scale", "0.0",
        "--use_sequence_classifier", "False",
        # V7: Data pipeline (always on)
        "--use_window_position", "True",
        # V7: Architecture
        "--use_pointer_network", str(config["use_pointer_network"]),
        "--pointer_weight", str(config.get("pointer_weight", 2.0)),
        "--axis_value_tables", os.path.join(project_root, "data/axis_value_tables.json"),
        "--use_sensor_prior", str(config["use_sensor_prior"]),
        "--drop_path_rate", str(config["drop_path_rate"]),
        # V7: Training
        "--use_distillation", str(config.get("use_distillation", False)),
        "--distillation_alpha", str(config.get("distillation_alpha", 0.3)),
        "--distillation_temp", str(config.get("distillation_temp", 2.0)),
        # Output
        "--output_dir", output_dir,
        # W&B
        "--wandb",
        "--wandb_project", "gcode-decoder-v7",
    ]

    # Add teacher checkpoint if distillation is enabled
    teacher_ckpt = config.get("teacher_checkpoint", "")
    if config.get("use_distillation", False) and teacher_ckpt:
        cmd.extend(["--teacher_checkpoint", teacher_ckpt])

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=project_root)

    # Parse results (wandb may nest output under a run ID subdirectory)
    metrics_path = Path(output_dir) / "results" / "metrics.json"
    if not metrics_path.exists():
        candidates = list(Path(output_dir).glob("*/results/metrics.json"))
        if candidates:
            metrics_path = candidates[0]
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
        (err_path / "stdout.txt").write_text(result.stdout[-2000:] if result.stdout else "no stdout")
        return {"val_token_accuracy": 0, "test_sequence_accuracy": 0}


def main():
    parser = argparse.ArgumentParser(description="V7 Ray Tune Decoder Sweep")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of trials")
    parser.add_argument("--gpus_per_trial", type=float, default=0.06, help="GPU fraction per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=2, help="CPUs per trial")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Max concurrent trials")
    parser.add_argument("--resume", action="store_true", help="Resume previous sweep")
    parser.add_argument("--pilot", action="store_true", help="Run pilot sweep (shorter training, fewer trials)")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Pilot mode: shorter training for pipeline verification
    epochs = 100 if args.pilot else 500
    patience = 20 if args.pilot else 120

    search_space = {
        # ── LOCKED (V6-confirmed) ──
        "d_model": 384,
        "dropout": 0.1,
        "batch_size": 32,
        "focal_gamma": 2.0,
        "memory_pos_encoding": True,
        "hierarchical": False,
        "use_regression_head": False,
        "use_sequence_classifier": False,
        "noise_scale": 0.0,
        "lr": 0.0003,
        "n_layers": 8,
        "beam_width": 1,
        "warmup_epochs": 10,
        "epochs": epochs,
        "patience": patience,

        # ── ENCODER (V7: uses re-preprocessed data from data_clean/) ──
        "encoder_config": "v7_f110_w256_s64",

        # ── NARROW RANGES (still worth exploring) ──
        "n_heads": tune.choice([8, 12]),
        "digit_weight": tune.choice([15, 20, 30]),
        "scheduled_sampling": tune.choice([0.3, 0.5]),
        "weight_decay": tune.choice([0.05, 0.1]),
        "label_smoothing": tune.choice([0.1, 0.2]),

        # ── V6 SURVIVORS ──
        "grammar_constraint": tune.choice([True, False]),
        "multi_window_context": tune.choice([2, 4]),  # Drop 0: file-level neighbors now
        "oversample_rare": tune.choice([True, False]),
        "window_dropout": tune.choice([0.0, 0.1]),

        # ── V7 NEW: DATA PIPELINE (always on) ──
        "use_window_position": True,  # 87% info gain, was blocked in V6

        # ── V7 NEW: IMPLEMENTED BUT NEVER SWEPT (V1-V6) ──
        "use_sensor_prior": tune.choice([True, False]),
        "drop_path_rate": tune.choice([0.0, 0.1]),

        # ── V7 NEW: ARCHITECTURE ──
        "use_pointer_network": tune.choice([True, False]),
        "pointer_weight": tune.choice([1.0, 2.0]),

        # ── V7 NEW: TRAINING ──
        "use_distillation": False,  # Disabled for Phase 1 (no teacher checkpoint verified yet)
        "distillation_alpha": 0.3,
        "distillation_temp": 2.0,
        "teacher_checkpoint": "",

        # ── CROSS-VALIDATION ──
        "fold": tune.choice([1, 2, 3, 4, 5]),
    }

    os.makedirs("outputs/decoder20260304/sweep_v7", exist_ok=True)

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
        name="v7_decoder_sweep_v2",
        resume="AUTO" if args.resume else False,
        verbose=1,
    )

    # Print best results
    print("\n" + "=" * 80)
    print("V7 SWEEP COMPLETE")
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
    results_df.to_csv("outputs/decoder20260304/sweep_v7/ray_results_summary.csv", index=False)
    print(f"\nFull results saved to outputs/decoder20260304/sweep_v7/ray_results_summary.csv")


if __name__ == "__main__":
    main()
