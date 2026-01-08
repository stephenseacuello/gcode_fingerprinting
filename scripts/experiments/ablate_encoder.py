#!/usr/bin/env python3
"""
Encoder Ablation Study Script (Phase 1.6)

This script runs quick ablation experiments comparing different encoder
configurations to measure the impact of Phase 1.6 encoder improvements:

1. Baseline: Standard MM-DTAE-LSTM (frozen)
2. +MultiHeadPooling: Multi-head attention pooling
3. +PartialUnfreeze: Last 2 layers unfrozen with LR scaling
4. +MultiScale: Full multi-scale temporal encoder
5. +Auxiliary: Auxiliary supervision heads
6. Full: All Phase 1.6 improvements combined

Usage:
    python scripts/ablate_encoder.py \
        --split-dir outputs/processed_v3 \
        --vocab-path data/vocabulary_4digit_full.json \
        --encoder-path models/encoder_best.pt \
        --output-dir outputs/encoder_ablation \
        --epochs 10 \
        --quick

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Configuration presets for ablation
ABLATION_CONFIGS = {
    "baseline": {
        "description": "Standard MM-DTAE-LSTM encoder (frozen)",
        "flags": [],
    },
    "multihead_pooling": {
        "description": "Baseline + Multi-head attention pooling",
        "flags": [
            "--use-enhanced-encoder",
            "--use-multihead-pooling",
            "--pooling-n-heads", "4",
            "--pooling-n-queries", "8",
        ],
    },
    "partial_unfreeze": {
        "description": "Baseline + Partial unfreezing (2 layers, 0.1x LR)",
        "flags": [
            "--unfreeze-encoder-layers", "2",
            "--encoder-lr-scale", "0.1",
        ],
    },
    "multiscale": {
        "description": "Multi-scale temporal encoder (dilated convs + BiLSTM)",
        "flags": [
            "--use-enhanced-encoder",
            "--use-multiscale-encoder",
            "--encoder-n-scales", "4",
            "--encoder-kernel-sizes", "3", "5", "7", "11",
            "--encoder-dilations", "1", "2", "4", "8",
        ],
    },
    "multiscale_pooling": {
        "description": "Multi-scale encoder + Multi-head pooling",
        "flags": [
            "--use-enhanced-encoder",
            "--use-multiscale-encoder",
            "--use-multihead-pooling",
            "--pooling-n-heads", "4",
            "--pooling-n-queries", "8",
        ],
    },
    "auxiliary": {
        "description": "Enhanced encoder + Auxiliary supervision heads",
        "flags": [
            "--use-enhanced-encoder",
            "--use-auxiliary-heads",
            "--auxiliary-loss-weight", "0.3",
        ],
    },
    "full": {
        "description": "All Phase 1.6 improvements combined",
        "flags": [
            "--use-enhanced-encoder",
            "--use-multiscale-encoder",
            "--use-multihead-pooling",
            "--pooling-n-heads", "4",
            "--pooling-n-queries", "8",
            "--use-auxiliary-heads",
            "--auxiliary-loss-weight", "0.3",
            "--unfreeze-encoder-layers", "2",
            "--encoder-lr-scale", "0.1",
        ],
    },
    "full_frozen": {
        "description": "All Phase 1.6 improvements (encoder frozen)",
        "flags": [
            "--use-enhanced-encoder",
            "--use-multiscale-encoder",
            "--use-multihead-pooling",
            "--pooling-n-heads", "4",
            "--pooling-n-queries", "8",
            "--use-auxiliary-heads",
            "--auxiliary-loss-weight", "0.3",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run encoder ablation study comparing Phase 1.6 improvements"
    )

    # Required paths
    parser.add_argument("--split-dir", type=str, required=True,
                        help="Path to processed data splits")
    parser.add_argument("--vocab-path", type=str, required=True,
                        help="Path to vocabulary JSON")
    parser.add_argument("--encoder-path", type=str, required=True,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for ablation results")

    # Ablation selection
    parser.add_argument("--configs", type=str, nargs="+",
                        default=list(ABLATION_CONFIGS.keys()),
                        help=f"Which configs to run. Options: {list(ABLATION_CONFIGS.keys())}")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        help="Configs to exclude from ablation")

    # Training settings
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs per ablation run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--d-model", type=int, default=192,
                        help="Decoder model dimension (default: 192)")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Decoder layers (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 epochs, smaller batch")

    # Logging
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str,
                        default="gcode-encoder-ablation",
                        help="WandB project name")

    # Misc
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    return parser.parse_args()


def run_ablation(
    config_name: str,
    config: Dict,
    args,
    run_idx: int,
    total_runs: int
) -> Optional[Dict]:
    """Run a single ablation configuration."""

    print(f"\n{'='*70}")
    print(f"ABLATION {run_idx}/{total_runs}: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}")

    # Build command
    run_name = f"ablation_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_subdir = Path(args.output_dir) / config_name

    cmd = [
        sys.executable,
        "scripts/train_sensor_multihead.py",
        "--split-dir", args.split_dir,
        "--vocab-path", args.vocab_path,
        "--encoder-path", args.encoder_path,
        "--output-dir", str(output_subdir),
        "--max-epochs", str(args.epochs if not args.quick else 5),
        "--batch-size", str(args.batch_size if not args.quick else 16),
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(max(5, args.epochs // 2)),
        "--print-every", "5",
        "--run-name", run_name,
    ]

    # Add config-specific flags
    cmd.extend(config["flags"])

    # Add WandB if enabled
    if args.use_wandb:
        cmd.extend([
            "--use-wandb",
            "--wandb-project", args.wandb_project,
        ])

    print(f"\nCommand: {' '.join(cmd)}")

    if args.dry_run:
        print("(Dry run - not executing)")
        return None

    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            print(f"ERROR: Ablation {config_name} failed with code {result.returncode}")
            return {"config": config_name, "status": "failed", "error": f"Exit code {result.returncode}"}

        # Load results
        results_path = output_subdir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            return {
                "config": config_name,
                "status": "success",
                "results": results,
            }
        else:
            return {"config": config_name, "status": "success", "results": None}

    except Exception as e:
        print(f"ERROR: Exception during {config_name}: {e}")
        return {"config": config_name, "status": "error", "error": str(e)}


def summarize_results(all_results: List[Dict], output_dir: Path):
    """Generate summary of ablation results."""

    print("\n" + "=" * 70)
    print("ENCODER ABLATION SUMMARY")
    print("=" * 70)

    summary = []
    for r in all_results:
        if r is None:
            continue

        config_name = r["config"]
        status = r["status"]

        if status == "success" and r.get("results"):
            test_metrics = r["results"].get("test_metrics", {})
            token_acc = test_metrics.get("token_acc", 0) * 100
            seq_acc = test_metrics.get("sequence_acc", 0) * 100
            summary.append({
                "config": config_name,
                "description": ABLATION_CONFIGS.get(config_name, {}).get("description", ""),
                "token_acc": token_acc,
                "seq_acc": seq_acc,
                "status": "success",
            })
            print(f"\n{config_name:20s}: Token={token_acc:5.1f}%  Seq={seq_acc:5.1f}%")
        else:
            summary.append({
                "config": config_name,
                "status": status,
                "error": r.get("error", "Unknown"),
            })
            print(f"\n{config_name:20s}: {status}")

    # Find best configuration
    successful = [s for s in summary if s["status"] == "success"]
    if successful:
        best_token = max(successful, key=lambda x: x.get("token_acc", 0))
        best_seq = max(successful, key=lambda x: x.get("seq_acc", 0))

        print(f"\n{'='*70}")
        print("BEST CONFIGURATIONS:")
        print(f"  Best Token Accuracy: {best_token['config']} ({best_token['token_acc']:.1f}%)")
        print(f"  Best Sequence Accuracy: {best_seq['config']} ({best_seq['seq_acc']:.1f}%)")

        # Compute improvements over baseline
        baseline = next((s for s in successful if s["config"] == "baseline"), None)
        if baseline:
            print(f"\nIMPROVEMENTS OVER BASELINE:")
            for s in successful:
                if s["config"] != "baseline":
                    token_delta = s["token_acc"] - baseline["token_acc"]
                    seq_delta = s["seq_acc"] - baseline["seq_acc"]
                    sign_t = "+" if token_delta >= 0 else ""
                    sign_s = "+" if seq_delta >= 0 else ""
                    print(f"  {s['config']:20s}: Token={sign_t}{token_delta:+5.1f}%  Seq={sign_s}{seq_delta:+5.1f}%")

    # Save summary
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": summary,
            "best_token": best_token["config"] if successful else None,
            "best_sequence": best_seq["config"] if successful else None,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs
    configs_to_run = [c for c in args.configs if c not in args.exclude]
    configs_to_run = [c for c in configs_to_run if c in ABLATION_CONFIGS]

    print("=" * 70)
    print("ENCODER ABLATION STUDY (Phase 1.6)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Configurations to run: {configs_to_run}")
    print(f"Epochs per run: {args.epochs if not args.quick else 5}")
    print(f"Quick mode: {args.quick}")

    # Save ablation config
    config_path = output_dir / "ablation_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "configs": {k: v for k, v in ABLATION_CONFIGS.items() if k in configs_to_run},
        }, f, indent=2)

    # Run ablations
    all_results = []
    for idx, config_name in enumerate(configs_to_run, 1):
        config = ABLATION_CONFIGS[config_name]
        result = run_ablation(
            config_name, config, args,
            run_idx=idx, total_runs=len(configs_to_run)
        )
        all_results.append(result)

    # Summarize results
    if not args.dry_run:
        summarize_results(all_results, output_dir)

    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
