#!/usr/bin/env python3
"""Stage 2 of the HP sweep: refine around Stage 1 winner.

Reads outputs/decoder20260511/audit/hp_sweep_summary.json (produced by
aggregate_hp_sweep.py), picks the winner by val_token_accuracy, then runs
10-12 refinement cells around it varying:

  - curriculum (off / 3phase)
  - scheduled_sampling (0 / 0.5 / 1.0)
  - label_smoothing (0 / 0.1 / 0.2)
  - dropout (winner_dropout / lower / higher)
  - n_layers (winner / +2)
  - focal_gamma (0 / 2)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "outputs/decoder20260511/audit/hp_sweep_summary.json"
SWEEP_ROOT = REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage2"
VOCAB = REPO / "data/gcode_vocab_v8.json"


def _parse_winner_tag(tag: str) -> dict:
    """Reverse-engineer hyperparameters from the Stage-1 tag.

    Tags follow the convention: lr_<lr>_b<batch>_d<dmodel>_w<lw>-<dw>
    """
    out = {"lr": 5e-5, "batch_size": 64, "d_model": 384, "n_layers": 8,
           "n_heads": 12, "dropout": 0.1, "weight_decay": 0.05,
           "legacy_weight": 1.0, "digit_weight": 1.0, "warmup_epochs": 10}
    # lr
    m = re.search(r"lr_([\d.e-]+)", tag)
    if m: out["lr"] = float(m.group(1))
    # batch
    m = re.search(r"_b(\d+)", tag)
    if m: out["batch_size"] = int(m.group(1))
    # d_model
    m = re.search(r"_d(\d+)", tag)
    if m:
        out["d_model"] = int(m.group(1))
        out["n_heads"] = 16 if out["d_model"] == 512 else 12
    # weights
    m = re.search(r"_w(\d+)-(\d+)", tag)
    if m:
        out["legacy_weight"] = float(m.group(1))
        out["digit_weight"] = float(m.group(2))
    return out


def run_cell(tag: str, base_kwargs: dict, **overrides) -> None:
    out_dir = SWEEP_ROOT / tag / "fold_1"
    if (out_dir / "decoder_checkpoint" / "best_decoder.pt").exists():
        print(f"  skip {tag}: already trained")
        return
    cfg = dict(base_kwargs)
    cfg.update(overrides)
    cmd = [
        "python3", "scripts/evaluation/run_decoder_quick_test.py",
        "--data_dir", "outputs/decoder20260511/preprocessed_f98/per_row/fold_1",
        "--encoder_config", "f98_w256_s64", "--fold", "1",
        "--vocab", str(VOCAB),
        "--output_dir", str(out_dir),
        "--epochs", "150", "--patience", "40",
        "--max_token_len", "32", "--seed", "42",
        "--memory_pos_encoding", "true", "--use_sensor_prior", "true",
        "--grammar_constraint", "true",
        "--use_window_position", "false",
        "--multi_window_context", "0", "--window_dropout", "0.0",
        "--device", "auto",
    ]
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]
    print(f"\n=== Stage 2 cell: {tag} ===")
    print("  " + " ".join(cmd[:8]) + " ... [config]")
    subprocess.run(cmd, cwd=REPO, check=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=SUMMARY)
    args = p.parse_args()

    if not args.summary.exists():
        print(f"no summary at {args.summary}; run Stage 1 first")
        return 1
    summary = json.loads(args.summary.read_text())
    winner = summary["winner_by_val_token"]
    print(f"Stage 1 winner: {winner['tag']}  val_tok={winner['best_val_token']:.4f}  "
          f"test_tok={winner.get('test_token_accuracy', 0):.4f}  num={winner.get('test_numeric_accuracy', 0):.4f}")

    base = _parse_winner_tag(winner["tag"])
    print(f"Inferred base config: {base}")

    # Stage 2 refinement cells (12 total).
    # Curriculum
    run_cell("curriculum_3phase", base, curriculum="3phase")
    # Scheduled sampling
    run_cell("scheduled_sampling_0.5", base, scheduled_sampling=0.5)
    run_cell("scheduled_sampling_1.0", base, scheduled_sampling=1.0)
    # Label smoothing
    run_cell("label_smooth_0.1", base, label_smoothing=0.1)
    run_cell("label_smooth_0.2", base, label_smoothing=0.2)
    # Dropout
    run_cell(f"dropout_0.05", base, dropout=0.05)
    run_cell(f"dropout_0.2", base, dropout=0.2)
    run_cell(f"dropout_0.3", base, dropout=0.3)
    # n_layers
    run_cell("n_layers_10", base, n_layers=10)
    run_cell("n_layers_12", base, n_layers=12)
    # Focal loss
    run_cell("focal_gamma_2", base, focal_gamma=2.0)
    # Combined: curriculum + scheduled sampling + label smoothing (best-guess stack)
    combined = dict(base)
    combined.update({"curriculum": "3phase", "scheduled_sampling": 0.5, "label_smoothing": 0.1})
    run_cell("combined_curric_ss_ls", base,
             curriculum="3phase", scheduled_sampling=0.5, label_smoothing=0.1)

    print(f"\nStage 2 done. Aggregating...")
    subprocess.run(["python3", "scripts/analysis/aggregate_hp_sweep.py",
                    "--sweep-root", str(SWEEP_ROOT)], cwd=REPO, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
