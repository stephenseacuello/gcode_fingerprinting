#!/usr/bin/env python3
"""Stage 3 architectural HP sweep with 2-GPU parallelism and W&B tracking.

Tests the architectural levers Stages 1-2 didn't explore:
  - --e2e encoder fine-tuning (biggest unknown)
  - --use_pointer_network
  - --use_regression_head
  - --cross_window_attention
  - alternative encoder configs (f110, f56)
  - smaller scheduled_sampling values (0.1, 0.2 — 0.5 was too aggressive in Stage 2)
  - multi-seed variance estimation on Stage-1 winner

All on V8 per_row fold 1. 150 epochs cap, patience 40. Each cell tagged with W&B.

Dispatches up to 2 cells in parallel on the 2 A6000 GPUs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
VOCAB = REPO / "data/gcode_vocab_v8.json"
SWEEP_ROOT = REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage3"
WANDB_PROJECT = "gcode-decoder-2026"

# Stage-1 winner config — base for most Stage 3 cells
WINNER = dict(
    lr=5e-5, batch_size=64, d_model=384, n_layers=8, n_heads=12,
    dropout=0.1, weight_decay=0.05,
    legacy_weight=3.0, digit_weight=1.0,  # winner had heavier legacy
    warmup_epochs=10,
    epochs=150, patience=40,
    max_token_len=32, seed=42,
)


def cmd_for(tag: str, **overrides) -> list[str]:
    cfg = dict(WINNER)
    cfg.update(overrides)
    out_dir = SWEEP_ROOT / tag / "fold_1"
    cmd = [
        "python3", "scripts/evaluation/run_decoder_quick_test.py",
        "--data_dir", "outputs/decoder20260511/preprocessed_f98/per_row/fold_1",
        "--fold", "1",
        "--vocab", str(VOCAB),
        "--output_dir", str(out_dir),
        "--memory_pos_encoding", "true", "--use_sensor_prior", "true",
        "--grammar_constraint", "true",
        "--use_window_position", "false",
        "--multi_window_context", "0", "--window_dropout", "0.0",
        "--wandb", "--wandb_project", WANDB_PROJECT,
        "--device", "auto",
    ]
    if "encoder_config" not in cfg:
        cmd += ["--encoder_config", "f98_w256_s64"]
    for k, v in cfg.items():
        if v == "":  # bare flag (e.g. --e2e)
            cmd += [f"--{k}"]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd, out_dir


# Stage 3 cells. Each is (tag, kwargs). Order is rough priority — most-impactful first.
CELLS = [
    # 1. The big one: encoder fine-tuning. Default e2e + smaller LR + smaller batch (memory).
    ("e2e_lr1e-5", dict(e2e="", lr=1e-5, batch_size=32, grad_accum=2)),
    # 2. e2e with even smaller LR
    ("e2e_lr5e-6", dict(e2e="", lr=5e-6, batch_size=32, grad_accum=2)),
    # 3. Pointer network on Stage-1 winner
    ("pointer_network", dict(use_pointer_network="true")),
    # 4. Regression head (alternative numeric path)
    ("regression_head", dict(use_regression_head="true", regression_weight=2.0)),
    # 5. Cross-window attention
    ("cross_window", dict(cross_window_attention="true", multi_window_context=2)),
    # 6. Scheduled sampling 0.1 (small)
    ("ss_0.1", dict(scheduled_sampling=0.1)),
    # 7. Scheduled sampling 0.2
    ("ss_0.2", dict(scheduled_sampling=0.2)),
    # 8. f110 encoder — all sensors (no exclusion)
    ("encoder_f110", dict(encoder_config="f110_w256_s64",
                          # f110 uses different data path; override data_dir
                          )),
    # 9. f56 encoder — minimal sensors (no proximity/pressure/color/magnetometer)
    ("encoder_f56", dict(encoder_config="f56_w256_s64")),
    # 10–12. Multi-seed on Stage-1 winner config (variance estimate)
    ("seed_2024", dict(seed=2024)),
    ("seed_123", dict(seed=123)),
    ("seed_456", dict(seed=456)),
    # 13. Combined best guess: pointer + cross_window
    ("pointer_x_cross_window", dict(use_pointer_network="true",
                                     cross_window_attention="true",
                                     multi_window_context=2)),
    # 14. Combined: regression head + scheduled sampling
    ("regression_x_ss", dict(use_regression_head="true",
                              regression_weight=2.0,
                              scheduled_sampling=0.1)),
]


def cell_done(out_dir: Path) -> bool:
    return (out_dir / "decoder_checkpoint" / "best_decoder.pt").exists()


def dispatch_parallel(cells: list[tuple[str, dict]], gpus: list[int]) -> None:
    """Run cells in parallel, one per GPU. Block until all done."""
    remaining = list(cells)
    running: dict[int, tuple[str, subprocess.Popen]] = {}

    def launch_on_gpu(gpu_id: int, tag: str, kwargs: dict) -> None:
        cmd, out_dir = cmd_for(tag, **kwargs)
        if cell_done(out_dir):
            print(f"[GPU {gpu_id}] skip {tag}: already trained")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "stage3_launch.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with open(log_path, "w") as logf:
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[gpu_id] = (tag, p)
        print(f"[GPU {gpu_id}] launched {tag} (pid={p.pid})")

    # Initial dispatch
    for gpu_id in gpus:
        if remaining:
            tag, kwargs = remaining.pop(0)
            launch_on_gpu(gpu_id, tag, kwargs)

    # Wait + dispatch loop
    while running:
        time.sleep(30)
        finished_gpus = []
        for gpu_id, (tag, p) in list(running.items()):
            if p.poll() is not None:
                rc = p.returncode
                print(f"[GPU {gpu_id}] finished {tag} (rc={rc})")
                finished_gpus.append(gpu_id)
        for gpu_id in finished_gpus:
            del running[gpu_id]
            if remaining:
                tag, kwargs = remaining.pop(0)
                launch_on_gpu(gpu_id, tag, kwargs)

    print("All Stage 3 cells finished.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--only", nargs="+", default=None, help="Run only these cell tags")
    p.add_argument("--skip", nargs="+", default=None, help="Skip these cell tags")
    args = p.parse_args()

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    cells = CELLS
    if args.only:
        cells = [(t, k) for t, k in cells if t in args.only]
    if args.skip:
        cells = [(t, k) for t, k in cells if t not in args.skip]

    print(f"Stage 3 plan: {len(cells)} cells across GPUs {args.gpus}")
    for tag, _ in cells:
        print(f"  - {tag}")

    dispatch_parallel(cells, args.gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
