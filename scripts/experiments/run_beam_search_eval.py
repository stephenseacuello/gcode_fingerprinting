#!/usr/bin/env python3
"""Run beam search evaluation on existing V8 f98 pilot + V7 f110 sweep best checkpoints."""
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = Path("outputs/decoder20260304")
VOCAB = "data/gcode_vocab_712.json"
SCRIPT = "scripts/evaluation/run_decoder_quick_test.py"

# V7 f110 sweep best per fold
V7_BEST_TRIALS = {
    1: "sweep_v7/784c43e0a2/aaw0ez60",
    2: "sweep_v7/6cd9cb3cdd/wp1edmba",
    3: "sweep_v7/7b24e99e90/oufw903c",
    4: "sweep_v7/085435298f/r5tcwf1p",
    5: "sweep_v7/c3355bff40/i0rc1805",
}


def run_beam_eval(encoder_config, fold, checkpoint, output_dir, trial_config=None):
    """Run beam search eval with correct architecture from trial config."""
    # Default to V7 best config
    cfg = {
        "d_model": 384, "n_layers": 8, "n_heads": 12, "dropout": 0.1,
        "memory_pos_encoding": True, "use_window_position": True,
        "use_sensor_prior": True, "grammar_constraint": False,
        "multi_window_context": 2, "use_pointer_network": False,
        "drop_path_rate": 0.0, "focal_gamma": 2.0, "label_smoothing": 0.1,
        "digit_weight": 15,
    }

    # Override with trial-specific config if available
    if trial_config:
        for key in ["d_model", "n_layers", "n_heads", "dropout",
                     "multi_window_context", "drop_path_rate",
                     "focal_gamma", "label_smoothing", "digit_weight"]:
            if key in trial_config:
                cfg[key] = trial_config[key]
        for key in ["memory_pos_encoding", "use_window_position",
                     "use_sensor_prior", "grammar_constraint",
                     "use_pointer_network"]:
            if key in trial_config:
                cfg[key] = trial_config[key]

    cmd = [
        sys.executable, SCRIPT,
        "--encoder_config", encoder_config,
        "--fold", str(fold),
        "--vocab", VOCAB,
        "--output_dir", output_dir,
        "--checkpoint", checkpoint,
        "--eval_only",
        "--beam_width", "1",  # Will evaluate greedy + beam=3 + beam=5 automatically
        "--d_model", str(cfg["d_model"]),
        "--n_layers", str(cfg["n_layers"]),
        "--n_heads", str(cfg["n_heads"]),
        "--dropout", str(cfg["dropout"]),
        "--memory_pos_encoding", str(cfg["memory_pos_encoding"]),
        "--use_window_position", str(cfg["use_window_position"]),
        "--use_sensor_prior", str(cfg["use_sensor_prior"]),
        "--grammar_constraint", str(cfg["grammar_constraint"]),
        "--multi_window_context", str(cfg["multi_window_context"]),
        "--use_pointer_network", str(cfg["use_pointer_network"]),
        "--drop_path_rate", str(cfg["drop_path_rate"]),
        "--focal_gamma", str(cfg["focal_gamma"]),
        "--label_smoothing", str(cfg["label_smoothing"]),
        "--digit_weight", str(cfg["digit_weight"]),
        "--max_token_len", "16",
    ]

    # Add axis_value_tables if pointer network
    if cfg.get("use_pointer_network"):
        cmd.extend(["--axis_value_tables", "data/axis_value_tables.json"])

    env = dict(__import__('os').environ, PYTHONPATH="src")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return encoder_config, fold, result.stdout, result.stderr, result.returncode


def main():
    jobs = []

    # V8 f98 pilot (all use same config)
    for fold in range(1, 6):
        ckpt = str(BASE / f"v8_pilot/fold_{fold}/decoder_checkpoint/best_decoder.pt")
        outdir = str(BASE / f"v8_beam/f98_fold_{fold}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        jobs.append(("v8_f98_w256_s64", fold, ckpt, outdir, None))

    # V7 f110 sweep best (per-trial config)
    for fold, trial_path in V7_BEST_TRIALS.items():
        trial_dir = BASE / trial_path
        ckpt = str(trial_dir / "decoder_checkpoint" / "best_decoder.pt")
        metrics_file = trial_dir / "results" / "metrics.json"
        outdir = str(BASE / f"v8_beam/f110_fold_{fold}")
        Path(outdir).mkdir(parents=True, exist_ok=True)

        trial_config = None
        if metrics_file.exists():
            with open(metrics_file) as f:
                trial_config = json.load(f).get("training_config", {})

        jobs.append(("v7_f110_w256_s64", fold, ckpt, outdir, trial_config))

    print(f"Launching {len(jobs)} beam search evaluations...")

    # Run up to 5 in parallel (GPU memory is fine for eval-only)
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_beam_eval, *job): job for job in jobs}
        for future in as_completed(futures):
            enc_cfg, fold, stdout, stderr, rc = future.result()
            label = "f98" if "f98" in enc_cfg else "f110"
            if rc == 0:
                # Extract beam results from output
                lines = stdout.split('\n')
                beam_results = []
                for line in lines:
                    if 'sequence_accuracy' in line:
                        beam_results.append(line.strip())
                print(f"  {label} fold {fold}: OK")
                for br in beam_results:
                    print(f"    {br}")
            else:
                print(f"  {label} fold {fold}: FAILED (rc={rc})")
                # Print last 20 lines of stderr
                err_lines = stderr.strip().split('\n')[-20:]
                for line in err_lines:
                    print(f"    {line}")

    # Collect and print summary
    print("\n" + "="*70)
    print("BEAM SEARCH RESULTS SUMMARY")
    print("="*70)

    for config_label, config_prefix in [("V8 f98", "f98"), ("V7 f110", "f110")]:
        print(f"\n--- {config_label} ---")
        for fold in range(1, 6):
            result_dir = BASE / f"v8_beam/{config_prefix}_fold_{fold}/results"
            print(f"  Fold {fold}:")
            for bw in [1, 3, 5]:
                metrics_file = result_dir / f"beam_{bw}_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        d = json.load(f)
                    tm = d["test_metrics"]
                    print(f"    beam={bw}: seq={tm['sequence_accuracy']:.4f} "
                          f"tok={tm['token_accuracy']:.4f} num={tm['numeric_accuracy']:.4f}")
                else:
                    print(f"    beam={bw}: (no results)")


if __name__ == "__main__":
    main()
