#!/usr/bin/env python3
"""
Full per-modality ablation pipeline: preprocess → train encoder → train decoder.

For each modality, removes that modality's channels, trains a new encoder,
then trains the decoder on top. Reports reconstruction accuracy per modality.

Usage:
    python scripts/experiments/run_full_modality_ablation.py --gpu 0,1
"""
import subprocess
import sys
import json
import os
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

# Modality definitions: name → exclude flags for preprocessing
MODALITY_ABLATIONS = OrderedDict([
    # Single modality removal
    ('no_accel',     ['--exclude-accelerometer']),
    ('no_gyro',      ['--exclude-gyroscope']),
    ('no_mag',       ['--exclude-magnetometer']),
    ('no_pressure',  ['--exclude-pressure']),
    ('no_temp',      ['--exclude-temperature']),
    ('no_prox',      ['--exclude-proximity']),
    ('no_color',     ['--exclude-color']),
    ('no_audio',     ['--exclude-audio']),
    ('no_elec',      ['--exclude-electrical']),
    # Nested removal (cumulative, following confound priority)
    ('no_prox_pres',          ['--exclude-proximity', '--exclude-pressure']),
    ('no_prox_pres_col',      ['--exclude-proximity', '--exclude-pressure', '--exclude-color']),
    ('no_prox_pres_col_mag',  ['--exclude-proximity', '--exclude-pressure', '--exclude-color', '--exclude-magnetometer']),
])

# Encoder training params (matching encoder paper)
ENCODER_EPOCHS = 300
ENCODER_PATIENCE = 50
ENCODER_LR = 1e-3
ENCODER_D_MODEL = 256

# Decoder training params (V7 best config)
DECODER_EPOCHS = 500
DECODER_PATIENCE = 150

N_FOLDS = 5
WINDOW_SIZE = 256
STRIDE = 64


def run_cmd(cmd, desc="", timeout=3600):
    """Run a command and return stdout."""
    print(f"  [{desc}] {' '.join(cmd[:5])}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, 'PYTHONPATH': f'src:{os.environ.get("PYTHONPATH", "")}'})
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
    return result


def preprocess_fold(config_name, exclude_flags, fold, output_dir):
    """Preprocess one fold with modality exclusion."""
    fold_dir = output_dir / f"fold_{fold}"
    if (fold_dir / "train_sequences.npz").exists():
        return True

    fold_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "romesh_changes/run_preprocessing_cv_fold.py",
        "--data-dir", "data_clean",
        "--output-dir", str(fold_dir),
        "--vocab-path", "data/gcode_vocab_712.json",
        "--sensor-report", "outputs/sensor_consistency_report.json",
        "--threshold", "93.0",
        "--fold", str(fold),
        "--n-folds", str(N_FOLDS),
        "--window-size", str(WINDOW_SIZE),
        "--stride", str(STRIDE),
    ] + exclude_flags

    result = run_cmd(cmd, f"preprocess {config_name} fold {fold}")
    return result.returncode == 0


def train_encoder_fold(config_name, preproc_dir, checkpoint_dir, fold, device):
    """Train encoder for one fold using run_modality_ablation.py."""
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists():
        return True

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = preproc_dir / f"fold_{fold}"

    cmd = [
        sys.executable, "scripts/evaluation/run_modality_ablation.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(checkpoint_dir),  # checkpoint_dir IS the per-fold encoder dir
        "--iteration", f"fold_{fold}",
        "--seed", "42",
        "--max_epochs", str(ENCODER_EPOCHS),
        "--patience", str(ENCODER_PATIENCE),
        "--lr", str(ENCODER_LR),
    ]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device),
           'PYTHONPATH': f'src:{os.environ.get("PYTHONPATH", "")}'}

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(f"  ENCODER ERROR: {result.stderr[-300:]}")
    return result.returncode == 0


def train_decoder_fold(config_name, preproc_dir, encoder_ckpt, fold, output_dir, device):
    """Train decoder for one fold."""
    metrics_path = output_dir / "results" / "metrics.json"
    if metrics_path.exists():
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/evaluation/run_decoder_quick_test.py",
        "--vocab", "data/gcode_vocab_712.json",
        "--encoder_ckpt", str(encoder_ckpt),
        "--data_dir", str(preproc_dir / f"fold_{fold}"),
        "--fold", str(fold),
        "--epochs", str(DECODER_EPOCHS),
        "--patience", str(DECODER_PATIENCE),
        "--d_model", "384",
        "--n_layers", "8",
        "--n_heads", "12",
        "--lr", "0.0003",
        "--batch_size", "32",
        "--dropout", "0.1",
        "--digit_weight", "15.0",
        "--label_smoothing", "0.1",
        "--scheduled_sampling", "0.5",
        "--focal_gamma", "2.0",
        "--weight_decay", "0.1",
        "--warmup_epochs", "10",
        "--legacy_weight", "1.0",
        "--memory_pos_encoding", "True",
        "--grammar_constraint", "True",
        "--multi_window_context", "2",
        "--use_window_position", "True",
        "--use_sensor_prior", "True",
        "--use_pointer_network", "False",
        "--use_regression_head", "False",
        "--hierarchical", "False",
        "--drop_path_rate", "0.0",
        "--oversample_rare", "False",
        "--window_dropout", "0.1",
        "--noise_scale", "0.0",
        "--use_sequence_classifier", "False",
        "--use_distillation", "False",
        "--pointer_weight", "2.0",
        "--axis_value_tables", "data/axis_value_tables.json",
        "--seed", "42",
        "--output_dir", str(output_dir),
    ]

    result = run_cmd(cmd, f"decoder {config_name} fold {fold}", timeout=7200)
    return result.returncode == 0


def get_results(config_dir):
    """Get per-fold results for a config."""
    accs = []
    for fold in range(1, N_FOLDS + 1):
        mf = config_dir / f"fold_{fold}" / "results" / "metrics.json"
        try:
            with open(mf) as f:
                m = json.load(f)
            accs.append(m['test_metrics']['sequence_accuracy'] * 100)
        except:
            pass
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU(s) to use, comma-separated')
    parser.add_argument('--configs', type=str, default='all', help='Comma-separated config names, or "all"')
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpu.split(',')]

    if args.configs == 'all':
        configs = list(MODALITY_ABLATIONS.keys())
    else:
        configs = args.configs.split(',')

    base = Path("outputs/decoder20260304/modality_ablation")
    base.mkdir(parents=True, exist_ok=True)

    log_path = base / "ablation_log.txt"
    print(f"=== Modality Ablation Started: {datetime.now()} ===")
    print(f"Configs: {configs}")
    print(f"GPUs: {gpus}")
    print(f"Log: {log_path}")

    for ci, config_name in enumerate(configs):
        exclude_flags = MODALITY_ABLATIONS[config_name]
        gpu = gpus[ci % len(gpus)]
        config_dir = base / config_name
        preproc_dir = config_dir / "preprocessed"

        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name} ({exclude_flags}) on GPU {gpu}")
        print(f"{'='*60}")

        # Check if already done
        existing = get_results(config_dir)
        if len(existing) == N_FOLDS:
            mean = statistics.mean(existing)
            std = statistics.stdev(existing)
            print(f"  SKIP: Already complete. {mean:.1f} ± {std:.1f}%")
            continue

        for fold in range(1, N_FOLDS + 1):
            print(f"\n  --- Fold {fold}/{N_FOLDS} ---")

            # Step 1: Preprocess
            ok = preprocess_fold(config_name, exclude_flags, fold, preproc_dir)
            if not ok:
                print(f"  FAILED: preprocessing fold {fold}")
                continue

            # Get feature count
            meta_path = preproc_dir / f"fold_{fold}" / "metadata.json"
            with open(meta_path) as f:
                n_features = json.load(f)['n_continuous_features']
            print(f"  Features: {n_features}")

            # Step 2: Train encoder
            # The modality ablation script saves to: output_dir/checkpoint/best_model.pt
            encoder_base = config_dir / "encoders" / f"fold_{fold}"
            encoder_ckpt = encoder_base / "checkpoint" / "best_model.pt"

            # Check if we can reuse an existing encoder from the encoder paper
            existing_encoder = None
            if config_name == 'no_prox_pres' and n_features == 98:
                existing_encoder = Path(f"outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/fold_{fold}/encoder/checkpoint/best_model.pt")
            elif config_name == 'no_prox_pres_col_mag' and n_features == 56:
                existing_encoder = Path(f"outputs/experiments_2026_02_25/no_proximity_no_pressure_no_color_no_magnetometer_w256_s64_cv/fold_{fold}/encoder/checkpoint/best_model.pt")

            if existing_encoder and existing_encoder.exists():
                encoder_ckpt = existing_encoder
                print(f"  Reusing existing encoder: {existing_encoder}")
            elif not encoder_ckpt.exists():
                ok = train_encoder_fold(config_name, preproc_dir, encoder_base, fold, gpu)
                if not ok:
                    print(f"  FAILED: encoder training fold {fold}")
                    continue

            # Step 3: Train decoder
            decoder_dir = config_dir / f"fold_{fold}"
            ok = train_decoder_fold(config_name, preproc_dir, encoder_ckpt, fold, decoder_dir, gpu)
            if not ok:
                print(f"  FAILED: decoder training fold {fold}")
                continue

        # Results
        accs = get_results(config_dir)
        if accs:
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0
            print(f"\n  RESULT: {config_name} = {mean:.1f} ± {std:.1f}% (n={len(accs)})")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_results = {}
    for config_name in MODALITY_ABLATIONS:
        config_dir = base / config_name
        accs = get_results(config_dir)
        if accs:
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0
            all_results[config_name] = {'mean': mean, 'std': std, 'n': len(accs), 'folds': accs}
            print(f"  {config_name:<30} {mean:>5.1f} ± {std:.1f}%")

    with open(base / "modality_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Complete: {datetime.now()} ===")


if __name__ == '__main__':
    main()
