#!/usr/bin/env python3
"""
Run ensemble training followed by all experiments.
One command to run everything and go home.

Usage:
    python scripts/run_ensemble_and_experiments.py \
        --config configs/best_lambda_2k_vocab.json \
        --data-dir outputs/stratified_splits_2k_vocab \
        --vocab-path data/vocabulary_4digit_full.json \
        --encoder-path outputs/encoder_model.pt \
        --output-dir outputs/full_pipeline_2k_vocab
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: {description} failed with code {result.returncode}")
        return False
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description} - COMPLETE")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run ensemble + all experiments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--encoder-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/full_pipeline_2k_vocab')
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--skip-ensemble', action='store_true', help='Skip ensemble training')
    parser.add_argument('--skip-sensor', action='store_true', help='Skip sensor ablations')
    parser.add_argument('--skip-modality', action='store_true', help='Skip modality ablations')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baselines')
    parser.add_argument('--skip-architecture', action='store_true', help='Skip architecture ablations')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*70)
    print("# FULL PIPELINE: ENSEMBLE + ALL EXPERIMENTS")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Output: {output_dir}")
    print("#"*70)

    # 1. Ensemble training (3 seeds)
    if not args.skip_ensemble:
        run_command([
            sys.executable, 'scripts/experiments/train_ensemble_seeds.py',
            '--config', args.config,
            '--data-dir', args.data_dir,
            '--vocab-path', args.vocab_path,
            '--encoder-path', args.encoder_path,
            '--output-dir', str(output_dir / 'ensemble'),
            '--seeds', args.seeds,
        ], "STEP 1/5: Ensemble Training (3 seeds)")

    # 2. Sensor ablations
    if not args.skip_sensor:
        run_command([
            sys.executable, 'scripts/experiments/run_sensor_ablations.py',
            '--config', args.config,
            '--data-dir', args.data_dir,
            '--vocab-path', args.vocab_path,
            '--encoder-path', args.encoder_path,
            '--output-dir', str(output_dir / 'sensor_ablations'),
        ], "STEP 2/5: Sensor Ablations")

    # 3. Modality ablations
    if not args.skip_modality:
        run_command([
            sys.executable, 'scripts/experiments/run_modality_ablations.py',
            '--config', args.config,
            '--data-dir', args.data_dir,
            '--vocab-path', args.vocab_path,
            '--encoder-path', args.encoder_path,
            '--output-dir', str(output_dir / 'modality_ablations'),
        ], "STEP 3/5: Modality Ablations")

    # 4. Baseline comparisons
    if not args.skip_baselines:
        run_command([
            sys.executable, 'scripts/experiments/run_baseline_comparisons.py',
            '--split-dir', args.data_dir,
            '--vocab-path', args.vocab_path,
            '--encoder-path', args.encoder_path,
            '--output-dir', str(output_dir / 'baselines'),
            '--seeds', args.seeds,
        ], "STEP 4/5: Baseline Comparisons")

    # 5. Architecture ablations
    if not args.skip_architecture:
        run_command([
            sys.executable, 'scripts/experiments/run_architecture_ablations.py',
            '--config', args.config,
            '--data-dir', args.data_dir,
            '--vocab-path', args.vocab_path,
            '--encoder-path', args.encoder_path,
            '--output-dir', str(output_dir / 'architecture_ablations'),
        ], "STEP 5/5: Architecture Ablations")

    print("\n" + "#"*70)
    print("# FULL PIPELINE COMPLETE!")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Results saved to: {output_dir}")
    print("#"*70)
    print("\nNext: Update paper v20 with these results!")


if __name__ == '__main__':
    main()
