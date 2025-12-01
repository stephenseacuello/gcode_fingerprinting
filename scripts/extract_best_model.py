#!/usr/bin/env python3
"""
Extract the best model checkpoint from W&B sweep or specific run
"""
import os
import sys
from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Install with: pip install wandb")
    sys.exit(1)


def get_best_run_from_sweep(sweep_id: str, project: str = "gcode-fingerprinting",
                           entity: str = "seacuello-university-of-rhode-island"):
    """Get the best run from a sweep."""
    api = wandb.Api()

    # Get sweep
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Get best run
    best_run = sweep.best_run()

    if best_run:
        print(f"✅ Best run from sweep {sweep_id}:")
        print(f"   Run ID: {best_run.id}")
        print(f"   Run Name: {best_run.name}")

        # Get metrics
        summary = best_run.summary
        if 'val/param_type_acc' in summary:
            print(f"   Val Param Type Acc: {summary['val/param_type_acc']:.4f}")
        if 'val/overall_acc' in summary:
            print(f"   Val Overall Acc: {summary['val/overall_acc']:.4f}")
        if 'val/command_acc' in summary:
            print(f"   Val Command Acc: {summary['val/command_acc']:.4f}")

        return best_run
    else:
        print(f"❌ No completed runs found in sweep {sweep_id}")
        return None


def get_specific_run(run_id: str, project: str = "gcode-fingerprinting",
                    entity: str = "seacuello-university-of-rhode-island"):
    """Get a specific run by ID."""
    api = wandb.Api()

    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"✅ Found run: {run.name} ({run.id})")

        # Get metrics
        summary = run.summary
        if summary:
            print("\n📊 Performance Metrics:")
            metrics_to_show = [
                ('val/param_type_acc', 'Param Type Acc'),
                ('val/command_acc', 'Command Acc'),
                ('val/param_value_acc', 'Param Value Acc'),
                ('val/overall_acc', 'Overall Acc'),
                ('val/loss', 'Val Loss')
            ]

            for metric_key, metric_name in metrics_to_show:
                if metric_key in summary:
                    if 'loss' in metric_key:
                        print(f"   {metric_name}: {summary[metric_key]:.4f}")
                    else:
                        print(f"   {metric_name}: {summary[metric_key]:.2%}")

        return run
    except Exception as e:
        print(f"❌ Could not find run {run_id}: {e}")
        return None


def get_latest_runs(project: str = "gcode-fingerprinting",
                   entity: str = "seacuello-university-of-rhode-island",
                   limit: int = 5):
    """Get the latest runs from the project."""
    api = wandb.Api()

    print(f"\n📋 Latest {limit} runs:")
    runs = api.runs(f"{entity}/{project}",
                   filters={"state": "finished"},
                   order="-created_at",
                   per_page=limit)

    best_acc = 0
    best_run = None

    for run in runs:
        summary = run.summary
        acc = summary.get('val/overall_acc', summary.get('val/param_type_acc', 0))

        print(f"\n   {run.name} ({run.id})")
        print(f"   Created: {run.created_at}")
        if 'val/param_type_acc' in summary:
            print(f"   Param Type Acc: {summary['val/param_type_acc']:.2%}")
        if 'val/overall_acc' in summary:
            print(f"   Overall Acc: {summary['val/overall_acc']:.2%}")

        if acc > best_acc:
            best_acc = acc
            best_run = run

    if best_run:
        print(f"\n🏆 Best run: {best_run.name} ({best_run.id}) with accuracy: {best_acc:.2%}")

    return best_run


def download_checkpoint(run, output_dir: Path):
    """Download checkpoint from W&B run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to download checkpoint
    checkpoint_path = output_dir / 'checkpoint_best.pt'

    try:
        # List files in run
        files = run.files()
        checkpoint_files = [f for f in files if 'checkpoint' in f.name and f.name.endswith('.pt')]

        if checkpoint_files:
            # Download the best checkpoint
            best_checkpoint = checkpoint_files[0]
            print(f"\n📥 Downloading checkpoint: {best_checkpoint.name}")
            best_checkpoint.download(root=str(output_dir), replace=True)

            # Rename if needed
            downloaded_path = output_dir / best_checkpoint.name
            if downloaded_path.exists() and downloaded_path != checkpoint_path:
                shutil.move(str(downloaded_path), str(checkpoint_path))

            print(f"✅ Checkpoint saved to: {checkpoint_path}")
            return checkpoint_path
        else:
            print("⚠️  No checkpoint files found in run")

            # Check if there's a local checkpoint with the run ID
            local_path = Path(f'outputs/multihead_v2/{run.id}/checkpoint_best.pt')
            if local_path.exists():
                print(f"✅ Found local checkpoint: {local_path}")
                return local_path

            return None

    except Exception as e:
        print(f"⚠️  Could not download checkpoint: {e}")

        # Try local fallback
        local_paths = [
            Path(f'outputs/multihead_v2/{run.id}/checkpoint_best.pt'),
            Path(f'outputs/{run.id}/checkpoint_best.pt'),
            Path(f'outputs/sweep_{run.sweep.id if hasattr(run, "sweep") else "unknown"}_best/checkpoint_best.pt')
        ]

        for local_path in local_paths:
            if local_path.exists():
                print(f"✅ Found local checkpoint: {local_path}")
                return local_path

        return None


def generate_analysis_commands(checkpoint_path: Path, run_id: str = None):
    """Generate commands to analyze the model."""
    print("\n" + "="*80)
    print("📝 ANALYSIS COMMANDS")
    print("="*80)

    commands = []

    # Performance analysis command
    analysis_cmd = f"""# Run comprehensive performance analysis
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/analyze_performance.py \\
    --checkpoint {checkpoint_path} \\
    --data-dir outputs/processed_hybrid \\
    --vocab-path data/vocabulary_1digit_hybrid.json \\
    --output-dir outputs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"""

    if run_id:
        analysis_cmd += f" \\\n    --wandb-run {run_id}"

    commands.append(analysis_cmd)

    # Visualization command
    viz_cmd = f"""# Run all visualizations
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \\
    --model-path {checkpoint_path} \\
    --data-dir outputs/processed_hybrid \\
    --vocab-path data/vocabulary_1digit_hybrid.json \\
    --output-dir outputs/visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')} \\
    --mode all"""

    commands.append(viz_cmd)

    # Individual visualization modes
    for mode in ['interpretation', 'interactive', 'production']:
        mode_cmd = f"""# Run {mode} visualization only
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/visualize_training.py \\
    --model-path {checkpoint_path} \\
    --mode {mode}"""
        commands.append(mode_cmd)

    # Print all commands
    for cmd in commands:
        print(f"\n{cmd}")

    # Save to file
    cmd_file = Path('run_analysis_commands.sh')
    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Commands to analyze the best model\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n\n")

    os.chmod(cmd_file, 0o755)
    print(f"\n✅ Commands saved to: {cmd_file}")
    print("   Run with: ./run_analysis_commands.sh")


def main():
    parser = argparse.ArgumentParser(description='Extract best model from W&B')
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='W&B sweep ID')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Specific W&B run ID')
    parser.add_argument('--latest', action='store_true',
                       help='Get the best from latest runs')
    parser.add_argument('--project', type=str, default='gcode-fingerprinting',
                       help='W&B project name')
    parser.add_argument('--entity', type=str, default='seacuello-university-of-rhode-island',
                       help='W&B entity/username')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save checkpoint')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip downloading checkpoint')

    args = parser.parse_args()

    print("="*80)
    print("🔍 W&B BEST MODEL EXTRACTOR")
    print("="*80)

    # Get the run
    run = None
    run_id = args.run_id

    if args.sweep_id:
        print(f"\n🔍 Looking for best run in sweep: {args.sweep_id}")
        run = get_best_run_from_sweep(args.sweep_id, args.project, args.entity)
        if run:
            run_id = run.id
    elif args.run_id:
        print(f"\n🔍 Looking for specific run: {args.run_id}")
        run = get_specific_run(args.run_id, args.project, args.entity)
    elif args.latest:
        print(f"\n🔍 Getting best from latest runs")
        run = get_latest_runs(args.project, args.entity)
        if run:
            run_id = run.id
    else:
        print("\n⚠️  Please specify --sweep-id, --run-id, or --latest")
        print("\nExample usage:")
        print("  python scripts/extract_best_model.py --sweep-id 27v7pl9i")
        print("  python scripts/extract_best_model.py --run-id m6ufwb81")
        print("  python scripts/extract_best_model.py --latest")
        sys.exit(1)

    if not run:
        print("\n❌ No run found")
        sys.exit(1)

    # Download or locate checkpoint
    checkpoint_path = None

    if not args.no_download:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f'outputs/best_model_{run_id}')

        checkpoint_path = download_checkpoint(run, output_dir)

    # If no checkpoint downloaded, try to find local one
    if not checkpoint_path:
        print("\n🔍 Looking for local checkpoint...")
        possible_paths = [
            Path(f'outputs/multihead_v2/{run_id}/checkpoint_best.pt'),
            Path(f'outputs/{run_id}/checkpoint_best.pt'),
            Path('outputs/best_config_training/checkpoint_best.pt'),
            Path('outputs/multihead_v2/checkpoint_best.pt')
        ]

        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                print(f"✅ Found local checkpoint: {checkpoint_path}")
                break

    if checkpoint_path and checkpoint_path.exists():
        # Get file size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"\n📦 Checkpoint Details:")
        print(f"   Path: {checkpoint_path}")
        print(f"   Size: {size_mb:.2f} MB")

        # Generate analysis commands
        generate_analysis_commands(checkpoint_path, run_id)
    else:
        print("\n❌ No checkpoint found. Please check your W&B run or local files.")

    print("\n" + "="*80)
    print("✅ DONE")
    print("="*80)


if __name__ == '__main__':
    main()