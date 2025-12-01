#!/usr/bin/env python3
"""
Deploy a checkpoint to the API server's default location.

Usage:
    python scripts/deploy_checkpoint.py \
        --source outputs/best_from_sweep/checkpoint_best.pt \
        --target outputs/production/checkpoint_best.pt
"""

import argparse
import shutil
import torch
from pathlib import Path
import sys


def validate_checkpoint(checkpoint_path: Path):
    """
    Validate that a checkpoint file can be loaded.

    Returns:
        dict: Checkpoint metadata if valid, None otherwise
    """
    try:
        print(f"Validating checkpoint: {checkpoint_path}")

        # Try to load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check for required keys
        required_keys = ['backbone_state_dict', 'multihead_state_dict']
        missing_keys = [k for k in required_keys if k not in checkpoint]

        if missing_keys:
            print(f"❌ Checkpoint missing required keys: {missing_keys}")
            return None

        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_acc': checkpoint.get('val_acc', 'N/A'),
            'config': checkpoint.get('config', {}),
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
        }

        print(f"✓ Checkpoint is valid")
        print(f"  Epoch: {metadata['epoch']}")
        print(f"  Val Accuracy: {metadata['val_acc']}")

        if 'hidden_dim' in metadata['config']:
            print(f"  Hidden Dim: {metadata['config']['hidden_dim']}")
        if 'num_layers' in metadata['config']:
            print(f"  Num Layers: {metadata['config']['num_layers']}")

        return metadata

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None


def deploy_checkpoint(source: Path, target: Path, create_backup: bool = True):
    """
    Copy checkpoint from source to target location.

    Args:
        source: Source checkpoint path
        target: Target checkpoint path
        create_backup: Whether to backup existing checkpoint at target
    """
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")

    # Create target directory if needed
    target.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing checkpoint
    if target.exists() and create_backup:
        backup_path = target.parent / f"{target.stem}_backup{target.suffix}"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(target, backup_path)

    # Copy checkpoint
    print(f"Copying checkpoint:")
    print(f"  From: {source}")
    print(f"  To:   {target}")
    shutil.copy2(source, target)
    print(f"✓ Checkpoint deployed successfully")

    return target


def update_api_default_path(checkpoint_path: Path):
    """
    Display instructions for updating API server to use new checkpoint.

    Note: This doesn't automatically modify server.py, just provides instructions.
    """
    print()
    print("=" * 80)
    print("API SERVER UPDATE")
    print("=" * 80)
    print()
    print("To use this checkpoint with the API server, you have two options:")
    print()
    print("Option 1: Update the default checkpoint path in server.py")
    print(f"  Edit: src/miracle/api/server.py")
    print(f"  Change default_checkpoint to: \"{checkpoint_path}\"")
    print()
    print("Option 2: Use the /load_checkpoint endpoint (if available)")
    print(f"  curl -X POST http://localhost:8000/load_checkpoint \\")
    print(f"    -H \"Content-Type: application/json\" \\")
    print(f"    -d '{{\"checkpoint_path\": \"{checkpoint_path}\"}}'")
    print()
    print("Option 3: Restart the API server")
    print("  The server will auto-load from the default path on startup")
    print()


def main():
    parser = argparse.ArgumentParser(description='Deploy checkpoint to API server')
    parser.add_argument('--source', type=str, required=True,
                       help='Source checkpoint path')
    parser.add_argument('--target', type=str, default=None,
                       help='Target checkpoint path (default: outputs/production/checkpoint_best.pt)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of existing checkpoint')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip checkpoint validation')

    args = parser.parse_args()

    # Set default target if not specified
    if args.target is None:
        args.target = 'outputs/production/checkpoint_best.pt'

    source = Path(args.source)
    target = Path(args.target)

    try:
        print("=" * 80)
        print("CHECKPOINT DEPLOYMENT")
        print("=" * 80)
        print()

        # Validate source checkpoint
        if not args.skip_validation:
            metadata = validate_checkpoint(source)
            if metadata is None:
                print("\n❌ Source checkpoint is invalid. Aborting deployment.")
                return 1
            print()

        # Deploy checkpoint
        deployed_path = deploy_checkpoint(
            source, target,
            create_backup=not args.no_backup
        )
        print()

        # Validate deployed checkpoint
        if not args.skip_validation:
            print("Validating deployed checkpoint...")
            metadata = validate_checkpoint(deployed_path)
            if metadata is None:
                print("\n❌ Deployed checkpoint is invalid!")
                return 1
            print()

        # Provide API update instructions
        update_api_default_path(deployed_path)

        print("=" * 80)
        print("DEPLOYMENT COMPLETE")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Restart API server to use new checkpoint")
        print("  2. Test prediction: curl -X POST http://localhost:8000/predict -d @test_payload.json")
        print("  3. Check API info: curl http://localhost:8000/info")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
