#!/usr/bin/env python3
"""
Train production-ready models with optimal hyperparameters.

This script trains final models using the best configurations found during
hyperparameter sweeps. Supports ensemble training with multiple random seeds.

Usage:
    # Train single production model
    python scripts/train_production.py --config configs/production_best.json --output models/production

    # Train ensemble (3 models with different seeds)
    python scripts/train_production.py --config configs/production_best.json --output models/production --ensemble 3

    # Load best config from W&B sweep
    python scripts/train_production.py --sweep-id <sweep_id> --output models/production

    # Auto-export to ONNX after training
    python scripts/train_production.py --config configs/production_best.json --output models/production --export-onnx
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainer:
    """Train production models with best hyperparameters."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        sweep_id: Optional[str] = None,
        output_dir: str = "models/production",
    ):
        """
        Initialize production trainer.

        Args:
            config_path: Path to config JSON
            sweep_id: W&B sweep ID to load best config from
            output_dir: Output directory for models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif sweep_id:
            self.config = self._load_best_from_sweep(sweep_id)
        else:
            raise ValueError("Must provide either config_path or sweep_id")

        logger.info(f"Loaded configuration: {json.dumps(self.config, indent=2)}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"✓ Loaded config from: {config_path}")
        return config

    def _load_best_from_sweep(self, sweep_id: str) -> Dict:
        """
        Load best configuration from W&B sweep.

        Args:
            sweep_id: W&B sweep ID (format: entity/project/sweep_id)

        Returns:
            Best configuration dictionary
        """
        logger.info(f"Loading best config from sweep: {sweep_id}")

        api = wandb.Api()
        sweep = api.sweep(sweep_id)

        # Get best run
        best_run = max(
            sweep.runs,
            key=lambda r: r.summary.get('val/overall_accuracy', 0)
        )

        logger.info(f"✓ Best run: {best_run.name}")
        logger.info(f"  Accuracy: {best_run.summary['val/overall_accuracy']:.2f}%")

        # Extract config
        config = dict(best_run.config)

        # Save for reference
        config_path = self.output_dir / "best_config_from_sweep.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✓ Config saved to: {config_path}")

        return config

    def train_single_model(
        self,
        seed: int = 42,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Train a single production model.

        Args:
            seed: Random seed
            model_name: Model name (default: model_seed{seed})

        Returns:
            Path to trained checkpoint
        """
        if model_name is None:
            model_name = f"model_seed{seed}"

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Training: {model_name}")
        logger.info("=" * 60)

        # Create model output directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Prepare training command
        cmd = [
            "python", "train_multihead.py",
            "--config", str(config_path),
            "--seed", str(seed),
            "--output", str(model_dir),
        ]

        # Add W&B project if specified
        if "wandb_project" in self.config:
            cmd.extend(["--wandb-project", self.config["wandb_project"]])

        # Run training
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
            )

            checkpoint_path = model_dir / "checkpoint_best.pt"

            if checkpoint_path.exists():
                logger.info(f"✓ Training complete: {checkpoint_path}")
                return str(checkpoint_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise

    def train_ensemble(
        self,
        num_models: int = 3,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Train ensemble of models with different seeds.

        Args:
            num_models: Number of models in ensemble
            base_seed: Base random seed

        Returns:
            List of checkpoint paths
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Training Ensemble ({num_models} models)")
        logger.info("=" * 60)

        checkpoints = []

        for i in range(num_models):
            seed = base_seed + i
            model_name = f"ensemble_model_{i+1}_seed{seed}"

            checkpoint = self.train_single_model(seed, model_name)
            checkpoints.append(checkpoint)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Ensemble Training Complete")
        logger.info("=" * 60)
        logger.info(f"Trained {len(checkpoints)} models:")

        for i, ckpt in enumerate(checkpoints, 1):
            logger.info(f"  {i}. {ckpt}")

        # Create ensemble config
        ensemble_config = {
            "num_models": num_models,
            "checkpoints": checkpoints,
            "base_seed": base_seed,
            "config": self.config,
        }

        ensemble_config_path = self.output_dir / "ensemble_config.json"
        with open(ensemble_config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)

        logger.info(f"\n✓ Ensemble config saved to: {ensemble_config_path}")

        return checkpoints

    def export_to_onnx(
        self,
        checkpoint_path: str,
        opset: int = 13,
        dynamic_batch: bool = True,
    ) -> str:
        """
        Export checkpoint to ONNX format.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            opset: ONNX opset version
            dynamic_batch: Enable dynamic batch size

        Returns:
            Path to ONNX model
        """
        logger.info(f"Exporting to ONNX: {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        onnx_path = checkpoint_path.parent / "model.onnx"

        cmd = [
            "python", "scripts/export_onnx.py",
            "--checkpoint", str(checkpoint_path),
            "--output", str(onnx_path),
            "--opset", str(opset),
        ]

        if dynamic_batch:
            cmd.append("--dynamic-batch")

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"✓ ONNX export complete: {onnx_path}")
            return str(onnx_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def quantize_model(
        self,
        onnx_path: str,
        calibration_data: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Quantize ONNX model.

        Args:
            onnx_path: Path to ONNX model
            calibration_data: Path to calibration data

        Returns:
            Dictionary of quantized model paths
        """
        logger.info(f"Quantizing model: {onnx_path}")

        onnx_path = Path(onnx_path)
        quantized_dir = onnx_path.parent / "quantized"

        cmd = [
            "python", "scripts/quantize_model.py",
            "--model", str(onnx_path),
            "--output", str(quantized_dir / "model_quantized.onnx"),
            "--compare",
        ]

        if calibration_data:
            cmd.extend(["--calibration-data", calibration_data])

        try:
            subprocess.run(cmd, check=True)

            # Find quantized models
            quantized_models = {
                "fp16": str(quantized_dir / "model_fp16.onnx"),
                "int8_dynamic": str(quantized_dir / "model_int8_dynamic.onnx"),
            }

            if calibration_data:
                quantized_models["int8_static"] = str(
                    quantized_dir / "model_int8_static.onnx"
                )

            logger.info("✓ Quantization complete")
            return quantized_models

        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e}")
            raise

    def create_deployment_package(
        self,
        checkpoint_path: str,
        include_quantized: bool = True,
    ) -> str:
        """
        Create deployment package with model and artifacts.

        Args:
            checkpoint_path: Path to checkpoint
            include_quantized: Include quantized models

        Returns:
            Path to deployment package
        """
        logger.info("Creating deployment package...")

        checkpoint_path = Path(checkpoint_path)
        package_dir = self.output_dir / "deployment_package"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        shutil.copy(checkpoint_path, package_dir / "model.pt")

        # Copy ONNX if exists
        onnx_path = checkpoint_path.parent / "model.onnx"
        if onnx_path.exists():
            shutil.copy(onnx_path, package_dir / "model.onnx")

        # Copy quantized models if requested
        if include_quantized:
            quantized_dir = checkpoint_path.parent / "quantized"
            if quantized_dir.exists():
                shutil.copytree(
                    quantized_dir,
                    package_dir / "quantized",
                    dirs_exist_ok=True,
                )

        # Copy config
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            shutil.copy(config_path, package_dir / "config.json")

        # Create README
        readme_content = f"""# G-Code Fingerprinting Model - Deployment Package

## Contents

- `model.pt`: PyTorch checkpoint
- `model.onnx`: ONNX model (FP32)
- `quantized/`: Quantized models (FP16, INT8)
- `config.json`: Model configuration

## Quick Start

### PyTorch Inference

```python
import torch

checkpoint = torch.load('model.pt')
model = checkpoint['model']
model.eval()

# Inference
with torch.no_grad():
    output = model(continuous, categorical)
```

### ONNX Inference

```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
outputs = session.run(None, {{
    'continuous': continuous_data,
    'categorical': categorical_data,
}})
```

### Using FastAPI Server

```bash
docker-compose up
curl http://localhost:8000/predict -X POST -d @sample_request.json
```

## Model Info

- **Checkpoint**: {checkpoint_path.name}
- **Config**: See config.json
- **Created**: {checkpoint_path.stat().st_mtime}

## Documentation

See full documentation at: https://github.com/your-repo/gcode-fingerprinting
"""

        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)

        logger.info(f"✓ Deployment package created: {package_dir}")

        return str(package_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train production-ready models"
    )

    # Config source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to config JSON file'
    )
    config_group.add_argument(
        '--sweep-id',
        type=str,
        help='W&B sweep ID to load best config from'
    )

    # Training options
    parser.add_argument(
        '--output',
        type=str,
        default='models/production',
        help='Output directory (default: models/production)'
    )
    parser.add_argument(
        '--ensemble',
        type=int,
        default=None,
        help='Train ensemble with N models (different seeds)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # Export options
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='Export to ONNX after training'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Quantize ONNX model (requires --export-onnx)'
    )
    parser.add_argument(
        '--calibration-data',
        type=str,
        help='Path to calibration data for INT8 quantization'
    )
    parser.add_argument(
        '--create-package',
        action='store_true',
        help='Create deployment package'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = ProductionTrainer(
        config_path=args.config,
        sweep_id=args.sweep_id,
        output_dir=args.output,
    )

    # Train model(s)
    if args.ensemble:
        checkpoints = trainer.train_ensemble(
            num_models=args.ensemble,
            base_seed=args.seed,
        )
        primary_checkpoint = checkpoints[0]
    else:
        primary_checkpoint = trainer.train_single_model(seed=args.seed)
        checkpoints = [primary_checkpoint]

    # Export to ONNX
    if args.export_onnx:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Exporting to ONNX")
        logger.info("=" * 60)

        for checkpoint in checkpoints:
            onnx_path = trainer.export_to_onnx(checkpoint)

            # Quantize if requested
            if args.quantize:
                trainer.quantize_model(
                    onnx_path,
                    calibration_data=args.calibration_data,
                )

    # Create deployment package
    if args.create_package:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Creating Deployment Package")
        logger.info("=" * 60)

        package_path = trainer.create_deployment_package(
            primary_checkpoint,
            include_quantized=args.quantize,
        )

        logger.info(f"\n✓ Package ready: {package_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Production Training Complete!")
    logger.info("=" * 60)

    if args.ensemble:
        logger.info(f"\nTrained {len(checkpoints)} models in ensemble")
    else:
        logger.info(f"\nTrained model: {primary_checkpoint}")


if __name__ == "__main__":
    main()
