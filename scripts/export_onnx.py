#!/usr/bin/env python3
"""
Export trained models to ONNX format for production deployment.

This script converts PyTorch checkpoints to ONNX format with optimization
and validation. Supports both full model and encoder-only export.

Usage:
    # Export full model
    python scripts/export_onnx.py --checkpoint outputs/checkpoint_best.pt --output models/production/model.onnx

    # Export with dynamic batch size
    python scripts/export_onnx.py --checkpoint outputs/checkpoint_best.pt --output models/production/model.onnx --dynamic-batch

    # Export encoder only (for fingerprinting)
    python scripts/export_onnx.py --checkpoint outputs/checkpoint_best.pt --output models/production/encoder.onnx --encoder-only

    # With opset version
    python scripts/export_onnx.py --checkpoint outputs/checkpoint_best.pt --output model.onnx --opset 14
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch models to ONNX format."""

    def __init__(
        self,
        checkpoint_path: str,
        opset_version: int = 13,
        dynamic_batch: bool = False,
    ):
        """
        Initialize ONNX exporter.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            opset_version: ONNX opset version (default: 13 for wide compatibility)
            dynamic_batch: Enable dynamic batch size
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch

        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model = self.checkpoint['model']
        self.model.eval()

        # Get model config
        self.config = self.checkpoint.get('config', {})

    def create_dummy_input(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create dummy input for tracing.

        Args:
            batch_size: Batch size for dummy input
            seq_len: Sequence length

        Returns:
            Tuple of (continuous, categorical) tensors
        """
        continuous = torch.randn(batch_size, seq_len, 135)
        categorical = torch.randint(0, 5, (batch_size, seq_len, 4))

        return continuous, categorical

    def export_full_model(
        self,
        output_path: str,
        batch_size: int = 1,
        seq_len: int = 64,
    ) -> None:
        """
        Export full model to ONNX.

        Args:
            output_path: Output ONNX file path
            batch_size: Batch size (use 1 if dynamic_batch=True)
            seq_len: Sequence length
        """
        logger.info("Exporting full model to ONNX...")

        # Create dummy input
        continuous, categorical = self.create_dummy_input(batch_size, seq_len)

        # Define dynamic axes
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                'continuous': {0: 'batch_size'},
                'categorical': {0: 'batch_size'},
                'type_logits': {0: 'batch_size'},
                'command_logits': {0: 'batch_size'},
                'param_type_logits': {0: 'batch_size'},
                'param_value_logits': {0: 'batch_size'},
            }

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (continuous, categorical),
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['continuous', 'categorical'],
                output_names=[
                    'type_logits',
                    'command_logits',
                    'param_type_logits',
                    'param_value_logits',
                ],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

        logger.info(f"✓ Model exported to: {output_path}")

        # Verify
        self._verify_onnx(output_path, continuous, categorical)

    def export_encoder_only(
        self,
        output_path: str,
        batch_size: int = 1,
        seq_len: int = 64,
    ) -> None:
        """
        Export encoder only (for fingerprinting).

        Args:
            output_path: Output ONNX file path
            batch_size: Batch size
            seq_len: Sequence length
        """
        logger.info("Exporting encoder-only to ONNX...")

        # Create wrapper that only uses encoder
        class EncoderWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.encoder = model.encoder
                self.sensor_embedding = model.sensor_embedding

            def forward(self, continuous, categorical):
                # Embed sensor data
                embedded = self.sensor_embedding(continuous, categorical)

                # Encode
                encoded = self.encoder(embedded)

                # Global average pooling for fingerprint
                fingerprint = encoded.mean(dim=1)

                return fingerprint

        wrapper = EncoderWrapper(self.model)
        wrapper.eval()

        # Create dummy input
        continuous, categorical = self.create_dummy_input(batch_size, seq_len)

        # Define dynamic axes
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                'continuous': {0: 'batch_size'},
                'categorical': {0: 'batch_size'},
                'fingerprint': {0: 'batch_size'},
            }

        # Export
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (continuous, categorical),
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['continuous', 'categorical'],
                output_names=['fingerprint'],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

        logger.info(f"✓ Encoder exported to: {output_path}")

        # Verify encoder
        self._verify_encoder(output_path, continuous, categorical, wrapper)

    def _verify_onnx(
        self,
        onnx_path: str,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> None:
        """
        Verify ONNX model matches PyTorch output.

        Args:
            onnx_path: Path to ONNX model
            continuous: Dummy continuous input
            categorical: Dummy categorical input
        """
        logger.info("Verifying ONNX model...")

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Run ONNX inference
        ort_session = ort.InferenceSession(onnx_path)

        ort_inputs = {
            'continuous': continuous.numpy(),
            'categorical': categorical.numpy(),
        }

        ort_outputs = ort_session.run(None, ort_inputs)

        # Run PyTorch inference
        with torch.no_grad():
            torch_outputs = self.model(continuous, categorical)

        # Compare outputs
        output_names = [
            'type_logits',
            'command_logits',
            'param_type_logits',
            'param_value_logits',
        ]

        for i, name in enumerate(output_names):
            torch_out = torch_outputs[i].numpy()
            onnx_out = ort_outputs[i]

            # Check shape
            assert torch_out.shape == onnx_out.shape, \
                f"{name} shape mismatch: {torch_out.shape} vs {onnx_out.shape}"

            # Check values (allow small numerical difference)
            max_diff = np.abs(torch_out - onnx_out).max()
            assert max_diff < 1e-4, \
                f"{name} max difference too large: {max_diff}"

            logger.info(f"  ✓ {name}: shape={torch_out.shape}, max_diff={max_diff:.6f}")

        logger.info("✓ ONNX model verified successfully")

    def _verify_encoder(
        self,
        onnx_path: str,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        wrapper: nn.Module,
    ) -> None:
        """Verify encoder ONNX model."""
        logger.info("Verifying encoder ONNX model...")

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Run ONNX inference
        ort_session = ort.InferenceSession(onnx_path)

        ort_inputs = {
            'continuous': continuous.numpy(),
            'categorical': categorical.numpy(),
        }

        ort_outputs = ort_session.run(None, ort_inputs)

        # Run PyTorch inference
        with torch.no_grad():
            torch_output = wrapper(continuous, categorical)

        # Compare
        torch_fp = torch_output.numpy()
        onnx_fp = ort_outputs[0]

        assert torch_fp.shape == onnx_fp.shape
        max_diff = np.abs(torch_fp - onnx_fp).max()
        assert max_diff < 1e-4

        logger.info(f"  ✓ fingerprint: shape={torch_fp.shape}, max_diff={max_diff:.6f}")
        logger.info("✓ Encoder ONNX model verified successfully")

    def save_metadata(
        self,
        output_path: str,
        model_type: str = "full",
    ) -> None:
        """
        Save model metadata alongside ONNX file.

        Args:
            output_path: ONNX file path
            model_type: "full" or "encoder"
        """
        metadata = {
            "model_type": model_type,
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "pytorch_checkpoint": str(self.checkpoint_path),
            "config": self.config,
            "input_shapes": {
                "continuous": [-1, 64, 135] if self.dynamic_batch else [1, 64, 135],
                "categorical": [-1, 64, 4] if self.dynamic_batch else [1, 64, 4],
            },
        }

        # Add output shapes
        if model_type == "full":
            vocab_size = self.config.get('vocab_size', 170)
            metadata["output_shapes"] = {
                "type_logits": [-1, 64, 2] if self.dynamic_batch else [1, 64, 2],
                "command_logits": [-1, 64, vocab_size] if self.dynamic_batch else [1, 64, vocab_size],
                "param_type_logits": [-1, 64, vocab_size] if self.dynamic_batch else [1, 64, vocab_size],
                "param_value_logits": [-1, 64, vocab_size] if self.dynamic_batch else [1, 64, vocab_size],
            }
        else:
            d_model = self.config.get('d_model', 128)
            metadata["output_shapes"] = {
                "fingerprint": [-1, d_model] if self.dynamic_batch else [1, d_model],
            }

        # Save
        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved to: {metadata_path}")


def get_model_size(onnx_path: str) -> Dict[str, float]:
    """
    Get ONNX model file size and parameter count.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with size metrics
    """
    model = onnx.load(onnx_path)

    # File size
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    # Parameter count
    param_count = 0
    for initializer in model.graph.initializer:
        shape = [dim for dim in initializer.dims]
        param_count += np.prod(shape)

    return {
        "file_size_mb": file_size_mb,
        "param_count": int(param_count),
        "param_count_millions": param_count / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output ONNX file path (.onnx)'
    )
    parser.add_argument(
        '--encoder-only',
        action='store_true',
        help='Export encoder only (for fingerprinting)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for export (default: 1)'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=64,
        help='Sequence length (default: 64)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize exporter
    exporter = ONNXExporter(
        checkpoint_path=args.checkpoint,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    # Export
    logger.info("=" * 60)
    logger.info("ONNX Export Configuration")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model type: {'encoder-only' if args.encoder_only else 'full'}")
    logger.info(f"Opset version: {args.opset}")
    logger.info(f"Dynamic batch: {args.dynamic_batch}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_len}")
    logger.info("=" * 60)

    if args.encoder_only:
        exporter.export_encoder_only(
            output_path=str(output_path),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        exporter.save_metadata(str(output_path), model_type="encoder")
    else:
        exporter.export_full_model(
            output_path=str(output_path),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        exporter.save_metadata(str(output_path), model_type="full")

    # Print model stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model Statistics")
    logger.info("=" * 60)

    stats = get_model_size(str(output_path))
    logger.info(f"File size: {stats['file_size_mb']:.2f} MB")
    logger.info(f"Parameters: {stats['param_count']:,} ({stats['param_count_millions']:.2f}M)")
    logger.info("=" * 60)

    logger.info("")
    logger.info("✓ Export complete!")


if __name__ == "__main__":
    main()
