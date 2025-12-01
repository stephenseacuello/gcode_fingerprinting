#!/usr/bin/env python3
"""
Quantize ONNX models for efficient inference.

Supports FP16 and INT8 quantization with optional calibration data.
Quantization reduces model size and improves inference speed with minimal
accuracy loss.

Usage:
    # FP16 quantization (fastest, minimal accuracy loss)
    python scripts/quantize_model.py --model models/production/model.onnx --output models/production/model_fp16.onnx --method fp16

    # INT8 quantization (smallest, requires calibration)
    python scripts/quantize_model.py --model models/production/model.onnx --output models/production/model_int8.onnx --method int8 --calibration-data data/preprocessed/test

    # Compare all quantization methods
    python scripts/quantize_model.py --model models/production/model.onnx --output models/production/model_quantized.onnx --compare
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.transformers.onnx_model import OnnxModel
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCodeCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for INT8 quantization."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_samples: int = 100,
    ):
        """
        Initialize calibration data reader.

        Args:
            data_dir: Directory with preprocessed data
            num_samples: Number of calibration samples
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.num_samples = num_samples
        self.current_idx = 0

        # Load or generate calibration data
        if self.data_dir and self.data_dir.exists():
            self._load_real_data()
        else:
            logger.warning("No calibration data provided, using random data")
            self._generate_random_data()

    def _load_real_data(self):
        """Load real sensor data for calibration."""
        logger.info(f"Loading calibration data from {self.data_dir}")

        try:
            # Load continuous and categorical data
            continuous_path = self.data_dir / "continuous_test.npy"
            categorical_path = self.data_dir / "categorical_test.npy"

            if continuous_path.exists() and categorical_path.exists():
                continuous = np.load(continuous_path).astype(np.float32)
                categorical = np.load(categorical_path).astype(np.int64)

                # Take subset
                n = min(self.num_samples, len(continuous))
                self.continuous_data = continuous[:n]
                self.categorical_data = categorical[:n]

                logger.info(f"✓ Loaded {n} calibration samples")
            else:
                raise FileNotFoundError("Calibration data files not found")

        except Exception as e:
            logger.warning(f"Failed to load real data: {e}")
            self._generate_random_data()

    def _generate_random_data(self):
        """Generate random calibration data."""
        logger.info(f"Generating {self.num_samples} random calibration samples")

        self.continuous_data = np.random.randn(
            self.num_samples, 64, 135
        ).astype(np.float32)

        self.categorical_data = np.random.randint(
            0, 5, size=(self.num_samples, 64, 4)
        ).astype(np.int64)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample."""
        if self.current_idx >= len(self.continuous_data):
            return None

        batch = {
            'continuous': self.continuous_data[self.current_idx:self.current_idx + 1],
            'categorical': self.categorical_data[self.current_idx:self.current_idx + 1],
        }

        self.current_idx += 1
        return batch

    def rewind(self):
        """Reset reader to beginning."""
        self.current_idx = 0


class ModelQuantizer:
    """Quantize ONNX models."""

    def __init__(self, model_path: str):
        """
        Initialize quantizer.

        Args:
            model_path: Path to ONNX model
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loaded model: {model_path}")

    def quantize_fp16(self, output_path: str) -> None:
        """
        Convert model to FP16 precision.

        Args:
            output_path: Output path for quantized model
        """
        logger.info("Quantizing to FP16...")

        # Load model
        model = onnx.load(str(self.model_path))

        # Convert weights to FP16
        from onnxconverter_common import float16

        model_fp16 = float16.convert_float_to_float16(model)

        # Save
        onnx.save(model_fp16, output_path)

        logger.info(f"✓ FP16 model saved to: {output_path}")

        # Compare sizes
        original_size = self.model_path.stat().st_size / (1024 * 1024)
        fp16_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - fp16_size / original_size) * 100

        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  FP16: {fp16_size:.2f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")

    def quantize_int8_dynamic(self, output_path: str) -> None:
        """
        Dynamic INT8 quantization (weights only).

        Args:
            output_path: Output path for quantized model
        """
        logger.info("Quantizing to INT8 (dynamic)...")

        quantize_dynamic(
            model_input=str(self.model_path),
            model_output=output_path,
            weight_type=QuantType.QUInt8,
        )

        logger.info(f"✓ INT8 dynamic model saved to: {output_path}")

        # Compare sizes
        original_size = self.model_path.stat().st_size / (1024 * 1024)
        int8_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - int8_size / original_size) * 100

        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  INT8: {int8_size:.2f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")

    def quantize_int8_static(
        self,
        output_path: str,
        calibration_reader: CalibrationDataReader,
    ) -> None:
        """
        Static INT8 quantization (weights + activations).

        Args:
            output_path: Output path for quantized model
            calibration_reader: Calibration data reader
        """
        logger.info("Quantizing to INT8 (static)...")

        quantize_static(
            model_input=str(self.model_path),
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

        logger.info(f"✓ INT8 static model saved to: {output_path}")

        # Compare sizes
        original_size = self.model_path.stat().st_size / (1024 * 1024)
        int8_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - int8_size / original_size) * 100

        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  INT8: {int8_size:.2f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")

    def benchmark_model(
        self,
        model_path: str,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.

        Args:
            model_path: Path to ONNX model
            num_runs: Number of inference runs

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {Path(model_path).name}...")

        # Create session
        session = ort.InferenceSession(model_path)

        # Create dummy input
        continuous = np.random.randn(1, 64, 135).astype(np.float32)
        categorical = np.random.randint(0, 5, size=(1, 64, 4)).astype(np.int64)

        inputs = {
            'continuous': continuous,
            'categorical': categorical,
        }

        # Warmup
        for _ in range(10):
            session.run(None, inputs)

        # Benchmark
        import time
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        results = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
        }

        logger.info(f"  Mean: {results['mean_ms']:.2f} ms")
        logger.info(f"  Median: {results['median_ms']:.2f} ms")
        logger.info(f"  Std: {results['std_ms']:.2f} ms")

        return results


def compare_quantization_methods(
    model_path: str,
    output_dir: str,
    calibration_data: Optional[str] = None,
) -> None:
    """
    Compare all quantization methods.

    Args:
        model_path: Path to original ONNX model
        output_dir: Output directory for quantized models
        calibration_data: Path to calibration data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantizer = ModelQuantizer(model_path)

    results = {}

    # Original model
    logger.info("")
    logger.info("=" * 60)
    logger.info("Original Model (FP32)")
    logger.info("=" * 60)
    results["fp32"] = quantizer.benchmark_model(model_path)

    # FP16
    logger.info("")
    logger.info("=" * 60)
    logger.info("FP16 Quantization")
    logger.info("=" * 60)

    try:
        fp16_path = output_dir / "model_fp16.onnx"
        quantizer.quantize_fp16(str(fp16_path))
        results["fp16"] = quantizer.benchmark_model(str(fp16_path))
    except Exception as e:
        logger.error(f"FP16 quantization failed: {e}")
        results["fp16"] = None

    # INT8 Dynamic
    logger.info("")
    logger.info("=" * 60)
    logger.info("INT8 Dynamic Quantization")
    logger.info("=" * 60)

    try:
        int8_dynamic_path = output_dir / "model_int8_dynamic.onnx"
        quantizer.quantize_int8_dynamic(str(int8_dynamic_path))
        results["int8_dynamic"] = quantizer.benchmark_model(str(int8_dynamic_path))
    except Exception as e:
        logger.error(f"INT8 dynamic quantization failed: {e}")
        results["int8_dynamic"] = None

    # INT8 Static
    if calibration_data:
        logger.info("")
        logger.info("=" * 60)
        logger.info("INT8 Static Quantization")
        logger.info("=" * 60)

        try:
            calibration_reader = GCodeCalibrationDataReader(calibration_data)
            int8_static_path = output_dir / "model_int8_static.onnx"
            quantizer.quantize_int8_static(
                str(int8_static_path),
                calibration_reader
            )
            results["int8_static"] = quantizer.benchmark_model(str(int8_static_path))
        except Exception as e:
            logger.error(f"INT8 static quantization failed: {e}")
            results["int8_static"] = None

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Quantization Comparison Summary")
    logger.info("=" * 60)

    for method, result in results.items():
        if result:
            speedup = results["fp32"]["mean_ms"] / result["mean_ms"]
            logger.info(f"{method.upper():15} | {result['mean_ms']:6.2f} ms | {speedup:4.2f}x speedup")
        else:
            logger.info(f"{method.upper():15} | Failed")

    # Save results
    results_path = output_dir / "quantization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models for efficient inference"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for quantized model'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['fp16', 'int8', 'int8-static'],
        default='fp16',
        help='Quantization method (default: fp16)'
    )
    parser.add_argument(
        '--calibration-data',
        type=str,
        help='Path to calibration data directory (for int8-static)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all quantization methods'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs (default: 100)'
    )

    args = parser.parse_args()

    if args.compare:
        # Compare all methods
        output_dir = Path(args.output).parent / "quantized"
        compare_quantization_methods(
            args.model,
            str(output_dir),
            args.calibration_data,
        )
    else:
        # Single method
        quantizer = ModelQuantizer(args.model)

        if args.method == 'fp16':
            quantizer.quantize_fp16(args.output)
        elif args.method == 'int8':
            quantizer.quantize_int8_dynamic(args.output)
        elif args.method == 'int8-static':
            if not args.calibration_data:
                logger.error("--calibration-data required for int8-static")
                return

            calibration_reader = GCodeCalibrationDataReader(args.calibration_data)
            quantizer.quantize_int8_static(args.output, calibration_reader)

        # Benchmark
        logger.info("")
        quantizer.benchmark_model(args.output, args.num_runs)

    logger.info("\n✓ Quantization complete!")


if __name__ == "__main__":
    main()
