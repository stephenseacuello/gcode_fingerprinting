#!/usr/bin/env python3
"""
Benchmark inference performance across model formats.

Measures latency, throughput, memory usage for:
- PyTorch (FP32)
- ONNX (FP32, FP16, INT8)
- Different batch sizes
- Different hardware (CPU, CUDA, MPS)

Usage:
    # Benchmark PyTorch checkpoint
    python scripts/benchmark_inference.py --checkpoint outputs/checkpoint_best.pt

    # Benchmark ONNX model
    python scripts/benchmark_inference.py --onnx models/production/model.onnx

    # Compare all formats
    python scripts/benchmark_inference.py --checkpoint outputs/checkpoint_best.pt --compare-all

    # Specific device
    python scripts/benchmark_inference.py --checkpoint outputs/checkpoint_best.pt --device cuda

    # Save results
    python scripts/benchmark_inference.py --checkpoint outputs/checkpoint_best.pt --output results/benchmark.json
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Benchmark inference performance."""

    def __init__(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        num_warmup: int = 10,
        num_runs: int = 100,
        seq_len: int = 64,
    ):
        """
        Initialize benchmark.

        Args:
            batch_sizes: Batch sizes to test
            num_warmup: Warmup iterations
            num_runs: Benchmark iterations
            seq_len: Sequence length
        """
        self.batch_sizes = batch_sizes
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.seq_len = seq_len

        # System info
        self.system_info = self._get_system_info()

        logger.info("Benchmark Configuration:")
        logger.info(f"  Batch sizes: {batch_sizes}")
        logger.info(f"  Warmup runs: {num_warmup}")
        logger.info(f"  Benchmark runs: {num_runs}")
        logger.info(f"  Sequence length: {seq_len}")

    def _get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        }

        # GPU info
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            info["cuda_available"] = False

        # MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            info["mps_available"] = True
        else:
            info["mps_available"] = False

        return info

    def create_dummy_data(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy input data."""
        continuous = torch.randn(batch_size, self.seq_len, 135)
        categorical = torch.randint(0, 5, (batch_size, self.seq_len, 4))

        return continuous, categorical

    def benchmark_pytorch(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> Dict[str, List[Dict]]:
        """
        Benchmark PyTorch model.

        Args:
            checkpoint_path: Path to checkpoint
            device: Device (cpu, cuda, mps)

        Returns:
            Benchmark results per batch size
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Benchmarking PyTorch (device={device})")
        logger.info("=" * 60)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = checkpoint['model']
        model.to(device)
        model.eval()

        results = []

        for batch_size in self.batch_sizes:
            logger.info(f"\nBatch size: {batch_size}")

            # Create data
            continuous, categorical = self.create_dummy_data(batch_size)
            continuous = continuous.to(device)
            categorical = categorical.to(device)

            # Warmup
            with torch.no_grad():
                for _ in range(self.num_warmup):
                    _ = model(continuous, categorical)

            # Benchmark
            times = []
            memory_usage = []

            with torch.no_grad():
                for _ in range(self.num_runs):
                    # Clear cache
                    if device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                    # Measure time
                    start = time.perf_counter()
                    _ = model(continuous, categorical)

                    if device == "cuda":
                        torch.cuda.synchronize()

                    end = time.perf_counter()

                    times.append((end - start) * 1000)  # Convert to ms

                    # Memory
                    if device == "cuda":
                        memory_usage.append(
                            torch.cuda.max_memory_allocated() / (1024**2)  # MB
                        )

            # Compute statistics
            result = {
                "batch_size": batch_size,
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "throughput_samples_per_sec": batch_size / (np.mean(times) / 1000),
            }

            if memory_usage:
                result["memory_mb"] = float(np.mean(memory_usage))

            results.append(result)

            logger.info(f"  Mean: {result['mean_ms']:.2f} ms")
            logger.info(f"  Median: {result['median_ms']:.2f} ms")
            logger.info(f"  P95: {result['p95_ms']:.2f} ms")
            logger.info(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/s")

            if memory_usage:
                logger.info(f"  Memory: {result['memory_mb']:.1f} MB")

        return {
            "model_type": "pytorch",
            "device": device,
            "results": results,
        }

    def benchmark_onnx(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Benchmark ONNX model.

        Args:
            onnx_path: Path to ONNX model
            providers: ONNX Runtime providers

        Returns:
            Benchmark results
        """
        import onnxruntime as ort

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Benchmarking ONNX")
        logger.info("=" * 60)

        # Default providers
        if providers is None:
            providers = ['CPUExecutionProvider']

        logger.info(f"Providers: {providers}")

        # Create session
        session = ort.InferenceSession(onnx_path, providers=providers)

        results = []

        for batch_size in self.batch_sizes:
            logger.info(f"\nBatch size: {batch_size}")

            # Create data
            continuous, categorical = self.create_dummy_data(batch_size)

            inputs = {
                'continuous': continuous.numpy(),
                'categorical': categorical.numpy(),
            }

            # Warmup
            for _ in range(self.num_warmup):
                _ = session.run(None, inputs)

            # Benchmark
            times = []

            for _ in range(self.num_runs):
                start = time.perf_counter()
                _ = session.run(None, inputs)
                end = time.perf_counter()

                times.append((end - start) * 1000)

            # Statistics
            result = {
                "batch_size": batch_size,
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "throughput_samples_per_sec": batch_size / (np.mean(times) / 1000),
            }

            results.append(result)

            logger.info(f"  Mean: {result['mean_ms']:.2f} ms")
            logger.info(f"  Median: {result['median_ms']:.2f} ms")
            logger.info(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/s")

        return {
            "model_type": "onnx",
            "model_path": str(onnx_path),
            "providers": providers,
            "results": results,
        }

    def compare_all_formats(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> Dict:
        """
        Compare all available model formats.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            device: Device for PyTorch

        Returns:
            Comparison results
        """
        checkpoint_path = Path(checkpoint_path)

        all_results = {
            "system_info": self.system_info,
            "benchmarks": [],
        }

        # PyTorch
        pytorch_results = self.benchmark_pytorch(str(checkpoint_path), device)
        all_results["benchmarks"].append(pytorch_results)

        # ONNX variants
        model_dir = checkpoint_path.parent

        # FP32 ONNX
        onnx_fp32 = model_dir / "model.onnx"
        if onnx_fp32.exists():
            onnx_results = self.benchmark_onnx(str(onnx_fp32))
            onnx_results["variant"] = "fp32"
            all_results["benchmarks"].append(onnx_results)

        # FP16 ONNX
        onnx_fp16 = model_dir / "quantized" / "model_fp16.onnx"
        if onnx_fp16.exists():
            onnx_fp16_results = self.benchmark_onnx(str(onnx_fp16))
            onnx_fp16_results["variant"] = "fp16"
            all_results["benchmarks"].append(onnx_fp16_results)

        # INT8 Dynamic
        onnx_int8_dynamic = model_dir / "quantized" / "model_int8_dynamic.onnx"
        if onnx_int8_dynamic.exists():
            onnx_int8_results = self.benchmark_onnx(str(onnx_int8_dynamic))
            onnx_int8_results["variant"] = "int8_dynamic"
            all_results["benchmarks"].append(onnx_int8_results)

        # Print comparison
        self._print_comparison(all_results)

        return all_results

    def _print_comparison(self, results: Dict) -> None:
        """Print comparison table."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("Performance Comparison (Batch Size = 1)")
        logger.info("=" * 80)

        # Extract batch_size=1 results
        comparison_data = []

        for benchmark in results["benchmarks"]:
            model_type = benchmark["model_type"]
            variant = benchmark.get("variant", "")

            # Find batch_size=1 result
            for result in benchmark["results"]:
                if result["batch_size"] == 1:
                    label = f"{model_type}_{variant}" if variant else model_type

                    comparison_data.append({
                        "label": label,
                        "mean_ms": result["mean_ms"],
                        "throughput": result["throughput_samples_per_sec"],
                    })
                    break

        # Sort by latency
        comparison_data.sort(key=lambda x: x["mean_ms"])

        # Print table
        logger.info(f"{'Model':<20} | {'Latency (ms)':>12} | {'Throughput':>12} | {'Speedup':>8}")
        logger.info("-" * 80)

        baseline = comparison_data[0]["mean_ms"]

        for data in comparison_data:
            speedup = baseline / data["mean_ms"]
            logger.info(
                f"{data['label']:<20} | "
                f"{data['mean_ms']:>12.2f} | "
                f"{data['throughput']:>10.1f}/s | "
                f"{speedup:>7.2f}x"
            )

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference performance"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--onnx',
        type=str,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all available formats'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for PyTorch (default: cpu)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16],
        help='Batch sizes to test (default: 1 2 4 8 16)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = InferenceBenchmark(
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
    )

    # Run benchmarks
    if args.compare_all and args.checkpoint:
        results = benchmark.compare_all_formats(args.checkpoint, args.device)

    elif args.checkpoint:
        results = benchmark.benchmark_pytorch(args.checkpoint, args.device)

    elif args.onnx:
        results = benchmark.benchmark_onnx(args.onnx)

    else:
        parser.error("Must provide --checkpoint or --onnx")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✓ Results saved to: {output_path}")

    logger.info("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
