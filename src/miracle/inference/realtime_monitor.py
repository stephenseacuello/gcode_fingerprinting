#!/usr/bin/env python3
"""
Real-time CNC Phase Detection and Anomaly Monitoring

This script demonstrates how to use the trained model for:
1. Real-time phase detection from live sensor data
2. Anomaly detection based on reconstruction error
3. Continuous monitoring with alerts

Usage:
    python src/miracle/inference/realtime_monitor.py \
        --checkpoint outputs/models/MM_DTAE_LSTM_20251116_165433_best.pt \
        --input-csv data/test_001_aligned.csv \
        --window-size 64 \
        --anomaly-threshold 0.3
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig


class RealtimeMonitor:
    """Real-time CNC phase detection and anomaly monitoring."""

    def __init__(
        self,
        checkpoint_path: str,
        window_size: int = 64,
        anomaly_threshold: float = 0.3,
        device: str = "cpu"
    ):
        """
        Initialize the monitor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            window_size: Size of sliding window (must match training)
            anomaly_threshold: Reconstruction error threshold for anomaly detection
            device: Device to run inference on
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        # PyTorch 2.6+ requires weights_only=False for models with numpy objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Create model config
        config_dict = checkpoint['config']
        self.config = ModelConfig(**config_dict)

        # Initialize and load model
        self.model = MM_DTAE_LSTM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Expected input: {self.config.sensor_dims[0]} continuous + {self.config.sensor_dims[1]} categorical features")

        # Phase names (customize based on your G-code)
        self.phase_names = {
            0: "Early Phase (Line 42)",
            1: "Middle Phase (Lines 50-60)",
            2: "Late Phase (Lines 73-96)"
        }

        # Statistics tracking
        self.phase_counts = {0: 0, 1: 0, 2: 0}
        self.anomaly_count = 0
        self.total_windows = 0

        # History for smoothing predictions
        self.prediction_history = []
        self.history_size = 5

    def preprocess_window(
        self,
        continuous_data: np.ndarray,
        categorical_data: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess a single window of data.

        Args:
            continuous_data: [window_size, n_continuous_features]
            categorical_data: [window_size, n_categorical_features]

        Returns:
            Tuple of (continuous_tensor, categorical_tensor, lengths)
        """
        # Convert to tensors
        cont_tensor = torch.from_numpy(continuous_data).float()
        cat_tensor = torch.from_numpy(categorical_data).long()

        # Add batch dimension
        cont_tensor = cont_tensor.unsqueeze(0)  # [1, T, C]
        cat_tensor = cat_tensor.unsqueeze(0)    # [1, T, K]

        # Lengths
        lengths = torch.tensor([self.window_size])

        # Move to device
        cont_tensor = cont_tensor.to(self.device)
        cat_tensor = cat_tensor.to(self.device)
        lengths = lengths.to(self.device)

        return cont_tensor, cat_tensor, lengths

    def predict_window(
        self,
        continuous_data: np.ndarray,
        categorical_data: np.ndarray,
        smooth: bool = True
    ) -> Dict:
        """
        Predict phase and compute anomaly score for a single window.

        Args:
            continuous_data: [window_size, n_continuous_features]
            categorical_data: [window_size, n_categorical_features]
            smooth: Whether to smooth predictions using history

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        cont_tensor, cat_tensor, lengths = self.preprocess_window(
            continuous_data, categorical_data
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model([cont_tensor, cat_tensor], lengths)

        # Classification
        logits = outputs['cls']  # [1, num_classes]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # Smoothing (majority vote over last N predictions)
        if smooth:
            self.prediction_history.append(predicted_class)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)

            # Majority vote
            from collections import Counter
            counts = Counter(self.prediction_history)
            smoothed_class = counts.most_common(1)[0][0]
        else:
            smoothed_class = predicted_class

        # Reconstruction error (anomaly score)
        reconstruction = outputs['recon'][0]  # [1, T, D] -> [T, D]
        recon_error = F.mse_loss(reconstruction, cont_tensor[0]).item()

        # Anomaly detection
        is_anomaly = recon_error > self.anomaly_threshold

        # Update statistics
        self.total_windows += 1
        self.phase_counts[smoothed_class] += 1
        if is_anomaly:
            self.anomaly_count += 1

        # Fingerprint
        fingerprint = outputs['fingerprint'].cpu().numpy()[0]  # [128]

        return {
            'predicted_class': predicted_class,
            'smoothed_class': smoothed_class,
            'phase_name': self.phase_names[smoothed_class],
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'reconstruction_error': recon_error,
            'is_anomaly': is_anomaly,
            'anomaly_score': recon_error / self.anomaly_threshold,
            'fingerprint': fingerprint,
            'fingerprint_norm': float(np.linalg.norm(fingerprint))
        }

    def monitor_stream(
        self,
        continuous_stream: np.ndarray,
        categorical_stream: np.ndarray,
        stride: int = 1,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Monitor a stream of sensor data with sliding windows.

        Args:
            continuous_stream: [n_timesteps, n_continuous_features]
            categorical_stream: [n_timesteps, n_categorical_features]
            stride: Number of timesteps to advance per window
            verbose: Print results in real-time

        Returns:
            List of prediction results for each window
        """
        n_timesteps = continuous_stream.shape[0]
        results = []

        print(f"\n{'='*80}")
        print(f"Starting real-time monitoring...")
        print(f"Total timesteps: {n_timesteps}")
        print(f"Window size: {self.window_size}")
        print(f"Stride: {stride}")
        print(f"Anomaly threshold: {self.anomaly_threshold}")
        print(f"{'='*80}\n")

        # Sliding window
        for start_idx in range(0, n_timesteps - self.window_size + 1, stride):
            end_idx = start_idx + self.window_size

            # Extract window
            cont_window = continuous_stream[start_idx:end_idx]
            cat_window = categorical_stream[start_idx:end_idx]

            # Predict
            result = self.predict_window(cont_window, cat_window)
            result['window_start'] = start_idx
            result['window_end'] = end_idx
            results.append(result)

            # Print status
            if verbose:
                self._print_status(result, start_idx, n_timesteps)

        # Final summary
        self._print_summary()

        return results

    def _print_status(self, result: Dict, current_idx: int, total: int):
        """Print real-time monitoring status."""
        phase = result['phase_name']
        conf = result['confidence']
        recon_err = result['reconstruction_error']

        # Progress
        progress = (current_idx / total) * 100

        # Status line
        status = f"[{progress:5.1f}%] Window {result['window_start']:4d}-{result['window_end']:4d} | "
        status += f"Phase: {phase:25s} | Conf: {conf:.3f} | "
        status += f"Recon Error: {recon_err:.4f}"

        # Anomaly alert
        if result['is_anomaly']:
            status += " | ⚠️  ANOMALY DETECTED!"

        print(status)

    def _print_summary(self):
        """Print monitoring session summary."""
        print(f"\n{'='*80}")
        print(f"MONITORING SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"Total windows processed: {self.total_windows}")
        print(f"\nPhase Distribution:")
        for phase_id, count in sorted(self.phase_counts.items()):
            percentage = (count / self.total_windows) * 100 if self.total_windows > 0 else 0
            print(f"  {self.phase_names[phase_id]:25s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\nAnomaly Detection:")
        anomaly_rate = (self.anomaly_count / self.total_windows) * 100 if self.total_windows > 0 else 0
        print(f"  Anomalies detected: {self.anomaly_count} ({anomaly_rate:.2f}%)")
        print(f"  Threshold used: {self.anomaly_threshold}")
        print(f"{'='*80}\n")

    def save_results(self, results: List[Dict], output_path: str):
        """Save monitoring results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame for easy analysis
        df_data = []
        for r in results:
            row = {
                'window_start': r['window_start'],
                'window_end': r['window_end'],
                'predicted_class': r['predicted_class'],
                'smoothed_class': r['smoothed_class'],
                'phase_name': r['phase_name'],
                'confidence': r['confidence'],
                'reconstruction_error': r['reconstruction_error'],
                'is_anomaly': r['is_anomaly'],
                'anomaly_score': r['anomaly_score']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Also save fingerprints separately
        fingerprints = np.array([r['fingerprint'] for r in results])
        fp_path = output_path.parent / (output_path.stem + "_fingerprints.npy")
        np.save(fp_path, fingerprints)
        print(f"Fingerprints saved to {fp_path}")


def load_and_preprocess_csv(csv_path: str, metadata_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess CSV file for monitoring.

    Args:
        csv_path: Path to aligned CSV file
        metadata_path: Optional path to preprocessing metadata (for column info)

    Returns:
        Tuple of (continuous_data, categorical_data)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Load metadata if provided
    if metadata_path:
        with open(metadata_path) as f:
            metadata = json.load(f)
        continuous_cols = metadata['continuous_cols']
        categorical_cols = metadata['categorical_cols']
    else:
        # Default columns (from your preprocessing)
        continuous_cols = [
            'x', 'y', 'z', 'a',
            'mo1en', 'mo1st', 'mo1tr',
            'mo2en', 'mo2st', 'mo2tr',
            'mo3en', 'mo3st', 'mo3tr',
            'mo4en', 'mo4st', 'mo4tr',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'pressure', 'temp', 'altitude',
            'r', 'g', 'b', 'c', 'colorTemp', 'lux'
        ]
        categorical_cols = ['sr', 'stat', 'unit', 'coor', 'momo', 'plan']

    # Extract features
    continuous_data = df[continuous_cols].values
    categorical_data = df[categorical_cols].values

    # Normalize continuous features (simple z-score)
    continuous_mean = np.mean(continuous_data, axis=0)
    continuous_std = np.std(continuous_data, axis=0) + 1e-8
    continuous_data = (continuous_data - continuous_mean) / continuous_std

    print(f"Loaded {len(df)} timesteps")
    print(f"Continuous features: {continuous_data.shape[1]}")
    print(f"Categorical features: {categorical_data.shape[1]}")

    return continuous_data, categorical_data


def main():
    parser = argparse.ArgumentParser(description="Real-time CNC Phase Detection and Anomaly Monitoring")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Path to input CSV file (aligned sensor data)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Optional path to preprocessing metadata.json")
    parser.add_argument("--window-size", type=int, default=64,
                        help="Window size (must match training)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for sliding window (1 = every timestep)")
    parser.add_argument("--anomaly-threshold", type=float, default=0.3,
                        help="Reconstruction error threshold for anomaly detection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="outputs/realtime_monitoring",
                        help="Output directory for results")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable prediction smoothing")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    continuous_data, categorical_data = load_and_preprocess_csv(
        args.input_csv,
        args.metadata
    )

    # Truncate to expected dimensions (115 continuous, 6 categorical)
    continuous_data = continuous_data[:, :115]
    categorical_data = categorical_data[:, :6]

    # Initialize monitor
    monitor = RealtimeMonitor(
        checkpoint_path=args.checkpoint,
        window_size=args.window_size,
        anomaly_threshold=args.anomaly_threshold,
        device=args.device
    )

    # Run monitoring
    results = monitor.monitor_stream(
        continuous_data,
        categorical_data,
        stride=args.stride,
        verbose=True
    )

    # Save results
    csv_name = Path(args.input_csv).stem
    output_csv = output_dir / f"{csv_name}_monitoring_results.csv"
    monitor.save_results(results, output_csv)

    print(f"\n✅ Monitoring complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
