"""
Training logger with TensorBoard, metrics tracking, and visualization.

Provides comprehensive logging for PyTorch model training including:
- TensorBoard integration for real-time monitoring
- JSON and CSV metrics export
- Automatic plot generation
- Training configuration saving
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Comprehensive training logger with TensorBoard, metrics, and plots.

    Creates a timestamped run directory and logs all training information:
    - TensorBoard logs for real-time monitoring
    - JSON metrics file (full history)
    - CSV metrics file (easy analysis)
    - Training configuration
    - Automatic plot generation at end

    Args:
        base_dir: Base directory for training outputs (e.g., 'outputs/training')
        run_name: Optional custom run name (default: auto-generated)
        model_name: Optional model name for auto-naming (default: None)

    Example:
        logger = TrainingLogger('outputs/training')
        logger.log_config(args)

        for epoch in range(epochs):
            # ... training ...
            logger.log_epoch(epoch, train_metrics, val_metrics, lr)

        logger.save_plots()
        logger.close()
    """

    def __init__(self, base_dir: Path | str, run_name: Optional[str] = None, model_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory with model name
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if model_name:
                run_name = f"{model_name}_{timestamp}"
            else:
                run_name = f"run_{timestamp}"

        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Create subdirectories
        self.tb_dir = self.run_dir / 'tensorboard'
        self.plots_dir = self.run_dir / 'plots'
        self.tb_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # Metrics storage
        self.metrics_history = []

        # File paths
        self.metrics_json = self.run_dir / 'metrics.json'
        self.metrics_csv = self.run_dir / 'metrics.csv'
        self.config_json = self.run_dir / 'config.json'

        print(f"✓ Training logger initialized: {self.run_dir}")
        print(f"  TensorBoard: tensorboard --logdir {self.tb_dir.parent}")

    def log_config(self, config: Dict[str, Any] | Any):
        """
        Save training configuration to JSON file.

        Args:
            config: Training configuration (dict or argparse.Namespace)
        """
        if not isinstance(config, dict):
            config = vars(config)

        # Add metadata
        config_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'run_dir': str(self.run_dir),
            'model_name': self.model_name,
            **config
        }

        with open(self.config_json, 'w') as f:
            json.dump(config_with_meta, f, indent=2)

        # Log to TensorBoard as text
        config_text = '\n'.join(f'{k}: {v}' for k, v in config.items())
        self.writer.add_text('config', config_text, 0)

        print(f"✓ Saved config to {self.config_json}")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float
    ):
        """
        Log metrics for one epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dict (e.g., {'total': 0.5, 'recon': 0.3, 'cls': 0.2})
            val_metrics: Validation metrics dict
            learning_rate: Current learning rate
        """
        # Combine all metrics
        epoch_data = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }

        # Store in history
        self.metrics_history.append(epoch_data)

        # Log to TensorBoard
        self.writer.add_scalar('learning_rate', learning_rate, epoch)

        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

        # Save metrics to files after each epoch
        self._save_metrics()

    def _save_metrics(self):
        """Save metrics history to JSON and CSV files."""
        # Save JSON
        with open(self.metrics_json, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save CSV
        if self.metrics_history:
            keys = self.metrics_history[0].keys()
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.metrics_history)

    def save_plots(self):
        """Generate and save training visualization plots."""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Extract data
            epochs = [m['epoch'] for m in self.metrics_history]

            # Plot 1: Loss curves (train vs val)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Total loss
            axes[0].plot(epochs, [m['train_total'] for m in self.metrics_history],
                        label='Train', marker='o', markersize=3)
            axes[0].plot(epochs, [m['val_total'] for m in self.metrics_history],
                        label='Val', marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title('Total Loss (Train vs Val)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Learning rate
            axes[1].plot(epochs, [m['learning_rate'] for m in self.metrics_history],
                        color='green', marker='o', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'loss_and_lr.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Plot 2: Per-task losses
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Reconstruction loss
            axes[0].plot(epochs, [m['train_recon'] for m in self.metrics_history],
                        label='Train', marker='o', markersize=3)
            axes[0].plot(epochs, [m['val_recon'] for m in self.metrics_history],
                        label='Val', marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Reconstruction Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Classification loss
            axes[1].plot(epochs, [m['train_cls'] for m in self.metrics_history],
                        label='Train', marker='o', markersize=3)
            axes[1].plot(epochs, [m['val_cls'] for m in self.metrics_history],
                        label='Val', marker='s', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Classification Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'task_losses.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved training plots to {self.plots_dir}")

        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")

    def log_model_graph(self, model: torch.nn.Module, input_sample: tuple):
        """
        Log model graph to TensorBoard.

        Args:
            model: PyTorch model
            input_sample: Sample input tuple (mods, lengths)
        """
        try:
            self.writer.add_graph(model, input_sample)
            print("✓ Logged model graph to TensorBoard")
        except Exception as e:
            print(f"⚠️  Could not log model graph: {e}")

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"✓ Training logger closed")

    def get_run_dir(self) -> Path:
        """Get the current run directory path."""
        return self.run_dir
