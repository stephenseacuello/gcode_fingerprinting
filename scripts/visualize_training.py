#!/usr/bin/env python3
"""
Comprehensive Visualization Script for G-code Fingerprinting Model
Integrates all visualization modules for training monitoring, model interpretation,
interactive exploration, and production dashboard.
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import visualization modules
from miracle.visualization.training_monitor import TrainingMonitor
from miracle.visualization.model_interpreter import ModelInterpreter
from miracle.visualization.interactive_explorer import InteractiveExplorer
from miracle.visualization.production_dashboard import ProductionDashboard

# Import existing modules
from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device

# Import for integration with training
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Some features will be limited.")


class ComprehensiveVisualizer:
    """Main class that integrates all visualization modules."""

    def __init__(self, model_path: Optional[Path] = None,
                 data_path: Optional[Path] = None,
                 vocab_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize the comprehensive visualizer.

        Args:
            model_path: Path to saved model checkpoint
            data_path: Path to data directory
            vocab_path: Path to vocabulary file
            output_dir: Output directory for visualizations
        """
        self.model_path = model_path
        self.data_path = Path(data_path) if data_path else Path('outputs/processed_hybrid')
        self.vocab_path = Path(vocab_path) if vocab_path else Path('data/vocabulary_1digit_hybrid.json')
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Load vocabulary
        self.vocab = self._load_vocabulary()
        self.decomposer = TokenDecomposer(self.vocab_path)

        # Initialize visualization modules
        self.training_monitor = TrainingMonitor(save_dir=self.output_dir / 'training')
        self.model_interpreter = None  # Will be initialized when model is loaded
        self.interactive_explorer = InteractiveExplorer(save_dir=self.output_dir / 'interactive')
        self.production_dashboard = ProductionDashboard(save_dir=self.output_dir / 'production')

        # Load model if path provided
        self.model = None
        self.backbone = None
        if model_path:
            self.load_model(model_path)

    def _load_vocabulary(self) -> dict:
        """Load vocabulary from file."""
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r') as f:
                vocab_data = json.load(f)
                if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                    return vocab_data['vocab']
                return vocab_data
        else:
            print(f"⚠️  Vocabulary not found at {self.vocab_path}")
            return {}

    def load_model(self, checkpoint_path: Path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        print(f"Loading model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config
            config = {
                'hidden_dim': 256,
                'num_layers': 5,
                'num_heads': 8
            }

        # Infer sensor dimensions from checkpoint if available
        sensor_dims = None
        if 'backbone_state_dict' in checkpoint:
            backbone_state = checkpoint['backbone_state_dict']

            # Try to infer dimensions from encoder weights
            if 'encoders.0.proj.0.weight' in backbone_state:
                n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1]
                print(f"Detected continuous dimension: {n_continuous}")
            else:
                n_continuous = 219  # Fallback based on error message
                print(f"Warning: Could not detect continuous dimension, using {n_continuous}")

            if 'encoders.1.proj.0.weight' in backbone_state:
                n_categorical = backbone_state['encoders.1.proj.0.weight'].shape[1]
                print(f"Detected categorical dimension: {n_categorical}")
            else:
                n_categorical = 4  # Fallback based on error message
                print(f"Warning: Could not detect categorical dimension, using {n_categorical}")

            sensor_dims = [n_continuous, n_categorical]
        else:
            # If no backbone state dict, try loading from dataset
            try:
                val_data_path = self.data_path / 'val_sequences.npz'
                if val_data_path.exists():
                    val_data = np.load(val_data_path, allow_pickle=True)
                    n_continuous = val_data['continuous'].shape[-1]
                    n_categorical = val_data['categorical'].shape[-1]
                    sensor_dims = [n_continuous, n_categorical]
                    print(f"Inferred dimensions from dataset: continuous={n_continuous}, categorical={n_categorical}")
                else:
                    print("Warning: Could not find validation data to infer dimensions")
            except Exception as e:
                print(f"Warning: Could not load dataset to infer dimensions: {e}")

        # Final fallback
        if sensor_dims is None:
            sensor_dims = [219, 4]  # Based on the error message
            print(f"Warning: Using fallback sensor dimensions: {sensor_dims}")

        # Create backbone
        backbone_config = ModelConfig(
            sensor_dims=sensor_dims,
            d_model=config['hidden_dim'],
            lstm_layers=config['num_layers'],
            gcode_vocab=len(self.vocab),
            n_heads=config['num_heads'],
            dropout=0.1
        )
        self.backbone = MM_DTAE_LSTM(backbone_config).to(self.device)

        # Create multi-head model
        self.model = MultiHeadGCodeLM(
            d_model=config['hidden_dim'],
            n_commands=self.decomposer.n_commands,
            n_param_types=self.decomposer.n_param_types,
            n_param_values=self.decomposer.n_param_values,
            nhead=config['num_heads'],
            num_layers=config['num_layers'],
            vocab_size=len(self.vocab)
        ).to(self.device)

        # Load state dicts
        if 'backbone_state_dict' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        if 'multihead_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['multihead_state_dict'])

        print(f"✅ Model loaded successfully with sensor_dims={sensor_dims}")

        # Initialize model interpreter with loaded model
        self.model_interpreter = ModelInterpreter(
            self.backbone,
            device=self.device,
            save_dir=self.output_dir / 'interpretation'
        )

    def visualize_training_run(self, train_loader, val_loader, epochs: int = 10):
        """
        Visualize a training run with real-time monitoring.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs to simulate
        """
        print("\n" + "="*50)
        print("TRAINING VISUALIZATION")
        print("="*50)

        # Create training dashboard
        fig, axes = self.training_monitor.create_dashboard()

        # Training loop with visualization
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Simulate training metrics (replace with actual training)
            train_metrics = {
                'loss': 2.0 * np.exp(-epoch/5) + np.random.normal(0, 0.1),
                'command_acc': min(0.95, 0.5 + epoch/20),
                'param_type_acc': min(0.90, 0.4 + epoch/25),
                'param_value_acc': min(0.85, 0.3 + epoch/30),
                'operation_acc': min(0.92, 0.45 + epoch/22),
                'overall_acc': min(0.88, 0.35 + epoch/27)
            }

            val_metrics = {k: v * 0.95 for k, v in train_metrics.items()}

            loss_components = {
                'command': 0.5 * np.exp(-epoch/8),
                'param_type': 0.3 * np.exp(-epoch/10),
                'param_value': 0.4 * np.exp(-epoch/7),
                'operation': 0.2 * np.exp(-epoch/12),
                'grammar': 0.1 * np.exp(-epoch/15)
            }

            lr = 1e-3 * (0.95 ** (epoch // 5))

            # Update monitor
            self.training_monitor.update_metrics(epoch, train_metrics, val_metrics,
                                                loss_components, lr)

            # Update dashboard
            self.training_monitor.update_dashboard(
                fig, axes,
                patience=20,
                current_patience=max(0, epoch - 5)
            )

            # Save periodically
            if epoch % 5 == 0:
                self.training_monitor.save_dashboard(fig, epoch)

            # Update production dashboard
            self.production_dashboard.update_performance_metrics(
                accuracy=val_metrics['overall_acc'],
                f1_score=val_metrics['overall_acc'] * 0.95,
                precision=val_metrics['overall_acc'] * 0.97,
                recall=val_metrics['overall_acc'] * 0.93
            )

            # Log to W&B if available
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'learning_rate': lr,
                    'epoch': epoch
                })

        # Save final metrics
        self.training_monitor.export_metrics_json()
        print(self.training_monitor.generate_training_report())

    def interpret_model(self, dataloader):
        """
        Run model interpretation visualizations.

        Args:
            dataloader: DataLoader for analysis
        """
        if self.model_interpreter is None:
            print("⚠️  Model not loaded. Skipping interpretation.")
            return

        print("\n" + "="*50)
        print("MODEL INTERPRETATION")
        print("="*50)

        # Feature importance
        print("Computing feature importance...")
        importance = self.model_interpreter.compute_feature_importance(dataloader, n_samples=50)
        fig = self.model_interpreter.plot_feature_importance(importance)
        fig.savefig(self.output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')

        # Saliency maps
        print("Generating saliency maps...")
        fig = self.model_interpreter.plot_saliency_maps(dataloader, n_samples=6)
        fig.savefig(self.output_dir / 'saliency_maps.png', dpi=150, bbox_inches='tight')

        # Confidence distributions
        print("Analyzing prediction confidence...")
        confidences = self.model_interpreter.analyze_prediction_confidence(dataloader, n_samples=100)
        fig = self.model_interpreter.plot_confidence_distributions(confidences)
        fig.savefig(self.output_dir / 'confidence_distributions.png', dpi=150, bbox_inches='tight')

        # Token error analysis
        print("Analyzing token errors...")
        error_df = self.model_interpreter.analyze_errors_by_token(dataloader, self.vocab, n_samples=100)
        fig = self.model_interpreter.plot_token_error_analysis(error_df)
        fig.savefig(self.output_dir / 'token_error_analysis.png', dpi=150, bbox_inches='tight')

        # Save error analysis
        error_df.to_csv(self.output_dir / 'token_error_analysis.csv', index=False)
        print(f"✅ Model interpretation complete. Results saved to {self.output_dir}")

    def create_interactive_visualizations(self, dataloader):
        """
        Create interactive visualizations.

        Args:
            dataloader: DataLoader for analysis
        """
        print("\n" + "="*50)
        print("INTERACTIVE VISUALIZATIONS")
        print("="*50)

        # Extract embeddings for visualization
        print("Extracting embeddings...")
        embeddings = []
        labels = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 20:  # Limit samples
                    break

                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device).float()
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                if self.backbone:
                    mods = [continuous, categorical]
                    output = self.backbone(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1])
                    memory = output['memory']

                    # Average pool over time
                    pooled = memory.mean(dim=1)
                    embeddings.append(pooled.cpu().numpy())

                    # Use operation type as label if available
                    if 'operation_type' in batch:
                        labels.extend(batch['operation_type'].numpy())
                    else:
                        labels.extend([0] * pooled.shape[0])

        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.array(labels)

            # Create 3D embeddings
            print("Creating 3D embedding visualization...")
            fig = self.interactive_explorer.create_3d_embeddings(
                embeddings, labels, method='tsne'
            )
            fig.write_html(self.output_dir / 'embeddings_3d.html')

            # Create PCA version
            fig_pca = self.interactive_explorer.create_3d_embeddings(
                embeddings, labels, method='pca'
            )
            fig_pca.write_html(self.output_dir / 'embeddings_3d_pca.html')

        # Create metrics dashboard
        print("Creating metrics dashboard...")
        metrics_history = {
            'Train Loss': list(2.0 * np.exp(-np.arange(50)/10) + np.random.normal(0, 0.1, 50)),
            'Val Loss': list(2.2 * np.exp(-np.arange(50)/12) + np.random.normal(0, 0.1, 50)),
            'Command Acc': list(np.minimum(0.95, 0.5 + np.arange(50)/100)),
            'Param Type Acc': list(np.minimum(0.90, 0.45 + np.arange(50)/110))
        }

        fig = self.interactive_explorer.create_metric_dashboard(metrics_history)
        fig.write_html(self.output_dir / 'metrics_dashboard.html')

        print(f"✅ Interactive visualizations saved to {self.output_dir}")

    def monitor_production(self):
        """Create production monitoring dashboard."""
        print("\n" + "="*50)
        print("PRODUCTION MONITORING")
        print("="*50)

        # Create dashboard
        fig, axes = self.production_dashboard.create_monitoring_dashboard()

        # Simulate production data
        print("Simulating production monitoring...")
        for i in range(100):
            # Performance metrics
            self.production_dashboard.update_performance_metrics(
                accuracy=0.92 + np.random.normal(0, 0.02),
                f1_score=0.89 + np.random.normal(0, 0.02),
                precision=0.91 + np.random.normal(0, 0.02),
                recall=0.88 + np.random.normal(0, 0.02)
            )

            # Latency
            self.production_dashboard.update_latency(
                latency_ms=25 + np.random.exponential(5),
                batch_size=32
            )

            # Throughput
            self.production_dashboard.update_throughput(
                samples_per_second=1000 + np.random.normal(0, 100)
            )

        # Add model comparisons
        self.production_dashboard.compare_models('Baseline', {
            'accuracy': 0.88, 'f1_score': 0.85, 'precision': 0.87, 'recall': 0.84
        })
        self.production_dashboard.compare_models('Enhanced', {
            'accuracy': 0.92, 'f1_score': 0.89, 'precision': 0.91, 'recall': 0.88
        })
        self.production_dashboard.compare_models('Latest', {
            'accuracy': 0.94, 'f1_score': 0.91, 'precision': 0.93, 'recall': 0.90
        })

        # Detect data drift
        current_features = np.random.randn(100, 8)
        reference_features = np.random.randn(100, 8) + 0.1  # Slight shift
        drift_scores = self.production_dashboard.detect_data_drift(
            current_features, reference_features
        )

        # Update all plots
        self.production_dashboard.plot_performance_timeline(axes['performance'])
        self.production_dashboard.plot_latency_distribution(axes['latency'])
        self.production_dashboard.plot_throughput_timeline(axes['throughput'])
        self.production_dashboard.plot_error_rate(axes['error_rate'])
        self.production_dashboard.plot_data_drift(axes['data_drift'], drift_scores)
        self.production_dashboard.plot_model_comparison(axes['model_comparison'])
        self.production_dashboard.plot_alerts(axes['alerts'])
        self.production_dashboard.plot_distribution_comparison(
            axes['distribution'], current_features[:, 0:1], reference_features[:, 0:1]
        )

        # Save dashboard
        fig.savefig(self.output_dir / 'production_dashboard.png', dpi=150, bbox_inches='tight')

        # Generate and save report
        report = self.production_dashboard.generate_performance_report()
        print(report)

        with open(self.output_dir / 'production_report.txt', 'w') as f:
            f.write(report)

        self.production_dashboard.save_monitoring_data()
        print(f"✅ Production monitoring complete. Results saved to {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive visualization for G-code fingerprinting')

    parser.add_argument('--model-path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='outputs/processed_hybrid',
                       help='Path to data directory')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_1digit_hybrid.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'training', 'interpretation', 'interactive', 'production'],
                       help='Visualization mode')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for data loading')

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = ComprehensiveVisualizer(
        model_path=Path(args.model_path) if args.model_path else None,
        data_path=Path(args.data_dir),
        vocab_path=Path(args.vocab_path),
        output_dir=Path(args.output_dir)
    )

    # Load data
    print(f"Loading data from {args.data_dir}")
    val_dataset = GCodeDataset(Path(args.data_dir) / 'val_sequences.npz')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Run visualizations based on mode
    if args.mode in ['all', 'training']:
        train_dataset = GCodeDataset(Path(args.data_dir) / 'train_sequences.npz')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        visualizer.visualize_training_run(train_loader, val_loader, epochs=10)

    if args.mode in ['all', 'interpretation']:
        visualizer.interpret_model(val_loader)

    if args.mode in ['all', 'interactive']:
        visualizer.create_interactive_visualizations(val_loader)

    if args.mode in ['all', 'production']:
        visualizer.monitor_production()

    print(f"\n✅ All visualizations complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()