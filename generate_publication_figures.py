"""
Generate publication-quality figures from trained model results.

Creates comprehensive visualizations for research papers/presentations.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from typing import Dict, List, Optional

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


class PublicationFigureGenerator:
    """Generate publication-quality figures."""

    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load history
        history_path = self.model_dir / 'history.json'
        if history_path.exists():
            try:
                with open(history_path) as f:
                    self.history = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: history.json is corrupted (likely from old training with NaN issues)")
                print(f"   Error: {e}")
                print(f"   Skipping training curve plots. Other figures will still be generated.")
                self.history = None
        else:
            self.history = None
            print(f"Warning: No history.json found in {model_dir}")

        # Load config
        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = None
            print(f"Warning: No config.json found in {model_dir}")

    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        if not self.history:
            print("Skipping training curves - no history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        train_history = self.history['train']
        val_history = self.history['val']
        epochs = range(1, len(train_history) + 1)

        # Total loss
        ax = axes[0, 0]
        train_total = [h.get('total', h.get('loss', 0)) for h in train_history]
        val_total = [h.get('total', h.get('loss', 0)) for h in val_history]
        ax.plot(epochs, train_total, label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_total, label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # G-code loss
        ax = axes[0, 1]
        train_gcode = [h.get('gcode', 0) for h in train_history]
        val_gcode = [h.get('gcode', 0) for h in val_history]
        ax.plot(epochs, train_gcode, label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_gcode, label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('G-code Loss')
        ax.set_title('G-code Prediction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reconstruction loss
        ax = axes[1, 0]
        train_recon = [h.get('recon', 0) for h in train_history]
        val_recon = [h.get('recon', 0) for h in val_history]
        ax.plot(epochs, train_recon, label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_recon, label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Sensor Reconstruction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Contrastive loss
        ax = axes[1, 1]
        train_contrast = [h.get('contrast', 0) for h in train_history]
        val_contrast = [h.get('contrast', 0) for h in val_history]
        ax.plot(epochs, train_contrast, label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_contrast, label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Contrastive Loss')
        ax.set_title('Fingerprint Contrastive Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'training_curves.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training_curves.pdf/png")

    def plot_loss_components(self):
        """Plot stacked area chart of loss components."""
        if not self.history:
            print("Skipping loss components - no history available")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        train_history = self.history['train']
        epochs = range(1, len(train_history) + 1)

        # Extract loss components
        gcode_loss = np.array([h.get('gcode', 0) for h in train_history])
        recon_loss = np.array([h.get('recon', 0) for h in train_history])
        contrast_loss = np.array([h.get('contrast', 0) for h in train_history])
        fp_loss = np.array([h.get('fp', 0) for h in train_history])

        # Stack them
        ax.fill_between(epochs, 0, gcode_loss, label='G-code', alpha=0.7)
        ax.fill_between(epochs, gcode_loss, gcode_loss + recon_loss, label='Reconstruction', alpha=0.7)
        ax.fill_between(epochs, gcode_loss + recon_loss,
                        gcode_loss + recon_loss + contrast_loss, label='Contrastive', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Contribution')
        ax.set_title('Training Loss Components Over Time (Stacked)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_components_stacked.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'loss_components_stacked.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved loss_components_stacked.pdf/png")

    def plot_model_config(self):
        """Create a text summary figure of model configuration."""
        if not self.config:
            print("Skipping model config - no config available")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # Extract key info
        model_cfg = self.config.get('model_config', {})
        train_args = self.config.get('training_args', {})
        metadata = self.config.get('metadata', {})

        # Create text
        config_text = "MODEL CONFIGURATION\n" + "="*50 + "\n\n"

        config_text += "Architecture:\n"
        config_text += f"  • Hidden dimension (d_model): {model_cfg.get('d_model', 'N/A')}\n"
        config_text += f"  • LSTM layers: {model_cfg.get('lstm_layers', 'N/A')}\n"
        config_text += f"  • Attention heads: {model_cfg.get('n_heads', 'N/A')}\n"
        config_text += f"  • Fingerprint dimension: {model_cfg.get('fp_dim', 'N/A')}\n"
        config_text += f"  • Vocabulary size: {model_cfg.get('gcode_vocab', 'N/A')}\n\n"

        config_text += "Training Hyperparameters:\n"
        config_text += f"  • Learning rate: {train_args.get('lr', 'N/A')}\n"
        config_text += f"  • Batch size: {train_args.get('batch_size', 'N/A')}\n"
        config_text += f"  • Epochs: {train_args.get('epochs', 'N/A')}\n"
        config_text += f"  • Optimizer: {train_args.get('optimizer', 'adamw')}\n"
        config_text += f"  • Weight decay: {train_args.get('weight_decay', 0.01)}\n"
        config_text += f"  • Scheduler: {train_args.get('scheduler', 'None')}\n"
        config_text += f"  • Grad clip: {train_args.get('grad_clip', 1.0)}\n\n"

        config_text += "Dataset:\n"
        config_text += f"  • Continuous features: {metadata.get('n_continuous_features', 'N/A')}\n"
        config_text += f"  • Categorical features: {metadata.get('n_categorical_features', 'N/A')}\n"
        config_text += f"  • Train samples: {metadata.get('n_train', 'N/A')}\n"
        config_text += f"  • Validation samples: {metadata.get('n_val', 'N/A')}\n"
        config_text += f"  • Test samples: {metadata.get('n_test', 'N/A')}\n"
        config_text += f"  • Window size: {metadata.get('window_size', 'N/A')}\n"
        config_text += f"  • Stride: {metadata.get('stride', 'N/A')}\n"

        ax.text(0.1, 0.95, config_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_configuration.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'model_configuration.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved model_configuration.pdf/png")

    def create_architecture_diagram(self):
        """Create a simplified architecture diagram."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Title
        ax.text(5, 11.5, 'MM-DTAE-LSTM Architecture',
                ha='center', fontsize=16, fontweight='bold')

        # Input layer
        ax.add_patch(plt.Rectangle((1, 10), 3, 0.6, facecolor='lightblue', edgecolor='black', linewidth=2))
        ax.text(2.5, 10.3, 'Sensor Inputs', ha='center', va='center', fontweight='bold')
        ax.text(0.5, 9.7, 'Continuous (233)', fontsize=9)
        ax.text(4.5, 9.7, 'Categorical (4)', fontsize=9)

        # Modality encoders
        ax.arrow(2.5, 10, 0, -0.8, head_width=0.2, head_length=0.1, fc='black')
        ax.add_patch(plt.Rectangle((1, 8.5), 3, 0.6, facecolor='lightgreen', edgecolor='black', linewidth=2))
        ax.text(2.5, 8.8, 'Per-Modality Encoders', ha='center', va='center', fontweight='bold')

        # Cross-modal fusion
        ax.arrow(2.5, 8.5, 0, -0.8, head_width=0.2, head_length=0.1, fc='black')
        ax.add_patch(plt.Rectangle((1, 7), 3, 0.6, facecolor='lightyellow', edgecolor='black', linewidth=2))
        ax.text(2.5, 7.3, 'Cross-Modal Fusion', ha='center', va='center', fontweight='bold')
        ax.text(2.5, 6.8, '+ Context Embeddings', ha='center', fontsize=9)

        # DTAE
        ax.arrow(2.5, 7, 0, -0.8, head_width=0.2, head_length=0.1, fc='black')
        ax.add_patch(plt.Rectangle((1, 5.5), 3, 0.6, facecolor='lightcoral', edgecolor='black', linewidth=2))
        ax.text(2.5, 5.8, 'Denoising Transformer', ha='center', va='center', fontweight='bold')
        ax.text(2.5, 5.3, 'AutoEncoder (DTAE)', ha='center', fontsize=9)

        # LSTM
        ax.arrow(2.5, 5.5, 0, -0.8, head_width=0.2, head_length=0.1, fc='black')
        ax.add_patch(plt.Rectangle((1, 4), 3, 0.6, facecolor='plum', edgecolor='black', linewidth=2))
        ax.text(2.5, 4.3, 'Bidirectional LSTM', ha='center', va='center', fontweight='bold')

        # Output heads
        ax.arrow(2.5, 4, 0, -0.5, head_width=0.2, head_length=0.1, fc='black')

        # Multi-task outputs
        heads = [
            ('G-code\nPrediction', 0.5, 'lightblue'),
            ('Sensor\nReconstruction', 2.5, 'lightgreen'),
            ('Fingerprint\nEmbedding', 4.5, 'lightyellow'),
            ('Anomaly\nDetection', 6.5, 'lightcoral'),
            ('Classification', 8.5, 'plum')
        ]

        for name, x, color in heads:
            ax.add_patch(plt.Rectangle((x, 1.5), 1.8, 1, facecolor=color, edgecolor='black', linewidth=2))
            ax.text(x + 0.9, 2, name, ha='center', va='center', fontsize=9, fontweight='bold')
            ax.arrow(2.5, 3.5, x + 0.9 - 2.5, -1.8, head_width=0.15, head_length=0.1,
                    fc='gray', alpha=0.5, linestyle='--')

        # Add legend
        legend_y = 0.5
        ax.text(5, legend_y + 0.5, 'Key Features:', fontweight='bold', fontsize=10)
        ax.text(5, legend_y, '• Multi-modal sensor fusion', fontsize=9)
        ax.text(5, legend_y - 0.3, '• Modality dropout for robustness', fontsize=9)
        ax.text(5, legend_y - 0.6, '• Multi-task learning', fontsize=9)
        ax.text(5, legend_y - 0.9, '• Adaptive loss weighting', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_diagram.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'architecture_diagram.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved architecture_diagram.pdf/png")

    def generate_all(self):
        """Generate all publication figures."""
        print(f"\nGenerating publication figures...")
        print(f"Model directory: {self.model_dir}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)

        self.plot_training_curves()
        self.plot_loss_components()
        self.plot_model_config()
        self.create_architecture_diagram()

        print("="*60)
        print(f"✅ All figures saved to: {self.output_dir}")
        print("\nGenerated figures:")
        for fig_file in sorted(self.output_dir.glob("*.pdf")):
            print(f"  • {fig_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument('--model-dir', type=Path, required=True,
                        help="Directory containing trained model (with history.json and config.json)")
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/figures/publication'),
                        help="Output directory for figures")

    args = parser.parse_args()

    generator = PublicationFigureGenerator(args.model_dir, args.output_dir)
    generator.generate_all()


if __name__ == '__main__':
    main()
