"""
Enhanced Training Monitor for Real-time Visualization
Provides comprehensive training insights including loss components,
learning rate schedules, gradient flow, and performance metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path
from datetime import datetime
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class TrainingMonitor:
    """Enhanced training monitor with real-time visualization capabilities."""

    def __init__(self,
                 fig_size: Tuple[int, int] = (20, 12),
                 update_interval: int = 100,
                 history_length: int = 1000,
                 save_dir: Optional[Path] = None):
        """
        Initialize the training monitor.

        Args:
            fig_size: Figure size for the dashboard
            update_interval: Update interval in milliseconds for animation
            history_length: Number of historical points to keep
            save_dir: Directory to save figures
        """
        self.fig_size = fig_size
        self.update_interval = update_interval
        self.history_length = history_length
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/training_monitor")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data storage
        self.reset_history()

        # Color scheme
        self.colors = {
            'train': '#1f77b4',
            'val': '#ff7f0e',
            'loss': '#2ca02c',
            'acc': '#d62728',
            'lr': '#9467bd',
            'gradient': '#8c564b',
            'command': '#e377c2',
            'param_type': '#7f7f7f',
            'param_value': '#bcbd22',
            'operation': '#17becf',
            'grammar': '#ff9800',
            'reconstruction': '#4caf50'
        }

    def reset_history(self):
        """Reset all history buffers."""
        self.epochs = deque(maxlen=self.history_length)
        self.train_losses = deque(maxlen=self.history_length)
        self.val_losses = deque(maxlen=self.history_length)
        self.train_metrics = {
            'command_acc': deque(maxlen=self.history_length),
            'param_type_acc': deque(maxlen=self.history_length),
            'param_value_acc': deque(maxlen=self.history_length),
            'operation_acc': deque(maxlen=self.history_length),
            'overall_acc': deque(maxlen=self.history_length)
        }
        self.val_metrics = {
            'command_acc': deque(maxlen=self.history_length),
            'param_type_acc': deque(maxlen=self.history_length),
            'param_value_acc': deque(maxlen=self.history_length),
            'operation_acc': deque(maxlen=self.history_length),
            'overall_acc': deque(maxlen=self.history_length)
        }
        self.loss_components = {
            'command': deque(maxlen=self.history_length),
            'param_type': deque(maxlen=self.history_length),
            'param_value': deque(maxlen=self.history_length),
            'operation': deque(maxlen=self.history_length),
            'grammar': deque(maxlen=self.history_length)
        }
        self.learning_rates = deque(maxlen=self.history_length)
        self.gradient_norms = {}  # Layer-wise gradient norms
        self.batch_times = deque(maxlen=100)  # Recent batch processing times
        self.gpu_memory = deque(maxlen=100)  # GPU memory usage

    def create_dashboard(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, Dict]:
        """
        Create the training dashboard with multiple panels.

        Returns:
            Figure and axes dictionary
        """
        if figsize is None:
            figsize = self.fig_size

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        axes = {
            'loss_curves': fig.add_subplot(gs[0, :2]),
            'accuracy_curves': fig.add_subplot(gs[0, 2:]),
            'loss_components': fig.add_subplot(gs[1, :2]),
            'lr_schedule': fig.add_subplot(gs[1, 2:]),
            'gradient_flow': fig.add_subplot(gs[2, :2]),
            'confusion_preview': fig.add_subplot(gs[2, 2:]),
            'training_speed': fig.add_subplot(gs[3, 0]),
            'memory_usage': fig.add_subplot(gs[3, 1]),
            'per_head_acc': fig.add_subplot(gs[3, 2]),
            'early_stopping': fig.add_subplot(gs[3, 3])
        }

        # Set titles
        axes['loss_curves'].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        axes['accuracy_curves'].set_title('Accuracy Metrics', fontsize=12, fontweight='bold')
        axes['loss_components'].set_title('Loss Component Breakdown', fontsize=12, fontweight='bold')
        axes['lr_schedule'].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes['gradient_flow'].set_title('Gradient Flow (Layer-wise)', fontsize=12, fontweight='bold')
        axes['confusion_preview'].set_title('Token Confusion Preview', fontsize=12, fontweight='bold')
        axes['training_speed'].set_title('Training Speed', fontsize=12, fontweight='bold')
        axes['memory_usage'].set_title('GPU Memory', fontsize=12, fontweight='bold')
        axes['per_head_acc'].set_title('Per-Head Accuracy', fontsize=12, fontweight='bold')
        axes['early_stopping'].set_title('Early Stopping Status', fontsize=12, fontweight='bold')

        return fig, axes

    def update_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                      loss_components: Optional[Dict] = None, lr: Optional[float] = None):
        """Update stored metrics with new data."""
        self.epochs.append(epoch)

        # Update losses
        if 'loss' in train_metrics:
            self.train_losses.append(train_metrics['loss'])
        if 'loss' in val_metrics:
            self.val_losses.append(val_metrics['loss'])

        # Update accuracies
        for key in self.train_metrics:
            if key in train_metrics:
                self.train_metrics[key].append(train_metrics[key])
            if key in val_metrics:
                self.val_metrics[key].append(val_metrics[key])

        # Update loss components
        if loss_components:
            for key in self.loss_components:
                if key in loss_components:
                    self.loss_components[key].append(loss_components[key])

        # Update learning rate
        if lr is not None:
            self.learning_rates.append(lr)

    def plot_loss_curves(self, ax: plt.Axes):
        """Plot training and validation loss curves."""
        ax.clear()
        epochs = list(self.epochs)

        if epochs and self.train_losses:
            ax.plot(epochs, list(self.train_losses),
                   label='Train Loss', color=self.colors['train'], linewidth=2)
        if epochs and self.val_losses:
            ax.plot(epochs, list(self.val_losses),
                   label='Val Loss', color=self.colors['val'], linewidth=2)

            # Mark best validation loss
            if self.val_losses:
                best_epoch = epochs[np.argmin(list(self.val_losses))]
                best_loss = min(self.val_losses)
                ax.scatter([best_epoch], [best_loss], color='red', s=100,
                          marker='*', zorder=5, label=f'Best: {best_loss:.4f}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def plot_accuracy_curves(self, ax: plt.Axes):
        """Plot accuracy metrics over time."""
        ax.clear()
        epochs = list(self.epochs)

        if not epochs:
            return

        # Plot validation accuracies
        metrics_to_plot = ['command_acc', 'param_type_acc', 'overall_acc']
        for i, metric in enumerate(metrics_to_plot):
            if self.val_metrics[metric]:
                ax.plot(epochs, list(self.val_metrics[metric]),
                       label=metric.replace('_', ' ').title(),
                       linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def plot_loss_components(self, ax: plt.Axes):
        """Plot stacked area chart of loss components."""
        ax.clear()
        epochs = list(self.epochs)

        if not epochs:
            return

        # Prepare data for stacked area chart
        components = []
        labels = []
        colors = []

        for comp_name, values in self.loss_components.items():
            if values and len(values) == len(epochs):
                components.append(list(values))
                labels.append(comp_name.title())
                colors.append(self.colors.get(comp_name, '#333333'))

        if components:
            ax.stackplot(epochs, *components, labels=labels, colors=colors, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Component Value')
            ax.legend(loc='upper right', ncol=2)
            ax.grid(True, alpha=0.3)

    def plot_lr_schedule(self, ax: plt.Axes):
        """Plot learning rate schedule with annotations."""
        ax.clear()
        epochs = list(self.epochs)
        lrs = list(self.learning_rates)

        if not epochs or not lrs:
            return

        ax.plot(epochs, lrs, color=self.colors['lr'], linewidth=2)
        ax.fill_between(epochs, lrs, alpha=0.3, color=self.colors['lr'])

        # Annotate warmup, decay, etc.
        if len(lrs) > 1:
            # Detect warmup
            if lrs[0] < lrs[min(5, len(lrs)-1)]:
                ax.axvspan(epochs[0], epochs[min(5, len(epochs)-1)],
                          alpha=0.2, color='orange', label='Warmup')

            # Detect plateau/drops
            for i in range(1, len(lrs)):
                if lrs[i] < lrs[i-1] * 0.9:  # Significant drop
                    ax.axvline(x=epochs[i], color='red', linestyle='--',
                             alpha=0.5, label='LR Drop' if i == 1 else '')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

    def plot_gradient_flow(self, ax: plt.Axes, model: Optional[torch.nn.Module] = None):
        """Plot gradient flow through layers."""
        ax.clear()

        if not self.gradient_norms:
            ax.text(0.5, 0.5, 'No gradient data available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Get latest gradient norms
        layers = []
        mean_grads = []
        max_grads = []

        for name, values in self.gradient_norms.items():
            if values:
                layers.append(name.split('.')[-1][:10])  # Truncate long names
                mean_grads.append(np.mean(values[-10:]))  # Average of last 10
                max_grads.append(np.max(values[-10:]))

        if layers:
            x = np.arange(len(layers))
            width = 0.35

            ax.bar(x - width/2, mean_grads, width, label='Mean',
                  color=self.colors['gradient'], alpha=0.7)
            ax.bar(x + width/2, max_grads, width, label='Max',
                  color=self.colors['gradient'], alpha=0.4)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Gradient Norm')
            ax.set_xticks(x)
            ax.set_xticklabels(layers, rotation=45, ha='right')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    def plot_training_speed(self, ax: plt.Axes):
        """Plot training speed metrics."""
        ax.clear()

        if self.batch_times:
            times = list(self.batch_times)
            samples_per_sec = 32 / np.array(times)  # Assuming batch_size=32

            ax.plot(samples_per_sec, color=self.colors['train'], linewidth=2)
            ax.fill_between(range(len(samples_per_sec)), samples_per_sec,
                          alpha=0.3, color=self.colors['train'])

            mean_speed = np.mean(samples_per_sec)
            ax.axhline(y=mean_speed, color='red', linestyle='--',
                      label=f'Avg: {mean_speed:.1f} samples/s')

            ax.set_xlabel('Batch')
            ax.set_ylabel('Samples/sec')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No speed data', ha='center', va='center',
                   transform=ax.transAxes)

        ax.grid(True, alpha=0.3)

    def plot_memory_usage(self, ax: plt.Axes):
        """Plot GPU memory usage."""
        ax.clear()

        if torch.cuda.is_available():
            # Get current GPU memory
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3

            self.gpu_memory.append((allocated, reserved))

            if self.gpu_memory:
                allocated_mem = [m[0] for m in self.gpu_memory]
                reserved_mem = [m[1] for m in self.gpu_memory]

                ax.plot(allocated_mem, label='Allocated', color='green', linewidth=2)
                ax.plot(reserved_mem, label='Reserved', color='orange', linewidth=2)
                ax.fill_between(range(len(allocated_mem)), allocated_mem,
                              alpha=0.3, color='green')

                ax.set_xlabel('Step')
                ax.set_ylabel('Memory (GB)')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'GPU not available', ha='center', va='center',
                   transform=ax.transAxes)

        ax.grid(True, alpha=0.3)

    def plot_per_head_accuracy(self, ax: plt.Axes):
        """Plot per-head accuracy breakdown."""
        ax.clear()

        if not self.val_metrics['command_acc']:
            ax.text(0.5, 0.5, 'No accuracy data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Get latest accuracies
        heads = ['Command', 'Param Type', 'Param Value', 'Operation']
        latest_acc = [
            self.val_metrics['command_acc'][-1] if self.val_metrics['command_acc'] else 0,
            self.val_metrics['param_type_acc'][-1] if self.val_metrics['param_type_acc'] else 0,
            self.val_metrics['param_value_acc'][-1] if self.val_metrics['param_value_acc'] else 0,
            self.val_metrics['operation_acc'][-1] if self.val_metrics['operation_acc'] else 0
        ]

        colors_list = [self.colors['command'], self.colors['param_type'],
                      self.colors['param_value'], self.colors['operation']]

        bars = ax.bar(heads, latest_acc, color=colors_list, alpha=0.7)

        # Add value labels on bars
        for bar, val in zip(bars, latest_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.2%}', ha='center', va='bottom')

        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

    def plot_early_stopping_status(self, ax: plt.Axes, patience: int = 10,
                                   current_patience: int = 0):
        """Visualize early stopping status."""
        ax.clear()

        # Create patience indicator
        remaining = patience - current_patience
        used = current_patience

        # Pie chart showing patience usage
        sizes = [remaining, used]
        colors_list = ['green' if remaining > 3 else 'orange', 'red']
        labels = [f'Remaining: {remaining}', f'Used: {used}']

        if remaining > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                               autopct='%1.0f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'EARLY STOPPING\nTRIGGERED',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color='red')

        ax.set_title(f'Patience: {current_patience}/{patience}')

    def update_dashboard(self, fig: plt.Figure, axes: Dict, **kwargs):
        """Update all dashboard panels."""
        self.plot_loss_curves(axes['loss_curves'])
        self.plot_accuracy_curves(axes['accuracy_curves'])
        self.plot_loss_components(axes['loss_components'])
        self.plot_lr_schedule(axes['lr_schedule'])
        self.plot_gradient_flow(axes['gradient_flow'], model=kwargs.get('model'))
        self.plot_training_speed(axes['training_speed'])
        self.plot_memory_usage(axes['memory_usage'])
        self.plot_per_head_accuracy(axes['per_head_acc'])
        self.plot_early_stopping_status(axes['early_stopping'],
                                       patience=kwargs.get('patience', 10),
                                       current_patience=kwargs.get('current_patience', 0))

        # Update confusion preview if provided
        if 'confusion_matrix' in kwargs and kwargs['confusion_matrix'] is not None:
            axes['confusion_preview'].clear()
            im = axes['confusion_preview'].imshow(kwargs['confusion_matrix'][:10, :10],
                                                  cmap='Blues', aspect='auto')
            axes['confusion_preview'].set_xlabel('Predicted')
            axes['confusion_preview'].set_ylabel('True')
            axes['confusion_preview'].set_title('Token Confusion (Top 10x10)')

        plt.tight_layout()

    def save_dashboard(self, fig: plt.Figure, epoch: int):
        """Save the dashboard to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.save_dir / f'training_dashboard_epoch_{epoch}_{timestamp}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {filename}")

    def create_animated_dashboard(self):
        """Create an animated dashboard that updates in real-time."""
        fig, axes = self.create_dashboard()

        def animate(frame):
            # In real use, this would pull latest data from training
            self.update_dashboard(fig, axes)
            return axes.values()

        anim = FuncAnimation(fig, animate, interval=self.update_interval,
                           blit=False, cache_frame_data=False)
        return fig, anim

    def export_metrics_json(self, filepath: Optional[Path] = None):
        """Export all metrics to JSON for later analysis."""
        if filepath is None:
            filepath = self.save_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        metrics = {
            'epochs': list(self.epochs),
            'train_losses': list(self.train_losses),
            'val_losses': list(self.val_losses),
            'train_metrics': {k: list(v) for k, v in self.train_metrics.items()},
            'val_metrics': {k: list(v) for k, v in self.val_metrics.items()},
            'loss_components': {k: list(v) for k, v in self.loss_components.items()},
            'learning_rates': list(self.learning_rates)
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics exported to {filepath}")

    def generate_training_report(self) -> str:
        """Generate a text summary of training progress."""
        if not self.epochs:
            return "No training data available."

        report = []
        report.append("="*50)
        report.append("TRAINING PROGRESS REPORT")
        report.append("="*50)

        # Best metrics
        if self.val_losses:
            best_loss_idx = np.argmin(list(self.val_losses))
            report.append(f"Best Val Loss: {self.val_losses[best_loss_idx]:.4f} @ Epoch {self.epochs[best_loss_idx]}")

        if self.val_metrics['overall_acc']:
            best_acc_idx = np.argmax(list(self.val_metrics['overall_acc']))
            report.append(f"Best Overall Acc: {self.val_metrics['overall_acc'][best_acc_idx]:.2%} @ Epoch {self.epochs[best_acc_idx]}")

        # Current metrics
        report.append(f"\nCurrent Epoch: {self.epochs[-1]}")
        if self.train_losses:
            report.append(f"Current Train Loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            report.append(f"Current Val Loss: {self.val_losses[-1]:.4f}")

        # Per-head accuracies
        report.append("\nCurrent Per-Head Accuracies:")
        for metric in ['command_acc', 'param_type_acc', 'param_value_acc', 'operation_acc']:
            if self.val_metrics[metric]:
                report.append(f"  {metric.replace('_', ' ').title()}: {self.val_metrics[metric][-1]:.2%}")

        # Training speed
        if self.batch_times:
            avg_time = np.mean(list(self.batch_times))
            report.append(f"\nAvg Batch Time: {avg_time:.3f}s")
            report.append(f"Throughput: {32/avg_time:.1f} samples/sec")

        return "\n".join(report)


def example_usage():
    """Example of how to use the TrainingMonitor."""
    monitor = TrainingMonitor()

    # Create dashboard
    fig, axes = monitor.create_dashboard()

    # Simulate training loop
    for epoch in range(50):
        # Simulate metrics
        train_metrics = {
            'loss': 2.0 * np.exp(-epoch/10) + np.random.normal(0, 0.1),
            'command_acc': min(0.95, 0.5 + epoch/100 + np.random.normal(0, 0.05)),
            'param_type_acc': min(0.90, 0.4 + epoch/80 + np.random.normal(0, 0.05)),
            'param_value_acc': min(0.85, 0.3 + epoch/70 + np.random.normal(0, 0.05)),
            'operation_acc': min(0.92, 0.45 + epoch/90 + np.random.normal(0, 0.05)),
            'overall_acc': min(0.88, 0.35 + epoch/75 + np.random.normal(0, 0.05))
        }

        val_metrics = {k: v * 0.95 for k, v in train_metrics.items()}

        loss_components = {
            'command': 0.5 * np.exp(-epoch/15),
            'param_type': 0.3 * np.exp(-epoch/12),
            'param_value': 0.4 * np.exp(-epoch/10),
            'operation': 0.2 * np.exp(-epoch/20),
            'grammar': 0.1 * np.exp(-epoch/25)
        }

        lr = 1e-3 * (0.95 ** (epoch // 10))

        # Update monitor
        monitor.update_metrics(epoch, train_metrics, val_metrics, loss_components, lr)

        # Update dashboard every 5 epochs
        if epoch % 5 == 0:
            monitor.update_dashboard(fig, axes, patience=20, current_patience=min(epoch//3, 19))
            plt.pause(0.01)

    # Save final dashboard
    monitor.save_dashboard(fig, epoch)

    # Export metrics
    monitor.export_metrics_json()

    # Print report
    print(monitor.generate_training_report())

    plt.show()


if __name__ == "__main__":
    example_usage()