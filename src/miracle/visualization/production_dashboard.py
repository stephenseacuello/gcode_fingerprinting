"""
Production Dashboard for Model Monitoring
Provides real-time monitoring, data drift detection, performance tracking,
and model comparison tools for production deployments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class ProductionDashboard:
    """Production monitoring and comparison dashboard."""

    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.1,
                 save_dir: Optional[Path] = None):
        """
        Initialize the production dashboard.

        Args:
            window_size: Size of monitoring window
            alert_threshold: Threshold for triggering alerts
            save_dir: Directory to save monitoring data
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/production_dashboard")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring buffers
        self.performance_buffer = deque(maxlen=window_size)
        self.latency_buffer = deque(maxlen=window_size)
        self.throughput_buffer = deque(maxlen=window_size)
        self.error_buffer = deque(maxlen=window_size)

        # Data distribution tracking
        self.reference_distributions = {}
        self.current_distributions = {}

        # Model comparison data
        self.model_performances = {}

        # Alert system
        self.alerts = []

    def create_monitoring_dashboard(self, figsize: Tuple[int, int] = (20, 12)) -> Tuple[plt.Figure, Dict]:
        """
        Create the production monitoring dashboard.

        Returns:
            Figure and axes dictionary
        """
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        # Create grid
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        axes = {
            'performance': fig.add_subplot(gs[0, :2]),
            'latency': fig.add_subplot(gs[0, 2:]),
            'throughput': fig.add_subplot(gs[1, :2]),
            'error_rate': fig.add_subplot(gs[1, 2:]),
            'data_drift': fig.add_subplot(gs[2, :2]),
            'model_comparison': fig.add_subplot(gs[2, 2:]),
            'alerts': fig.add_subplot(gs[3, :2]),
            'distribution': fig.add_subplot(gs[3, 2:])
        }

        # Set titles
        for ax_name, title in [
            ('performance', 'Model Performance'),
            ('latency', 'Inference Latency'),
            ('throughput', 'Throughput (samples/sec)'),
            ('error_rate', 'Error Rate'),
            ('data_drift', 'Data Drift Detection'),
            ('model_comparison', 'Model Comparison'),
            ('alerts', 'System Alerts'),
            ('distribution', 'Feature Distributions')
        ]:
            axes[ax_name].set_title(title, fontsize=12, fontweight='bold')

        return fig, axes

    def update_performance_metrics(self, accuracy: float, f1_score: float,
                                  precision: float, recall: float,
                                  timestamp: Optional[datetime] = None):
        """
        Update performance metrics.

        Args:
            accuracy: Current accuracy
            f1_score: Current F1 score
            precision: Current precision
            recall: Current recall
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }

        self.performance_buffer.append(metrics)

        # Check for performance degradation
        if len(self.performance_buffer) > 100:
            recent_acc = np.mean([m['accuracy'] for m in list(self.performance_buffer)[-100:]])
            baseline_acc = np.mean([m['accuracy'] for m in list(self.performance_buffer)[-500:-100]])

            if baseline_acc - recent_acc > self.alert_threshold:
                self.add_alert('Performance Degradation',
                             f'Accuracy dropped by {baseline_acc - recent_acc:.2%}',
                             'high')

    def update_latency(self, latency_ms: float, batch_size: int = 1):
        """Update latency metrics."""
        self.latency_buffer.append({
            'timestamp': datetime.now(),
            'latency': latency_ms,
            'batch_size': batch_size,
            'per_sample': latency_ms / batch_size
        })

        # Check for latency spikes
        if len(self.latency_buffer) > 10:
            recent_latency = np.mean([l['per_sample'] for l in list(self.latency_buffer)[-10:]])
            avg_latency = np.mean([l['per_sample'] for l in self.latency_buffer])

            if recent_latency > avg_latency * 2:
                self.add_alert('Latency Spike',
                             f'Latency increased to {recent_latency:.1f}ms',
                             'medium')

    def update_throughput(self, samples_per_second: float):
        """Update throughput metrics."""
        self.throughput_buffer.append({
            'timestamp': datetime.now(),
            'throughput': samples_per_second
        })

    def detect_data_drift(self, current_features: np.ndarray,
                         reference_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Detect data drift using statistical tests.

        Args:
            current_features: Current feature distributions
            reference_features: Reference feature distributions

        Returns:
            Dictionary with drift scores
        """
        drift_scores = {}

        if reference_features is not None:
            # Store reference if not exists
            if 'reference' not in self.reference_distributions:
                self.reference_distributions['reference'] = reference_features

            ref = self.reference_distributions['reference']

            # Kolmogorov-Smirnov test for each feature
            for i in range(min(current_features.shape[1], ref.shape[1])):
                ks_statistic, p_value = stats.ks_2samp(current_features[:, i], ref[:, i])
                drift_scores[f'feature_{i}'] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }

            # Jensen-Shannon divergence for overall distribution
            # Create histograms
            current_hist, _ = np.histogram(current_features.flatten(), bins=50, density=True)
            ref_hist, _ = np.histogram(ref.flatten(), bins=50, density=True)

            # Normalize
            current_hist = current_hist / current_hist.sum()
            ref_hist = ref_hist / ref_hist.sum()

            js_divergence = jensenshannon(current_hist, ref_hist)
            drift_scores['overall'] = {
                'js_divergence': js_divergence,
                'drift_detected': js_divergence > self.alert_threshold
            }

            if drift_scores['overall']['drift_detected']:
                self.add_alert('Data Drift Detected',
                             f'JS divergence: {js_divergence:.3f}',
                             'high')

        return drift_scores

    def add_alert(self, alert_type: str, message: str, severity: str = 'low'):
        """
        Add an alert to the system.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity ('low', 'medium', 'high')
        """
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)

        # Keep only recent alerts
        cutoff = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff]

    def compare_models(self, model_name: str, metrics: Dict[str, float]):
        """
        Add model performance for comparison.

        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
        """
        self.model_performances[model_name] = {
            'metrics': metrics,
            'timestamp': datetime.now()
        }

    def plot_performance_timeline(self, ax: plt.Axes):
        """Plot performance metrics over time."""
        ax.clear()

        if not self.performance_buffer:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Extract data
        timestamps = [m['timestamp'] for m in self.performance_buffer]
        accuracy = [m['accuracy'] for m in self.performance_buffer]
        f1_score = [m['f1_score'] for m in self.performance_buffer]

        # Plot
        ax.plot(timestamps, accuracy, label='Accuracy', linewidth=2, color='blue')
        ax.plot(timestamps, f1_score, label='F1 Score', linewidth=2, color='green')

        # Add rolling average
        if len(accuracy) > 20:
            window = 20
            rolling_acc = pd.Series(accuracy).rolling(window).mean()
            ax.plot(timestamps, rolling_acc, '--', alpha=0.5, color='blue',
                   label='Accuracy (MA)')

        ax.set_xlabel('Time')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Rotate x-labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def plot_latency_distribution(self, ax: plt.Axes):
        """Plot latency distribution and percentiles."""
        ax.clear()

        if not self.latency_buffer:
            ax.text(0.5, 0.5, 'No latency data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        latencies = [l['per_sample'] for l in self.latency_buffer]

        # Plot histogram
        ax.hist(latencies, bins=30, alpha=0.7, color='orange', edgecolor='black')

        # Add percentile lines
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        ax.axvline(p50, color='green', linestyle='--', label=f'P50: {p50:.1f}ms')
        ax.axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.1f}ms')
        ax.axvline(p99, color='red', linestyle='--', label=f'P99: {p99:.1f}ms')

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_throughput_timeline(self, ax: plt.Axes):
        """Plot throughput over time."""
        ax.clear()

        if not self.throughput_buffer:
            ax.text(0.5, 0.5, 'No throughput data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        timestamps = [t['timestamp'] for t in self.throughput_buffer]
        throughput = [t['throughput'] for t in self.throughput_buffer]

        ax.plot(timestamps, throughput, linewidth=2, color='green')
        ax.fill_between(timestamps, throughput, alpha=0.3, color='green')

        # Add average line
        avg_throughput = np.mean(throughput)
        ax.axhline(avg_throughput, color='red', linestyle='--',
                  label=f'Avg: {avg_throughput:.1f} samples/s')

        ax.set_xlabel('Time')
        ax.set_ylabel('Samples/second')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def plot_error_rate(self, ax: plt.Axes):
        """Plot error rate over time."""
        ax.clear()

        if not self.performance_buffer:
            ax.text(0.5, 0.5, 'No error data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        timestamps = [m['timestamp'] for m in self.performance_buffer]
        error_rate = [1 - m['accuracy'] for m in self.performance_buffer]

        ax.plot(timestamps, error_rate, linewidth=2, color='red')
        ax.fill_between(timestamps, error_rate, alpha=0.3, color='red')

        # Add threshold line
        ax.axhline(self.alert_threshold, color='orange', linestyle='--',
                  label=f'Alert Threshold: {self.alert_threshold:.1%}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Error Rate')
        ax.set_ylim([0, max(max(error_rate) * 1.1, self.alert_threshold * 2)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def plot_data_drift(self, ax: plt.Axes, drift_scores: Optional[Dict] = None):
        """Plot data drift detection results."""
        ax.clear()

        if not drift_scores:
            ax.text(0.5, 0.5, 'No drift data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Extract feature drift scores
        features = []
        ks_stats = []
        p_values = []

        for key, value in drift_scores.items():
            if key.startswith('feature_'):
                features.append(key)
                ks_stats.append(value['ks_statistic'])
                p_values.append(value['p_value'])

        if features:
            # Create bar plot
            x = np.arange(len(features))
            bars = ax.bar(x, ks_stats, color=['red' if p < 0.05 else 'green'
                                             for p in p_values])

            ax.set_xlabel('Feature')
            ax.set_ylabel('KS Statistic')
            ax.set_xticks(x)
            ax.set_xticklabels([f'F{i}' for i in range(len(features))], rotation=45)

            # Add significance line
            ax.axhline(0.05, color='orange', linestyle='--', label='Significance')

            ax.legend()
            ax.set_title(f"Data Drift Detection (JS: {drift_scores.get('overall', {}).get('js_divergence', 0):.3f})")

        ax.grid(True, alpha=0.3, axis='y')

    def plot_model_comparison(self, ax: plt.Axes):
        """Plot model comparison chart."""
        ax.clear()

        if not self.model_performances:
            ax.text(0.5, 0.5, 'No model comparison data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Prepare data
        models = list(self.model_performances.keys())
        metrics_names = list(next(iter(self.model_performances.values()))['metrics'].keys())

        # Create grouped bar chart
        x = np.arange(len(metrics_names))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [self.model_performances[model]['metrics'].get(m, 0)
                     for m in metrics_names]
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

    def plot_alerts(self, ax: plt.Axes):
        """Plot recent alerts."""
        ax.clear()

        if not self.alerts:
            ax.text(0.5, 0.5, 'No active alerts', ha='center', va='center',
                   transform=ax.transAxes, color='green', fontsize=12)
            return

        # Group alerts by severity
        high_alerts = [a for a in self.alerts if a['severity'] == 'high']
        medium_alerts = [a for a in self.alerts if a['severity'] == 'medium']
        low_alerts = [a for a in self.alerts if a['severity'] == 'low']

        # Create alert summary
        y_pos = 0.9
        for alerts, color, label in [
            (high_alerts, 'red', 'HIGH'),
            (medium_alerts, 'orange', 'MEDIUM'),
            (low_alerts, 'yellow', 'LOW')
        ]:
            if alerts:
                for alert in alerts[-3:]:  # Show last 3 of each type
                    time_ago = (datetime.now() - alert['timestamp']).total_seconds() / 60
                    text = f"[{label}] {alert['type']}: {alert['message']} ({time_ago:.0f}m ago)"
                    ax.text(0.05, y_pos, text, transform=ax.transAxes,
                           color=color, fontsize=9, fontweight='bold')
                    y_pos -= 0.1

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')

    def plot_distribution_comparison(self, ax: plt.Axes, current_dist: Optional[np.ndarray] = None,
                                    reference_dist: Optional[np.ndarray] = None):
        """Plot distribution comparison."""
        ax.clear()

        if current_dist is None or reference_dist is None:
            ax.text(0.5, 0.5, 'No distribution data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Create violin plots
        data_to_plot = [reference_dist.flatten(), current_dist.flatten()]
        parts = ax.violinplot(data_to_plot, positions=[0, 1], widths=0.7,
                             showmeans=True, showmedians=True)

        # Color the violins
        colors = ['green', 'orange']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Reference', 'Current'])
        ax.set_ylabel('Value Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    def generate_performance_report(self) -> str:
        """Generate text performance report."""
        report = []
        report.append("="*50)
        report.append("PRODUCTION MONITORING REPORT")
        report.append("="*50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Performance summary
        if self.performance_buffer:
            recent_perf = list(self.performance_buffer)[-100:]
            avg_acc = np.mean([m['accuracy'] for m in recent_perf])
            avg_f1 = np.mean([m['f1_score'] for m in recent_perf])

            report.append("PERFORMANCE METRICS (Last 100 samples):")
            report.append(f"  Average Accuracy: {avg_acc:.2%}")
            report.append(f"  Average F1 Score: {avg_f1:.3f}")
            report.append("")

        # Latency summary
        if self.latency_buffer:
            latencies = [l['per_sample'] for l in self.latency_buffer]
            report.append("LATENCY METRICS:")
            report.append(f"  P50: {np.percentile(latencies, 50):.1f}ms")
            report.append(f"  P95: {np.percentile(latencies, 95):.1f}ms")
            report.append(f"  P99: {np.percentile(latencies, 99):.1f}ms")
            report.append("")

        # Throughput summary
        if self.throughput_buffer:
            avg_throughput = np.mean([t['throughput'] for t in self.throughput_buffer])
            report.append(f"AVERAGE THROUGHPUT: {avg_throughput:.1f} samples/sec")
            report.append("")

        # Active alerts
        if self.alerts:
            report.append(f"ACTIVE ALERTS ({len(self.alerts)}):")
            for severity in ['high', 'medium', 'low']:
                severity_alerts = [a for a in self.alerts if a['severity'] == severity]
                if severity_alerts:
                    report.append(f"  {severity.upper()}: {len(severity_alerts)} alerts")
                    for alert in severity_alerts[-3:]:
                        report.append(f"    - {alert['type']}: {alert['message']}")
        else:
            report.append("NO ACTIVE ALERTS")

        report.append("")
        report.append("="*50)

        return "\n".join(report)

    def save_monitoring_data(self):
        """Save monitoring data to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save performance metrics
        if self.performance_buffer:
            perf_df = pd.DataFrame(list(self.performance_buffer))
            perf_df.to_csv(self.save_dir / f'performance_{timestamp}.csv', index=False)

        # Save alerts
        if self.alerts:
            with open(self.save_dir / f'alerts_{timestamp}.json', 'w') as f:
                json.dump(self.alerts, f, default=str, indent=2)

        # Save report
        report = self.generate_performance_report()
        with open(self.save_dir / f'report_{timestamp}.txt', 'w') as f:
            f.write(report)

        print(f"Monitoring data saved to {self.save_dir}")


def example_usage():
    """Example of how to use the ProductionDashboard."""
    dashboard = ProductionDashboard()

    # Create dashboard
    fig, axes = dashboard.create_monitoring_dashboard()

    # Simulate production monitoring
    for i in range(100):
        # Update performance
        dashboard.update_performance_metrics(
            accuracy=0.92 + np.random.normal(0, 0.02),
            f1_score=0.89 + np.random.normal(0, 0.02),
            precision=0.91 + np.random.normal(0, 0.02),
            recall=0.88 + np.random.normal(0, 0.02)
        )

        # Update latency
        dashboard.update_latency(
            latency_ms=25 + np.random.exponential(5),
            batch_size=32
        )

        # Update throughput
        dashboard.update_throughput(
            samples_per_second=1000 + np.random.normal(0, 100)
        )

        # Simulate data drift occasionally
        if i % 30 == 0:
            current = np.random.randn(100, 8)
            reference = np.random.randn(100, 8)
            drift_scores = dashboard.detect_data_drift(current, reference)

            # Plot data drift
            dashboard.plot_data_drift(axes['data_drift'], drift_scores)

    # Add model comparisons
    dashboard.compare_models('Model_A', {
        'accuracy': 0.92, 'f1_score': 0.89, 'precision': 0.91, 'recall': 0.88
    })
    dashboard.compare_models('Model_B', {
        'accuracy': 0.94, 'f1_score': 0.91, 'precision': 0.93, 'recall': 0.90
    })
    dashboard.compare_models('Model_C', {
        'accuracy': 0.90, 'f1_score': 0.87, 'precision': 0.89, 'recall': 0.86
    })

    # Add some alerts
    dashboard.add_alert('Model Update', 'New model deployed successfully', 'low')
    dashboard.add_alert('Performance Warning', 'Accuracy below threshold', 'medium')

    # Update all plots
    dashboard.plot_performance_timeline(axes['performance'])
    dashboard.plot_latency_distribution(axes['latency'])
    dashboard.plot_throughput_timeline(axes['throughput'])
    dashboard.plot_error_rate(axes['error_rate'])
    dashboard.plot_model_comparison(axes['model_comparison'])
    dashboard.plot_alerts(axes['alerts'])

    # Example distributions
    current_dist = np.random.randn(1000, 1)
    reference_dist = np.random.randn(1000, 1) + 0.1
    dashboard.plot_distribution_comparison(axes['distribution'], current_dist, reference_dist)

    plt.tight_layout()
    plt.show()

    # Generate report
    print(dashboard.generate_performance_report())

    # Save data
    dashboard.save_monitoring_data()


if __name__ == "__main__":
    example_usage()