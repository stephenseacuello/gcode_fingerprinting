"""
Visualization utilities for G-Code fingerprinting project.

This module provides reusable plotting functions for:
- Training metrics and learning curves
- Confusion matrices and error analysis
- Hyperparameter sweep visualizations
- Model predictions and comparisons
"""

from .training_plots import (
    plot_learning_curves,
    plot_loss_curves,
    plot_metric_comparison,
)

from .sweep_plots import (
    plot_parameter_importance,
    plot_parallel_coordinates,
    plot_metric_distributions,
)

__all__ = [
    # Training visualizations
    'plot_learning_curves',
    'plot_loss_curves',
    'plot_metric_comparison',

    # Sweep visualizations
    'plot_parameter_importance',
    'plot_parallel_coordinates',
    'plot_metric_distributions',
]
