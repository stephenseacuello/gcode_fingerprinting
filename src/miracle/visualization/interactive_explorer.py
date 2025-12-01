"""
Interactive Explorer for Deep Learning Models
Provides interactive visualizations using Plotly for exploring
embeddings, confusion matrices, and model predictions.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class InteractiveExplorer:
    """Interactive visualization tools for model exploration."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize the interactive explorer.

        Args:
            save_dir: Directory to save HTML visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/interactive_explorer")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Plotly theme
        self.template = 'plotly_white'

        # Color palettes
        self.colors = px.colors.qualitative.Plotly
        self.diverging_colors = px.colors.diverging.RdBu

    def create_3d_embeddings(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
                            method: str = 'tsne', perplexity: int = 30,
                            hover_text: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive 3D embedding visualization.

        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            labels: Optional labels for coloring
            method: 'tsne' or 'pca'
            perplexity: t-SNE perplexity parameter
            hover_text: Optional hover text for each point

        Returns:
            Plotly figure object
        """
        # Reduce dimensionality to 3D
        if method == 'tsne':
            reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
            coords = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=3)
            coords = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create DataFrame
        df = pd.DataFrame(coords, columns=['X', 'Y', 'Z'])

        if labels is not None:
            df['Label'] = labels
            color = 'Label'
        else:
            color = None

        if hover_text is not None:
            df['Text'] = hover_text
            hover_data = ['Text']
        else:
            hover_data = None

        # Create 3D scatter plot
        fig = px.scatter_3d(df, x='X', y='Y', z='Z',
                           color=color,
                           hover_data=hover_data,
                           template=self.template,
                           title=f'3D Embeddings ({method.upper()})',
                           labels={'X': f'{method.upper()} 1',
                                  'Y': f'{method.upper()} 2',
                                  'Z': f'{method.upper()} 3'})

        # Update layout for better interactivity
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        return fig

    def create_interactive_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                           class_names: Optional[List[str]] = None,
                                           normalize: bool = True) -> go.Figure:
        """
        Create interactive confusion matrix with drill-down capability.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            normalize: Whether to normalize the matrix

        Returns:
            Plotly figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2%'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        # Create class names if not provided
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        # Create hover text with details
        hover_text = []
        for i in range(cm.shape[0]):
            hover_row = []
            for j in range(cm.shape[1]):
                if normalize:
                    text = f'True: {class_names[i]}<br>' \
                          f'Pred: {class_names[j]}<br>' \
                          f'Rate: {cm[i, j]:.2%}<br>' \
                          f'Count: {int(cm[i, j] * y_true[y_true == i].shape[0])}'
                else:
                    text = f'True: {class_names[i]}<br>' \
                          f'Pred: {class_names[j]}<br>' \
                          f'Count: {cm[i, j]}'
                hover_row.append(text)
            hover_text.append(hover_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Rate' if normalize else 'Count')
        ))

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    text = f'{cm[i, j]:.1%}'
                else:
                    text = str(int(cm[i, j]))
                fig.add_annotation(
                    x=j, y=i, text=text,
                    showarrow=False,
                    font=dict(color='white' if cm[i, j] > cm.max() / 2 else 'black')
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='True',
            template=self.template,
            width=800,
            height=800
        )

        return fig

    def create_token_prediction_timeline(self, predictions: List[Dict],
                                        ground_truth: List[str],
                                        sample_id: int = 0) -> go.Figure:
        """
        Create timeline visualization of token predictions.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth tokens
            sample_id: Sample identifier

        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Confidence Scores', 'Token Types', 'Prediction vs Truth'),
            vertical_spacing=0.1,
            row_heights=[0.3, 0.3, 0.4]
        )

        positions = list(range(len(predictions)))

        # Extract data
        confidences = [p.get('confidence', 0.5) for p in predictions]
        token_types = [p.get('type', 'unknown') for p in predictions]
        predicted_tokens = [p.get('predicted', '') for p in predictions]

        # Plot 1: Confidence scores
        fig.add_trace(
            go.Scatter(x=positions, y=confidences,
                      mode='lines+markers',
                      name='Confidence',
                      line=dict(color='blue', width=2),
                      marker=dict(size=6)),
            row=1, col=1
        )

        # Add confidence threshold
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                     annotation_text="High Confidence", row=1, col=1)

        # Plot 2: Token types (categorical)
        type_map = {'command': 1, 'param_type': 2, 'param_value': 3, 'special': 0}
        type_values = [type_map.get(t, -1) for t in token_types]

        fig.add_trace(
            go.Scatter(x=positions, y=type_values,
                      mode='markers',
                      name='Token Type',
                      marker=dict(size=10, color=type_values,
                                colorscale='Viridis'),
                      text=token_types,
                      hovertemplate='Type: %{text}<extra></extra>'),
            row=2, col=1
        )

        # Plot 3: Prediction comparison
        # Create a comparison view
        correct = [1 if pred == gt else 0 for pred, gt in zip(predicted_tokens, ground_truth)]

        fig.add_trace(
            go.Scatter(x=positions, y=[1] * len(positions),
                      mode='markers+text',
                      name='Ground Truth',
                      marker=dict(size=15, color='green', symbol='square'),
                      text=ground_truth,
                      textposition="top center",
                      hovertemplate='Truth: %{text}<extra></extra>'),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=positions, y=[0] * len(positions),
                      mode='markers+text',
                      name='Predicted',
                      marker=dict(size=15, color=['green' if c else 'red' for c in correct],
                                symbol='circle'),
                      text=predicted_tokens,
                      textposition="bottom center",
                      hovertemplate='Predicted: %{text}<extra></extra>'),
            row=3, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Token Position", row=3, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)
        fig.update_yaxes(title_text="Type", ticktext=['Special', 'Command', 'Param Type', 'Param Value'],
                        tickvals=[0, 1, 2, 3], row=2, col=1)
        fig.update_yaxes(title_text="", showticklabels=False, row=3, col=1)

        fig.update_layout(
            title=f'Token Prediction Timeline - Sample {sample_id}',
            template=self.template,
            height=800,
            showlegend=True
        )

        return fig

    def create_hyperparameter_parallel_coordinates(self, sweep_results: pd.DataFrame,
                                                   metric_column: str = 'val_acc') -> go.Figure:
        """
        Create parallel coordinates plot for hyperparameter exploration.

        Args:
            sweep_results: DataFrame with sweep results
            metric_column: Metric column to color by

        Returns:
            Plotly figure object
        """
        # Select hyperparameter columns
        param_columns = [col for col in sweep_results.columns
                        if col not in ['val_acc', 'train_acc', 'loss', 'run_id', 'epoch']]

        # Normalize metric for coloring
        metric_values = sweep_results[metric_column].values
        norm_metric = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())

        # Create dimensions
        dimensions = []
        for col in param_columns:
            if sweep_results[col].dtype in ['float64', 'int64']:
                dimensions.append(
                    dict(range=[sweep_results[col].min(), sweep_results[col].max()],
                        label=col,
                        values=sweep_results[col])
                )
            else:
                # Categorical parameter
                unique_vals = sweep_results[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                dimensions.append(
                    dict(range=[0, len(unique_vals) - 1],
                        label=col,
                        values=[mapping[val] for val in sweep_results[col]],
                        tickvals=list(range(len(unique_vals))),
                        ticktext=list(unique_vals))
                )

        # Add metric dimension
        dimensions.append(
            dict(range=[metric_values.min(), metric_values.max()],
                label=metric_column,
                values=metric_values)
        )

        # Create parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=metric_values,
                         colorscale='Viridis',
                         showscale=True,
                         colorbar=dict(title=metric_column)),
                dimensions=dimensions
            )
        )

        fig.update_layout(
            title='Hyperparameter Parallel Coordinates',
            template=self.template,
            width=1200,
            height=600
        )

        return fig

    def create_loss_surface_3d(self, param1_values: np.ndarray, param2_values: np.ndarray,
                              loss_values: np.ndarray, param1_name: str = 'Param 1',
                              param2_name: str = 'Param 2') -> go.Figure:
        """
        Create 3D loss surface visualization.

        Args:
            param1_values: First parameter values
            param2_values: Second parameter values
            loss_values: Loss values (2D grid)
            param1_name: Name of first parameter
            param2_name: Name of second parameter

        Returns:
            Plotly figure object
        """
        # Create surface plot
        fig = go.Figure(data=[go.Surface(x=param1_values, y=param2_values, z=loss_values,
                                        colorscale='Viridis')])

        # Find minimum
        min_idx = np.unravel_index(loss_values.argmin(), loss_values.shape)
        min_loss = loss_values[min_idx]
        min_p1 = param1_values[min_idx[1]]
        min_p2 = param2_values[min_idx[0]]

        # Add minimum point
        fig.add_trace(go.Scatter3d(
            x=[min_p1], y=[min_p2], z=[min_loss],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'Minimum: {min_loss:.4f}',
            text=f'{param1_name}: {min_p1:.4f}<br>{param2_name}: {min_p2:.4f}<br>Loss: {min_loss:.4f}',
            hovertemplate='%{text}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title='Loss Surface Visualization',
            scene=dict(
                xaxis_title=param1_name,
                yaxis_title=param2_name,
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template=self.template,
            width=900,
            height=700
        )

        return fig

    def create_attention_heatmap_interactive(self, attention_weights: np.ndarray,
                                            source_tokens: List[str],
                                            target_tokens: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive attention heatmap.

        Args:
            attention_weights: Attention weight matrix
            source_tokens: Source token labels
            target_tokens: Target token labels (if None, use source)

        Returns:
            Plotly figure object
        """
        if target_tokens is None:
            target_tokens = source_tokens

        # Create hover text
        hover_text = []
        for i, src in enumerate(source_tokens):
            hover_row = []
            for j, tgt in enumerate(target_tokens):
                text = f'Source: {src}<br>Target: {tgt}<br>Weight: {attention_weights[i, j]:.4f}'
                hover_row.append(text)
            hover_text.append(hover_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=target_tokens,
            y=source_tokens,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Attention Weight')
        ))

        # Update layout
        fig.update_layout(
            title='Attention Weights Heatmap',
            xaxis_title='Target Tokens',
            yaxis_title='Source Tokens',
            template=self.template,
            width=800,
            height=800
        )

        return fig

    def create_metric_dashboard(self, metrics_history: Dict[str, List[float]]) -> go.Figure:
        """
        Create comprehensive metrics dashboard.

        Args:
            metrics_history: Dictionary of metric histories

        Returns:
            Plotly figure object
        """
        # Create subplots
        n_metrics = len(metrics_history)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        subplot_titles = list(metrics_history.keys())
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )

        # Add traces
        for idx, (metric_name, values) in enumerate(metrics_history.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            epochs = list(range(len(values)))

            # Add main line
            fig.add_trace(
                go.Scatter(x=epochs, y=values,
                          mode='lines+markers',
                          name=metric_name,
                          line=dict(width=2),
                          marker=dict(size=4)),
                row=row, col=col
            )

            # Add smoothed line
            if len(values) > 10:
                # Simple moving average
                window = 5
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                fig.add_trace(
                    go.Scatter(x=epochs[window-1:], y=smoothed,
                              mode='lines',
                              name=f'{metric_name} (smoothed)',
                              line=dict(width=2, dash='dash'),
                              opacity=0.7),
                    row=row, col=col
                )

            # Add best point
            if 'loss' in metric_name.lower():
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            fig.add_trace(
                go.Scatter(x=[epochs[best_idx]], y=[values[best_idx]],
                          mode='markers',
                          name=f'Best: {values[best_idx]:.4f}',
                          marker=dict(size=12, color='red', symbol='star')),
                row=row, col=col
            )

        # Update layout
        fig.update_layout(
            title='Training Metrics Dashboard',
            template=self.template,
            height=300 * n_rows,
            showlegend=False
        )

        # Update axes
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text="Epoch", row=i, col=j)

        return fig

    def create_grammar_constraint_graph(self, valid_transitions: Dict[str, List[str]]) -> go.Figure:
        """
        Create interactive grammar constraint visualization.

        Args:
            valid_transitions: Dictionary of valid token transitions

        Returns:
            Plotly figure object
        """
        # Create nodes and edges
        nodes = list(set(valid_transitions.keys()) |
                    set(sum(valid_transitions.values(), [])))

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(size=10, color=[]),
            text=[],
            textposition="top center"
        )

        edge_trace = go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )

        # Simple circular layout
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n_nodes
            x = np.cos(angle)
            y = np.sin(angle)

            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
            node_trace['marker']['color'] += (i,)

            # Add edges
            if node in valid_transitions:
                for target in valid_transitions[node]:
                    if target in nodes:
                        target_idx = nodes.index(target)
                        target_angle = 2 * np.pi * target_idx / n_nodes
                        target_x = np.cos(target_angle)
                        target_y = np.sin(target_angle)

                        edge_trace['x'] += (x, target_x, None)
                        edge_trace['y'] += (y, target_y, None)

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title='Grammar Constraint Graph',
            showlegend=False,
            hovermode='closest',
            template=self.template,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=900,
            height=900
        )

        return fig

    def save_all_interactive_plots(self, model_outputs: Dict, save_html: bool = True):
        """
        Generate and save all interactive visualizations.

        Args:
            model_outputs: Dictionary with model outputs and metrics
            save_html: Whether to save as HTML files
        """
        print("Generating interactive visualizations...")

        # Example: Create 3D embeddings
        if 'embeddings' in model_outputs:
            fig = self.create_3d_embeddings(
                model_outputs['embeddings'],
                labels=model_outputs.get('labels'),
                method='tsne'
            )
            if save_html:
                fig.write_html(self.save_dir / '3d_embeddings.html')

        # Example: Create confusion matrix
        if 'y_true' in model_outputs and 'y_pred' in model_outputs:
            fig = self.create_interactive_confusion_matrix(
                model_outputs['y_true'],
                model_outputs['y_pred'],
                class_names=model_outputs.get('class_names')
            )
            if save_html:
                fig.write_html(self.save_dir / 'confusion_matrix.html')

        # Example: Create metrics dashboard
        if 'metrics_history' in model_outputs:
            fig = self.create_metric_dashboard(model_outputs['metrics_history'])
            if save_html:
                fig.write_html(self.save_dir / 'metrics_dashboard.html')

        print(f"Interactive visualizations saved to {self.save_dir}")


def example_usage():
    """Example of how to use the InteractiveExplorer."""
    explorer = InteractiveExplorer()

    # Example 1: 3D Embeddings
    embeddings = np.random.randn(500, 50)
    labels = np.random.randint(0, 5, 500)
    fig = explorer.create_3d_embeddings(embeddings, labels, method='pca')
    fig.show()

    # Example 2: Interactive Confusion Matrix
    y_true = np.random.randint(0, 10, 1000)
    y_pred = y_true.copy()
    y_pred[:100] = np.random.randint(0, 10, 100)  # Add some errors

    fig = explorer.create_interactive_confusion_matrix(
        y_true, y_pred,
        class_names=[f'Class_{i}' for i in range(10)]
    )
    fig.show()

    # Example 3: Metrics Dashboard
    metrics_history = {
        'Train Loss': list(2.0 * np.exp(-np.arange(50)/10) + np.random.normal(0, 0.1, 50)),
        'Val Loss': list(2.2 * np.exp(-np.arange(50)/12) + np.random.normal(0, 0.1, 50)),
        'Train Acc': list(np.minimum(0.95, 0.5 + np.arange(50)/100 + np.random.normal(0, 0.05, 50))),
        'Val Acc': list(np.minimum(0.90, 0.45 + np.arange(50)/110 + np.random.normal(0, 0.05, 50)))
    }

    fig = explorer.create_metric_dashboard(metrics_history)
    fig.show()

    # Example 4: Hyperparameter Parallel Coordinates
    sweep_data = {
        'learning_rate': np.random.uniform(0.0001, 0.01, 100),
        'batch_size': np.random.choice([16, 32, 64, 128], 100),
        'hidden_dim': np.random.choice([64, 128, 256, 512], 100),
        'dropout': np.random.uniform(0.1, 0.5, 100),
        'val_acc': np.random.uniform(0.7, 0.95, 100)
    }
    sweep_df = pd.DataFrame(sweep_data)

    fig = explorer.create_hyperparameter_parallel_coordinates(sweep_df, 'val_acc')
    fig.show()


if __name__ == "__main__":
    example_usage()