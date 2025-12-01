"""
Model Interpreter for Deep Learning Model Analysis
Provides visualization tools for understanding model predictions including
feature importance, saliency maps, confidence distributions, and error analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class ModelInterpreter:
    """Model interpretation and visualization tools."""

    def __init__(self, model: torch.nn.Module, device: str = 'cpu',
                 save_dir: Optional[Path] = None):
        """
        Initialize the model interpreter.

        Args:
            model: PyTorch model to interpret
            device: Device for computations
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/model_interpreter")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.cmap_importance = 'RdBu_r'
        self.cmap_saliency = 'hot'
        self.cmap_confidence = 'viridis'

    def compute_feature_importance(self, dataloader: torch.utils.data.DataLoader,
                                  n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute feature importance using gradient-based method.

        Args:
            dataloader: DataLoader with samples
            n_samples: Number of samples to use

        Returns:
            Dictionary with feature importance scores
        """
        self.model.eval()

        continuous_importance = []
        categorical_importance = []

        sample_count = 0

        with torch.enable_grad():
            for batch in dataloader:
                if sample_count >= n_samples:
                    break

                # Get inputs
                continuous = batch['continuous'].to(self.device).requires_grad_(True)
                categorical = batch['categorical'].to(self.device).float().float().requires_grad_(True)
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                # Forward pass
                mods = [continuous, categorical]
                output = self.model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1])

                # Get loss or output magnitude
                if hasattr(output, 'logits'):
                    loss = output.logits.mean()
                else:
                    loss = output['memory'].mean()

                # Backward pass
                loss.backward()

                # Collect gradients
                if continuous.grad is not None:
                    continuous_importance.append(continuous.grad.abs().cpu().numpy())
                if categorical.grad is not None:
                    categorical_importance.append(categorical.grad.abs().cpu().numpy())

                sample_count += continuous.shape[0]

                # Clear gradients
                self.model.zero_grad()

        # Average importance scores
        importance = {}
        if continuous_importance:
            importance['continuous'] = np.mean(np.concatenate(continuous_importance, axis=0), axis=0)
        if categorical_importance:
            importance['categorical'] = np.mean(np.concatenate(categorical_importance, axis=0), axis=0)

        return importance

    def plot_feature_importance(self, importance: Dict[str, np.ndarray],
                               feature_names: Optional[Dict[str, List[str]]] = None,
                               top_k: int = 20) -> plt.Figure:
        """
        Plot feature importance scores.

        Args:
            importance: Feature importance dictionary
            feature_names: Optional feature names
            top_k: Number of top features to show

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Default feature names
        if feature_names is None:
            feature_names = {
                'continuous': [f'Sensor_{i}' for i in range(8)],
                'categorical': ['Operation_Type']
            }

        # Plot continuous features
        if 'continuous' in importance:
            ax = axes[0]
            # Average across time dimension if needed
            if importance['continuous'].ndim > 1:
                imp_values = importance['continuous'].mean(axis=0)
            else:
                imp_values = importance['continuous']

            # Get top features
            n_features = len(imp_values)
            indices = np.argsort(imp_values)[-min(top_k, n_features):]

            names = [feature_names['continuous'][i % len(feature_names['continuous'])]
                    for i in indices]
            values = imp_values[indices]

            # Create bar plot
            bars = ax.barh(range(len(values)), values)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Importance Score')
            ax.set_title('Continuous Feature Importance')

            # Color bars by value
            norm = plt.Normalize(vmin=values.min(), vmax=values.max())
            sm = plt.cm.ScalarMappable(cmap=self.cmap_importance, norm=norm)
            for bar, val in zip(bars, values):
                bar.set_color(sm.to_rgba(val))

        # Plot categorical features
        if 'categorical' in importance:
            ax = axes[1]
            imp_values = importance['categorical']

            # Ensure imp_values is 1D
            if len(imp_values.shape) > 1:
                imp_values = imp_values.mean(axis=0) if imp_values.shape[0] > 1 else imp_values.flatten()

            names = feature_names.get('categorical', [f'Cat_{i}' for i in range(len(imp_values))])

            # Ensure names and imp_values have same length
            if len(names) != len(imp_values):
                names = [f'Cat_{i}' for i in range(len(imp_values))]

            bars = ax.bar(range(len(imp_values)), imp_values)
            ax.set_xticks(range(len(imp_values)))
            ax.set_xticklabels(names, rotation=45)
            ax.set_ylabel('Importance Score')
            ax.set_title('Categorical Feature Importance')

            # Color bars
            for bar, val in zip(bars, imp_values):
                bar.set_color(plt.cm.RdBu_r(val / max(imp_values)))

        plt.tight_layout()
        return fig

    def generate_saliency_map(self, sample: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Generate saliency map for a single sample.

        Args:
            sample: Single sample dictionary

        Returns:
            Saliency map array
        """
        self.model.eval()

        # Prepare inputs
        continuous = sample['continuous'].unsqueeze(0).to(self.device).requires_grad_(True)
        categorical = sample['categorical'].unsqueeze(0).to(self.device).float()
        tokens = sample['tokens'].unsqueeze(0).to(self.device)
        lengths = sample['lengths'].unsqueeze(0).to(self.device)

        # Forward pass
        mods = [continuous, categorical]
        output = self.model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1])

        # Get prediction
        if hasattr(output, 'logits'):
            # Multi-head model
            logits = output.logits['command_logits']
            pred = logits.max(dim=-1)[0].sum()
        else:
            pred = output['memory'].max(dim=-1)[0].sum()

        # Backward pass
        pred.backward()

        # Get gradients
        saliency = continuous.grad.abs().squeeze().cpu().numpy()

        return saliency

    def plot_saliency_maps(self, dataloader: torch.utils.data.DataLoader,
                          n_samples: int = 6) -> plt.Figure:
        """
        Plot saliency maps for multiple samples.

        Args:
            dataloader: DataLoader with samples
            n_samples: Number of samples to visualize

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        sample_idx = 0
        for batch in dataloader:
            if sample_idx >= n_samples:
                break

            # Get single sample
            sample = {k: v[0] for k, v in batch.items()}

            # Generate saliency map
            saliency = self.generate_saliency_map(sample)

            # Plot
            ax = axes[sample_idx]
            im = ax.imshow(saliency.T, cmap=self.cmap_saliency, aspect='auto')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature')
            ax.set_title(f'Sample {sample_idx + 1} Saliency Map')
            plt.colorbar(im, ax=ax, fraction=0.046)

            sample_idx += 1

        plt.suptitle('Saliency Maps - Important Time Points for Predictions', fontsize=14)
        plt.tight_layout()
        return fig

    def analyze_prediction_confidence(self, dataloader: torch.utils.data.DataLoader,
                                     n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Analyze prediction confidence distributions.

        Args:
            dataloader: DataLoader with samples
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with confidence metrics
        """
        self.model.eval()

        confidences = {
            'command': [],
            'param_type': [],
            'param_value': [],
            'operation': [],
            'entropy': []
        }

        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= n_samples:
                    break

                # Get inputs
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device).float()
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                # Forward pass
                mods = [continuous, categorical]
                output = self.model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1])

                if hasattr(output, 'logits'):
                    # Get probabilities for each head
                    if 'command_logits' in output.logits:
                        probs = F.softmax(output.logits['command_logits'], dim=-1)
                        conf = probs.max(dim=-1)[0].cpu().numpy()
                        confidences['command'].extend(conf.flatten())

                        # Calculate entropy
                        ent = entropy(probs.cpu().numpy(), axis=-1)
                        confidences['entropy'].extend(ent.flatten())

                    if 'param_type_logits' in output.logits:
                        probs = F.softmax(output.logits['param_type_logits'], dim=-1)
                        conf = probs.max(dim=-1)[0].cpu().numpy()
                        confidences['param_type'].extend(conf.flatten())

                    if 'param_value_logits' in output.logits:
                        probs = F.softmax(output.logits['param_value_logits'], dim=-1)
                        conf = probs.max(dim=-1)[0].cpu().numpy()
                        confidences['param_value'].extend(conf.flatten())

                    if 'operation_logits' in output.logits:
                        probs = F.softmax(output.logits['operation_logits'], dim=-1)
                        conf = probs.max(dim=-1)[0].cpu().numpy()
                        confidences['operation'].extend(conf.flatten())

                sample_count += continuous.shape[0]

        # Convert to arrays
        for key in confidences:
            if confidences[key]:
                confidences[key] = np.array(confidences[key])

        return confidences

    def plot_confidence_distributions(self, confidences: Dict[str, np.ndarray]) -> plt.Figure:
        """
        Plot confidence score distributions.

        Args:
            confidences: Dictionary with confidence scores

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        heads = ['command', 'param_type', 'param_value', 'operation']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (head, color) in enumerate(zip(heads, colors)):
            ax = axes[idx // 2, idx % 2]

            if head in confidences and len(confidences[head]) > 0:
                data = confidences[head]

                # Plot histogram
                ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')

                # Add statistics
                mean_conf = np.mean(data)
                median_conf = np.median(data)
                ax.axvline(mean_conf, color='red', linestyle='--',
                          label=f'Mean: {mean_conf:.3f}')
                ax.axvline(median_conf, color='green', linestyle='--',
                          label=f'Median: {median_conf:.3f}')

                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{head.title()} Prediction Confidence')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {head} data', ha='center', va='center',
                       transform=ax.transAxes)

        plt.suptitle('Prediction Confidence Distributions', fontsize=14)
        plt.tight_layout()
        return fig

    def analyze_errors_by_token(self, dataloader: torch.utils.data.DataLoader,
                               vocab: Dict[str, int], n_samples: int = 1000) -> pd.DataFrame:
        """
        Analyze which tokens are hardest to predict.

        Args:
            dataloader: DataLoader with samples
            vocab: Vocabulary dictionary
            n_samples: Number of samples

        Returns:
            DataFrame with error analysis
        """
        self.model.eval()

        # Reverse vocab for token names
        id_to_token = {v: k for k, v in vocab.items()}

        token_stats = {}

        with torch.no_grad():
            sample_count = 0
            for batch in dataloader:
                if sample_count >= n_samples:
                    break

                # Get inputs
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device).float()
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                # Forward pass
                tgt_in = tokens[:, :-1]
                tgt_out = tokens[:, 1:]

                mods = [continuous, categorical]
                output = self.model(mods=mods, lengths=lengths, gcode_in=tgt_in)

                if hasattr(output, 'logits'):
                    # Get predictions
                    if 'command_logits' in output.logits:
                        preds = output.logits['command_logits'].argmax(dim=-1)
                    else:
                        preds = output['memory'].argmax(dim=-1)

                    # Compare with targets
                    for i in range(tgt_out.shape[0]):
                        for j in range(tgt_out.shape[1]):
                            token_id = tgt_out[i, j].item()
                            if token_id == 0:  # Skip padding
                                continue

                            token_name = id_to_token.get(token_id, f'UNK_{token_id}')

                            if token_name not in token_stats:
                                token_stats[token_name] = {
                                    'count': 0,
                                    'correct': 0,
                                    'errors': []
                                }

                            token_stats[token_name]['count'] += 1

                            if preds[i, j].item() == token_id:
                                token_stats[token_name]['correct'] += 1
                            else:
                                pred_name = id_to_token.get(preds[i, j].item(), 'UNK')
                                token_stats[token_name]['errors'].append(pred_name)

                sample_count += continuous.shape[0]

        # Create DataFrame
        rows = []
        for token, stats in token_stats.items():
            if stats['count'] > 0:
                accuracy = stats['correct'] / stats['count']
                # Most common error
                if stats['errors']:
                    most_common_error = max(set(stats['errors']),
                                          key=stats['errors'].count)
                else:
                    most_common_error = 'None'

                rows.append({
                    'token': token,
                    'count': stats['count'],
                    'accuracy': accuracy,
                    'error_rate': 1 - accuracy,
                    'most_common_error': most_common_error
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('error_rate', ascending=False)

        return df

    def plot_token_error_analysis(self, error_df: pd.DataFrame, top_k: int = 20) -> plt.Figure:
        """
        Plot token error analysis.

        Args:
            error_df: DataFrame with error analysis
            top_k: Number of tokens to show

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Handle empty DataFrame
        if error_df.empty:
            axes[0].text(0.5, 0.5, 'No token errors found',
                        ha='center', va='center', fontsize=12)
            axes[0].set_title('Token Error Analysis')
            axes[1].text(0.5, 0.5, 'No data available',
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('Token Accuracy Distribution')
            plt.tight_layout()
            return fig

        # Get top error tokens
        top_errors = error_df.head(top_k)

        # Plot 1: Error rates
        ax = axes[0]
        bars = ax.barh(range(len(top_errors)), top_errors['error_rate'].values)
        ax.set_yticks(range(len(top_errors)))
        ax.set_yticklabels(top_errors['token'].values)
        ax.set_xlabel('Error Rate')
        ax.set_title(f'Top {top_k} Hardest Tokens to Predict')

        # Color bars by error rate
        for bar, val in zip(bars, top_errors['error_rate'].values):
            color = plt.cm.Reds(val)
            bar.set_color(color)

        # Plot 2: Token frequency vs accuracy
        ax = axes[1]
        scatter = ax.scatter(error_df['count'], error_df['accuracy'],
                           c=error_df['error_rate'], cmap='RdYlGn_r',
                           alpha=0.6, s=50)
        ax.set_xlabel('Token Frequency')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Token Frequency vs Prediction Accuracy')
        ax.set_xscale('log')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error Rate')

        # Annotate interesting points
        for _, row in top_errors.head(5).iterrows():
            ax.annotate(row['token'], (row['count'], row['accuracy']),
                       fontsize=8, alpha=0.7)

        plt.tight_layout()
        return fig

    def visualize_attention_patterns(self, sample: Dict[str, torch.Tensor]) -> plt.Figure:
        """
        Visualize attention patterns for a sample.

        Args:
            sample: Single sample dictionary

        Returns:
            Figure object
        """
        self.model.eval()

        # Prepare inputs
        continuous = sample['continuous'].unsqueeze(0).to(self.device)
        categorical = sample['categorical'].unsqueeze(0).to(self.device)
        tokens = sample['tokens'].unsqueeze(0).to(self.device)
        lengths = sample['lengths'].unsqueeze(0).to(self.device)

        # Forward pass with attention
        mods = [continuous, categorical]

        # Hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            if hasattr(output, 'attention_weights'):
                attention_weights.append(output.attention_weights.cpu().numpy())

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        # Forward pass
        output = self.model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1])

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Plot attention patterns
        if attention_weights:
            n_layers = len(attention_weights)
            fig, axes = plt.subplots(1, min(n_layers, 4), figsize=(16, 4))

            if n_layers == 1:
                axes = [axes]

            for idx, (ax, attn) in enumerate(zip(axes, attention_weights[:4])):
                # Average over heads and batch
                if attn.ndim > 2:
                    attn = attn.mean(axis=(0, 1))

                im = ax.imshow(attn, cmap='Blues', aspect='auto')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                ax.set_title(f'Layer {idx + 1} Attention')
                plt.colorbar(im, ax=ax, fraction=0.046)

            plt.suptitle('Attention Patterns Across Layers', fontsize=14)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No attention weights captured',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        return fig

    def create_failure_gallery(self, dataloader: torch.utils.data.DataLoader,
                              vocab: Dict[str, int], n_failures: int = 12) -> plt.Figure:
        """
        Create a gallery of failure cases with explanations.

        Args:
            dataloader: DataLoader with samples
            vocab: Vocabulary dictionary
            n_failures: Number of failures to show

        Returns:
            Figure object
        """
        self.model.eval()

        # Reverse vocab
        id_to_token = {v: k for k, v in vocab.items()}

        failures = []

        with torch.no_grad():
            for batch in dataloader:
                if len(failures) >= n_failures:
                    break

                # Get inputs
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device).float()
                tokens = batch['tokens'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                # Forward pass
                tgt_in = tokens[:, :-1]
                tgt_out = tokens[:, 1:]

                mods = [continuous, categorical]
                output = self.model(mods=mods, lengths=lengths, gcode_in=tgt_in)

                if hasattr(output, 'logits'):
                    # Get predictions
                    preds = {}
                    for key in ['command_logits', 'param_type_logits', 'param_value_logits']:
                        if key in output.logits:
                            preds[key] = output.logits[key].argmax(dim=-1)

                    # Find failures
                    for i in range(tgt_out.shape[0]):
                        for j in range(min(tgt_out.shape[1], 10)):  # Check first 10 positions
                            if tgt_out[i, j].item() == 0:  # Skip padding
                                continue

                            # Check if any prediction is wrong
                            is_failure = False
                            for pred_key in preds:
                                if preds[pred_key][i, j].item() != tgt_out[i, j].item():
                                    is_failure = True
                                    break

                            if is_failure and len(failures) < n_failures:
                                # Collect failure info
                                failure_info = {
                                    'true_token': id_to_token.get(tgt_out[i, j].item(), 'UNK'),
                                    'pred_token': id_to_token.get(preds[pred_key][i, j].item(), 'UNK'),
                                    'position': j,
                                    'sensor_data': continuous[i, j].cpu().numpy()
                                }
                                failures.append(failure_info)

        # Create gallery
        n_cols = 4
        n_rows = (len(failures) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else axes

        for idx, (ax, failure) in enumerate(zip(axes, failures)):
            # Plot sensor data
            ax.plot(failure['sensor_data'], alpha=0.7)
            ax.set_title(f"True: {failure['true_token']}\nPred: {failure['pred_token']}",
                        fontsize=10)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(failures), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Failure Case Gallery - Misclassified Examples', fontsize=14)
        plt.tight_layout()
        return fig

    def save_all_visualizations(self, dataloader: torch.utils.data.DataLoader,
                              vocab: Dict[str, int]):
        """
        Generate and save all visualizations.

        Args:
            dataloader: DataLoader with samples
            vocab: Vocabulary dictionary
        """
        print("Generating model interpretation visualizations...")

        # Feature importance
        print("Computing feature importance...")
        importance = self.compute_feature_importance(dataloader, n_samples=100)
        fig = self.plot_feature_importance(importance)
        fig.savefig(self.save_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Saliency maps
        print("Generating saliency maps...")
        fig = self.plot_saliency_maps(dataloader, n_samples=6)
        fig.savefig(self.save_dir / 'saliency_maps.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Confidence distributions
        print("Analyzing prediction confidence...")
        confidences = self.analyze_prediction_confidence(dataloader, n_samples=500)
        fig = self.plot_confidence_distributions(confidences)
        fig.savefig(self.save_dir / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Token error analysis
        print("Analyzing token errors...")
        error_df = self.analyze_errors_by_token(dataloader, vocab, n_samples=500)
        fig = self.plot_token_error_analysis(error_df)
        fig.savefig(self.save_dir / 'token_error_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Save error DataFrame
        error_df.to_csv(self.save_dir / 'token_errors.csv', index=False)

        # Failure gallery
        print("Creating failure gallery...")
        fig = self.create_failure_gallery(dataloader, vocab, n_failures=12)
        fig.savefig(self.save_dir / 'failure_gallery.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"All visualizations saved to {self.save_dir}")


def example_usage():
    """Example of how to use the ModelInterpreter."""
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(8, 128, 2, batch_first=True)
            self.head = torch.nn.Linear(128, 50)

        def forward(self, mods, lengths, gcode_in):
            continuous = mods[0]
            out, _ = self.lstm(continuous)
            logits = self.head(out)
            return {'logits': {'command_logits': logits}, 'memory': out}

    model = DummyModel()
    interpreter = ModelInterpreter(model)

    # Create dummy data
    dataset = []
    for _ in range(100):
        dataset.append({
            'continuous': torch.randn(64, 8),
            'categorical': torch.zeros(64, 1),
            'tokens': torch.randint(0, 50, (65,)),
            'lengths': torch.tensor([64])
        })

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Create dummy vocab
    vocab = {f'TOKEN_{i}': i for i in range(50)}

    # Test visualizations
    importance = interpreter.compute_feature_importance(dataloader, n_samples=10)
    fig = interpreter.plot_feature_importance(importance)
    plt.show()

    confidences = interpreter.analyze_prediction_confidence(dataloader, n_samples=50)
    fig = interpreter.plot_confidence_distributions(confidences)
    plt.show()


if __name__ == "__main__":
    example_usage()