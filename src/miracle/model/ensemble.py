"""
Ensemble prediction module for combining multiple decoder models.

This module implements:
1. EnsembleDecoder - Average predictions from multiple models
2. Test-time augmentation (TTA)
3. Weighted ensemble with learned or manual weights

Usage:
    from miracle.model.ensemble import EnsembleDecoder

    # Load multiple checkpoints
    ensemble = EnsembleDecoder.from_checkpoints(
        checkpoint_paths=['model1.pt', 'model2.pt', 'model3.pt'],
        model_class=SensorMultiHeadDecoder,
        model_kwargs={...},
        device='cuda'
    )

    # Inference
    outputs = ensemble(tokens, sensor_embeddings, operation_type)

Author: Claude Code
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Callable


class EnsembleDecoder(nn.Module):
    """
    Combines predictions from multiple decoder models.

    Supports:
    - Simple averaging of log probabilities
    - Weighted averaging with custom weights
    - Temperature scaling per model
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
        voting_mode: str = 'soft',  # 'soft' (avg logits) or 'hard' (majority vote)
    ):
        """
        Initialize ensemble.

        Args:
            models: List of trained decoder models
            weights: Optional weights for each model (will be normalized)
            temperature: Temperature for softmax (default: 1.0)
            voting_mode: 'soft' for probability averaging, 'hard' for majority vote
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.temperature = temperature
        self.voting_mode = voting_mode

        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        self.register_buffer('weights', torch.tensor(weights))

        # Put all models in eval mode
        for model in self.models:
            model.eval()

    def forward(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Average predictions across all models.

        Args:
            tokens: Input token sequence [B, L]
            sensor_embeddings: Sensor features [B, T_s, sensor_dim]
            operation_type: Operation type [B]
            **kwargs: Additional arguments passed to models

        Returns:
            Dict with averaged logits for all output heads
        """
        all_outputs = []

        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                outputs = model(tokens, sensor_embeddings, operation_type, **kwargs)
            all_outputs.append(outputs)

        # Ensemble by weighted averaging
        ensemble_outputs = {}

        # Standard classification heads
        for key in ['legacy_logits', 'raw_legacy_logits', 'type_logits',
                    'command_logits', 'param_type_logits']:
            if key in all_outputs[0]:
                stacked = torch.stack([o[key] for o in all_outputs], dim=0)  # [N, B, L, C]

                if self.voting_mode == 'soft':
                    # Weighted average of log-softmax
                    log_probs = F.log_softmax(stacked / self.temperature, dim=-1)
                    weighted = (log_probs * self.weights.view(-1, 1, 1, 1)).sum(dim=0)
                    ensemble_outputs[key] = weighted
                else:
                    # Hard voting - take most common prediction
                    predictions = stacked.argmax(dim=-1)  # [N, B, L]
                    # Mode across models (most common prediction)
                    mode_pred, _ = torch.mode(predictions, dim=0)  # [B, L]
                    # Convert back to logits (one-hot)
                    vocab_size = stacked.shape[-1]
                    ensemble_outputs[key] = F.one_hot(mode_pred, vocab_size).float()

        # Digit logits - 5D tensor [B, L, n_digits, 10]
        if 'digit_logits' in all_outputs[0]:
            stacked = torch.stack([o['digit_logits'] for o in all_outputs], dim=0)
            # Apply weights: [N, B, L, n_digits, 10]
            weights_expanded = self.weights.view(-1, 1, 1, 1, 1)
            if self.voting_mode == 'soft':
                log_probs = F.log_softmax(stacked / self.temperature, dim=-1)
                ensemble_outputs['digit_logits'] = (log_probs * weights_expanded).sum(dim=0)
            else:
                ensemble_outputs['digit_logits'] = stacked.mean(dim=0)

        # Sign logits - 4D tensor [B, L, 3]
        if 'sign_logits' in all_outputs[0]:
            stacked = torch.stack([o['sign_logits'] for o in all_outputs], dim=0)
            weights_expanded = self.weights.view(-1, 1, 1, 1)
            if self.voting_mode == 'soft':
                log_probs = F.log_softmax(stacked / self.temperature, dim=-1)
                ensemble_outputs['sign_logits'] = (log_probs * weights_expanded).sum(dim=0)
            else:
                ensemble_outputs['sign_logits'] = stacked.mean(dim=0)

        return ensemble_outputs

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        model_class: type,
        model_kwargs: Dict[str, Any],
        device: str = 'cpu',
        weights: Optional[List[float]] = None,
    ) -> 'EnsembleDecoder':
        """
        Load ensemble from multiple checkpoint files.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            model_class: Class of the model (e.g., SensorMultiHeadDecoder)
            model_kwargs: Kwargs to initialize the model
            device: Device to load models on
            weights: Optional weights for each model

        Returns:
            EnsembleDecoder instance
        """
        models = []

        for path in checkpoint_paths:
            print(f"Loading model from {path}...")
            model = model_class(**model_kwargs)

            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            models.append(model)

        ensemble = cls(models, weights=weights)
        ensemble.to(device)

        print(f"Created ensemble of {len(models)} models")
        return ensemble


class TestTimeAugmentation:
    """
    Apply test-time augmentation (TTA) for more robust predictions.

    Augments input sensor data multiple times and averages predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        n_augmentations: int = 5,
        noise_level: float = 0.01,
        scale_range: tuple = (0.98, 1.02),
    ):
        """
        Initialize TTA.

        Args:
            model: Trained decoder model
            n_augmentations: Number of augmentations to apply
            noise_level: Gaussian noise standard deviation
            scale_range: Range for random scaling (min, max)
        """
        self.model = model
        self.n_augmentations = n_augmentations
        self.noise_level = noise_level
        self.scale_range = scale_range

    def augment_sensors(self, sensor_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to sensor embeddings."""
        # Random Gaussian noise
        noise = torch.randn_like(sensor_embeddings) * self.noise_level
        augmented = sensor_embeddings + noise

        # Random scaling
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        augmented = augmented * scale

        return augmented

    def __call__(
        self,
        tokens: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        operation_type: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply TTA and average predictions.

        Args:
            tokens: Input token sequence [B, L]
            sensor_embeddings: Sensor features [B, T_s, sensor_dim]
            operation_type: Operation type [B]

        Returns:
            Dict with averaged logits
        """
        all_predictions = []

        # Original prediction
        with torch.no_grad():
            outputs = self.model(tokens, sensor_embeddings, operation_type, **kwargs)
            all_predictions.append(outputs)

        # Augmented predictions
        for _ in range(self.n_augmentations):
            aug_sensors = self.augment_sensors(sensor_embeddings)
            with torch.no_grad():
                outputs = self.model(tokens, aug_sensors, operation_type, **kwargs)
                all_predictions.append(outputs)

        # Average predictions
        averaged_outputs = {}
        for key in all_predictions[0].keys():
            if isinstance(all_predictions[0][key], torch.Tensor):
                stacked = torch.stack([o[key] for o in all_predictions], dim=0)
                averaged_outputs[key] = stacked.mean(dim=0)

        return averaged_outputs


def create_diversity_ensemble(
    checkpoint_paths: List[str],
    model_class: type,
    model_kwargs: Dict[str, Any],
    val_loader,
    device: str = 'cpu',
) -> EnsembleDecoder:
    """
    Create ensemble with weights optimized for prediction diversity.

    Models that make different mistakes get higher weights.

    Args:
        checkpoint_paths: List of checkpoint paths
        model_class: Model class
        model_kwargs: Model initialization kwargs
        val_loader: Validation dataloader
        device: Device

    Returns:
        EnsembleDecoder with optimized weights
    """
    # First load all models with equal weights
    ensemble = EnsembleDecoder.from_checkpoints(
        checkpoint_paths, model_class, model_kwargs, device
    )

    # Collect predictions from each model
    all_predictions = [[] for _ in range(len(checkpoint_paths))]
    all_targets = []

    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        sensor_embeddings = batch['sensor_embeddings'].to(device)
        operation_type = batch['operation_type'].to(device)
        targets = batch['targets'].to(device)

        for i, model in enumerate(ensemble.models):
            with torch.no_grad():
                outputs = model(tokens, sensor_embeddings, operation_type)
                preds = outputs['legacy_logits'].argmax(dim=-1)
                all_predictions[i].append(preds.cpu())

        all_targets.append(targets.cpu())

    # Concatenate
    for i in range(len(all_predictions)):
        all_predictions[i] = torch.cat(all_predictions[i], dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute pairwise disagreement
    n_models = len(checkpoint_paths)
    disagreement = torch.zeros(n_models, n_models)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Count positions where models disagree
            disagree_mask = all_predictions[i] != all_predictions[j]
            disagreement[i, j] = disagreement[j, i] = disagree_mask.float().mean()

    # Weight by average disagreement (more disagreement = more diverse)
    avg_disagreement = disagreement.sum(dim=1) / (n_models - 1)

    # Also weight by individual accuracy
    accuracies = []
    for i in range(n_models):
        correct = (all_predictions[i] == all_targets).float().mean()
        accuracies.append(correct.item())

    accuracies = torch.tensor(accuracies)

    # Combined weight: accuracy * diversity
    combined_weights = accuracies * (1 + avg_disagreement)
    combined_weights = combined_weights / combined_weights.sum()

    print("Diversity-optimized weights:")
    for i, (w, acc, div) in enumerate(zip(combined_weights, accuracies, avg_disagreement)):
        print(f"  Model {i}: weight={w:.3f}, acc={acc:.3f}, diversity={div:.3f}")

    # Update ensemble weights
    ensemble.weights = combined_weights.to(device)

    return ensemble
