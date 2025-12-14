"""
Loss functions for G-code prediction training.

Includes:
- Cross-entropy loss for token prediction
- Reconstruction loss for sensor data
- Contrastive loss for fingerprinting
- Combined multi-task losses
- Digit-by-digit loss for numeric value prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

__all__ = [
    "GCodeLoss",
    "ReconstructionLoss",
    "ContrastiveLoss",
    "MultiTaskLoss",
    "LabelSmoothingCrossEntropy",
    "MultiHeadGCodeLoss",
    "MultiHeadFocalLoss",
    "FocalLoss",
    "DigitLoss",
]


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] or [B, T, C]
            targets: [B] or [B, T]
        """
        # Flatten if needed
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        else:
            C = logits.size(-1)

        # Mask for valid targets
        mask = targets != self.ignore_index

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot encoding with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (C - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[~mask] = 0.0

        # KL divergence
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss[mask].mean()

        return loss


class GCodeLoss(nn.Module):
    """Loss for G-code token prediction with class imbalance handling."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss

        if use_focal_loss:
            # Use focal loss for severe class imbalance
            self.criterion = FocalLoss(
                alpha=0.25,
                gamma=focal_gamma,
                ignore_index=pad_token_id
            )
        elif label_smoothing > 0:
            # Label smoothing doesn't support class weights, so use standard CE if weights provided
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing
                )
            else:
                self.criterion = LabelSmoothingCrossEntropy(
                    smoothing=label_smoothing,
                    ignore_index=pad_token_id
                )
        else:
            # Standard cross-entropy with optional class weights
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=pad_token_id
            )

    def forward(
        self,
        logits: torch.Tensor,  # [B, T, V]
        targets: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """Compute cross-entropy loss for token prediction."""
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = self.criterion(logits_flat, targets_flat)

        return loss


class ReconstructionLoss(nn.Module):
    """Loss for sensor data reconstruction."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        reconstructed: torch.Tensor,  # [B, T, D]
        original: torch.Tensor,        # [B, T, D]
        mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        if mask is not None:
            # Apply mask (True = PAD positions to ignore)
            mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
            reconstructed = reconstructed.masked_fill(mask_expanded, 0.0)
            original = original.masked_fill(mask_expanded, 0.0)

            # Compute loss only on valid positions
            n_valid = (~mask).sum() * reconstructed.size(-1)
            loss = self.criterion(reconstructed, original) * (reconstructed.numel() / n_valid)
        else:
            loss = self.criterion(reconstructed, original)

        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for fingerprint embeddings."""

    def __init__(self, temperature: float = 0.07, similarity: str = "cosine"):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, D]
        labels: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        """
        Compute contrastive loss (NT-Xent / SimCLR style).

        Embeddings from same class should be similar,
        embeddings from different classes should be dissimilar.
        """
        B = embeddings.size(0)
        device = embeddings.device

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        if self.similarity == "cosine":
            sim_matrix = embeddings @ embeddings.T  # [B, B]
        else:
            # Euclidean distance
            dist_matrix = torch.cdist(embeddings, embeddings)
            sim_matrix = -dist_matrix

        sim_matrix = sim_matrix / self.temperature

        # Create label mask (same class = positive pairs)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        label_matrix.fill_diagonal_(False)  # Exclude self-comparisons

        # For each sample, positive samples are those with same label
        positives = label_matrix.float()
        negatives = (~label_matrix).float()

        # Mask out diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix * mask.float() + (~mask).float() * (-1e9)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)

        # Sum over positives in numerator, all negatives in denominator
        pos_sum = (exp_sim * positives).sum(dim=1)
        neg_sum = (exp_sim * negatives).sum(dim=1)

        # Loss = -log(sum_pos / (sum_pos + sum_neg))
        loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))
        loss = loss.mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for the full model.

    Combines:
    - G-code token prediction loss (with optional class weights and focal loss)
    - Reconstruction loss
    - Contrastive fingerprint loss
    - Classification loss (optional)

    Args:
        vocab_size: Size of the G-code vocabulary
        pad_token_id: Token ID to ignore in loss computation
        task_weights: Manual weights for each task
        adaptive: Use uncertainty-based adaptive task weighting
        class_weights: Optional weights for handling class imbalance in G-code prediction
        use_focal_loss: Use focal loss instead of cross-entropy for G-code prediction
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor for G-code prediction (0.0-1.0)
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        task_weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        # Loss functions
        self.gcode_loss = GCodeLoss(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
        self.recon_loss = ReconstructionLoss(loss_type="mse")
        self.contrast_loss = ContrastiveLoss()
        self.cls_loss = nn.CrossEntropyLoss()

        # Task weights
        if task_weights is None:
            task_weights = {
                "gcode": 1.0,
                "recon": 0.5,
                "contrast": 0.3,
                "cls": 0.5,
            }
        self.task_weights = task_weights

        # Adaptive weighting (uncertainty-based)
        self.adaptive = adaptive
        if adaptive:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1))
                for k in task_weights.keys()
            })

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model_output: Dictionary with model predictions
            targets: Dictionary with ground truth

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        losses = {}

        # G-code token prediction loss
        if "gcode_logits" in model_output and "gcode_tokens" in targets:
            # CRITICAL: Handle teacher forcing alignment
            # Model predicts NEXT token, so logits is [B, T-1, V]
            # Need to align targets by shifting: tokens[:, 1:] -> [B, T-1]
            logits = model_output["gcode_logits"]
            token_targets = targets["gcode_tokens"]

            # If shapes don't match, assume teacher forcing and shift targets
            if logits.size(1) == token_targets.size(1) - 1:
                token_targets = token_targets[:, 1:]  # Shift to next tokens

            losses["gcode"] = self.gcode_loss(
                logits,
                token_targets
            )

        # Reconstruction loss
        if "recon" in model_output and "sensor_data" in targets:
            pad_mask = targets.get("pad_mask", None)
            losses["recon"] = self.recon_loss(
                model_output["recon"],
                targets["sensor_data"],
                mask=pad_mask
            )

        # Contrastive loss
        if "fingerprint" in model_output and "labels" in targets:
            losses["contrast"] = self.contrast_loss(
                model_output["fingerprint"],
                targets["labels"]
            )

        # Classification loss
        if "cls" in model_output and "labels" in targets:
            losses["cls"] = self.cls_loss(
                model_output["cls"],
                targets["labels"]
            )

        # Combine losses
        if self.adaptive:
            # Adaptive uncertainty-based weighting
            total_loss = 0.0
            for key, loss_val in losses.items():
                if key in self.log_vars:
                    precision = torch.exp(-self.log_vars[key])
                    total_loss = total_loss + precision * loss_val + self.log_vars[key]
        else:
            # Fixed weighting
            total_loss = sum(
                self.task_weights.get(k, 1.0) * v
                for k, v in losses.items()
            )

        # Convert to dict for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] or [B, T, C]
            targets: [B] or [B, T]
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Flatten if needed
        if logits.dim() == 3:
            B, T, C = logits.shape
            probs = probs.reshape(-1, C)
            targets = targets.reshape(-1)

        # Gather target probabilities
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
            ignore_index=self.ignore_index
        )

        # Focal loss
        loss = self.alpha * focal_weight * ce_loss
        loss = loss.mean()

        return loss


class PerDigitFocalLoss(nn.Module):
    """Focal loss for single digit position (10 classes)."""

    def __init__(self, gamma: float = 5.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, 10] unnormalized scores
            targets: [N] digit class (0-9)
        """
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # High gamma focal weight - focuses strongly on hard examples
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal loss
        loss = focal_weight * ce_loss
        return loss.mean()


class DigitLoss(nn.Module):
    """
    Loss for digit-by-digit numeric value prediction.

    Combines:
    1. Cross-entropy for sign prediction (3 classes: +, -, zero)
    2. Per-digit focal loss for each digit position (10 classes: 0-9)
    3. Optional auxiliary MSE loss for numerical guidance
    """

    def __init__(
        self,
        n_digit_positions: int = 6,
        aux_loss_weight: float = 0.1,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = True,
        focal_gamma: float = 5.0,
    ):
        """
        Args:
            n_digit_positions: Number of digit positions (int + decimal)
            aux_loss_weight: Weight for auxiliary regression loss
            label_smoothing: Label smoothing for digit CE loss (ignored if focal)
            use_focal_loss: Use per-digit focal loss (recommended for class imbalance)
            focal_gamma: Gamma for focal loss (higher = more focus on hard examples)
        """
        super().__init__()
        self.n_digit_positions = n_digit_positions
        self.aux_loss_weight = aux_loss_weight
        self.use_focal_loss = use_focal_loss

        self.sign_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        if use_focal_loss:
            # Per-digit focal loss with high gamma to combat class imbalance
            self.digit_loss_fns = nn.ModuleList([
                PerDigitFocalLoss(gamma=focal_gamma, ignore_index=-1)
                for _ in range(n_digit_positions)
            ])
        else:
            self.digit_loss_fn = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=label_smoothing,
            )
        self.aux_loss_fn = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        numeric_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute digit-by-digit loss.

        Args:
            logits: Dict with:
                - 'digit_sign_logits': [B, T, 3]
                - 'digit_logits': [B, T, n_positions, 10]
                - 'digit_aux_value': [B, T, 1]
            targets: Dict with:
                - 'sign_targets': [B, T] (0=+, 1=-, 2=zero)
                - 'digit_targets': [B, T, n_positions] (0-9)
                - 'param_value_raw': [B, T] raw numeric values
            numeric_mask: [B, T] True for NUMERIC token positions

        Returns:
            total_loss: Combined digit loss
            loss_dict: Individual loss components
        """
        device = logits['digit_sign_logits'].device
        loss_dict = {}

        if not numeric_mask.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {'sign_loss': 0.0, 'digit_loss': 0.0, 'aux_loss': 0.0, 'total': 0.0}

        # 1. Sign loss
        sign_logits = logits['digit_sign_logits']  # [B, T, 3]
        sign_targets = targets.get('sign_targets', torch.zeros_like(numeric_mask, dtype=torch.long))

        sign_logits_flat = sign_logits[numeric_mask]  # [N, 3]
        sign_targets_flat = sign_targets[numeric_mask]  # [N]
        sign_loss = self.sign_loss_fn(sign_logits_flat, sign_targets_flat)
        loss_dict['sign_loss'] = sign_loss.item()

        # 2. Digit losses (per-position focal loss or standard CE)
        digit_logits = logits['digit_logits']  # [B, T, n_positions, 10]
        digit_targets = targets.get('digit_targets',
                                     torch.zeros((*numeric_mask.shape, self.n_digit_positions),
                                                dtype=torch.long, device=device))

        digit_loss = torch.tensor(0.0, device=device)
        n_positions = digit_logits.size(2)

        for pos in range(n_positions):
            pos_logits = digit_logits[:, :, pos, :][numeric_mask]  # [N, 10]
            pos_targets = digit_targets[:, :, pos][numeric_mask]  # [N]

            if self.use_focal_loss:
                pos_loss = self.digit_loss_fns[pos](pos_logits, pos_targets)
            else:
                pos_loss = self.digit_loss_fn(pos_logits, pos_targets)

            digit_loss = digit_loss + pos_loss
            loss_dict[f'digit_{pos}_loss'] = pos_loss.item()

        digit_loss = digit_loss / n_positions
        loss_dict['digit_loss'] = digit_loss.item()

        # 3. Auxiliary regression loss
        if 'digit_aux_value' in logits and 'param_value_raw' in targets:
            aux_value = logits['digit_aux_value'].squeeze(-1)  # [B, T]
            value_targets = targets['param_value_raw']  # [B, T]

            aux_value_flat = aux_value[numeric_mask]  # [N]
            value_targets_flat = value_targets[numeric_mask]  # [N]
            aux_loss = self.aux_loss_fn(aux_value_flat, value_targets_flat)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        loss_dict['aux_loss'] = aux_loss.item()

        # Total loss
        total_loss = sign_loss + digit_loss + self.aux_loss_weight * aux_loss
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class MultiHeadFocalLoss(nn.Module):
    """Focal loss for multi-head classification with per-class alpha weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Per-class weights [num_classes] (e.g., inverse frequency)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] unnormalized scores
            targets: [N] class indices
        """
        # Filter out ignored indices
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Per-class alpha weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal loss
        loss = focal_weight * ce_loss
        return loss.mean()


class MultiHeadGCodeLoss(nn.Module):
    """
    Loss for multi-head G-code prediction.

    Computes separate cross-entropy losses for each prediction head:
    1. Token type (SPECIAL/COMMAND/PARAMETER/NUMERIC)
    2. Command ID (for commands)
    3. Parameter type ID (for parameters and numeric values)
    4. Parameter value - supports two modes:
       a. Direct regression (legacy)
       b. Digit-by-digit prediction (recommended)

    Loss is weighted combination of all heads, with masking to only
    compute losses for relevant heads based on token type.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        type_weight: float = 1.0,
        command_weight: float = 3.0,  # Higher weight for commands (rarer)
        param_type_weight: float = 2.0,
        param_value_weight: float = 1.0,
        operation_weight: float = 2.0,  # Weight for operation type classification
        label_smoothing: float = 0.0,
        command_class_weights: Optional[torch.Tensor] = None,  # Per-class weights for commands
        param_type_class_weights: Optional[torch.Tensor] = None,
        param_value_class_weights: Optional[torch.Tensor] = None,
        operation_class_weights: Optional[torch.Tensor] = None,  # Per-class weights for operation type
        use_focal_loss: bool = False,  # Enable focal loss
        focal_gamma: float = 2.0,  # Focal loss gamma parameter
        use_digit_loss: bool = False,  # Use digit-by-digit loss instead of regression
        n_digit_positions: int = 6,  # Number of digit positions
        digit_aux_weight: float = 0.1,  # Weight for auxiliary regression in digit loss
    ):
        """
        Args:
            pad_token_id: Token ID to ignore in loss computation
            type_weight: Weight for token type prediction
            command_weight: Weight for command prediction (higher = more emphasis)
            param_type_weight: Weight for parameter type prediction
            param_value_weight: Weight for parameter value prediction
            operation_weight: Weight for operation type classification
            label_smoothing: Label smoothing factor
            command_class_weights: Per-class weights for command head (e.g., inverse frequency)
            param_type_class_weights: Per-class weights for parameter type head
            param_value_class_weights: Per-class weights for parameter value head
            operation_class_weights: Per-class weights for operation type head (prevents mode collapse)
            use_focal_loss: If True, use focal loss instead of cross-entropy
            focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
            use_digit_loss: If True, use digit-by-digit loss for numeric values
            n_digit_positions: Number of digit positions (for digit loss)
            digit_aux_weight: Weight for auxiliary regression in digit loss
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.type_weight = type_weight
        self.command_weight = command_weight
        self.param_type_weight = param_type_weight
        self.param_value_weight = param_value_weight
        self.operation_weight = operation_weight
        self.use_focal_loss = use_focal_loss
        self.use_digit_loss = use_digit_loss

        # Digit loss (for digit-by-digit prediction)
        if use_digit_loss:
            self.digit_loss_fn = DigitLoss(
                n_digit_positions=n_digit_positions,
                aux_loss_weight=digit_aux_weight,
                label_smoothing=label_smoothing,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
            )
        else:
            self.digit_loss_fn = None

        # Loss functions for each head
        if use_focal_loss:
            # FOCAL LOSS MODE: Better for class imbalance
            self.type_loss_fn = MultiHeadFocalLoss(
                gamma=focal_gamma, alpha=None, ignore_index=-1
            )
            self.command_loss_fn = MultiHeadFocalLoss(
                gamma=focal_gamma, alpha=command_class_weights, ignore_index=-1
            )
            self.param_type_loss_fn = MultiHeadFocalLoss(
                gamma=focal_gamma, alpha=param_type_class_weights, ignore_index=-1
            )
            self.param_value_loss_fn = MultiHeadFocalLoss(
                gamma=focal_gamma, alpha=param_value_class_weights, ignore_index=-1
            )
            self.operation_loss_fn = MultiHeadFocalLoss(
                gamma=focal_gamma, alpha=operation_class_weights, ignore_index=-1
            )
        elif label_smoothing > 0:
            self.type_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, ignore_index=-1
            )
            self.command_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, ignore_index=-1
            )
            self.param_type_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, ignore_index=-1
            )
            self.param_value_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, ignore_index=-1
            )
            # Operation loss doesn't use label smoothing (small number of classes)
            # Use per-class weights to prevent mode collapse on minority classes
            self.operation_loss_fn = nn.CrossEntropyLoss(weight=operation_class_weights, ignore_index=-1)
        else:
            self.type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            self.command_loss_fn = nn.CrossEntropyLoss(
                weight=command_class_weights, ignore_index=-1
            )
            self.param_type_loss_fn = nn.CrossEntropyLoss(
                weight=param_type_class_weights, ignore_index=-1
            )
            self.param_value_loss_fn = nn.CrossEntropyLoss(
                weight=param_value_class_weights, ignore_index=-1
            )
            # Use per-class weights to prevent mode collapse on minority classes
            self.operation_loss_fn = nn.CrossEntropyLoss(weight=operation_class_weights, ignore_index=-1)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-head loss.

        Args:
            logits: Dictionary with:
                - 'type_logits': [B, T, 4]
                - 'command_logits': [B, T, n_commands]
                - 'param_type_logits': [B, T, n_param_types]
                - 'param_value_logits': [B, T, n_param_values]
            targets: Dictionary with:
                - 'type': [B, T] token types (0-3)
                - 'command_id': [B, T] command IDs
                - 'param_type_id': [B, T] parameter type IDs
                - 'param_value_id': [B, T] parameter value IDs

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        B, T = targets['type'].shape

        # Create padding mask (PAD tokens have type=0)
        pad_mask = targets['type'] == 0  # [B, T]

        # 1. Token type loss (always computed)
        type_logits = logits['type_logits'].reshape(-1, 4)
        type_targets = targets['type'].reshape(-1)
        # Mask PAD positions with -1 (will be ignored by loss)
        type_targets = type_targets.masked_fill(pad_mask.reshape(-1), -1)
        type_loss = self.type_loss_fn(type_logits, type_targets)

        # 2. Command loss (only for COMMAND tokens, type=1)
        command_mask = (targets['type'] == 1) & (~pad_mask)  # [B, T]
        command_logits = logits['command_logits'].reshape(-1, logits['command_logits'].size(-1))
        command_targets = targets['command_id'].reshape(-1)
        # Mask non-command positions with -1
        command_targets = command_targets.masked_fill(~command_mask.reshape(-1), -1)

        if command_mask.any():
            command_loss = self.command_loss_fn(command_logits, command_targets)
        else:
            command_loss = torch.tensor(0.0, device=type_loss.device)

        # 3. Parameter type loss (for PARAMETER and NUMERIC tokens, type=2 or 3)
        param_type_mask = ((targets['type'] == 2) | (targets['type'] == 3)) & (~pad_mask)  # [B, T]
        param_type_logits = logits['param_type_logits'].reshape(-1, logits['param_type_logits'].size(-1))
        param_type_targets = targets['param_type_id'].reshape(-1)
        # Mask non-parameter positions with -1
        param_type_targets = param_type_targets.masked_fill(~param_type_mask.reshape(-1), -1)

        if param_type_mask.any():
            param_type_loss = self.param_type_loss_fn(param_type_logits, param_type_targets)
        else:
            param_type_loss = torch.tensor(0.0, device=type_loss.device)

        # 4. Parameter value loss (only for NUMERIC tokens, type=3)
        param_value_mask = (targets['type'] == 3) & (~pad_mask)  # [B, T]

        # Initialize digit loss dict
        digit_loss_dict = {}

        # Check for digit-by-digit prediction first (recommended mode)
        if self.use_digit_loss and 'digit_sign_logits' in logits and 'digit_logits' in logits:
            # DIGIT MODE: Predict each digit separately (eliminates mode collapse)
            digit_loss, digit_loss_dict = self.digit_loss_fn(logits, targets, param_value_mask)
            param_value_loss = digit_loss

        elif 'param_value_regression' in logits:
            # REGRESSION MODE: Direct continuous value prediction
            param_value_regression_pred = logits['param_value_regression'].squeeze(-1)  # [B, T]

            # Get regression targets (raw numeric values)
            if 'param_value_raw' in targets:
                param_value_regression_targets = targets['param_value_raw']  # [B, T]
            else:
                # Fallback: use zeros (should not happen in production)
                param_value_regression_targets = torch.zeros_like(param_value_regression_pred)

            # Only compute regression loss on NUMERIC tokens
            if param_value_mask.any():
                # Flatten and mask
                regression_pred_flat = param_value_regression_pred.reshape(-1)
                regression_tgt_flat = param_value_regression_targets.reshape(-1)
                regression_mask_flat = param_value_mask.reshape(-1)

                # Huber loss only on valid positions (robust to outliers)
                param_value_loss = F.huber_loss(
                    regression_pred_flat[regression_mask_flat],
                    regression_tgt_flat[regression_mask_flat],
                    delta=10.0  # Higher tolerance for G-code values
                )
            else:
                param_value_loss = torch.tensor(0.0, device=type_loss.device)

        elif 'param_value_logits' in logits:
            # LEGACY MODE: Standard bucketing (single param_value_logits)
            param_value_logits = logits['param_value_logits'].reshape(-1, logits['param_value_logits'].size(-1))
            param_value_targets = targets['param_value_id'].reshape(-1)
            # Mask non-numeric positions with -1
            param_value_targets = param_value_targets.masked_fill(~param_value_mask.reshape(-1), -1)

            if param_value_mask.any():
                param_value_loss = self.param_value_loss_fn(param_value_logits, param_value_targets)
            else:
                param_value_loss = torch.tensor(0.0, device=type_loss.device)
        else:
            # No param_value prediction (should not happen)
            param_value_loss = torch.tensor(0.0, device=type_loss.device)

        # 5. Operation type loss (per-sequence classification)
        # Only computed if operation_logits and operation_type targets are present
        operation_loss = torch.tensor(0.0, device=type_loss.device)
        if 'operation_logits' in logits and 'operation_type' in targets:
            operation_logits = logits['operation_logits']  # [B, n_operation_types]
            operation_targets = targets['operation_type']  # [B]
            operation_loss = self.operation_loss_fn(operation_logits, operation_targets)

        # Combine losses with weights
        total_loss = (
            self.type_weight * type_loss +
            self.command_weight * command_loss +
            self.param_type_weight * param_type_loss +
            self.param_value_weight * param_value_loss +
            self.operation_weight * operation_loss
        )

        # Loss dictionary for logging
        loss_dict = {
            'type': type_loss.item(),
            'command': command_loss.item(),
            'param_type': param_type_loss.item(),
            'param_value': param_value_loss.item(),
            'operation': operation_loss.item(),
            'total': total_loss.item(),
        }

        # Add digit-specific losses if using digit mode
        if digit_loss_dict:
            for key, value in digit_loss_dict.items():
                loss_dict[f'digit_{key}'] = value

        return total_loss, loss_dict
