"""
Data augmentation for G-code fingerprinting.

Implements various augmentation strategies to address class imbalance:
- Sensor noise injection
- Temporal shifting/warping
- Magnitude scaling
- Feature dropout/masking
- Mixup augmentation
- Time warping (DTW-inspired)
- Cutout (temporal masking)
- Token-level mixup (NEW)
- Adversarial perturbations (NEW)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
import random
from scipy.interpolate import interp1d


class DataAugmenter:
    """
    Data augmentation for sensor data and G-code sequences.

    Strategies:
    1. Sensor noise: Add Gaussian noise to continuous features
    2. Temporal shift: Shift sequence by ±N timesteps
    3. Magnitude scaling: Scale sensor values by small factor
    4. Time warping: Non-linear temporal distortion
    5. Feature dropout: Randomly zero out feature dimensions
    6. Cutout: Zero out random time segments
    7. Jitter: Add time-dependent noise
    """

    def __init__(
        self,
        noise_level: float = 0.01,  # Conservative: was 0.02
        shift_range: int = 1,  # Conservative: was 2
        scale_range: tuple = (0.98, 1.02),  # Conservative: was (0.95, 1.05)
        augment_prob: float = 0.3,  # Conservative: was 0.5
        # New augmentation params (conservative defaults)
        time_warp_sigma: float = 0.1,  # Conservative: was 0.2
        feature_dropout_prob: float = 0.05,  # Conservative: was 0.1
        cutout_length: int = 2,  # Conservative: was 3
        jitter_sigma: float = 0.005,  # Conservative: was 0.01
    ):
        """
        Args:
            noise_level: Standard deviation for Gaussian noise (fraction of signal)
            shift_range: Maximum temporal shift in timesteps (±shift_range)
            scale_range: Range for random magnitude scaling (min, max)
            augment_prob: Probability of applying each augmentation
            time_warp_sigma: Strength of time warping distortion
            feature_dropout_prob: Probability of dropping each feature
            cutout_length: Maximum length of cutout window
            jitter_sigma: Standard deviation for time-dependent jitter
        """
        self.noise_level = noise_level
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.augment_prob = augment_prob
        self.time_warp_sigma = time_warp_sigma
        self.feature_dropout_prob = feature_dropout_prob
        self.cutout_length = cutout_length
        self.jitter_sigma = jitter_sigma

    def add_sensor_noise(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to sensor readings.

        Args:
            continuous: [T, D] sensor data

        Returns:
            Augmented sensor data [T, D]
        """
        noise = torch.randn_like(continuous) * self.noise_level
        return continuous + noise

    def temporal_shift(self, continuous: torch.Tensor, shift: Optional[int] = None) -> torch.Tensor:
        """
        Shift sequence by ±N timesteps with padding.

        Args:
            continuous: [T, D] sensor data
            shift: Number of timesteps to shift (if None, random)

        Returns:
            Shifted sensor data [T, D]
        """
        if shift is None:
            shift = np.random.randint(-self.shift_range, self.shift_range + 1)

        if shift == 0:
            return continuous

        T, D = continuous.shape

        if shift > 0:
            # Shift forward: repeat first row at start
            padding = continuous[0:1, :].repeat(shift, 1)  # [shift, D]
            shifted = torch.cat([padding, continuous[:-shift, :]], dim=0)  # [T, D]
            return shifted
        else:
            # Shift backward: repeat last row at end
            abs_shift = abs(shift)
            padding = continuous[-1:, :].repeat(abs_shift, 1)  # [abs_shift, D]
            shifted = torch.cat([continuous[abs_shift:, :], padding], dim=0)  # [T, D]
            return shifted

    def scale_magnitude(self, continuous: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
        """
        Scale sensor magnitudes by a random factor.

        Args:
            continuous: [T, D] sensor data
            scale: Scaling factor (if None, random from scale_range)

        Returns:
            Scaled sensor data [T, D]
        """
        if scale is None:
            scale = np.random.uniform(*self.scale_range)

        return continuous * scale

    def time_warp(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to create non-linear temporal distortions.

        Uses smooth random curves to warp the time axis, simulating
        variations in machining speed.

        Args:
            continuous: [T, D] sensor data

        Returns:
            Time-warped sensor data [T, D]
        """
        T, D = continuous.shape

        if T < 4:  # Too short for warping
            return continuous

        # Create smooth random warp path using cubic spline
        num_knots = max(2, T // 10)
        orig_steps = np.linspace(0, T - 1, num=num_knots)
        random_warps = np.random.normal(loc=1.0, scale=self.time_warp_sigma, size=num_knots)
        random_warps = np.cumsum(random_warps)
        random_warps = random_warps * (T - 1) / random_warps[-1]  # Normalize to [0, T-1]

        # Create interpolation function for warp path
        warp_func = interp1d(orig_steps, random_warps, kind='linear', fill_value='extrapolate')

        # Apply warp to each timestep
        warped_steps = warp_func(np.arange(T))
        warped_steps = np.clip(warped_steps, 0, T - 1)

        # Interpolate continuous data at warped time points
        continuous_np = continuous.numpy()
        warped_data = np.zeros_like(continuous_np)

        for d in range(D):
            interp_func = interp1d(np.arange(T), continuous_np[:, d], kind='linear', fill_value='extrapolate')
            warped_data[:, d] = interp_func(warped_steps)

        return torch.from_numpy(warped_data).float()

    def feature_dropout(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out entire feature dimensions.

        Encourages the model to not rely on any single sensor channel.

        Args:
            continuous: [T, D] sensor data

        Returns:
            Feature-dropped sensor data [T, D]
        """
        T, D = continuous.shape

        # Create dropout mask for features
        mask = torch.ones(D)
        dropout_indices = torch.rand(D) < self.feature_dropout_prob
        mask[dropout_indices] = 0.0

        # Scale remaining features to maintain expected value
        if mask.sum() > 0:
            scale_factor = D / mask.sum()
            mask = mask * scale_factor

        return continuous * mask.unsqueeze(0)

    def cutout(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Zero out a random contiguous time segment (temporal masking).

        Forces the model to learn from partial sequences.

        Args:
            continuous: [T, D] sensor data

        Returns:
            Cutout sensor data [T, D]
        """
        T, D = continuous.shape
        result = continuous.clone()

        # Random cutout length
        cut_len = np.random.randint(1, min(self.cutout_length + 1, T // 2))

        # Random start position
        start = np.random.randint(0, T - cut_len)

        # Zero out the segment
        result[start:start + cut_len, :] = 0.0

        return result

    def jitter(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Add time-dependent jitter noise.

        Different from uniform noise - creates smooth noise patterns
        that simulate sensor drift or calibration variations.

        Args:
            continuous: [T, D] sensor data

        Returns:
            Jittered sensor data [T, D]
        """
        T, D = continuous.shape

        # Create smooth noise by interpolating random points
        num_control = max(2, T // 5)
        control_points = np.random.randn(num_control, D) * self.jitter_sigma

        # Interpolate to full length
        x_control = np.linspace(0, T - 1, num_control)
        x_full = np.arange(T)

        jitter_noise = np.zeros((T, D))
        for d in range(D):
            interp_func = interp1d(x_control, control_points[:, d], kind='cubic', fill_value='extrapolate')
            jitter_noise[:, d] = interp_func(x_full)

        return continuous + torch.from_numpy(jitter_noise).float()

    def permute_segments(self, continuous: torch.Tensor, num_segments: int = 4) -> torch.Tensor:
        """
        Randomly permute temporal segments.

        Breaks temporal dependencies to encourage learning local patterns.

        Args:
            continuous: [T, D] sensor data
            num_segments: Number of segments to create and shuffle

        Returns:
            Segment-permuted sensor data [T, D]
        """
        T, D = continuous.shape

        if T < num_segments:
            return continuous

        # Split into segments
        segment_len = T // num_segments
        segments = []
        for i in range(num_segments):
            start = i * segment_len
            end = start + segment_len if i < num_segments - 1 else T
            segments.append(continuous[start:end, :])

        # Shuffle segments
        random.shuffle(segments)

        return torch.cat(segments, dim=0)

    def adversarial_perturbation(
        self,
        continuous: torch.Tensor,
        epsilon: float = 0.01,
        gradient_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Apply adversarial perturbation to sensor data.

        Uses FGSM-style perturbation if gradient_fn is provided,
        otherwise uses random direction perturbation.

        Args:
            continuous: [T, D] sensor data
            epsilon: Maximum perturbation magnitude
            gradient_fn: Optional function that computes gradients w.r.t. input

        Returns:
            Adversarially perturbed sensor data [T, D]
        """
        if gradient_fn is not None:
            # FGSM-style: perturb in gradient direction
            continuous_grad = continuous.clone().requires_grad_(True)
            grad = gradient_fn(continuous_grad)
            perturbation = epsilon * torch.sign(grad)
        else:
            # Random direction perturbation (when no gradient available)
            # Generate smooth random perturbation
            T, D = continuous.shape
            random_dir = torch.randn_like(continuous)
            # Normalize to unit norm per feature
            norms = random_dir.norm(dim=0, keepdim=True) + 1e-8
            random_dir = random_dir / norms
            perturbation = epsilon * random_dir * continuous.abs().mean()

        return continuous + perturbation

    def augment_sample(self, sample: Dict, strong: bool = False) -> Dict:
        """
        Apply random augmentations to a sample.

        Args:
            sample: Dictionary with sensor features (supports both 'continuous' and 'sensor_features' keys)
            strong: If True, apply more aggressive augmentation

        Returns:
            Augmented sample dictionary
        """
        # Support both key names for sensor data
        sensor_key = 'sensor_features' if 'sensor_features' in sample else 'continuous'
        continuous = sample[sensor_key].clone()

        # Determine augmentation probability (higher for strong mode)
        aug_prob = self.augment_prob * 1.5 if strong else self.augment_prob

        # Basic augmentations (always available)
        if random.random() < aug_prob:
            continuous = self.add_sensor_noise(continuous)

        if random.random() < aug_prob:
            continuous = self.temporal_shift(continuous)

        if random.random() < aug_prob:
            continuous = self.scale_magnitude(continuous)

        # Advanced augmentations (applied with lower probability - conservative)
        if random.random() < aug_prob * 0.25:  # Was 0.5
            try:
                continuous = self.time_warp(continuous)
            except Exception:
                pass  # Skip if time warp fails

        if random.random() < aug_prob * 0.15:  # Was 0.3
            continuous = self.feature_dropout(continuous)

        if random.random() < aug_prob * 0.15:  # Was 0.3
            continuous = self.cutout(continuous)

        if random.random() < aug_prob * 0.2:  # Was 0.4
            continuous = self.jitter(continuous)

        # Strong mode only: segment permutation (can hurt temporal learning)
        if strong and random.random() < 0.2:
            continuous = self.permute_segments(continuous)

        # Adversarial perturbation (very low probability, strong mode only)
        if strong and random.random() < 0.1:
            continuous = self.adversarial_perturbation(continuous, epsilon=0.01)

        # Return augmented sample - preserve all original keys and update sensor data
        result = dict(sample)  # Copy all original keys
        result[sensor_key] = continuous  # Update sensor data with augmented version
        return result


class AugmentedGCodeDataset(torch.utils.data.Dataset):
    """
    Augmented G-code dataset with class-aware oversampling.

    This wrapper:
    1. Identifies sequences with rare tokens (G-commands)
    2. Oversamples these sequences N times
    3. Applies sensor augmentation on-the-fly
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        oversample_rare: bool = True,
        oversample_factor: int = 3,
        rare_token_ids: Optional[list] = None,
        augmenter: Optional[DataAugmenter] = None,
        augment: bool = True,
    ):
        """
        Args:
            base_dataset: Base GCodeDataset to wrap
            oversample_rare: Whether to oversample rare token sequences
            oversample_factor: How many times to repeat rare sequences
            rare_token_ids: List of rare token IDs (G-commands, M-commands)
            augmenter: DataAugmenter instance (creates default if None)
            augment: Whether to apply augmentation
        """
        self.base_dataset = base_dataset
        self.oversample_rare = oversample_rare
        self.oversample_factor = oversample_factor
        self.augment = augment

        # Create augmenter
        if augmenter is None:
            self.augmenter = DataAugmenter()
        else:
            self.augmenter = augmenter

        # Find rare token sequences
        if oversample_rare and rare_token_ids is not None:
            self.indices = self._create_oversampled_indices(rare_token_ids)
        else:
            # No oversampling: 1:1 mapping
            self.indices = list(range(len(base_dataset)))

    def _create_oversampled_indices(self, rare_token_ids: list) -> list:
        """
        Create index mapping with oversampling for rare sequences.

        Args:
            rare_token_ids: List of rare token IDs

        Returns:
            List of indices (with repetitions for rare sequences)
        """
        rare_token_set = set(rare_token_ids)
        rare_indices = []

        # Scan dataset for rare tokens
        for i in range(len(self.base_dataset)):
            tokens = self.base_dataset.tokens[i]
            # Check if any rare token appears
            if any(int(tok) in rare_token_set for tok in tokens if tok != 0):
                rare_indices.append(i)

        print(f"Found {len(rare_indices)} sequences with rare tokens")

        # Create oversampled index list
        indices = list(range(len(self.base_dataset)))

        # Add rare sequences multiple times
        for _ in range(self.oversample_factor - 1):
            indices.extend(rare_indices)

        print(f"Dataset size: {len(self.base_dataset)} → {len(indices)} (with {self.oversample_factor}x oversampling)")

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        # Get real index from oversampled index
        real_idx = self.indices[idx]

        # Get base sample
        sample = self.base_dataset[real_idx]

        # Apply augmentation if enabled
        if self.augment:
            sample = self.augmenter.augment_sample(sample)

        return sample


class TokenMixup:
    """
    Token-level mixup for similar operations.

    Mixes sensor embeddings between samples of the same operation type
    to create synthetic training examples.
    """

    # Define which operations are similar and can be mixed
    SIMILAR_OPS = {
        # Normal operations
        0: [0, 1],       # adaptive <-> adaptive150025
        1: [0, 1],
        2: [2, 3],       # face <-> face150025
        3: [2, 3],
        4: [4, 5],       # pocket <-> pocket150025
        5: [4, 5],
        # Damage operations (can mix among themselves)
        6: [6, 7, 8],    # damageadaptive <-> damageface <-> damagepocket
        7: [6, 7, 8],
        8: [6, 7, 8],
    }

    def __init__(
        self,
        alpha: float = 0.2,
        mixup_prob: float = 0.3,
        same_op_only: bool = True,
    ):
        """
        Args:
            alpha: Beta distribution parameter for mixup ratio
            mixup_prob: Probability of applying mixup
            same_op_only: If True, only mix within same operation type
        """
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.same_op_only = same_op_only

    def mixup_batch(
        self,
        sensor_features: torch.Tensor,
        tokens: torch.Tensor,
        operation_types: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Apply mixup to a batch of samples.

        Args:
            sensor_features: [B, T, D] sensor embeddings
            tokens: [B, L] input tokens
            operation_types: [B] operation type indices
            targets: Optional target dictionary

        Returns:
            Mixed sensor features, tokens, labels, and optionally mixed targets
        """
        if random.random() > self.mixup_prob:
            return sensor_features, tokens, operation_types, targets

        B = sensor_features.size(0)
        device = sensor_features.device

        # Sample mixup coefficient from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Find mixup partners based on operation similarity
        mix_indices = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            op_type = operation_types[i].item()
            if self.same_op_only:
                # Find samples with same operation type
                same_op_mask = (operation_types == op_type)
                candidates = same_op_mask.nonzero(as_tuple=True)[0]
            else:
                # Find samples with similar operation type
                similar_ops = self.SIMILAR_OPS.get(op_type, [op_type])
                similar_mask = torch.zeros(B, dtype=torch.bool, device=device)
                for sim_op in similar_ops:
                    similar_mask |= (operation_types == sim_op)
                candidates = similar_mask.nonzero(as_tuple=True)[0]

            if len(candidates) > 1:
                # Remove self from candidates
                candidates = candidates[candidates != i]
                if len(candidates) > 0:
                    mix_indices[i] = candidates[torch.randint(len(candidates), (1,))]
                else:
                    mix_indices[i] = i
            else:
                mix_indices[i] = i

        # Mix sensor features
        mixed_sensors = lam * sensor_features + (1 - lam) * sensor_features[mix_indices]

        # For tokens and targets, we use the original (hard labels)
        # The mixup is only applied to continuous sensor features

        return mixed_sensors, tokens, operation_types, targets


class TemperatureSampler:
    """
    Temperature-based batch sampler for upweighting rare operations.

    Uses softmax temperature to control the sampling distribution:
    - Lower temperature -> more uniform sampling
    - Higher temperature -> more frequency-based sampling
    """

    def __init__(
        self,
        labels: List[int],
        temperature: float = 0.5,
        base_weight: float = 1.0,
    ):
        """
        Args:
            labels: List of operation type labels for all samples
            temperature: Sampling temperature (lower = more uniform)
            base_weight: Minimum weight for any class
        """
        self.labels = np.array(labels)
        self.temperature = temperature
        self.base_weight = base_weight

        # Compute class frequencies
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))

        # Compute sampling weights using temperature
        self._compute_weights()

    def _compute_weights(self):
        """Compute sample weights based on inverse frequency with temperature."""
        n_samples = len(self.labels)
        max_count = max(self.class_counts.values())

        # Inverse frequency with temperature scaling
        class_weights = {}
        for cls, count in self.class_counts.items():
            # Apply temperature: higher temp = closer to original freq
            # Lower temp = more uniform
            weight = (max_count / count) ** (1.0 / self.temperature)
            class_weights[cls] = max(weight, self.base_weight)

        # Assign weights to samples
        self.weights = np.array([class_weights[label] for label in self.labels])

        # Normalize
        self.weights = self.weights / self.weights.sum() * n_samples

    def get_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """Get a WeightedRandomSampler with computed weights."""
        return torch.utils.data.WeightedRandomSampler(
            weights=self.weights.tolist(),
            num_samples=len(self.labels),
            replacement=True,
        )

    def get_batch_weights(self, batch_labels: torch.Tensor) -> torch.Tensor:
        """
        Get per-sample weights for a batch.

        Useful for weighted loss computation within a batch.

        Args:
            batch_labels: [B] operation type indices

        Returns:
            [B] weights for each sample
        """
        device = batch_labels.device
        weights = torch.tensor(
            [self.weights[self.labels == label.item()].mean()
             for label in batch_labels],
            device=device,
        )
        return weights


def get_rare_token_ids(vocab_path: str) -> list:
    """
    Get list of rare token IDs (G-commands, M-commands) from vocabulary.

    Args:
        vocab_path: Path to vocabulary JSON file

    Returns:
        List of rare token IDs
    """
    import json
    from pathlib import Path

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))

    rare_tokens = []
    for token, token_id in vocab.items():
        # G-commands: G0, G1, G2, G3, G90, etc.
        if isinstance(token, str) and token.startswith('G') and token[1:].isdigit():
            rare_tokens.append(token_id)
        # M-commands: M3, M5, etc.
        elif isinstance(token, str) and token.startswith('M') and token[1:].isdigit():
            rare_tokens.append(token_id)

    print(f"Found {len(rare_tokens)} rare token IDs (G/M commands)")
    return rare_tokens


# Example usage
if __name__ == '__main__':
    from pathlib import Path
    import sys

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from miracle.dataset.dataset import GCodeDataset

    # Load dataset
    data_dir = Path('outputs/processed_v2')
    train_data = GCodeDataset(data_dir / 'train_sequences.npz')

    # Get rare token IDs
    rare_ids = get_rare_token_ids('data/gcode_vocab_v2.json')

    # Create augmented dataset
    aug_dataset = AugmentedGCodeDataset(
        train_data,
        oversample_rare=True,
        oversample_factor=3,
        rare_token_ids=rare_ids,
        augment=True
    )

    print(f"\nOriginal dataset: {len(train_data)} samples")
    print(f"Augmented dataset: {len(aug_dataset)} samples")

    # Test augmentation
    sample = aug_dataset[0]
    print(f"\nSample shape:")
    print(f"  continuous: {sample['continuous'].shape}")
    print(f"  tokens: {sample['tokens'].shape}")
    print(f"  length: {sample['length']}")
