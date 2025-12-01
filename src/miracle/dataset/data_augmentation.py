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
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
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

    def augment_sample(self, sample: Dict, strong: bool = False) -> Dict:
        """
        Apply random augmentations to a sample.

        Args:
            sample: Dictionary with 'continuous', 'categorical', 'tokens', 'length'
            strong: If True, apply more aggressive augmentation

        Returns:
            Augmented sample dictionary
        """
        continuous = sample['continuous'].clone()

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

        # Return augmented sample (tokens and categorical unchanged)
        return {
            'continuous': continuous,
            'categorical': sample['categorical'],
            'tokens': sample['tokens'],
            'residuals': sample.get('residuals', torch.zeros(1)),  # Pass through residuals
            'param_value_raw': sample.get('param_value_raw', torch.zeros(1)),  # Pass through param_value_raw
            'length': sample['length'],
            'gcode_text': sample.get('gcode_text', ''),
            'operation_type': sample.get('operation_type', 0)  # Pass through operation_type
        }


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
