"""
Unit tests for data augmentation module.

Tests cover:
- DataAugmenter class (noise, shift, scaling)
- AugmentedGCodeDataset class (oversampling)
- get_rare_token_ids utility function
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.dataset.data_augmentation import (
    DataAugmenter,
    AugmentedGCodeDataset,
    get_rare_token_ids
)


@pytest.mark.unit
class TestDataAugmenter:
    """Test suite for DataAugmenter class."""

    def test_sensor_noise_preserves_shape(self):
        """Test that noise injection preserves tensor shape."""
        augmenter = DataAugmenter(noise_level=0.02)
        continuous = torch.randn(64, 135)  # [T, D]

        augmented = augmenter.add_sensor_noise(continuous)

        assert augmented.shape == continuous.shape, "Noise should preserve shape"

    def test_sensor_noise_adds_gaussian_noise(self):
        """Test that noise injection adds Gaussian noise with correct statistics."""
        torch.manual_seed(42)
        augmenter = DataAugmenter(noise_level=0.02)
        continuous = torch.ones(1000, 135) * 10.0  # Constant signal

        augmented = augmenter.add_sensor_noise(continuous)
        noise = augmented - continuous

        # Check noise statistics
        assert abs(noise.mean().item()) < 0.01, "Noise should have mean ~0"
        assert abs(noise.std().item() - 0.02) < 0.01, "Noise should have std ~0.02"

    def test_temporal_shift_preserves_shape(self):
        """Test that temporal shifting preserves sequence length."""
        augmenter = DataAugmenter(shift_range=2)
        continuous = torch.randn(64, 135)

        shifted = augmenter.temporal_shift(continuous, shift=2)

        assert shifted.shape == continuous.shape, "Shift should preserve shape"

    def test_temporal_shift_forward(self):
        """Test forward (positive) temporal shift."""
        augmenter = DataAugmenter(shift_range=5)
        continuous = torch.arange(10).unsqueeze(1).float()  # [[0], [1], ..., [9]]

        shifted = augmenter.temporal_shift(continuous, shift=3)

        # After shift=3, first 3 elements should be 0 (padding)
        assert torch.all(shifted[:3] == 0.0), "First 3 should be padded with first value (0)"
        # Remaining should be shifted
        assert torch.all(shifted[3:] == continuous[:-3]), "Rest should be shifted forward"

    def test_temporal_shift_backward(self):
        """Test backward (negative) temporal shift."""
        augmenter = DataAugmenter(shift_range=5)
        continuous = torch.arange(10).unsqueeze(1).float()

        shifted = augmenter.temporal_shift(continuous, shift=-3)

        # After shift=-3, elements should be shifted backward
        assert torch.all(shifted[:-3] == continuous[3:]), "Should shift backward"
        # Last 3 should be padded with last value (9)
        assert torch.all(shifted[-3:] == 9.0), "Last 3 should be padded with last value"

    def test_temporal_shift_zero(self):
        """Test that shift=0 returns identical tensor."""
        augmenter = DataAugmenter(shift_range=2)
        continuous = torch.randn(64, 135)

        shifted = augmenter.temporal_shift(continuous, shift=0)

        assert torch.allclose(shifted, continuous), "Shift=0 should return unchanged tensor"

    def test_temporal_shift_range(self):
        """Test that random shift is within specified range."""
        augmenter = DataAugmenter(shift_range=2)
        continuous = torch.randn(64, 135)

        # Test multiple random shifts
        shifts_observed = []
        for _ in range(50):
            # We can't directly observe the shift, but we can test it doesn't crash
            # and preserves shape
            shifted = augmenter.temporal_shift(continuous, shift=None)
            assert shifted.shape == continuous.shape

    def test_magnitude_scaling_range(self):
        """Test that scaling factor is within specified range."""
        augmenter = DataAugmenter(scale_range=(0.95, 1.05))
        continuous = torch.ones(64, 135) * 100.0

        # Test multiple random scalings
        for _ in range(20):
            scaled = augmenter.scale_magnitude(continuous, scale=None)
            ratio = (scaled / continuous).mean().item()
            assert 0.95 <= ratio <= 1.05, f"Scale {ratio} should be in [0.95, 1.05]"

    def test_magnitude_scaling_preserves_shape(self):
        """Test that magnitude scaling preserves tensor shape."""
        augmenter = DataAugmenter(scale_range=(0.9, 1.1))
        continuous = torch.randn(64, 135)

        scaled = augmenter.scale_magnitude(continuous, scale=1.05)

        assert scaled.shape == continuous.shape, "Scaling should preserve shape"

    def test_magnitude_scaling_factor(self):
        """Test that scaling applies correct factor."""
        augmenter = DataAugmenter()
        continuous = torch.ones(64, 135) * 10.0

        scaled = augmenter.scale_magnitude(continuous, scale=1.2)

        expected = continuous * 1.2
        assert torch.allclose(scaled, expected), "Scaling should multiply by factor"

    def test_augment_sample_preserves_labels(self):
        """Test that augmentation preserves G-code labels and categorical features."""
        augmenter = DataAugmenter(augment_prob=1.0)  # Always augment
        sample = {
            'continuous': torch.randn(64, 135),
            'categorical': torch.randint(0, 5, (64, 4)),
            'tokens': torch.randint(0, 170, (64,)),
            'length': 64,
            'gcode_text': ['G00', 'X10', 'Y20']
        }

        augmented = augmenter.augment_sample(sample)

        # Labels should be unchanged
        assert torch.all(augmented['categorical'] == sample['categorical']), \
            "Categorical features should not change"
        assert torch.all(augmented['tokens'] == sample['tokens']), \
            "Tokens should not change"
        assert augmented['length'] == sample['length'], "Length should not change"
        assert augmented['gcode_text'] == sample['gcode_text'], "G-code text should not change"

    def test_augment_sample_changes_continuous(self):
        """Test that augmentation modifies continuous features."""
        torch.manual_seed(42)
        augmenter = DataAugmenter(augment_prob=1.0)  # Always augment
        sample = {
            'continuous': torch.ones(64, 135) * 10.0,
            'categorical': torch.zeros(64, 4),
            'tokens': torch.zeros(64),
            'length': 64,
            'gcode_text': []
        }

        augmented = augmenter.augment_sample(sample)

        # Continuous should be modified (with high probability)
        assert not torch.allclose(augmented['continuous'], sample['continuous']), \
            "Continuous features should be augmented"

    def test_augment_sample_no_augmentation(self):
        """Test that augment_prob=0 leaves sample unchanged."""
        augmenter = DataAugmenter(augment_prob=0.0)  # Never augment
        sample = {
            'continuous': torch.randn(64, 135),
            'categorical': torch.zeros(64, 4),
            'tokens': torch.zeros(64),
            'length': 64,
            'gcode_text': []
        }

        augmented = augmenter.augment_sample(sample)

        # Should be identical (clone, so different object)
        assert torch.allclose(augmented['continuous'], sample['continuous']), \
            "With augment_prob=0, continuous should be unchanged"


@pytest.mark.unit
class TestAugmentedGCodeDataset:
    """Test suite for AugmentedGCodeDataset class."""

    @pytest.fixture
    def mock_base_dataset(self):
        """Create a mock base dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=10)
        # Initialize all with non-rare tokens (high values, assuming rare are low)
        dataset.tokens = torch.full((10, 64), 100, dtype=torch.long)
        # Pad with zeros after first few tokens
        dataset.tokens[:, 10:] = 0

        # Add some rare tokens (e.g., token ID 5 and 10 are rare) to specific sequences
        dataset.tokens[2, :5] = torch.tensor([5, 10, 100, 100, 0])  # Seq 2 has rare
        dataset.tokens[5, :5] = torch.tensor([5, 100, 100, 0, 0])   # Seq 5 has rare

        def getitem(idx):
            return {
                'continuous': torch.randn(64, 135),
                'categorical': torch.zeros(64, 4),
                'tokens': dataset.tokens[idx],
                'length': 64,
                'gcode_text': []
            }

        dataset.__getitem__ = Mock(side_effect=getitem)
        return dataset

    def test_no_oversampling_size(self, mock_base_dataset):
        """Test that without oversampling, dataset size is unchanged."""
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=False,
            augment=False
        )

        assert len(aug_dataset) == len(mock_base_dataset), \
            "Without oversampling, size should match base dataset"

    def test_oversampling_increases_size(self, mock_base_dataset):
        """Test that oversampling increases dataset size."""
        rare_token_ids = [5, 10]  # These appear in sequences 2 and 5
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=True,
            oversample_factor=3,
            rare_token_ids=rare_token_ids,
            augment=False
        )

        # Base: 10 sequences
        # Rare: 2 sequences (indices 2, 5)
        # Expected: 10 + 2*(3-1) = 14
        assert len(aug_dataset) == 14, \
            f"Expected 14 samples with 3x oversampling, got {len(aug_dataset)}"

    def test_oversampling_correct_indices(self, mock_base_dataset):
        """Test that oversampling creates correct index mapping."""
        rare_token_ids = [5, 10]
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=True,
            oversample_factor=3,
            rare_token_ids=rare_token_ids,
            augment=False
        )

        # Count occurrences of each index
        from collections import Counter
        index_counts = Counter(aug_dataset.indices)

        # Rare sequences (2, 5) should appear 3 times
        assert index_counts[2] == 3, "Rare sequence 2 should appear 3 times"
        assert index_counts[5] == 3, "Rare sequence 5 should appear 3 times"

        # Non-rare sequences should appear once
        for idx in [0, 1, 3, 4, 6, 7, 8, 9]:
            assert index_counts[idx] == 1, f"Non-rare sequence {idx} should appear once"

    def test_augmentation_applied(self, mock_base_dataset):
        """Test that augmentation is applied when enabled."""
        augmenter = DataAugmenter(augment_prob=1.0, noise_level=0.1)
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=False,
            augmenter=augmenter,
            augment=True
        )

        # Get same sample twice (should be different due to random augmentation)
        torch.manual_seed(42)
        sample1 = aug_dataset[0]

        torch.manual_seed(99)  # Different seed
        sample2 = aug_dataset[0]

        # Continuous should be different (due to augmentation randomness)
        assert not torch.allclose(sample1['continuous'], sample2['continuous']), \
            "Augmentation should produce different results with different seeds"

    def test_no_augmentation(self, mock_base_dataset):
        """Test that augmentation can be disabled."""
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=False,
            augment=False
        )

        # Get sample
        sample = aug_dataset[0]

        # Should just be pass-through from base dataset
        assert sample is not None

    def test_getitem_returns_correct_structure(self, mock_base_dataset):
        """Test that __getitem__ returns correct sample structure."""
        aug_dataset = AugmentedGCodeDataset(
            mock_base_dataset,
            oversample_rare=False,
            augment=False
        )

        sample = aug_dataset[0]

        # Check structure
        assert 'continuous' in sample
        assert 'categorical' in sample
        assert 'tokens' in sample
        assert 'length' in sample
        assert 'gcode_text' in sample

        # Check shapes
        assert sample['continuous'].shape == (64, 135)
        assert sample['categorical'].shape == (64, 4)
        assert sample['tokens'].shape == (64,)


@pytest.mark.unit
class TestGetRareTokenIds:
    """Test suite for get_rare_token_ids utility function."""

    def test_get_rare_token_ids_g_commands(self, temp_dir):
        """Test that G-commands are correctly identified as rare."""
        vocab = {
            'vocab': {
                'PAD': 0,
                'G0': 1,
                'G1': 2,
                'G90': 3,
                'X': 4,
                'NUM_X_10': 5,
            }
        }

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        rare_ids = get_rare_token_ids(str(vocab_path))

        # Should identify G0, G1, G90 as rare (IDs 1, 2, 3)
        assert set(rare_ids) == {1, 2, 3}, f"Expected {{1, 2, 3}}, got {set(rare_ids)}"

    def test_get_rare_token_ids_m_commands(self, temp_dir):
        """Test that M-commands are correctly identified as rare."""
        vocab = {
            'vocab': {
                'PAD': 0,
                'M3': 1,
                'M5': 2,
                'X': 3,
            }
        }

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        rare_ids = get_rare_token_ids(str(vocab_path))

        # Should identify M3, M5 as rare (IDs 1, 2)
        assert set(rare_ids) == {1, 2}

    def test_get_rare_token_ids_excludes_parameters(self, temp_dir):
        """Test that parameters (X, Y, Z) are not considered rare."""
        vocab = {
            'vocab': {
                'G0': 0,
                'X': 1,
                'Y': 2,
                'Z': 3,
                'NUM_X_10': 4,
            }
        }

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        rare_ids = get_rare_token_ids(str(vocab_path))

        # Only G0 should be rare
        assert rare_ids == [0], "Only G-commands should be rare"

    def test_get_rare_token_ids_token2id_format(self, temp_dir):
        """Test compatibility with token2id format (alternative vocab format)."""
        vocab = {
            'token2id': {
                'G0': 0,
                'G1': 1,
                'M3': 2,
            }
        }

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        rare_ids = get_rare_token_ids(str(vocab_path))

        # Should work with token2id format
        assert set(rare_ids) == {0, 1, 2}

    def test_get_rare_token_ids_empty_vocab(self, temp_dir):
        """Test behavior with empty vocabulary."""
        vocab = {'vocab': {}}

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        rare_ids = get_rare_token_ids(str(vocab_path))

        assert rare_ids == [], "Empty vocab should return empty list"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
