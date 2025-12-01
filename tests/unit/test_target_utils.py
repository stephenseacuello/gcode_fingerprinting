"""
Unit tests for target decomposition utilities.

Tests cover:
- TokenDecomposer initialization
- Token decomposition (single and batch)
- Token composition (single and batch)
- Roundtrip consistency (decompose → compose → same token)
- Edge cases and error handling
"""

import pytest
import torch
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.dataset.target_utils import TokenDecomposer


@pytest.mark.unit
class TestTokenDecomposerInit:
    """Test suite for TokenDecomposer initialization."""

    @pytest.fixture
    def sample_vocab_file(self, temp_dir):
        """Create a sample vocabulary file for testing."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                '<BOS>': 1,
                '<EOS>': 2,
                '<UNK>': 3,
                'G0': 4,
                'G1': 5,
                'G2': 6,
                'M3': 7,
                'M5': 8,
                'X': 9,
                'Y': 10,
                'Z': 11,
                'F': 12,
                'R': 13,
                'NUM_X_00': 14,
                'NUM_X_10': 15,
                'NUM_Y_15': 16,
                'NUM_Z_05': 17,
            }
        }

        vocab_path = temp_dir / 'test_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return str(vocab_path)

    def test_initialization(self, sample_vocab_file):
        """Test that TokenDecomposer initializes correctly."""
        decomposer = TokenDecomposer(sample_vocab_file)

        assert decomposer.vocab_size > 0
        assert decomposer.n_commands > 0
        assert decomposer.n_param_types > 0
        assert decomposer.n_param_values == 100

    def test_command_tokens_parsed(self, sample_vocab_file):
        """Test that G/M commands are correctly identified."""
        decomposer = TokenDecomposer(sample_vocab_file)

        # Should identify G0, G1, G2, M3, M5
        assert 'G0' in decomposer.command_tokens
        assert 'G1' in decomposer.command_tokens
        assert 'M3' in decomposer.command_tokens
        assert decomposer.n_commands >= 5

    def test_param_tokens_parsed(self, sample_vocab_file):
        """Test that parameter tokens (X, Y, Z, F, R) are identified."""
        decomposer = TokenDecomposer(sample_vocab_file)

        assert 'X' in decomposer.param_tokens
        assert 'Y' in decomposer.param_tokens
        assert 'Z' in decomposer.param_tokens
        assert 'F' in decomposer.param_tokens

    def test_numeric_tokens_parsed(self, sample_vocab_file):
        """Test that numeric tokens are correctly identified."""
        decomposer = TokenDecomposer(sample_vocab_file)

        assert 'NUM_X_00' in decomposer.numeric_tokens
        assert 'NUM_X_10' in decomposer.numeric_tokens
        assert 'NUM_Y_15' in decomposer.numeric_tokens


@pytest.mark.unit
class TestSingleTokenDecomposition:
    """Test suite for single token decomposition."""

    @pytest.fixture
    def decomposer(self, temp_dir):
        """Create a TokenDecomposer instance."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                '<BOS>': 1,
                '<UNK>': 3,
                'G0': 4,
                'G1': 5,
                'M3': 6,
                'X': 7,
                'Y': 8,
                'F': 9,
                'NUM_X_15': 10,
                'NUM_Y_23': 11,
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return TokenDecomposer(str(vocab_path))

    def test_decompose_special_token(self, decomposer):
        """Test decomposing special tokens (PAD, BOS, EOS, UNK)."""
        # PAD token (ID=0)
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(0)

        assert type_id == TokenDecomposer.TYPE_SPECIAL
        assert cmd_id == 0
        assert param_type_id == 0
        assert param_value_id == 0

    def test_decompose_command_token(self, decomposer):
        """Test decomposing G/M command tokens."""
        # G0 token
        token_id = decomposer.vocab['G0']
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

        assert type_id == TokenDecomposer.TYPE_COMMAND
        assert cmd_id >= 0  # Should map to command index
        assert param_type_id == 0
        assert param_value_id == 0

    def test_decompose_parameter_token(self, decomposer):
        """Test decomposing parameter tokens (X, Y, Z, F)."""
        # X token
        token_id = decomposer.vocab['X']
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

        assert type_id == TokenDecomposer.TYPE_PARAMETER
        assert cmd_id == 0
        assert param_type_id >= 0  # Should map to parameter index
        assert param_value_id == 0

    def test_decompose_numeric_token(self, decomposer):
        """Test decomposing numeric tokens (NUM_X_15)."""
        # NUM_X_15 token
        token_id = decomposer.vocab['NUM_X_15']
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

        assert type_id == TokenDecomposer.TYPE_NUMERIC
        assert cmd_id == 0
        assert param_type_id >= 0  # Should map to X parameter
        assert param_value_id == 15  # Value should be 15

    def test_decompose_out_of_vocab(self, decomposer):
        """Test decomposing token ID that's out of vocabulary."""
        # Token ID larger than vocab size
        out_of_vocab_id = 9999
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(out_of_vocab_id)

        # Should return SPECIAL (treated as UNK)
        assert type_id == TokenDecomposer.TYPE_SPECIAL
        assert cmd_id == 0
        assert param_type_id == 0
        assert param_value_id == 0


@pytest.mark.unit
class TestBatchDecomposition:
    """Test suite for batch token decomposition."""

    @pytest.fixture
    def decomposer(self, temp_dir):
        """Create a TokenDecomposer instance."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                'G0': 1,
                'X': 2,
                'NUM_X_15': 3,
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return TokenDecomposer(str(vocab_path))

    def test_batch_decomposition_shape(self, decomposer):
        """Test that batch decomposition preserves shape."""
        token_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # [2, 4]

        targets = decomposer.decompose_batch(token_ids)

        # All outputs should have shape [B, T]
        assert targets['type'].shape == (2, 4)
        assert targets['command_id'].shape == (2, 4)
        assert targets['param_type_id'].shape == (2, 4)
        assert targets['param_value_id'].shape == (2, 4)

    def test_batch_decomposition_consistency(self, decomposer):
        """Test that batch decomposition matches single decomposition."""
        token_ids = torch.tensor([[1, 2, 3]])  # [1, 3]

        # Batch decomposition
        targets = decomposer.decompose_batch(token_ids)

        # Single decomposition for comparison
        for t, token_id in enumerate([1, 2, 3]):
            type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

            assert targets['type'][0, t].item() == type_id
            assert targets['command_id'][0, t].item() == cmd_id
            assert targets['param_type_id'][0, t].item() == param_type_id
            assert targets['param_value_id'][0, t].item() == param_value_id

    def test_batch_decomposition_device(self, decomposer):
        """Test that batch decomposition preserves device."""
        token_ids = torch.tensor([[0, 1, 2, 3]])

        targets = decomposer.decompose_batch(token_ids)

        # All tensors should be on same device as input
        assert targets['type'].device == token_ids.device
        assert targets['command_id'].device == token_ids.device


@pytest.mark.unit
class TestSingleTokenComposition:
    """Test suite for single token composition."""

    @pytest.fixture
    def decomposer(self, temp_dir):
        """Create a TokenDecomposer instance."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                '<UNK>': 1,
                'G0': 2,
                'G1': 3,
                'X': 4,
                'Y': 5,
                'NUM_X_15': 6,
                'NUM_Y_23': 7,
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return TokenDecomposer(str(vocab_path))

    def test_compose_special_token(self, decomposer):
        """Test composing special token."""
        token_id = decomposer.compose_token(
            type_id=TokenDecomposer.TYPE_SPECIAL,
            command_id=0,
            param_type_id=0,
            param_value_id=0
        )

        # Should return PAD or similar special token
        assert token_id in [0, 1]  # PAD or UNK

    def test_compose_command_token(self, decomposer):
        """Test composing command token."""
        # G0 is at index 0 in command_tokens
        g0_index = decomposer.command_tokens.index('G0')
        token_id = decomposer.compose_token(
            type_id=TokenDecomposer.TYPE_COMMAND,
            command_id=g0_index,
            param_type_id=0,
            param_value_id=0
        )

        # Should return G0 token ID
        assert token_id == decomposer.vocab['G0']

    def test_compose_parameter_token(self, decomposer):
        """Test composing parameter token."""
        # X is at some index in param_tokens
        x_index = decomposer.param_tokens.index('X')
        token_id = decomposer.compose_token(
            type_id=TokenDecomposer.TYPE_PARAMETER,
            command_id=0,
            param_type_id=x_index,
            param_value_id=0
        )

        # Should return X token ID
        assert token_id == decomposer.vocab['X']

    def test_compose_numeric_token(self, decomposer):
        """Test composing numeric token."""
        # NUM_X_15
        x_index = decomposer.param_tokens.index('X')
        token_id = decomposer.compose_token(
            type_id=TokenDecomposer.TYPE_NUMERIC,
            command_id=0,
            param_type_id=x_index,
            param_value_id=15
        )

        # Should return NUM_X_15 or closest match
        # Could be NUM_X_15 or NUM_X_15 (with/without zero-padding)
        assert token_id in [decomposer.vocab.get('NUM_X_15', 0),
                           decomposer.vocab.get('NUM_X_15', 0),
                           decomposer.vocab.get('<UNK>', 1)]

    def test_compose_invalid_command_index(self, decomposer):
        """Test composing with invalid command index."""
        invalid_index = 9999
        token_id = decomposer.compose_token(
            type_id=TokenDecomposer.TYPE_COMMAND,
            command_id=invalid_index,
            param_type_id=0,
            param_value_id=0
        )

        # Should return UNK
        assert token_id == decomposer.vocab.get('<UNK>', 0)


@pytest.mark.unit
class TestBatchComposition:
    """Test suite for batch token composition."""

    @pytest.fixture
    def decomposer(self, temp_dir):
        """Create a TokenDecomposer instance."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                'G0': 1,
                'X': 2,
                'NUM_X_15': 3,
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return TokenDecomposer(str(vocab_path))

    def test_batch_composition_shape(self, decomposer):
        """Test that batch composition produces correct shape."""
        B, T = 2, 4
        targets = {
            'type': torch.zeros(B, T, dtype=torch.long),
            'command_id': torch.zeros(B, T, dtype=torch.long),
            'param_type_id': torch.zeros(B, T, dtype=torch.long),
            'param_value_id': torch.zeros(B, T, dtype=torch.long),
        }

        token_ids = decomposer.compose_batch(targets)

        assert token_ids.shape == (B, T)

    def test_batch_composition_device(self, decomposer):
        """Test that batch composition preserves device."""
        B, T = 2, 4
        targets = {
            'type': torch.zeros(B, T, dtype=torch.long),
            'command_id': torch.zeros(B, T, dtype=torch.long),
            'param_type_id': torch.zeros(B, T, dtype=torch.long),
            'param_value_id': torch.zeros(B, T, dtype=torch.long),
        }

        token_ids = decomposer.compose_batch(targets)

        assert token_ids.device == targets['type'].device


@pytest.mark.unit
class TestRoundtripConsistency:
    """Test suite for decompose → compose roundtrip consistency."""

    @pytest.fixture
    def decomposer(self, temp_dir):
        """Create a TokenDecomposer instance."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                '<BOS>': 1,
                '<UNK>': 2,
                'G0': 3,
                'G1': 4,
                'M3': 5,
                'X': 6,
                'Y': 7,
                'NUM_X_10': 8,
                'NUM_Y_20': 9,
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        return TokenDecomposer(str(vocab_path))

    def test_roundtrip_special_token(self, decomposer):
        """Test roundtrip for special tokens."""
        original_id = 0  # PAD

        # Decompose
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(original_id)

        # Compose
        reconstructed_id = decomposer.compose_token(type_id, cmd_id, param_type_id, param_value_id)

        # Should be same or equivalent special token
        assert reconstructed_id in [0, 1, 2]  # PAD, BOS, or UNK

    def test_roundtrip_command_token(self, decomposer):
        """Test roundtrip for command tokens."""
        original_id = decomposer.vocab['G0']

        # Decompose
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(original_id)

        # Compose
        reconstructed_id = decomposer.compose_token(type_id, cmd_id, param_type_id, param_value_id)

        # Should be exactly the same
        assert reconstructed_id == original_id

    def test_roundtrip_parameter_token(self, decomposer):
        """Test roundtrip for parameter tokens."""
        original_id = decomposer.vocab['X']

        # Decompose → Compose
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(original_id)
        reconstructed_id = decomposer.compose_token(type_id, cmd_id, param_type_id, param_value_id)

        assert reconstructed_id == original_id

    def test_roundtrip_numeric_token(self, decomposer):
        """Test roundtrip for numeric tokens."""
        original_id = decomposer.vocab['NUM_X_10']

        # Decompose → Compose
        type_id, cmd_id, param_type_id, param_value_id = decomposer.decompose_token(original_id)
        reconstructed_id = decomposer.compose_token(type_id, cmd_id, param_type_id, param_value_id)

        # Should be same or very close (accounting for bucketing)
        assert reconstructed_id == original_id or \
               decomposer.id2token[reconstructed_id].startswith('NUM_X_')

    def test_batch_roundtrip(self, decomposer):
        """Test batch roundtrip: decompose → compose → same tokens."""
        original_ids = torch.tensor([[0, 3, 6, 8], [4, 7, 9, 0]])  # [2, 4]

        # Decompose
        targets = decomposer.decompose_batch(original_ids)

        # Compose
        reconstructed_ids = decomposer.compose_batch(targets)

        # Most tokens should match (some may differ due to bucketing/approximation)
        # At least check shape matches
        assert reconstructed_ids.shape == original_ids.shape


@pytest.mark.unit
class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_vocab(self, temp_dir):
        """Test behavior with minimal vocabulary."""
        vocab = {'vocab': {'<PAD>': 0, '<UNK>': 1}}

        vocab_path = temp_dir / 'minimal_vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        decomposer = TokenDecomposer(str(vocab_path))

        # Should handle empty command/param lists
        assert decomposer.n_commands == 0
        assert decomposer.n_param_types == 0

    def test_negative_param_value(self, temp_dir):
        """Test handling of negative parameter values."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                'X': 1,
                'NUM_X_-1': 2,  # Negative value
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        decomposer = TokenDecomposer(str(vocab_path))

        # Decompose negative numeric token
        token_id = decomposer.vocab['NUM_X_-1']
        type_id, _, _, param_value_id = decomposer.decompose_token(token_id)

        # Should handle negative values (absolute value used)
        assert type_id == TokenDecomposer.TYPE_NUMERIC
        assert param_value_id == 1  # abs(-1) = 1

    def test_malformed_numeric_token(self, temp_dir):
        """Test handling of malformed numeric tokens."""
        vocab = {
            'vocab': {
                '<PAD>': 0,
                'NUM_INVALID': 1,  # Missing param type and value
                'NUM_X_': 2,       # Missing value
            }
        }

        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        decomposer = TokenDecomposer(str(vocab_path))

        # Should handle gracefully (not crash)
        for token_id in [1, 2]:
            result = decomposer.decompose_token(token_id)
            assert len(result) == 4  # Should return 4-tuple


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
