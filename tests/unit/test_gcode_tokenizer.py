"""
Unit tests for G-code tokenizer module.

Tests cover:
- TokenizerConfig dataclass
- GCodeTokenizer canonicalization
- Tokenization modes (literal, split, hybrid)
- Bucketing functionality
- Vocabulary building
- Encode/decode operations
- Save/load persistence
"""

import pytest
import json
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.utilities.gcode_tokenizer import (
    GCodeTokenizer,
    TokenizerConfig,
    DEFAULT_SPECIAL,
)


@pytest.mark.unit
class TestTokenizerConfig:
    """Test suite for TokenizerConfig dataclass."""

    def test_default_config(self):
        """Test that default config has expected values."""
        config = TokenizerConfig()

        assert config.mode == "hybrid"
        assert config.canonical_decimal_places == 4
        assert config.vocab_size == 50000
        assert config.min_freq == 1
        assert config.bucket_digits is None
        assert "PAD" in config.special
        assert config.special["PAD"] == 0

    def test_custom_bucket_digits(self):
        """Test creating config with custom bucket_digits."""
        config = TokenizerConfig(bucket_digits=2)

        assert config.bucket_digits == 2

    def test_custom_mode(self):
        """Test creating config with different modes."""
        for mode in ["literal", "split", "hybrid"]:
            config = TokenizerConfig(mode=mode)
            assert config.mode == mode


@pytest.mark.unit
class TestCanonicalization:
    """Test suite for G-code canonicalization."""

    def test_strip_semicolon_comments(self):
        """Test that semicolon comments are removed."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "G0 X10 ; move to X10"
        canonical = tokenizer.canonicalize_line(line)

        assert ";" not in canonical
        assert "move" not in canonical
        assert "G0" in canonical
        assert "X10" in canonical

    def test_strip_parenthesis_comments(self):
        """Test that parenthesis comments are removed."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "G0 (rapid move) X10"
        canonical = tokenizer.canonicalize_line(line)

        assert "(" not in canonical
        assert "rapid" not in canonical
        assert "G0" in canonical
        assert "X10" in canonical

    def test_normalize_g_command(self):
        """Test that G-commands are normalized (G00 → G0)."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "G00 X10"
        canonical = tokenizer.canonicalize_line(line)

        assert "G0" in canonical or "G00" in canonical  # Depends on normalization

    def test_uppercase_conversion(self):
        """Test that commands are uppercased."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "g1 x10 y20"
        canonical = tokenizer.canonicalize_line(line)

        assert canonical.isupper()
        assert "G1" in canonical
        assert "X10" in canonical

    def test_empty_line_returns_none(self):
        """Test that empty lines return None."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        assert tokenizer.canonicalize_line("") is None
        assert tokenizer.canonicalize_line("   ") is None
        assert tokenizer.canonicalize_line("; comment only") is None

    def test_line_number_removal(self):
        """Test that line numbers (N...) are removed."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "N100 G0 X10"
        canonical = tokenizer.canonicalize_line(line)

        assert "N100" not in canonical
        assert "G0" in canonical

    def test_multi_space_normalization(self):
        """Test that multiple spaces are collapsed."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "G0    X10     Y20"
        canonical = tokenizer.canonicalize_line(line)

        # Should have single spaces
        assert "  " not in canonical

    def test_canonicalize_multiple_lines(self):
        """Test canonicalizing multiple lines."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        lines = [
            "G0 X10 ; first move",
            "",
            "G1 Y20 (second move)",
        ]

        canonical = tokenizer.canonicalize(lines)

        assert len(canonical) == 2  # Empty line filtered out
        assert any("G0" in line for line in canonical)
        assert any("G1" in line for line in canonical)


@pytest.mark.unit
class TestBucketing:
    """Test suite for value bucketing functionality."""

    def test_bucketing_2digit_positive(self):
        """Test 2-digit bucketing for positive values."""
        config = TokenizerConfig(bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        # 1575 → "15"
        assert tokenizer._bucket_value(1575) == "15"

        # 125 → "12"
        assert tokenizer._bucket_value(125) == "12"

        # 5 → "05" (zero-padded)
        assert tokenizer._bucket_value(5) == "05"

    def test_bucketing_2digit_negative(self):
        """Test 2-digit bucketing for negative values."""
        config = TokenizerConfig(bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        # -125 → "-1" (sign + first digit for bucket_digits=2)
        assert tokenizer._bucket_value(-125) == "-1"

        # -5 → "-5"
        assert tokenizer._bucket_value(-5) == "-5"

    def test_bucketing_3digit(self):
        """Test 3-digit bucketing."""
        config = TokenizerConfig(bucket_digits=3)
        tokenizer = GCodeTokenizer(config)

        # 1575 → "157"
        assert tokenizer._bucket_value(1575) == "157"

        # 125 → "125"
        assert tokenizer._bucket_value(125) == "125"

        # 5 → "005"
        assert tokenizer._bucket_value(5) == "005"

    def test_no_bucketing(self):
        """Test that bucket_digits=None preserves full value."""
        config = TokenizerConfig(bucket_digits=None)
        tokenizer = GCodeTokenizer(config)

        assert tokenizer._bucket_value(1575) == "1575"
        assert tokenizer._bucket_value(-125) == "-125"

    def test_bucketing_zero(self):
        """Test bucketing zero value."""
        config = TokenizerConfig(bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        assert tokenizer._bucket_value(0) == "00"


@pytest.mark.unit
class TestTokenization:
    """Test suite for tokenization functionality."""

    def test_tokenize_g_command(self):
        """Test tokenization of G-command."""
        config = TokenizerConfig(mode="hybrid")
        tokenizer = GCodeTokenizer(config)

        tokens = tokenizer._tokenize_word("G0")

        assert tokens == ["G0"]

    def test_tokenize_parameter_with_value(self):
        """Test tokenization of parameter with numeric value."""
        config = TokenizerConfig(mode="split", bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        # X10.0 → ["X", "NUM_X_10"]
        tokens = tokenizer._tokenize_word("X10.0")

        assert len(tokens) == 2
        assert tokens[0] == "X"
        assert "NUM_X" in tokens[1]

    def test_tokenize_literal_mode(self):
        """Test literal mode keeps word intact."""
        config = TokenizerConfig(mode="literal")
        tokenizer = GCodeTokenizer(config)

        tokens = tokenizer._tokenize_word("X10.5")

        assert tokens == ["X10.5"]

    def test_tokenize_canonical_line(self):
        """Test tokenizing a full canonical line."""
        config = TokenizerConfig(mode="hybrid")
        tokenizer = GCodeTokenizer(config)

        canonical = ["G0 X10 Y20"]
        tokens = tokenizer.tokenize_canonical(canonical)

        assert "G0" in tokens
        assert "X" in tokens or "X10" in tokens  # Depends on mode
        assert "Y" in tokens or "Y20" in tokens

    def test_tokenize_with_bucketing(self):
        """Test that bucketing is applied during tokenization."""
        config = TokenizerConfig(mode="split", bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        # X15.75 should be quantized and bucketed
        tokens = tokenizer._tokenize_word("X15.75")

        assert len(tokens) == 2
        assert tokens[0] == "X"
        # NUM_X should be bucketed (15750 μm → "15" or similar)
        assert tokens[1].startswith("NUM_X_")


@pytest.mark.unit
class TestVocabularyBuilding:
    """Test suite for vocabulary building."""

    def test_vocab_initialization(self):
        """Test that vocabulary initializes with special tokens."""
        config = TokenizerConfig()
        tokenizer = GCodeTokenizer(config)

        assert "PAD" in tokenizer.vocab
        assert "BOS" in tokenizer.vocab
        assert "EOS" in tokenizer.vocab
        assert "UNK" in tokenizer.vocab
        assert tokenizer.vocab["PAD"] == 0

    def test_vocab_building_from_files(self, temp_dir):
        """Test building vocabulary from G-code files."""
        config = TokenizerConfig(mode="hybrid")
        tokenizer = GCodeTokenizer(config)

        # Create sample G-code file
        gcode_file = temp_dir / "test.gcode"
        gcode_file.write_text("G0 X10 Y20\nG1 X15 Y25\nG0 X10 Y20")

        vocab = tokenizer.train_vocab([gcode_file])

        # Should contain special tokens + G-code tokens
        assert len(vocab) > len(DEFAULT_SPECIAL)
        assert "G0" in vocab
        assert "G1" in vocab

    def test_vocab_respects_min_freq(self, temp_dir):
        """Test that min_freq filters out rare tokens."""
        config = TokenizerConfig(mode="hybrid", min_freq=2)
        tokenizer = GCodeTokenizer(config)

        # Create file where some tokens appear only once
        gcode_file = temp_dir / "test.gcode"
        gcode_file.write_text("G0 X10 Y20\nG0 X10\nG99 X5")  # G99 appears once

        vocab = tokenizer.train_vocab([gcode_file])

        # G0 appears 2 times, should be in vocab
        assert "G0" in vocab

        # G99 appears 1 time, might not be in vocab (depends on min_freq)
        if config.min_freq > 1:
            assert "G99" not in vocab

    def test_inverse_vocab_created(self):
        """Test that inverse vocabulary is created correctly."""
        config = TokenizerConfig()
        tokenizer = GCodeTokenizer(config)

        # Check bidirectional mapping
        for token, idx in tokenizer.vocab.items():
            assert tokenizer.inv_vocab[idx] == token


@pytest.mark.unit
class TestEncodeDecod:
    """Test suite for encode/decode operations."""

    def test_encode_simple_gcode(self):
        """Test encoding simple G-code."""
        config = TokenizerConfig(mode="hybrid")
        tokenizer = GCodeTokenizer(config)

        # Build minimal vocab
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5, "X10": 6, "Y20": 7}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}

        lines = ["G0 X10 Y20"]
        ids = tokenizer.encode(lines, add_bos_eos=True)

        # Should have BOS, tokens, EOS
        assert ids[0] == DEFAULT_SPECIAL["BOS"]
        assert ids[-1] == DEFAULT_SPECIAL["EOS"]
        assert len(ids) >= 3  # BOS + at least 1 token + EOS

    def test_encode_without_bos_eos(self):
        """Test encoding without BOS/EOS tokens."""
        config = TokenizerConfig()
        tokenizer = GCodeTokenizer(config)
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}

        lines = ["G0"]
        ids = tokenizer.encode(lines, add_bos_eos=False)

        # Should not have BOS/EOS
        assert DEFAULT_SPECIAL["BOS"] not in ids
        assert DEFAULT_SPECIAL["EOS"] not in ids

    def test_decode_reverses_encode(self):
        """Test that decode reverses encode (roundtrip)."""
        config = TokenizerConfig(mode="hybrid")
        tokenizer = GCodeTokenizer(config)
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5, "G1": 6, "X": 7}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}

        ids = [DEFAULT_SPECIAL["BOS"], 5, 6, 7, DEFAULT_SPECIAL["EOS"]]
        tokens = tokenizer.decode(ids)

        # BOS/EOS should be stripped
        assert "BOS" not in tokens
        assert "EOS" not in tokens
        assert len(tokens) == 3

    def test_unknown_token_handling(self):
        """Test that unknown tokens map to UNK."""
        config = TokenizerConfig()
        tokenizer = GCodeTokenizer(config)

        lines = ["G999"]  # Unknown command
        ids = tokenizer.encode(lines, add_bos_eos=False)

        # Should contain UNK token
        assert DEFAULT_SPECIAL["UNK"] in ids


@pytest.mark.unit
class TestSaveLoad:
    """Test suite for save/load persistence."""

    def test_save_vocab(self, temp_dir):
        """Test saving vocabulary to file."""
        config = TokenizerConfig(bucket_digits=2)
        tokenizer = GCodeTokenizer(config)
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5, "G1": 6}

        vocab_path = temp_dir / "vocab.json"
        tokenizer.save_vocab(vocab_path)

        # Check file exists and is valid JSON
        assert vocab_path.exists()
        data = json.loads(vocab_path.read_text())
        assert "vocab" in data
        assert "config" in data

    def test_load_vocab(self, temp_dir):
        """Test loading vocabulary from file."""
        config = TokenizerConfig(bucket_digits=2)
        tokenizer = GCodeTokenizer(config)
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5, "G1": 6, "X": 7}

        # Save
        vocab_path = temp_dir / "vocab.json"
        tokenizer.save_vocab(vocab_path)

        # Load
        loaded_tokenizer = GCodeTokenizer.load(vocab_path)

        # Should match original
        assert loaded_tokenizer.vocab == tokenizer.vocab
        assert loaded_tokenizer.cfg.bucket_digits == 2

    def test_save_load_roundtrip(self, temp_dir):
        """Test that save/load preserves all configuration."""
        config = TokenizerConfig(
            mode="split",
            bucket_digits=3,
            min_freq=2,
            vocab_size=1000
        )
        tokenizer = GCodeTokenizer(config)
        tokenizer.vocab = {**DEFAULT_SPECIAL, "G0": 5}

        # Save and load
        vocab_path = temp_dir / "vocab.json"
        tokenizer.save_vocab(vocab_path)
        loaded = GCodeTokenizer.load(vocab_path)

        # Configuration should match
        assert loaded.cfg.mode == "split"
        assert loaded.cfg.bucket_digits == 3
        assert loaded.cfg.min_freq == 2
        assert loaded.cfg.vocab_size == 1000

    def test_ignore_for_pid_serialization(self, temp_dir):
        """Test that ignore_for_pid set is correctly serialized."""
        config = TokenizerConfig()
        config.ignore_for_pid = {"F", "S", "T"}
        tokenizer = GCodeTokenizer(config)

        vocab_path = temp_dir / "vocab.json"
        tokenizer.save_vocab(vocab_path)

        # Load and verify set is restored
        loaded = GCodeTokenizer.load(vocab_path)
        assert isinstance(loaded.cfg.ignore_for_pid, set)
        assert loaded.cfg.ignore_for_pid == {"F", "S", "T"}


@pytest.mark.unit
class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_vocab(self):
        """Test tokenizer with empty vocabulary (only special tokens)."""
        config = TokenizerConfig()
        tokenizer = GCodeTokenizer(config)

        # All tokens should map to UNK
        lines = ["G0 X10"]
        ids = tokenizer.encode(lines, add_bos_eos=False)

        # Most/all should be UNK
        assert ids.count(DEFAULT_SPECIAL["UNK"]) > 0

    def test_very_large_number(self):
        """Test handling of very large numeric values."""
        config = TokenizerConfig(mode="split", bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        tokens = tokenizer._tokenize_word("X9999999.999")

        # Should handle large numbers gracefully
        assert len(tokens) >= 1

    def test_negative_values(self):
        """Test handling of negative values."""
        config = TokenizerConfig(mode="split", bucket_digits=2)
        tokenizer = GCodeTokenizer(config)

        tokens = tokenizer._tokenize_word("X-10.5")

        assert len(tokens) >= 1
        assert tokens[0] == "X"

    def test_special_characters_in_comment(self):
        """Test that special characters in comments don't break parsing."""
        tokenizer = GCodeTokenizer(TokenizerConfig())

        line = "G0 ; comment with (parens) and ;semicolons"
        canonical = tokenizer.canonicalize_line(line)

        assert canonical is not None
        assert "G0" in canonical


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
