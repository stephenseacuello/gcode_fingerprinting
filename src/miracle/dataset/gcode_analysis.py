"""
G-code token analysis for raw data exploration.

This module provides comprehensive analysis of G-code commands BEFORE tokenization:
- Token frequency and vocabulary analysis
- Token sequence pattern detection
- Co-occurrence analysis
- Rare token identification
- Command complexity metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Counter as CounterType
from collections import Counter, defaultdict
import re

__all__ = [
    "GCodeAnalyzer",
    "analyze_token_frequencies",
    "compute_vocabulary_stats",
    "analyze_token_sequences",
    "identify_rare_tokens",
    "compute_cooccurrence_matrix",
]


class GCodeAnalyzer:
    """Comprehensive analyzer for G-code commands."""

    def __init__(self, csv_files: List[Path], gcode_column: str = 'gcode_text'):
        """
        Initialize analyzer with list of CSV files.

        Args:
            csv_files: List of paths to aligned CSV files
            gcode_column: Name of column containing G-code text
        """
        self.csv_files = csv_files
        self.gcode_column = gcode_column
        self.all_gcodes = []
        self.statistics = {}

    def load_all_gcodes(self) -> None:
        """Load all G-code commands from CSV files."""
        print(f"Loading G-code commands from {len(self.csv_files)} files...")

        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)

            # Try both possible column names
            if self.gcode_column in df.columns:
                gcodes = df[self.gcode_column].dropna().tolist()
            elif 'gcode_string' in df.columns:
                gcodes = df['gcode_string'].dropna().tolist()
            else:
                print(f"  Warning: No G-code column found in {csv_path.name}")
                continue

            self.all_gcodes.extend(gcodes)

        print(f"  Loaded {len(self.all_gcodes):,} G-code commands")

    def compute_token_frequencies(self) -> Counter:
        """
        Compute frequency distribution of G-code tokens.

        Returns:
            Counter with token frequencies
        """
        if not self.all_gcodes:
            raise ValueError("No G-code data loaded. Call load_all_gcodes() first.")

        # Count occurrences
        token_counts = Counter(self.all_gcodes)
        self.statistics['token_frequencies'] = token_counts

        return token_counts

    def compute_vocabulary_stats(self) -> Dict:
        """
        Compute vocabulary statistics.

        Returns:
            Dictionary with vocabulary metrics
        """
        if not self.all_gcodes:
            raise ValueError("No G-code data loaded. Call load_all_gcodes() first.")

        token_counts = self.statistics.get('token_frequencies', self.compute_token_frequencies())

        # Vocabulary size
        vocab_size = len(token_counts)
        total_tokens = len(self.all_gcodes)
        unique_tokens = len(set(self.all_gcodes))

        # Most common tokens
        most_common_10 = token_counts.most_common(10)
        coverage_top_10 = sum(count for _, count in most_common_10) / total_tokens

        # Least common tokens
        least_common = [token for token, count in token_counts.items() if count == 1]
        singleton_count = len(least_common)

        # Token length analysis
        token_lengths = [len(token) for token in token_counts.keys()]

        stats = {
            'vocab_size': vocab_size,
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'type_token_ratio': vocab_size / total_tokens,  # Lexical diversity
            'most_common_10': most_common_10,
            'coverage_top_10': coverage_top_10,
            'singleton_count': singleton_count,
            'singleton_pct': 100 * singleton_count / vocab_size,
            'mean_token_length': np.mean(token_lengths),
            'std_token_length': np.std(token_lengths),
            'min_token_length': min(token_lengths),
            'max_token_length': max(token_lengths),
        }

        self.statistics['vocabulary_stats'] = stats
        return stats

    def analyze_token_sequences(self, window_size: int = 10) -> Dict:
        """
        Analyze token sequence patterns using sliding windows.

        Args:
            window_size: Size of sequence window

        Returns:
            Dictionary with sequence statistics
        """
        if not self.all_gcodes:
            raise ValueError("No G-code data loaded. Call load_all_gcodes() first.")

        # Create sequences
        sequences = []
        for i in range(len(self.all_gcodes) - window_size + 1):
            seq = tuple(self.all_gcodes[i:i+window_size])
            sequences.append(seq)

        # Count sequence patterns
        sequence_counts = Counter(sequences)

        # Find most common sequences
        most_common_seqs = sequence_counts.most_common(20)

        # Compute sequence diversity
        unique_sequences = len(sequence_counts)
        total_sequences = len(sequences)

        stats = {
            'window_size': window_size,
            'total_sequences': total_sequences,
            'unique_sequences': unique_sequences,
            'sequence_diversity': unique_sequences / total_sequences if total_sequences > 0 else 0,
            'most_common_sequences': most_common_seqs[:10],  # Top 10
            'repeated_sequences': sum(1 for count in sequence_counts.values() if count > 1),
            'singleton_sequences': sum(1 for count in sequence_counts.values() if count == 1),
        }

        self.statistics['sequence_analysis'] = stats
        return stats

    def identify_rare_tokens(self, threshold: int = 10) -> Dict:
        """
        Identify rare tokens that appear less than threshold times.

        Args:
            threshold: Minimum occurrence count

        Returns:
            Dictionary with rare token information
        """
        token_counts = self.statistics.get('token_frequencies', self.compute_token_frequencies())

        rare_tokens = {token: count for token, count in token_counts.items() if count < threshold}
        rare_token_list = sorted(rare_tokens.items(), key=lambda x: x[1])

        stats = {
            'threshold': threshold,
            'num_rare_tokens': len(rare_tokens),
            'pct_rare_tokens': 100 * len(rare_tokens) / len(token_counts),
            'rare_tokens': rare_token_list,
            'total_rare_occurrences': sum(rare_tokens.values()),
            'pct_rare_occurrences': 100 * sum(rare_tokens.values()) / len(self.all_gcodes),
        }

        self.statistics['rare_tokens'] = stats
        return stats

    def compute_cooccurrence_matrix(self, top_n: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        Compute co-occurrence matrix for top N tokens.

        Args:
            top_n: Number of most frequent tokens to analyze

        Returns:
            co_occurrence_matrix: [top_n, top_n] matrix
            token_labels: List of token names
        """
        token_counts = self.statistics.get('token_frequencies', self.compute_token_frequencies())

        # Get top N tokens
        top_tokens = [token for token, _ in token_counts.most_common(top_n)]

        # Build co-occurrence matrix
        n = len(top_tokens)
        cooccur_matrix = np.zeros((n, n), dtype=np.int64)

        # Count co-occurrences within a sliding window
        window = 5  # Look at ±5 tokens

        for i, token in enumerate(top_tokens):
            # Find all positions of this token
            positions = [idx for idx, t in enumerate(self.all_gcodes) if t == token]

            for pos in positions:
                # Look at nearby tokens
                start = max(0, pos - window)
                end = min(len(self.all_gcodes), pos + window + 1)

                for nearby_pos in range(start, end):
                    if nearby_pos == pos:
                        continue  # Skip self

                    nearby_token = self.all_gcodes[nearby_pos]
                    if nearby_token in top_tokens:
                        j = top_tokens.index(nearby_token)
                        cooccur_matrix[i, j] += 1

        self.statistics['cooccurrence_matrix'] = cooccur_matrix
        self.statistics['cooccurrence_labels'] = top_tokens

        return cooccur_matrix, top_tokens

    def analyze_command_complexity(self) -> pd.DataFrame:
        """
        Analyze complexity of G-code commands.

        Returns:
            DataFrame with complexity metrics per command type
        """
        token_counts = self.statistics.get('token_frequencies', self.compute_token_frequencies())

        complexity_list = []

        for token, count in token_counts.items():
            # Parse command type (first characters before numbers)
            match = re.match(r'^([A-Za-z]+)', token)
            command_type = match.group(1) if match else 'UNKNOWN'

            # Count parameters (rough heuristic: count spaces and numbers)
            num_parts = len(token.split())
            num_numbers = len(re.findall(r'-?\d+\.?\d*', token))

            complexity_dict = {
                'token': token,
                'command_type': command_type,
                'frequency': count,
                'length': len(token),
                'num_parts': num_parts,
                'num_parameters': num_numbers,
                'has_coordinates': any(c in token for c in ['X', 'Y', 'Z']),
                'has_feedrate': 'F' in token,
                'has_spindle': 'S' in token,
            }
            complexity_list.append(complexity_dict)

        complexity_df = pd.DataFrame(complexity_list)
        self.statistics['command_complexity'] = complexity_df

        return complexity_df

    def analyze_per_file_tokens(self) -> pd.DataFrame:
        """
        Analyze token distribution per file.

        Returns:
            DataFrame with per-file token statistics
        """
        file_stats = []

        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)

            # Get G-code column
            if self.gcode_column in df.columns:
                gcodes = df[self.gcode_column].dropna()
            elif 'gcode_string' in df.columns:
                gcodes = df['gcode_string'].dropna()
            else:
                continue

            # Compute stats
            token_counts = Counter(gcodes)

            stats_dict = {
                'file': csv_path.stem,
                'num_commands': len(gcodes),
                'num_unique_commands': len(token_counts),
                'type_token_ratio': len(token_counts) / len(gcodes) if len(gcodes) > 0 else 0,
                'most_common_command': token_counts.most_common(1)[0][0] if token_counts else None,
                'most_common_count': token_counts.most_common(1)[0][1] if token_counts else 0,
            }
            file_stats.append(stats_dict)

        file_stats_df = pd.DataFrame(file_stats)
        self.statistics['per_file_tokens'] = file_stats_df

        return file_stats_df

    def detect_face_vs_pocket_patterns(self) -> Dict:
        """
        Attempt to detect patterns differentiating face milling vs pocket operations.

        Returns:
            Dictionary with pattern analysis
        """
        # Heuristic: Face milling typically has simpler, more repetitive patterns
        # Pocket milling has more complex paths with frequent direction changes

        token_counts = self.statistics.get('token_frequencies', self.compute_token_frequencies())

        # Count commands with Z motion (pockets often have depth changes)
        z_commands = sum(count for token, count in token_counts.items() if 'Z' in token)
        z_ratio = z_commands / len(self.all_gcodes) if self.all_gcodes else 0

        # Count arc commands (G2/G3) - pockets may have rounded corners
        arc_commands = sum(count for token, count in token_counts.items()
                          if 'G2' in token or 'G3' in token)
        arc_ratio = arc_commands / len(self.all_gcodes) if self.all_gcodes else 0

        # Count rapid moves (G0) vs linear moves (G1)
        rapid_commands = sum(count for token, count in token_counts.items() if 'G0' in token)
        linear_commands = sum(count for token, count in token_counts.items() if 'G1' in token)
        rapid_to_linear = rapid_commands / linear_commands if linear_commands > 0 else 0

        stats = {
            'z_motion_ratio': z_ratio,
            'arc_command_ratio': arc_ratio,
            'rapid_to_linear_ratio': rapid_to_linear,
            'total_commands': len(self.all_gcodes),
            'interpretation': self._interpret_patterns(z_ratio, arc_ratio, rapid_to_linear),
        }

        self.statistics['face_vs_pocket'] = stats
        return stats

    def _interpret_patterns(self, z_ratio: float, arc_ratio: float, rapid_to_linear: float) -> str:
        """Helper to interpret face vs pocket patterns."""
        if z_ratio > 0.1:
            return "Likely contains pocket operations (high Z motion)"
        elif arc_ratio > 0.05:
            return "Likely contains pocket operations (arc commands present)"
        elif rapid_to_linear < 0.1:
            return "Likely face milling (low rapid-to-linear ratio)"
        else:
            return "Mixed or uncertain operation type"


# Standalone utility functions

def analyze_token_frequencies(gcode_list: List[str]) -> Counter:
    """
    Compute token frequency distribution.

    Args:
        gcode_list: List of G-code command strings

    Returns:
        Counter with frequencies
    """
    return Counter(gcode_list)


def compute_vocabulary_stats(token_counts: Counter) -> Dict:
    """
    Compute vocabulary statistics from token counts.

    Args:
        token_counts: Counter with token frequencies

    Returns:
        Dictionary with statistics
    """
    vocab_size = len(token_counts)
    total_tokens = sum(token_counts.values())

    return {
        'vocab_size': vocab_size,
        'total_tokens': total_tokens,
        'most_common_10': token_counts.most_common(10),
        'singleton_count': sum(1 for count in token_counts.values() if count == 1),
    }


def analyze_token_sequences(gcode_list: List[str], window_size: int = 10) -> Dict:
    """
    Analyze sequential patterns in G-code.

    Args:
        gcode_list: List of G-code commands
        window_size: Sequence window size

    Returns:
        Dictionary with sequence statistics
    """
    sequences = []
    for i in range(len(gcode_list) - window_size + 1):
        seq = tuple(gcode_list[i:i+window_size])
        sequences.append(seq)

    sequence_counts = Counter(sequences)

    return {
        'total_sequences': len(sequences),
        'unique_sequences': len(sequence_counts),
        'most_common': sequence_counts.most_common(10),
    }


def identify_rare_tokens(token_counts: Counter, threshold: int = 10) -> List[Tuple[str, int]]:
    """
    Identify rare tokens below threshold.

    Args:
        token_counts: Counter with token frequencies
        threshold: Minimum occurrence count

    Returns:
        List of (token, count) tuples for rare tokens
    """
    rare = [(token, count) for token, count in token_counts.items() if count < threshold]
    return sorted(rare, key=lambda x: x[1])


def compute_cooccurrence_matrix(gcode_list: List[str], top_n: int = 20, window: int = 5) -> Tuple[np.ndarray, List[str]]:
    """
    Compute co-occurrence matrix for top tokens.

    Args:
        gcode_list: List of G-code commands
        top_n: Number of top tokens to analyze
        window: Co-occurrence window size

    Returns:
        co_occurrence_matrix, token_labels
    """
    token_counts = Counter(gcode_list)
    top_tokens = [token for token, _ in token_counts.most_common(top_n)]

    n = len(top_tokens)
    cooccur_matrix = np.zeros((n, n), dtype=np.int64)

    for i, token in enumerate(top_tokens):
        positions = [idx for idx, t in enumerate(gcode_list) if t == token]

        for pos in positions:
            start = max(0, pos - window)
            end = min(len(gcode_list), pos + window + 1)

            for nearby_pos in range(start, end):
                if nearby_pos == pos:
                    continue

                nearby_token = gcode_list[nearby_pos]
                if nearby_token in top_tokens:
                    j = top_tokens.index(nearby_token)
                    cooccur_matrix[i, j] += 1

    return cooccur_matrix, top_tokens
