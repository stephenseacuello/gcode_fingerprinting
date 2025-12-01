#!/usr/bin/env python3
"""
Context utilities for extracting and preparing context embeddings.

This module provides utilities for:
1. Defining context specifications (vocabulary sizes) for categorical variables
2. Extracting context IDs from preprocessed data
3. Creating context dictionaries for model input

Context Variables:
------------------
1. TinyG Status Report Fields (6 variables):
   - stat: TinyG machine status
   - unit: Inches/mm
   - dist: Absolute/incremental positioning
   - coor: Work coordinate system (G54/G55/etc.)
   - momo: Motion mode
   - plane: Active plane selection

2. Experimental Parameters (6 variables - to be added to preprocessing):
   - material: Material type
   - toolpath_type: Tool path variant (1-3)
   - tool_diameter: Tool diameter category
   - toolpath_param: Tool path parameter category
   - speed: Spindle speed category
   - feed_rate: Feed rate category
"""

from typing import Dict, Optional, List
import json
from pathlib import Path
import numpy as np
import pandas as pd


# Default context specifications (vocabulary sizes)
# These are based on typical CNC operations and TinyG status reports
DEFAULT_CONTEXT_SPECS = {
    # TinyG Status Report Fields
    "stat": 16,          # TinyG has ~10 status codes (0-9), use 16 for safety
    "unit": 3,           # 0=unknown, 1=inches, 2=mm
    "dist": 3,           # 0=unknown, 1=absolute, 2=incremental
    "coor": 10,          # G54-G59 (6 WCS) + G53/G59.1-G59.3 = ~10
    "momo": 16,          # Motion modes: G0, G1, G2, G3, etc. (~10-12 codes)
    "plane": 4,          # 0=unknown, 1=XY (G17), 2=XZ (G18), 3=YZ (G19)

    # Experimental Parameters (to be added)
    "material": 8,       # e.g., wood, plastic, aluminum, steel, etc.
    "toolpath_type": 4,  # 3 toolpath types + unknown
    "tool_diameter": 16, # Categorical bins for tool diameter
    "toolpath_param": 16,# Categorical bins for toolpath parameter
    "speed": 32,         # Categorical bins for spindle speed
    "feed_rate": 32,     # Categorical bins for feed rate
}


def get_context_specs_from_metadata(
    metadata_path: Path,
    include_experimental: bool = False,
) -> Dict[str, int]:
    """
    Extract context specifications from preprocessing metadata.

    Args:
        metadata_path: Path to metadata.json from preprocessing
        include_experimental: Whether to include experimental parameters
                            (material, toolpath, etc.)

    Returns:
        Dictionary mapping context variable names to vocabulary sizes
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    context_specs = {}

    # Get TinyG status field mappings from metadata
    categorical_mappings = metadata.get('categorical_mappings', {})

    for field_name in ['stat', 'unit', 'dist', 'coor', 'momo', 'plane']:
        if field_name in categorical_mappings:
            # Vocabulary size = max mapped value + 1 (for 0=unknown)
            vocab_size = max(categorical_mappings[field_name].values()) + 1
            context_specs[field_name] = vocab_size
        elif field_name in DEFAULT_CONTEXT_SPECS:
            # Fall back to default if not in metadata
            context_specs[field_name] = DEFAULT_CONTEXT_SPECS[field_name]

    # Add experimental parameters if requested
    if include_experimental:
        for param in ['material', 'toolpath_type', 'tool_diameter',
                      'toolpath_param', 'speed', 'feed_rate']:
            if param in categorical_mappings:
                vocab_size = max(categorical_mappings[param].values()) + 1
                context_specs[param] = vocab_size
            elif param in DEFAULT_CONTEXT_SPECS:
                context_specs[param] = DEFAULT_CONTEXT_SPECS[param]

    return context_specs


def create_context_dict_from_batch(
    batch_data: Dict,
    context_fields: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract context IDs from a batch of data.

    Args:
        batch_data: Dictionary containing batch data with context fields
        context_fields: List of context field names to extract

    Returns:
        Dictionary mapping field names to tensors of shape [B] or [B,T]
        Returns None if no context fields are available
    """
    ctx_ids = {}

    for field in context_fields:
        if field in batch_data:
            ctx_ids[field] = batch_data[field]

    return ctx_ids if ctx_ids else None


def add_experimental_params_to_csv(
    csv_path: Path,
    output_path: Path,
    material: str,
    toolpath_type: int,
    tool_diameter: float,
    toolpath_param: float,
    speed: float,
    feed_rate: float,
) -> None:
    """
    Add experimental parameters to an aligned CSV file.

    This is a utility to augment existing test data with experimental
    context that wasn't originally captured.

    Args:
        csv_path: Path to existing aligned CSV
        output_path: Path to write augmented CSV
        material: Material type (e.g., "wood", "plastic", "aluminum")
        toolpath_type: Tool path variant (1, 2, or 3)
        tool_diameter: Tool diameter in mm
        toolpath_param: Tool path parameter value
        speed: Spindle speed in RPM
        feed_rate: Feed rate in mm/min
    """
    df = pd.read_csv(csv_path)

    # Add experimental parameters as new columns
    df['material'] = material
    df['toolpath_type'] = toolpath_type
    df['tool_diameter'] = tool_diameter
    df['toolpath_param'] = toolpath_param
    df['speed'] = speed
    df['feed_rate'] = feed_rate

    # Save augmented CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved augmented CSV to {output_path}")
    print(f"  Added {6} experimental parameter columns")


def categorize_continuous_context(
    value: float,
    bins: List[float],
) -> int:
    """
    Convert a continuous context variable into a categorical bin.

    Args:
        value: The continuous value to categorize
        bins: List of bin edges (e.g., [0, 1000, 2000, 3000] for speed)

    Returns:
        Bin index (0 for unknown/out of range, 1-N for bins)
    """
    if pd.isna(value):
        return 0  # Unknown

    for i, edge in enumerate(bins[:-1]):
        if bins[i] <= value < bins[i + 1]:
            return i + 1  # Bins start at 1

    # Value is >= last bin edge
    if value >= bins[-1]:
        return len(bins)

    return 0  # Unknown/out of range


def create_speed_bins(num_bins: int = 31) -> List[float]:
    """Create reasonable bins for spindle speed (RPM)."""
    # Typical CNC spindle speeds: 0 - 24000 RPM
    return list(np.linspace(0, 24000, num_bins))


def create_feed_rate_bins(num_bins: int = 31) -> List[float]:
    """Create reasonable bins for feed rate (mm/min)."""
    # Typical feed rates: 0 - 3000 mm/min
    return list(np.linspace(0, 3000, num_bins))


def create_tool_diameter_bins(num_bins: int = 15) -> List[float]:
    """Create reasonable bins for tool diameter (mm)."""
    # Common tool diameters: 1mm - 16mm
    return list(np.linspace(1, 16, num_bins))


# Example usage in preprocessing
def example_usage():
    """
    Example of how to use these utilities in a preprocessing pipeline.
    """
    # 1. Get context specs from metadata
    metadata_path = Path("outputs/preprocessing/test_001_aligned/metadata.json")
    context_specs = get_context_specs_from_metadata(metadata_path)
    print("Context specs:", context_specs)

    # 2. Augment existing CSV with experimental params
    csv_path = Path("data/test_001_aligned.csv")
    output_path = Path("data/test_001_aligned_with_context.csv")

    add_experimental_params_to_csv(
        csv_path=csv_path,
        output_path=output_path,
        material="wood",
        toolpath_type=1,
        tool_diameter=6.35,  # 1/4 inch
        toolpath_param=5.0,
        speed=10000,
        feed_rate=1000,
    )

    # 3. During dataset loading, extract context
    # (This would be integrated into your dataset class)
    batch_data = {
        'X_cont': np.random.randn(8, 64, 20),
        'X_cat': np.random.randint(0, 5, (8, 64, 6)),
        'stat': np.array([3, 3, 3, 3, 5, 5, 5, 5]),  # [B]
        'unit': np.array([2, 2, 2, 2, 2, 2, 2, 2]),  # [B]
        'material': np.array([1, 1, 1, 1, 2, 2, 2, 2]),  # [B]
    }

    ctx = create_context_dict_from_batch(
        batch_data,
        context_fields=['stat', 'unit', 'material']
    )
    print("Context IDs:", ctx)


if __name__ == '__main__':
    # Show default context specs
    print("Default Context Specifications:")
    print("=" * 50)
    for name, vocab_size in DEFAULT_CONTEXT_SPECS.items():
        print(f"  {name:20s}: {vocab_size:3d} categories")

    print("\n" + "=" * 50)
    print("Total context variables: 12")
    print("  - TinyG status fields: 6")
    print("  - Experimental params: 6")
