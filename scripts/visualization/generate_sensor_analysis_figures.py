#!/usr/bin/env python3
"""
Generate Sensor Analysis Figures for G-code Fingerprinting Paper v18.

Generates publication-ready figures for ASME IMECE submission:
1. Modality importance bar chart with per-channel efficiency
2. Modality×Location heatmap showing critical sensors

Usage:
    python scripts/visualization/generate_sensor_analysis_figures.py

Author: Claude Code
Date: January 2026
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
ABLATION_DIR = BASE_DIR / "outputs" / "archive" / "2024-12_best_runs" / "sensor_multihead_v3"
OUTPUT_DIR = ABLATION_DIR / "visualizations" / "paper" / "figures"


def setup_asme_style():
    """Configure matplotlib for ASME publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    if HAS_SEABORN:
        sns.set_palette('colorblind')


def load_ablation_data():
    """Load all ablation JSON files."""
    data = {}

    # Main modality ablation
    with open(ABLATION_DIR / "ablation_results.json") as f:
        data['modality'] = json.load(f)

    # Location ablation
    with open(ABLATION_DIR / "ablations" / "location_ablation" / "location_ablation.json") as f:
        data['location'] = json.load(f)

    # Modality×Location matrix
    with open(ABLATION_DIR / "ablations" / "modality_location_ablation" / "modality_location_ablation.json") as f:
        data['matrix'] = json.load(f)

    return data


def generate_modality_importance_chart(data: dict, output_path: Path):
    """
    Generate horizontal bar chart showing modality importance.
    Shows both total accuracy drop and per-channel efficiency.
    """
    setup_asme_style()

    ablations = data['modality']['ablations']
    baseline = data['modality']['baseline']

    # Extract and sort modalities by total impact
    modalities = []
    for name, info in ablations.items():
        modalities.append({
            'name': name.replace('_', ' ').title(),
            'drop': info['drop_pct'],
            'channels': info['n_channels'],
            'per_channel': info['drop_pct'] / info['n_channels'] if info['n_channels'] > 0 else 0
        })

    # Sort by total drop (descending)
    modalities = sorted(modalities, key=lambda x: x['drop'], reverse=True)

    # Define colors by sensor category
    color_map = {
        'Environmental': '#2ecc71',  # Green
        'Gyroscope': '#3498db',      # Blue
        'Magnetometer': '#9b59b6',   # Purple
        'Accelerometer': '#e74c3c',  # Red
        'Color': '#f39c12',          # Orange
        'Position': '#1abc9c',       # Teal
        'Audio': '#34495e',          # Dark gray
        'Motor Current': '#95a5a6',  # Light gray
    }

    colors = [color_map.get(m['name'], '#7f8c8d') for m in modalities]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left plot: Total accuracy drop
    y_pos = np.arange(len(modalities))
    names = [m['name'] for m in modalities]
    drops = [m['drop'] for m in modalities]

    bars1 = ax1.barh(y_pos, drops, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Accuracy Drop (%)')
    ax1.set_title('(a) Total Modality Impact')
    ax1.invert_yaxis()

    # Add value labels
    for i, (bar, drop) in enumerate(zip(bars1, drops)):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{drop:.1f}%', va='center', ha='left', fontsize=7)

    # Add channel count annotations
    for i, m in enumerate(modalities):
        ax1.text(-0.5, i, f'({m["channels"]}ch)', va='center', ha='right',
                fontsize=6, color='gray')

    # Right plot: Per-channel efficiency
    per_channel = [m['per_channel'] for m in modalities]

    bars2 = ax2.barh(y_pos, per_channel, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Hide y labels on right plot
    ax2.set_xlabel('Accuracy Drop per Channel (%)')
    ax2.set_title('(b) Per-Channel Efficiency')
    ax2.invert_yaxis()

    # Add value labels
    for bar, pc in zip(bars2, per_channel):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{pc:.2f}%', va='center', ha='left', fontsize=7)

    # Highlight key finding: proximity has ~9x efficiency (0.74/0.08 = 9.25)
    ax2.annotate('~9× efficiency\nvs accelerometer',
                xy=(per_channel[0], 0), xytext=(0.5, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=7, ha='center')

    plt.tight_layout()

    # Save
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def generate_location_heatmap(data: dict, output_path: Path):
    """
    Generate heatmap showing modality×location interaction.
    Highlights critical sensors.
    """
    setup_asme_style()

    matrix_data = data['matrix']['matrix']
    modality_ablation = data['modality']['ablations']

    # Define modalities and locations (including Motor Current for completeness)
    modalities = ['Accelerometer', 'Gyroscope', 'Magnetometer', 'Proximity', 'Microphone', 'Color', 'Motor Current']
    locations = ['frame_b2', 'frame_l2', 'frame_l3', 'frame_r1', 'frame_r2',
                 'spindle1', 'spindle2', 'y_bed__1', 'y_bed__2', 'y_bed__3',
                 'y_bed__4', 'xa_motor', 'controller']

    # Build matrix
    heatmap_data = np.zeros((len(modalities), len(locations)))

    for i, mod in enumerate(modalities):
        for j, loc in enumerate(locations):
            # Special case: Motor Current is only at controller location
            if mod == 'Motor Current':
                if loc == 'controller':
                    # Motor current contributes 0% unique information (key finding!)
                    heatmap_data[i, j] = modality_ablation.get('motor_current', {}).get('drop_pct', 0.0)
                else:
                    heatmap_data[i, j] = np.nan
            elif mod in matrix_data and loc in matrix_data[mod]:
                drop_pct = matrix_data[mod][loc].get('drop_pct')
                if drop_pct is not None:
                    heatmap_data[i, j] = drop_pct
                else:
                    heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = np.nan

    # Create figure (slightly larger to accommodate Motor Current row and controller column)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Create masked array for NaN values
    masked_data = np.ma.masked_invalid(heatmap_data)

    # Plot heatmap
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(masked_data, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy Drop (%)', fontsize=8)

    # Format location labels (cleaner)
    loc_labels = [l.replace('frame_', 'F:').replace('spindle', 'Sp').replace('y_bed__', 'YB:').replace('xa_motor', 'XA').replace('controller', 'Ctrl')
                  for l in locations]

    # Set ticks
    ax.set_xticks(np.arange(len(locations)))
    ax.set_yticks(np.arange(len(modalities)))
    ax.set_xticklabels(loc_labels, rotation=45, ha='right')
    ax.set_yticklabels(modalities)

    # Add value annotations for significant cells
    for i in range(len(modalities)):
        for j in range(len(locations)):
            val = heatmap_data[i, j]
            # Show values > 0.5, but also show Motor Current 0% at controller (important finding)
            if not np.isnan(val) and (val > 0.5 or (modalities[i] == 'Motor Current' and locations[j] == 'controller')):
                text_color = 'white' if val > 1.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=6, color=text_color)

    # Highlight the critical sensor (Proximity @ frame_r1)
    # frame_r1 is index 3, Proximity is index 3
    rect = plt.Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor='gold', linewidth=2)
    ax.add_patch(rect)
    ax.annotate('Most critical:\n−2.77%', xy=(3, 3), xytext=(5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gold', lw=1.5),
                fontsize=7, fontweight='bold', color='darkred')

    # Highlight Motor Current 0% contribution (security finding)
    # Motor Current is index 6, controller is index 12
    rect2 = plt.Rectangle((11.5, 5.5), 1, 1, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect2)
    ax.annotate('0% contribution:\neasiest to spoof,\nno unique info', xy=(12, 6), xytext=(8, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=6, fontweight='bold', color='darkgreen')

    ax.set_xlabel('Sensor Location')
    ax.set_ylabel('Sensor Modality')
    ax.set_title('Modality × Location Interaction Matrix (Accuracy Drop %)')

    plt.tight_layout()

    # Save
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def generate_spoofing_difficulty_chart(output_path: Path):
    """
    Generate chart showing sensor spoofing difficulty ranking.
    For security analysis section.
    """
    setup_asme_style()

    # Data from security analysis
    sensors = [
        ('Proximity\n(distributed)', 'Very Hard', 4, 'Requires physical presence\nat multiple locations'),
        ('Accelerometer/\nGyroscope', 'Hard', 3, 'Coherent vibration injection\nacross 12 nodes'),
        ('Magnetometer', 'Moderate', 2, 'EMF injection possible\nbut spatially inconsistent'),
        ('Motor Current', 'Easy', 1, 'Single-point electrical\nsignal injection'),
    ]

    fig, ax = plt.subplots(figsize=(6, 3))

    colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']  # Green to red
    y_pos = np.arange(len(sensors))

    bars = ax.barh(y_pos, [s[2] for s in sensors], color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in sensors])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Easy', 'Moderate', 'Hard', 'Very Hard'])
    ax.set_xlabel('Spoofing Difficulty')
    ax.set_title('Sensor Spoofing Resistance Ranking')
    ax.set_xlim(0, 5)
    ax.invert_yaxis()

    # Add annotations
    for i, (name, difficulty, level, rationale) in enumerate(sensors):
        ax.text(4.2, i, rationale, va='center', ha='left', fontsize=6, style='italic')

    # Add note about motor current
    ax.annotate('0% unique contribution\n→ inherent robustness',
                xy=(1, 3), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8),
                fontsize=7, color='darkred', fontweight='bold')

    plt.tight_layout()

    # Save
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    """Generate all sensor analysis figures."""
    print("="*60)
    print("Generating Sensor Analysis Figures for Paper v18")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading ablation data...")
    data = load_ablation_data()
    print(f"  Baseline accuracy: {data['modality']['baseline']*100:.2f}%")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Modality importance bar chart
    generate_modality_importance_chart(
        data,
        OUTPUT_DIR / "sensor_modality_importance.pdf"
    )

    # 2. Location×Modality heatmap
    generate_location_heatmap(
        data,
        OUTPUT_DIR / "sensor_location_heatmap.pdf"
    )

    # 3. Spoofing difficulty chart
    generate_spoofing_difficulty_chart(
        OUTPUT_DIR / "sensor_spoofing_difficulty.pdf"
    )

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
