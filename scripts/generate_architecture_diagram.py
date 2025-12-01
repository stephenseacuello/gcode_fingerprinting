#!/usr/bin/env python3
"""
Generate publication-quality architecture diagrams for the G-code fingerprinting project.

This script creates high-resolution visualizations of:
1. Complete neural network architecture with layer details
2. Data pipeline flowchart
3. Hierarchical token decomposition
4. System overview diagram

Output formats: PNG, SVG, PDF (for publications)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import argparse


# Color palette
COLORS = {
    'backbone': '#4A90E2',  # Blue
    'lm': '#E24A4A',        # Red
    'loss': '#F5A623',      # Orange
    'api': '#7ED321',       # Green
    'data': '#BD10E0',      # Purple
    'text': '#333333',
    'light_gray': '#F5F5F5',
    'border': '#CCCCCC'
}


def setup_figure(figsize=(16, 10), dpi=300):
    """Create high-resolution figure."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax


def draw_box(ax, x, y, width, height, text, color, fontsize=10, style='round'):
    """Draw a fancy box with text."""
    if style == 'round':
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor=COLORS['border'],
            facecolor=color,
            linewidth=2,
            alpha=0.9
        )
    else:
        box = FancyBboxPatch(
            (x, y), width, height,
            edgecolor=COLORS['border'],
            facecolor=color,
            linewidth=2,
            alpha=0.9
        )

    ax.add_patch(box)

    # Add text
    ax.text(
        x + width/2, y + height/2,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight='bold',
        color='white' if color != COLORS['light_gray'] else COLORS['text'],
        wrap=True
    )

    return box


def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        linewidth=2,
        color=COLORS['text'],
        alpha=0.7,
        mutation_scale=20
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=8, style='italic', color=COLORS['text'])


def generate_model_architecture(output_dir):
    """Generate detailed model architecture diagram."""
    fig, ax = setup_figure(figsize=(18, 14))

    # Title
    ax.text(50, 95, 'Complete Model Architecture', ha='center', fontsize=20,
            fontweight='bold', color=COLORS['text'])

    # Input layer
    draw_box(ax, 5, 85, 15, 6, 'Sensor Data\n[B, T, 8]', COLORS['data'], fontsize=9)
    draw_box(ax, 25, 85, 15, 6, 'Context\n[B, n_cats]', COLORS['data'], fontsize=9)

    # Backbone - MM_DTAE_LSTM
    y_start = 75
    draw_box(ax, 5, y_start, 35, 4, 'Linear Modality Encoder', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 12.5, 85, 12.5, y_start + 4)
    draw_arrow(ax, 32.5, 85, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'Positional Encoding (Sinusoidal)', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 75, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'Cross-Modal Fusion (Attention)', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 69, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'Add Noise + Mask', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 63, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'DTAE Encoder (2-layer Transformer)', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 57, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'DTAE Decoder (Reconstruction)', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 51, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'LSTM Layers (2-6 layers)', COLORS['backbone'], fontsize=9)
    draw_arrow(ax, 22.5, 45, 22.5, y_start + 4)

    y_start -= 6
    draw_box(ax, 5, y_start, 35, 4, 'Contextualized Memory\n[B, T, d_model]', COLORS['backbone'], fontsize=8)

    # Language Model - MultiHeadGCodeLM
    lm_x = 50
    lm_y = 75
    draw_arrow(ax, 40, 33 + 4, lm_x, lm_y + 4, label='Memory', style='->')

    draw_box(ax, lm_x, lm_y, 30, 4, 'Token Embedding', COLORS['lm'], fontsize=9)

    lm_y -= 6
    draw_box(ax, lm_x, lm_y, 30, 4, 'Positional Encoding', COLORS['lm'], fontsize=9)
    draw_arrow(ax, lm_x + 15, lm_y + 10, lm_x + 15, lm_y + 4)

    lm_y -= 6
    draw_box(ax, lm_x, lm_y, 30, 4, 'Causal Transformer Decoder\n(2-5 layers)', COLORS['lm'], fontsize=8)
    draw_arrow(ax, lm_x + 15, lm_y + 10, lm_x + 15, lm_y + 4)

    lm_y -= 6
    draw_box(ax, lm_x, lm_y, 30, 4, 'Decoder Hidden States\n[B, T, d_model]', COLORS['lm'], fontsize=8)
    draw_arrow(ax, lm_x + 15, lm_y + 10, lm_x + 15, lm_y + 4)

    # 5 Prediction Heads
    head_y = 48
    head_width = 12
    head_spacing = 15
    head_start_x = 10

    heads = [
        ('Type\n[B,T,4]', COLORS['loss']),
        ('Command\n[B,T,15]', COLORS['loss']),
        ('ParamType\n[B,T,10]', COLORS['loss']),
        ('ParamValue\n[B,T,1]', COLORS['loss']),
        ('Operation\n[B,10]', COLORS['loss'])
    ]

    for i, (head_text, color) in enumerate(heads):
        head_x = head_start_x + i * head_spacing
        draw_box(ax, head_x, head_y, head_width, 5, head_text, color, fontsize=7)
        draw_arrow(ax, lm_x + 15, lm_y, head_x + head_width/2, head_y + 5)

    # Loss computation
    loss_y = 38
    loss_names = ['CrossEnt\nw=1.0', 'Focal\nw=5.0,γ=2.5', 'CrossEnt\nw=3.0', 'Huber\nw=1.0,δ=1.0', 'CrossEnt\nw=2.0']

    for i, loss_name in enumerate(loss_names):
        loss_x = head_start_x + i * head_spacing
        draw_box(ax, loss_x, loss_y, head_width, 5, loss_name, COLORS['api'], fontsize=6)
        draw_arrow(ax, loss_x + head_width/2, head_y, loss_x + head_width/2, loss_y + 5)

    # Total loss
    draw_box(ax, 30, 28, 20, 5, 'Total Loss\n(Weighted Sum)', COLORS['api'], fontsize=9)
    for i in range(5):
        loss_x = head_start_x + i * head_spacing
        draw_arrow(ax, loss_x + head_width/2, loss_y, 40, 33)

    # Optimizer
    draw_box(ax, 30, 18, 20, 5, 'AdamW Optimizer\nlr=5e-5, wd=0.05', COLORS['api'], fontsize=8)
    draw_arrow(ax, 40, 28, 40, 23)

    draw_box(ax, 30, 8, 20, 5, 'Gradient Clipping\nmax_norm=1.0', COLORS['api'], fontsize=8)
    draw_arrow(ax, 40, 18, 40, 13)

    # Metrics (right side)
    metrics_x = 85
    ax.text(metrics_x, 80, 'Performance', ha='left', fontsize=12, fontweight='bold')

    metrics = [
        'Type: 99.8%',
        'Command: 100%',
        'ParamType: 84.3%',
        'ParamValue: 56.2%',
        'Operation: 92%'
    ]

    for i, metric in enumerate(metrics):
        y_pos = 75 - i * 5
        ax.text(metrics_x, y_pos, metric, ha='left', fontsize=9, color=COLORS['text'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['data'], label='Input Data'),
        mpatches.Patch(facecolor=COLORS['backbone'], label='MM_DTAE_LSTM Backbone'),
        mpatches.Patch(facecolor=COLORS['lm'], label='MultiHeadGCodeLM'),
        mpatches.Patch(facecolor=COLORS['loss'], label='Prediction Heads'),
        mpatches.Patch(facecolor=COLORS['api'], label='Loss & Optimization')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'model_architecture')


def generate_data_pipeline(output_dir):
    """Generate data pipeline flowchart."""
    fig, ax = setup_figure(figsize=(16, 12))

    # Title
    ax.text(50, 95, 'Data Processing Pipeline', ha='center', fontsize=20,
            fontweight='bold', color=COLORS['text'])

    # Stage 1: Raw Data
    draw_box(ax, 35, 85, 30, 5, 'Raw CSV Files (100 files)\n8 sensors + G-code labels', COLORS['data'], fontsize=9)

    # Stage 2: Vocabulary Building
    draw_box(ax, 5, 75, 20, 5, 'Scan G-code\ncommands', COLORS['backbone'], fontsize=8)
    draw_box(ax, 27, 75, 20, 5, 'Apply 2-digit\nbucketing', COLORS['backbone'], fontsize=8)
    draw_box(ax, 49, 75, 20, 5, 'Build vocab\n170 tokens', COLORS['backbone'], fontsize=8)
    draw_box(ax, 71, 75, 24, 5, 'Save\ngcode_vocab_v2.json', COLORS['api'], fontsize=8)

    draw_arrow(ax, 50, 85, 15, 80)
    draw_arrow(ax, 25, 77.5, 27, 77.5)
    draw_arrow(ax, 47, 77.5, 49, 77.5)
    draw_arrow(ax, 69, 77.5, 71, 77.5)

    # Stage 3: Preprocessing
    y = 65
    steps = [
        ('Sliding Window\nsize=64, stride=16', 10),
        ('Handle NaN\nForward fill', 10),
        ('Normalize\nRobustScaler', 10),
        ('Tokenize\nHierarchical', 10),
        ('Extract Raw\nValues', 10),
        ('Train/Val/Test\n70/15/15', 10),
        ('Save .npz\n3160 files', 10)
    ]

    x_start = 5
    for i, (text, width) in enumerate(steps):
        x = x_start + i * 13
        draw_box(ax, x, y, width, 5, text, COLORS['backbone'], fontsize=7)
        if i > 0:
            draw_arrow(ax, x - 3, y + 2.5, x, y + 2.5)

    draw_arrow(ax, 50, 75, 50, 70)

    # Stage 4: Augmentation
    y = 52
    ax.text(50, y + 8, 'Data Augmentation (Training Only)', ha='center',
            fontsize=11, fontweight='bold')

    aug_techniques = [
        'Gaussian\nNoise', 'Time\nWarping', 'Magnitude\nScaling',
        'Class\nOversampling'
    ]

    for i, tech in enumerate(aug_techniques):
        x = 15 + i * 18
        draw_box(ax, x, y, 12, 4, tech, COLORS['loss'], fontsize=7)
        draw_arrow(ax, 50, 65, x + 6, y + 4)

    # Stage 5: Training
    draw_box(ax, 35, 40, 30, 5, 'DataLoader (Batch size 32)', COLORS['lm'], fontsize=9)
    draw_arrow(ax, 50, 52, 50, 45)

    draw_box(ax, 35, 30, 30, 5, 'Model Training Loop', COLORS['lm'], fontsize=9)
    draw_arrow(ax, 50, 40, 50, 35)

    # Stage 6: Outputs
    y = 20
    outputs = [
        ('Checkpoints', 15), ('Metrics', 15), ('Figures', 15), ('W&B Logs', 15)
    ]

    x_start = 20
    for i, (text, width) in enumerate(outputs):
        x = x_start + i * 17
        draw_box(ax, x, y, width, 4, text, COLORS['api'], fontsize=8)
        draw_arrow(ax, 50, 30, x + width/2, y + 4)

    # Statistics
    stats_x = 5
    stats_y = 10
    ax.text(stats_x, stats_y, 'Dataset Statistics:', ha='left', fontsize=10, fontweight='bold')
    stats = [
        'Total sequences: 3,160',
        'Train: 2,212 (70%)',
        'Val: 474 (15%)',
        'Test: 474 (15%)',
        'Sequence length: 64 timesteps',
        'Features: 8 sensors',
        'Vocab size: 170 tokens'
    ]

    for i, stat in enumerate(stats):
        ax.text(stats_x, stats_y - (i+1) * 1.2, f'• {stat}', ha='left', fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, 'data_pipeline')


def generate_token_decomposition(output_dir):
    """Generate hierarchical token decomposition diagram."""
    fig, ax = setup_figure(figsize=(14, 10))

    # Title
    ax.text(50, 95, 'Hierarchical Token Decomposition', ha='center', fontsize=20,
            fontweight='bold', color=COLORS['text'])

    # Traditional approach (problem)
    ax.text(25, 85, 'Traditional Approach', ha='center', fontsize=14, fontweight='bold')
    draw_box(ax, 10, 75, 30, 6, 'Single Softmax\n170-class vocabulary', COLORS['data'], fontsize=9)

    draw_arrow(ax, 25, 75, 25, 67)

    draw_box(ax, 10, 60, 30, 6, 'Severe Class Imbalance\nG1: 45%, X50: 0.3%', '#E24A4A', fontsize=8)

    draw_arrow(ax, 25, 60, 25, 52)

    draw_box(ax, 10, 45, 30, 6, 'Mode Collapse\nPoor rare token learning', '#E24A4A', fontsize=8)

    # Hierarchical approach (solution)
    ax.text(70, 85, 'Hierarchical Approach (Our Method)', ha='center', fontsize=14, fontweight='bold')

    # 5 heads
    heads = [
        ('Head 1\nToken Type\n4 classes\n99.8% acc', 55, 72),
        ('Head 2\nCommand\n15 classes\n100% acc', 65, 72),
        ('Head 3\nParam Type\n10 classes\n84.3% acc', 75, 72),
        ('Head 4\nParam Value\nRegression\n56.2% acc', 85, 72),
        ('Head 5\nOperation\n10 classes\n92% acc', 70, 62)
    ]

    for text, x, y in heads:
        draw_box(ax, x - 4, y, 8, 8, text, COLORS['api'], fontsize=7)

    # Convergence
    draw_box(ax, 63, 50, 14, 6, 'Token\nReconstruction', COLORS['api'], fontsize=9)

    for _, x, y in heads:
        draw_arrow(ax, x, y, 70, 56)

    # Example decompositions
    ax.text(50, 38, 'Example Decompositions', ha='center', fontsize=14, fontweight='bold')

    # Example 1
    ex1_x = 10
    ex1_y = 30
    draw_box(ax, ex1_x, ex1_y, 15, 3, "Token: 'X120.5'", COLORS['data'], fontsize=8)

    decomp1 = [
        'Type: PARAM (2/4)',
        'Command: PAD',
        'ParamType: X (0/10)',
        'ParamValue: 120.5',
        'Operation: adaptive'
    ]

    for i, d in enumerate(decomp1):
        y = ex1_y - (i+1) * 3.5
        color = COLORS['api'] if 'PAD' not in d else COLORS['light_gray']
        draw_box(ax, ex1_x, y, 15, 2.5, d, color, fontsize=7)
        draw_arrow(ax, ex1_x + 7.5, ex1_y - i * 3.5, ex1_x + 7.5, ex1_y - (i+1) * 3.5 + 2.5)

    # Example 2
    ex2_x = 42
    ex2_y = 30
    draw_box(ax, ex2_x, ex2_y, 15, 3, "Token: 'G1'", COLORS['data'], fontsize=8)

    decomp2 = [
        'Type: CMD (1/4)',
        'Command: G1 (1/15)',
        'ParamType: PAD',
        'ParamValue: PAD',
        'Operation: face'
    ]

    for i, d in enumerate(decomp2):
        y = ex2_y - (i+1) * 3.5
        color = COLORS['api'] if 'PAD' not in d else COLORS['light_gray']
        draw_box(ax, ex2_x, y, 15, 2.5, d, color, fontsize=7)
        draw_arrow(ax, ex2_x + 7.5, ex2_y - i * 3.5, ex2_x + 7.5, ex2_y - (i+1) * 3.5 + 2.5)

    # Example 3
    ex3_x = 74
    ex3_y = 30
    draw_box(ax, ex3_x, ex3_y, 15, 3, "Token: 'NUM_F_50'", COLORS['data'], fontsize=8)

    decomp3 = [
        'Type: NUMERIC (3/4)',
        'Command: PAD',
        'ParamType: F (3/10)',
        'ParamValue: 50.0',
        'Operation: pocket'
    ]

    for i, d in enumerate(decomp3):
        y = ex3_y - (i+1) * 3.5
        color = COLORS['api'] if 'PAD' not in d else COLORS['light_gray']
        draw_box(ax, ex3_x, y, 15, 2.5, d, color, fontsize=7)
        draw_arrow(ax, ex3_x + 7.5, ex3_y - i * 3.5, ex3_x + 7.5, ex3_y - (i+1) * 3.5 + 2.5)

    plt.tight_layout()
    save_figure(fig, output_dir, 'token_decomposition')


def generate_system_overview(output_dir):
    """Generate high-level system overview."""
    fig, ax = setup_figure(figsize=(16, 10))

    # Title
    ax.text(50, 95, 'System Overview', ha='center', fontsize=20,
            fontweight='bold', color=COLORS['text'])

    # Layer 1: Data
    ax.text(10, 82, 'Data Layer', ha='left', fontsize=12, fontweight='bold')
    draw_box(ax, 10, 72, 18, 6, 'Raw CSV Files\n100 files', COLORS['data'], fontsize=9)
    draw_box(ax, 30, 72, 18, 6, 'Preprocessing\nWindowing', COLORS['data'], fontsize=9)
    draw_box(ax, 50, 72, 18, 6, 'Processed Data\n3160 sequences', COLORS['data'], fontsize=9)

    draw_arrow(ax, 28, 75, 30, 75)
    draw_arrow(ax, 48, 75, 50, 75)

    # Layer 2: Model
    ax.text(10, 62, 'Model Layer', ha='left', fontsize=12, fontweight='bold')
    draw_box(ax, 10, 52, 18, 6, 'Dataset Loader\nPyTorch', COLORS['backbone'], fontsize=9)
    draw_box(ax, 30, 52, 18, 6, 'MM_DTAE_LSTM\nBackbone', COLORS['backbone'], fontsize=9)
    draw_box(ax, 50, 52, 18, 6, 'MultiHeadLM\n5 Heads', COLORS['lm'], fontsize=9)

    draw_arrow(ax, 59, 72, 19, 58)
    draw_arrow(ax, 28, 55, 30, 55)
    draw_arrow(ax, 48, 55, 50, 55)

    # Layer 3: Training
    ax.text(10, 42, 'Training Layer', ha='left', fontsize=12, fontweight='bold')
    draw_box(ax, 10, 32, 18, 6, 'Loss Function\nFocal + Huber', COLORS['loss'], fontsize=9)
    draw_box(ax, 30, 32, 18, 6, 'Optimizer\nAdamW', COLORS['loss'], fontsize=9)
    draw_box(ax, 50, 32, 18, 6, 'Checkpoints\nBest Model', COLORS['api'], fontsize=9)

    draw_arrow(ax, 59, 52, 19, 38)
    draw_arrow(ax, 28, 35, 30, 35)
    draw_arrow(ax, 48, 35, 50, 35)

    # Layer 4: Evaluation
    ax.text(10, 22, 'Evaluation Layer', ha='left', fontsize=12, fontweight='bold')
    draw_box(ax, 10, 12, 18, 6, 'Metrics\nAccuracy, F1', COLORS['api'], fontsize=9)
    draw_box(ax, 30, 12, 18, 6, 'Visualization\n14 Figures', COLORS['api'], fontsize=9)
    draw_box(ax, 50, 12, 18, 6, 'Analysis\nBootstrap CI', COLORS['api'], fontsize=9)

    draw_arrow(ax, 59, 32, 19, 18)

    # Layer 5: Deployment
    ax.text(72, 62, 'Deployment', ha='left', fontsize=12, fontweight='bold')
    draw_box(ax, 72, 52, 22, 6, 'FastAPI Server\nREST Endpoints', COLORS['api'], fontsize=9)
    draw_box(ax, 72, 42, 22, 6, 'Inference\n67 req/s (M1)', COLORS['api'], fontsize=9)

    draw_arrow(ax, 59, 35, 72, 55)
    draw_arrow(ax, 83, 52, 83, 48)

    # Performance metrics (bottom right)
    perf_x = 72
    perf_y = 30
    ax.text(perf_x, perf_y, 'Performance', ha='left', fontsize=12, fontweight='bold')

    metrics = [
        'Token Type: 99.8%',
        'Command: 100%',
        'Param Type: 84.3%',
        'Param Value: 56.2%',
        'Operation: 92%',
        '',
        'Latency: 15ms',
        'Model Size: 2.5-12M params'
    ]

    for i, metric in enumerate(metrics):
        if metric:
            ax.text(perf_x, perf_y - (i+1) * 2, f'• {metric}', ha='left', fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, 'system_overview')


def save_figure(fig, output_dir, name):
    """Save figure in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    formats = ['png', 'svg', 'pdf']
    for fmt in formats:
        filepath = output_path / f'{name}.{fmt}'
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {filepath}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate architecture diagrams for G-code fingerprinting project'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/architecture_diagrams',
        help='Output directory for generated diagrams'
    )
    parser.add_argument(
        '--diagrams',
        nargs='+',
        choices=['model', 'pipeline', 'decomposition', 'overview', 'all'],
        default=['all'],
        help='Which diagrams to generate'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nGenerating architecture diagrams...')
    print(f'Output directory: {output_dir}\n')

    diagrams_to_generate = args.diagrams
    if 'all' in diagrams_to_generate:
        diagrams_to_generate = ['model', 'pipeline', 'decomposition', 'overview']

    if 'model' in diagrams_to_generate:
        print('Generating model architecture diagram...')
        generate_model_architecture(output_dir)

    if 'pipeline' in diagrams_to_generate:
        print('Generating data pipeline diagram...')
        generate_data_pipeline(output_dir)

    if 'decomposition' in diagrams_to_generate:
        print('Generating token decomposition diagram...')
        generate_token_decomposition(output_dir)

    if 'overview' in diagrams_to_generate:
        print('Generating system overview diagram...')
        generate_system_overview(output_dir)

    print('\nDone! All diagrams generated successfully.')
    print(f'Output formats: PNG (300 DPI), SVG (vector), PDF (publication-ready)')


if __name__ == '__main__':
    main()
