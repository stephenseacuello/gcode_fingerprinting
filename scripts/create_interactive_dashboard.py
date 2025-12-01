#!/usr/bin/env python3
"""
Interactive G-Code Reconstruction Dashboard using Plotly
Creates an interactive HTML dashboard for exploring reconstruction results.
"""

import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np


def parse_demo_outputs(json_path: Path):
    """Parse the demo outputs JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_interactive_dashboard(samples, output_path: Path):
    """
    Create an interactive Plotly dashboard with multiple visualizations.
    """

    # Calculate metrics for all samples
    similarities = []
    pred_token_counts = []
    actual_token_counts = []
    valid_recon = []

    for sample in samples:
        matcher = SequenceMatcher(None, sample['reconstructed_gcode'], sample['actual_gcode'])
        similarities.append(matcher.ratio() * 100)
        pred_token_counts.append(len(sample['predicted_tokens'].split()))
        actual_token_counts.append(len(sample['actual_gcode'].split()))
        valid_recon.append('Valid' if sample['reconstructed_gcode'] != "[empty]" else 'Invalid')

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Similarity Score Distribution',
            'Valid vs Invalid Reconstructions',
            'Token Count Comparison',
            'Sample-by-Sample Similarity',
            'Reconstruction Success Rate',
            'Quality Score Matrix'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Similarity histogram
    fig.add_trace(
        go.Histogram(
            x=similarities,
            nbinsx=20,
            name='Similarity',
            marker_color='#3498db',
            hovertemplate='Similarity: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Valid/Invalid pie chart
    valid_count = sum(1 for v in valid_recon if v == 'Valid')
    invalid_count = len(valid_recon) - valid_count

    fig.add_trace(
        go.Pie(
            labels=['Valid', 'Invalid'],
            values=[valid_count, invalid_count],
            marker_colors=['#2ecc71', '#e74c3c'],
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Token count comparison
    sample_ids = [f"S{s['sample_id']}" for s in samples]

    fig.add_trace(
        go.Bar(
            x=sample_ids,
            y=pred_token_counts,
            name='Predicted',
            marker_color='#9b59b6',
            hovertemplate='Sample: %{x}<br>Predicted Tokens: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=sample_ids,
            y=actual_token_counts,
            name='Actual',
            marker_color='#16a085',
            hovertemplate='Sample: %{x}<br>Actual Tokens: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Sample-by-sample similarity scatter
    colors = ['#2ecc71' if v == 'Valid' else '#e74c3c' for v in valid_recon]

    fig.add_trace(
        go.Scatter(
            x=list(range(len(samples))),
            y=similarities,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=2, color='black')
            ),
            text=[f"Sample {s['sample_id']}<br>Predicted: {s['reconstructed_gcode']}<br>Actual: {s['actual_gcode']}"
                  for s in samples],
            hovertemplate='%{text}<br>Similarity: %{y:.1f}%<extra></extra>',
            name='Samples'
        ),
        row=2, col=2
    )

    # 5. Success rate bar
    success_rate = (valid_count / len(samples)) * 100
    failure_rate = 100 - success_rate

    fig.add_trace(
        go.Bar(
            x=['Success', 'Failure'],
            y=[success_rate, failure_rate],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f'{success_rate:.1f}%', f'{failure_rate:.1f}%'],
            textposition='auto',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )

    # 6. Quality score heatmap
    quality_matrix = []
    max_tokens = max(len(s['predicted_tokens'].split()) for s in samples)

    for sample in samples:
        pred_tokens = sample['predicted_tokens'].split()
        actual_tokens = sample['actual_gcode'].split()

        # Create quality row (1 = good match, 0 = poor match)
        quality_row = []
        for i in range(max_tokens):
            if i < len(pred_tokens):
                # Simple heuristic: token exists and reconstruction is valid
                if sample['reconstructed_gcode'] != "[empty]":
                    quality_row.append(0.5)
                else:
                    quality_row.append(0.2)
            else:
                quality_row.append(0)
        quality_matrix.append(quality_row)

    fig.add_trace(
        go.Heatmap(
            z=quality_matrix,
            colorscale='RdYlGn',
            hovertemplate='Sample: %{y}<br>Token Position: %{x}<br>Quality: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="G-Code Reconstruction: Interactive Dashboard",
        title_font_size=20,
        title_x=0.5,
        showlegend=True,
        height=1200,
        hovermode='closest',
        font=dict(size=11)
    )

    # Update axes
    fig.update_xaxes(title_text="Similarity Score (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)

    fig.update_xaxes(title_text="Sample ID", row=2, col=1)
    fig.update_yaxes(title_text="Token Count", row=2, col=1)

    fig.update_xaxes(title_text="Sample ID", row=2, col=2)
    fig.update_yaxes(title_text="Similarity (%)", row=2, col=2)

    fig.update_yaxes(title_text="Percentage (%)", row=3, col=1)

    fig.update_xaxes(title_text="Token Position", row=3, col=2)
    fig.update_yaxes(title_text="Sample ID", row=3, col=2)

    # Save to HTML
    html_file = output_path / 'interactive_dashboard.html'
    fig.write_html(
        html_file,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    )

    print(f"✓ Interactive dashboard saved: {html_file}")
    return html_file


def create_detailed_comparison_table(samples, output_path: Path):
    """
    Create an interactive table showing detailed sample-by-sample comparison.
    """

    # Prepare data
    sample_ids = [s['sample_id'] for s in samples]
    predicted_tokens = [s['predicted_tokens'] for s in samples]
    reconstructed = [s['reconstructed_gcode'] for s in samples]
    actual = [s['actual_gcode'] for s in samples]
    similarities = []

    for sample in samples:
        matcher = SequenceMatcher(None, sample['reconstructed_gcode'], sample['actual_gcode'])
        similarities.append(f"{matcher.ratio() * 100:.1f}%")

    # Create table
    fig = go.Figure(data=[go.Table(
        columnwidth=[50, 200, 150, 150, 80],
        header=dict(
            values=['<b>Sample ID</b>', '<b>Predicted Tokens</b>',
                   '<b>Reconstructed</b>', '<b>Actual G-Code</b>', '<b>Similarity</b>'],
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='center',
            height=35
        ),
        cells=dict(
            values=[sample_ids, predicted_tokens, reconstructed, actual, similarities],
            fill_color=[['white', '#ecf0f1'] * len(samples)],
            align=['center', 'left', 'left', 'left', 'center'],
            font=dict(size=10),
            height=30
        )
    )])

    fig.update_layout(
        title_text="G-Code Reconstruction: Detailed Comparison Table",
        title_font_size=16,
        title_x=0.5,
        height=800
    )

    # Save to HTML
    html_file = output_path / 'comparison_table.html'
    fig.write_html(html_file)

    print(f"✓ Comparison table saved: {html_file}")
    return html_file


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive G-code reconstruction dashboard'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='reports/paper_gcode_reconstruction/demo_outputs.json',
        help='Path to demo outputs JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/gcode_reconstruction',
        help='Output directory for HTML files'
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("INTERACTIVE G-CODE RECONSTRUCTION DASHBOARD")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    # Load data
    print("Loading data...")
    samples = parse_demo_outputs(input_path)
    print(f"✓ Loaded {len(samples)} samples\n")

    # Create visualizations
    print("Creating interactive visualizations...\n")

    dashboard_file = create_interactive_dashboard(samples, output_path)
    table_file = create_detailed_comparison_table(samples, output_path)

    print(f"\n{'='*80}")
    print("✅ INTERACTIVE DASHBOARD COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Open in browser:")
    print(f"  - {dashboard_file}")
    print(f"  - {table_file}")
    print("")


if __name__ == '__main__':
    main()
