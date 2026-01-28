#!/usr/bin/env python3
"""
Generate t-SNE visualizations of decoder token embeddings using REAL model weights.

Creates 4 figures:
1. All tokens colored by type (Special, Command, Parameter, Numeric)
2. Command tokens only
3. Parameter tokens only
4. Numeric tokens colored by value magnitude
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Paths
BASE_DIR = Path(__file__).parent.parent
VOCAB_PATH = BASE_DIR / 'data' / 'vocabulary_4digit_full.json'
MODEL_PATH = BASE_DIR / 'outputs' / 'production' / 'best_model' / 'best_model.pt'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'archive' / '2024-12_best_runs' / 'sensor_multihead_v3' / 'visualizations' / 'paper' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vocabulary():
    """Load vocabulary and categorize tokens."""
    with open(VOCAB_PATH, 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data['vocab']

    # Categorize tokens
    categories = {
        'special': [],
        'command': [],
        'parameter': [],
        'numeric': []
    }

    token_to_idx = {}
    idx_to_token = {}

    for token, idx in vocab.items():
        token_to_idx[token] = idx
        idx_to_token[idx] = token

        # Categorize
        if token in ['PAD', 'BOS', 'EOS', 'UNK', 'MASK', 'SOS']:
            categories['special'].append((token, idx))
        elif token in ['G0', 'G1', 'G2', 'G3', 'G53', 'G17', 'G18', 'G19', 'G20', 'G21',
                      'G28', 'G40', 'G41', 'G42', 'G43', 'G49', 'G54', 'G80', 'G90', 'G91',
                      'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M8', 'M9', 'M30']:
            categories['command'].append((token, idx))
        elif token in ['X', 'Y', 'Z', 'A', 'B', 'C', 'F', 'S', 'I', 'J', 'K', 'R', 'P', 'Q', 'E', 'N', 'O', 'T', 'H', 'D', 'L']:
            categories['parameter'].append((token, idx))
        else:
            categories['numeric'].append((token, idx))

    print(f"  Vocabulary loaded: {len(vocab)} tokens")
    print(f"    Special: {len(categories['special'])}")
    print(f"    Command: {len(categories['command'])}")
    print(f"    Parameter: {len(categories['parameter'])}")
    print(f"    Numeric: {len(categories['numeric'])}")

    return vocab, categories, idx_to_token


def load_real_embeddings():
    """Load real token embeddings from trained model checkpoint."""
    print(f"  Loading model from: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Extract token embedding weights
    embeddings = state_dict['token_embedding.weight'].numpy()

    print(f"  Loaded embeddings: shape {embeddings.shape}")
    return embeddings


def create_all_tokens_tsne(embeddings, categories, idx_to_token):
    """Create t-SNE of all tokens colored by type."""
    print("Creating all tokens t-SNE...")

    # Prepare data
    all_indices = []
    all_types = []

    for cat_name, tokens in categories.items():
        for token, idx in tokens:
            all_indices.append(idx)
            all_types.append(cat_name)

    X = embeddings[all_indices]

    # Run t-SNE
    print("  Running t-SNE on", len(X), "tokens...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(50, len(X) - 1), max_iter=2000)
    X_tsne = tsne.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'special': '#e74c3c', 'command': '#3498db', 'parameter': '#2ecc71', 'numeric': '#9b59b6'}
    labels = {'special': 'Special (SOS/EOS)', 'command': 'Commands (G/M)', 'parameter': 'Parameters (X/Y/Z)', 'numeric': 'Numeric Values'}

    for cat_name in ['numeric', 'parameter', 'command', 'special']:
        mask = [t == cat_name for t in all_types]
        if sum(mask) > 0:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=colors[cat_name], label=f"{labels[cat_name]} ({sum(mask)})",
                      alpha=0.6, s=15 if cat_name == 'numeric' else 80,
                      edgecolors='white', linewidth=0.3)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Learned Token Embedding Space', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'token_tsne_all.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'token_tsne_all.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'token_tsne_all.pdf'}")
    plt.close()


def create_command_tsne(embeddings, categories, idx_to_token):
    """Create t-SNE of command tokens only."""
    print("Creating command tokens t-SNE...")

    tokens = categories['command']
    if len(tokens) < 3:
        print("  Not enough command tokens for t-SNE")
        return

    indices = [idx for _, idx in tokens]
    names = [token for token, _ in tokens]
    X = embeddings[indices]

    # Run t-SNE (or PCA for small samples)
    if len(X) < 10:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method = "PCA"
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X) - 1), max_iter=1000)
        X_reduced = tsne.fit_transform(X)
        method = "t-SNE"

    # Color by G vs M codes
    colors = ['#3498db' if n.startswith('G') else '#e74c3c' for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1)

    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=9, fontweight='bold',
                   xytext=(8, 4), textcoords='offset points')

    ax.set_xlabel(f'{method} Dimension 1')
    ax.set_ylabel(f'{method} Dimension 2')
    ax.set_title('Command Token Embeddings', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='G-codes'),
                      Patch(facecolor='#e74c3c', label='M-codes')]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'token_tsne_commands.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'token_tsne_commands.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'token_tsne_commands.pdf'}")
    plt.close()


def create_parameter_tsne(embeddings, categories, idx_to_token):
    """Create t-SNE of parameter tokens only."""
    print("Creating parameter tokens t-SNE...")

    tokens = categories['parameter']
    if len(tokens) < 3:
        print("  Not enough parameter tokens for t-SNE")
        return

    indices = [idx for _, idx in tokens]
    names = [token for token, _ in tokens]
    X = embeddings[indices]

    # Use PCA for small number of samples
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(X)

    # Color by parameter type
    color_map = {
        'X': '#e74c3c', 'Y': '#3498db', 'Z': '#2ecc71',
        'F': '#f39c12', 'S': '#9b59b6',
        'I': '#1abc9c', 'J': '#16a085', 'K': '#1abc9c',
        'R': '#e67e22', 'P': '#95a5a6', 'Q': '#7f8c8d'
    }
    colors = [color_map.get(n, '#34495e') for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, s=200, alpha=0.9, edgecolors='black', linewidth=1.5)

    # Add labels inside circles
    for i, name in enumerate(names):
        ax.annotate(name, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=11, fontweight='bold',
                   ha='center', va='center', color='white')

    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_title('Parameter Token Embeddings', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'token_tsne_parameters.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'token_tsne_parameters.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'token_tsne_parameters.pdf'}")
    plt.close()


def create_numeric_tsne(embeddings, categories, idx_to_token):
    """Create t-SNE of numeric tokens colored by value."""
    print("Creating numeric tokens t-SNE...")

    tokens = categories['numeric']
    if len(tokens) < 30:
        print("  Not enough numeric tokens for t-SNE")
        return

    # Sample if too many (for faster t-SNE)
    np.random.seed(42)
    if len(tokens) > 1000:
        indices_to_use = np.random.choice(len(tokens), 1000, replace=False)
        tokens = [tokens[i] for i in indices_to_use]

    indices = [idx for _, idx in tokens]
    names = [token for token, _ in tokens]
    X = embeddings[indices]

    # Extract numeric values for coloring
    values = []
    for name in names:
        try:
            if 'NUM_' in name:
                parts = name.replace('NUM_', '').split('_')
                if len(parts) >= 2:
                    val_str = parts[1]
                    # Handle negative numbers (often encoded with 'm' prefix)
                    if val_str.startswith('m'):
                        val = -float(val_str[1:]) / 10000
                    else:
                        val = float(val_str) / 10000
                    values.append(val)
                else:
                    values.append(0)
            else:
                # Try to parse as float
                clean = name.replace('m', '-').replace('_', '.')
                values.append(float(clean))
        except:
            values.append(0)

    values = np.array(values)

    # Run t-SNE
    print(f"  Running t-SNE on {len(X)} numeric tokens...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(50, len(X) - 1), max_iter=2000)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by value magnitude
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=values, cmap='coolwarm',
                        s=20, alpha=0.7, edgecolors='white', linewidth=0.2)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Numeric Value', fontsize=10)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Numeric Token Embeddings (colored by value)', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'token_tsne_numeric.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'token_tsne_numeric.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'token_tsne_numeric.pdf'}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Token Embedding t-SNE (REAL WEIGHTS)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load vocabulary
    vocab, categories, idx_to_token = load_vocabulary()

    # Load REAL embeddings from trained model
    print("\nLoading real embeddings from trained model...")
    embeddings = load_real_embeddings()

    # Generate figures
    print()
    create_all_tokens_tsne(embeddings, categories, idx_to_token)
    create_command_tsne(embeddings, categories, idx_to_token)
    create_parameter_tsne(embeddings, categories, idx_to_token)
    create_numeric_tsne(embeddings, categories, idx_to_token)

    print()
    print("=" * 60)
    print("Token t-SNE generation complete (using real model weights)!")
    print("=" * 60)


if __name__ == '__main__':
    main()
