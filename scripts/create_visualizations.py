#!/usr/bin/env python3
"""
Create comprehensive visualizations for the G-code fingerprinting project.
Generates: loss curves, embeddings (t-SNE, UMAP), confusion matrices, attention maps.
"""
import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse

# Try importing optional dependencies
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: sklearn not available, t-SNE will be skipped")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap not available, UMAP will be skipped")

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.utilities.device import get_device, print_device_info
from torch.utils.data import DataLoader


def plot_loss_curves(history_path: Path, output_dir: Path):
    """Plot training and validation loss curves."""
    print("📊 Creating loss curves...")

    with open(history_path, 'r') as f:
        content = f.read()
        # Handle incomplete JSON by adding closing brackets if needed
        if not content.strip().endswith('}'):
            content = content.rsplit(',', 1)[0] + '\n  ]\n}'
        history = json.loads(content)

    train_losses = [epoch['gcode'] for epoch in history['train']]
    val_losses = [epoch['gcode'] for epoch in history['val']]
    epochs = list(range(1, len(train_losses) + 1))

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Full training curve
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Progress ({len(epochs)} Epochs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Log scale
    ax = axes[0, 1]
    ax.semilogy(epochs, train_losses, label='Training Loss', linewidth=2, alpha=0.8)
    ax.semilogy(epochs, val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Last 50 epochs (convergence)
    ax = axes[1, 0]
    start_idx = max(0, len(epochs) - 50)
    ax.plot(epochs[start_idx:], train_losses[start_idx:], label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs[start_idx:], val_losses[start_idx:], label='Validation Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Convergence (Last 50 Epochs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Improvement over time
    ax = axes[1, 1]
    initial_loss = train_losses[0]
    improvement = [(initial_loss - loss) / initial_loss * 100 for loss in train_losses]
    ax.plot(epochs, improvement, linewidth=2, color='green', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Loss Reduction Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved loss curves to {output_path}")
    plt.close()

    return train_losses, val_losses


def extract_embeddings(model, dataloader, device=None):
    """Extract embeddings from the model."""
    print("🔍 Extracting embeddings...")

    model.eval()
    all_embeddings = []
    all_labels = []
    all_gcode_texts = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].float().to(device)
            tokens = batch['tokens']
            lengths = batch['lengths'].to(device)
            gcode_texts = batch['gcode_texts']

            # Get embeddings
            mods = [continuous, categorical]
            outputs = model(mods=mods, lengths=lengths, modality_dropout_p=0.0)

            # Extract fingerprint embeddings if available
            if 'fingerprint' in outputs:
                embeddings = outputs['fingerprint'].cpu().numpy()
            else:
                # Use memory/hidden states
                embeddings = outputs['memory'][:, 0, :].cpu().numpy()  # Use first timestep

            all_embeddings.append(embeddings)
            all_labels.extend([t[0].item() for t in tokens])  # First token as label
            all_gcode_texts.extend(gcode_texts)

    all_embeddings = np.vstack(all_embeddings)
    print(f"✅ Extracted {len(all_embeddings)} embeddings of dimension {all_embeddings.shape[1]}")

    return all_embeddings, all_labels, all_gcode_texts


def plot_embeddings_tsne(embeddings, labels, gcode_texts, output_dir, vocab=None):
    """Plot t-SNE visualization of embeddings."""
    if not TSNE_AVAILABLE:
        print("⚠️  Skipping t-SNE (sklearn not available)")
        return

    print("📊 Creating t-SNE visualization...")

    # Load vocabulary and create reverse mapping (ID -> token text)
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_token = {v: k for k, v in vocab_data['vocab'].items()}

    # Decode labels to actual G-code text
    decoded_labels = [id_to_token.get(label_id, f'UNK_{label_id}') for label_id in labels]

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get unique labels (use decoded text)
    unique_labels_ids = list(set(labels))[:20]  # Limit to 20 for visibility
    unique_labels_text = [id_to_token.get(label_id, f'UNK_{label_id}') for label_id in unique_labels_ids]

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_ids)))

    for i, (label_id, label_text) in enumerate(zip(unique_labels_ids, unique_labels_text)):
        mask = np.array([l == label_id for l in labels])
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=label_text, alpha=0.6, s=50)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Embeddings', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'embeddings_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved t-SNE visualization to {output_path}")
    plt.close()


def plot_embeddings_umap(embeddings, labels, gcode_texts, output_dir, vocab=None):
    """Plot UMAP visualization of embeddings."""
    if not UMAP_AVAILABLE:
        print("⚠️  Skipping UMAP (umap-learn not available)")
        return

    print("📊 Creating UMAP visualization...")

    # Load vocabulary and create reverse mapping (ID -> token text)
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_token = {v: k for k, v in vocab_data['vocab'].items()}

    # Reduce to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1))
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get unique labels (use decoded text)
    unique_labels_ids = list(set(labels))[:20]
    unique_labels_text = [id_to_token.get(label_id, f'UNK_{label_id}') for label_id in unique_labels_ids]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_ids)))

    for i, (label_id, label_text) in enumerate(zip(unique_labels_ids, unique_labels_text)):
        mask = np.array([l == label_id for l in labels])
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=label_text, alpha=0.6, s=50)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('UMAP Visualization of Embeddings', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'embeddings_umap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved UMAP visualization to {output_path}")
    plt.close()


def plot_training_summary(history_path: Path, output_dir: Path):
    """Create summary statistics plot."""
    print("📊 Creating training summary...")

    with open(history_path, 'r') as f:
        content = f.read()
        if not content.strip().endswith('}'):
            content = content.rsplit(',', 1)[0] + '\n  ]\n}'
        history = json.loads(content)

    train_losses = [epoch['gcode'] for epoch in history['train']]
    val_losses = [epoch['gcode'] for epoch in history['val']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Statistics table
    ax = axes[0]
    ax.axis('off')

    stats = [
        ['Metric', 'Value'],
        ['Total Epochs', f"{len(train_losses)}"],
        ['Initial Train Loss', f"{train_losses[0]:.4f}"],
        ['Final Train Loss', f"{train_losses[-1]:.4f}"],
        ['Improvement', f"{(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%"],
        ['', ''],
        ['Initial Val Loss', f"{val_losses[0]:.4f}"],
        ['Final Val Loss', f"{val_losses[-1]:.4f}"],
        ['Best Val Loss', f"{min(val_losses):.4f}"],
        ['', ''],
        ['Overfitting', 'No' if val_losses[-1] <= train_losses[-1] * 1.1 else 'Possible'],
    ]

    table = ax.table(cellText=stats, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Training Statistics', fontsize=14, fontweight='bold', pad=20)

    # 2. Loss distribution
    ax = axes[1]
    ax.hist(train_losses, bins=30, alpha=0.6, label='Training', color='blue', edgecolor='black')
    ax.hist(val_losses, bins=30, alpha=0.6, label='Validation', color='orange', edgecolor='black')
    ax.set_xlabel('Loss Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning rate decay simulation (if we had LR schedule)
    ax = axes[2]
    epochs = list(range(1, len(train_losses) + 1))
    smoothing_window = min(10, len(train_losses) // 10)
    if smoothing_window > 1:
        smooth_train = np.convolve(train_losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        smooth_val = np.convolve(val_losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        smooth_epochs = epochs[smoothing_window-1:]
        ax.plot(smooth_epochs, smooth_train, label='Smoothed Train', linewidth=2, color='blue')
        ax.plot(smooth_epochs, smooth_val, label='Smoothed Val', linewidth=2, color='orange')
    else:
        ax.plot(epochs, train_losses, label='Training', linewidth=2, color='blue')
        ax.plot(epochs, val_losses, label='Validation', linewidth=2, color='orange')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'training_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training summary to {output_path}")
    plt.close()


def get_predictions(model, dataloader, device):
    """Get model predictions on a dataset."""
    all_predictions = []
    all_targets = []
    all_gcode_texts = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].float().to(device)
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)

            mods = [continuous, categorical]
            outputs = model(mods=mods, lengths=lengths, gcode_in=tokens, modality_dropout_p=0.0)

            if 'gcode_logits' in outputs:
                preds = torch.argmax(outputs['gcode_logits'], dim=-1)
                all_predictions.append(preds.cpu())
                all_targets.append(tokens.cpu())
                all_gcode_texts.extend(batch['gcode_texts'])

    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

    return all_predictions, all_targets, all_gcode_texts


def plot_confusion_matrix(predictions, targets, output_dir, top_n=30):
    """Plot confusion matrix for top N most common tokens."""
    print("📊 Creating confusion matrix...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_token = {v: k for k, v in vocab_data['vocab'].items()}

    # Flatten predictions and targets
    preds_flat = predictions.flatten().numpy()
    targets_flat = targets.flatten().numpy()

    # Get top N most common tokens
    unique, counts = np.unique(targets_flat, return_counts=True)
    top_indices = unique[np.argsort(counts)[-top_n:]]

    # Filter to only top tokens
    mask = np.isin(targets_flat, top_indices)
    preds_filtered = preds_flat[mask]
    targets_filtered = targets_flat[mask]

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets_filtered, preds_filtered, labels=top_indices)

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Plot
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    token_labels = [id_to_token.get(idx, f'T{idx}') for idx in top_indices]
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_yticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(token_labels, fontsize=9)

    # Annotate cells
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            if cm_norm[i, j] > 0.01:  # Only show significant values
                ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > thresh else "black",
                       fontsize=7)

    ax.set_title(f'Confusion Matrix (Top {top_n} Tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Token', fontsize=12)
    ax.set_xlabel('Predicted Token', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix to {output_path}")
    plt.close()


def plot_per_token_type_accuracy(predictions, targets, output_dir):
    """Plot accuracy breakdown by token type (G-codes, M-codes, numerics, etc.)."""
    print("📊 Creating per-token-type accuracy breakdown...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Categorize tokens
    g_ids = set(idx for token, idx in vocab.items() if token.startswith('G') and len(token) <= 3)
    m_ids = set(idx for token, idx in vocab.items() if token.startswith('M') and len(token) <= 3)
    axis_ids = set(idx for token, idx in vocab.items() if token in ['X', 'Y', 'Z', 'A', 'B', 'C'])
    num_ids = set(idx for token, idx in vocab.items() if token.startswith('NUM_'))
    special_ids = set([vocab.get('PAD', 0), vocab.get('BOS', 1), vocab.get('EOS', 2), vocab.get('UNK', 3)])

    # Flatten
    preds_flat = predictions.flatten().numpy()
    targets_flat = targets.flatten().numpy()

    # Calculate accuracies
    categories = {
        'G-Commands': g_ids,
        'M-Commands': m_ids,
        'Axis Params': axis_ids,
        'Numeric Values': num_ids,
        'Special Tokens': special_ids,
    }

    accuracies = {}
    counts = {}

    for cat_name, cat_ids in categories.items():
        mask = np.isin(targets_flat, list(cat_ids))
        if mask.sum() > 0:
            acc = (preds_flat[mask] == targets_flat[mask]).mean() * 100
            accuracies[cat_name] = acc
            counts[cat_name] = mask.sum()
        else:
            accuracies[cat_name] = 0
            counts[cat_name] = 0

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Accuracy
    categories_list = list(accuracies.keys())
    accs = [accuracies[c] for c in categories_list]
    colors = ['#2ecc71' if acc > 90 else '#f39c12' if acc > 70 else '#e74c3c' for acc in accs]

    bars = ax1.barh(categories_list, accs, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy by Token Type', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 105])
    ax1.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (cat, acc) in enumerate(zip(categories_list, accs)):
        ax1.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Plot 2: Sample counts
    cnts = [counts[c] for c in categories_list]
    ax2.barh(categories_list, cnts, color='#3498db', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Number of Tokens', fontsize=12)
    ax2.set_title('Token Count by Type', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (cat, cnt) in enumerate(zip(categories_list, cnts)):
        ax2.text(cnt + max(cnts)*0.02, i, f'{cnt:,}', va='center', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'token_type_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved token type accuracy to {output_path}")
    plt.close()


def plot_prediction_examples(predictions, targets, gcode_texts, output_dir, n_examples=10):
    """Show side-by-side prediction examples."""
    print("📊 Creating prediction examples gallery...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_token = {v: k for k, v in vocab_data['vocab'].items()}

    # Calculate accuracy per sequence
    accuracies = []
    for i in range(len(predictions)):
        acc = (predictions[i] == targets[i]).float().mean().item()
        accuracies.append(acc)

    # Get best, worst, and median examples
    sorted_indices = np.argsort(accuracies)
    example_indices = []
    example_indices.extend(sorted_indices[-3:].tolist())  # Best 3
    example_indices.extend(sorted_indices[:3].tolist())    # Worst 3
    example_indices.extend(sorted_indices[len(sorted_indices)//2:len(sorted_indices)//2+4].tolist())  # Middle 4

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(n_examples, 1, hspace=0.4)

    for plot_idx, seq_idx in enumerate(example_indices[:n_examples]):
        ax = fig.add_subplot(gs[plot_idx, 0])
        ax.axis('off')

        # Decode sequences
        pred_tokens = [id_to_token.get(idx.item(), 'UNK') for idx in predictions[seq_idx][:20]]
        true_tokens = [id_to_token.get(idx.item(), 'UNK') for idx in targets[seq_idx][:20]]

        # Create comparison text
        pred_text = ' '.join(pred_tokens)
        true_text = ' '.join(true_tokens)
        acc = accuracies[seq_idx] * 100

        # Color code
        if acc >= 90:
            color, status = '#2ecc71', '✅ Excellent'
        elif acc >= 70:
            color, status = '#f39c12', '⚠️  Good'
        else:
            color, status = '#e74c3c', '❌ Poor'

        # Display
        title = f"Example {plot_idx + 1} - Accuracy: {acc:.1f}% ({status})"
        info = f"True:      {true_text}\nPredicted: {pred_text}"

        ax.text(0.5, 0.5, info, transform=ax.transAxes,
               fontsize=9, verticalalignment='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2, pad=0.5))
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

    fig.suptitle('Prediction Examples Gallery (First 20 tokens shown)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'prediction_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved prediction examples to {output_path}")
    plt.close()


def plot_token_frequency_distribution(targets, output_dir):
    """Plot token frequency distribution showing class imbalance."""
    print("📊 Creating token frequency distribution...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id_to_token = {v: k for k, v in vocab.items()}

    # Get token frequencies
    targets_flat = targets.flatten().numpy()
    unique, counts = np.unique(targets_flat, return_counts=True)

    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    unique_sorted = unique[sorted_indices]
    counts_sorted = counts[sorted_indices]

    # Categorize tokens
    g_ids = set(idx for token, idx in vocab.items() if token.startswith('G') and len(token) <= 3)
    m_ids = set(idx for token, idx in vocab.items() if token.startswith('M') and len(token) <= 3)
    num_ids = set(idx for token, idx in vocab.items() if token.startswith('NUM_'))

    colors = []
    for token_id in unique_sorted[:50]:
        if token_id in g_ids:
            colors.append('#2ecc71')  # Green for G-codes
        elif token_id in m_ids:
            colors.append('#3498db')  # Blue for M-codes
        elif token_id in num_ids:
            colors.append('#95a5a6')  # Gray for numerics
        else:
            colors.append('#e74c3c')  # Red for others

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Top 50 tokens (linear scale)
    top_n = 50
    token_labels = [id_to_token.get(idx, f'T{idx}') for idx in unique_sorted[:top_n]]
    ax1.bar(range(top_n), counts_sorted[:top_n], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Token Frequency Distribution (Top {top_n} Tokens)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(token_labels, rotation=90, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='G-Commands'),
        Patch(facecolor='#3498db', label='M-Commands'),
        Patch(facecolor='#95a5a6', label='Numeric Values'),
        Patch(facecolor='#e74c3c', label='Other Tokens')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Plot 2: Log scale (all tokens)
    ax2.bar(range(len(unique_sorted)), counts_sorted, color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Token Rank', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_title('Full Token Distribution (Log Scale) - Shows Class Imbalance', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, which='both')

    # Highlight imbalance
    max_freq = counts_sorted[0]
    min_freq = counts_sorted[-1]
    ratio = max_freq / min_freq if min_freq > 0 else float('inf')
    ax2.text(0.5, 0.95, f'Imbalance Ratio: {ratio:.1f}:1 (max/min frequency)',
            transform=ax2.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            verticalalignment='top', horizontalalignment='center')

    plt.tight_layout()
    output_path = output_dir / 'token_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved token frequency distribution to {output_path}")
    plt.close()


def plot_error_analysis(predictions, targets, output_dir, top_n=20):
    """Analyze and visualize most common prediction errors."""
    print("📊 Creating error analysis...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_token = {v: k for k, v in vocab_data['vocab'].items()}

    # Flatten
    preds_flat = predictions.flatten().numpy()
    targets_flat = targets.flatten().numpy()

    # Find errors
    error_mask = preds_flat != targets_flat
    error_preds = preds_flat[error_mask]
    error_targets = targets_flat[error_mask]

    # Count error pairs
    error_pairs = list(zip(error_targets, error_preds))
    from collections import Counter
    error_counts = Counter(error_pairs)
    most_common_errors = error_counts.most_common(top_n)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Prepare data
    error_labels = []
    error_freqs = []

    for (true_id, pred_id), count in most_common_errors:
        true_token = id_to_token.get(true_id, f'T{true_id}')
        pred_token = id_to_token.get(pred_id, f'T{pred_id}')
        error_labels.append(f'{true_token} → {pred_token}')
        error_freqs.append(count)

    # Plot
    y_pos = np.arange(len(error_labels))
    bars = ax.barh(y_pos, error_freqs, color='#e74c3c', edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(error_labels, fontsize=10)
    ax.set_xlabel('Error Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Prediction Errors', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, error_freqs)):
        ax.text(freq + max(error_freqs)*0.01, i, str(freq), va='center', fontsize=9)

    # Add summary text
    total_errors = error_mask.sum()
    total_tokens = len(targets_flat)
    error_rate = (total_errors / total_tokens) * 100

    summary = f'Total Errors: {total_errors:,} / {total_tokens:,} ({error_rate:.2f}%)'
    ax.text(0.98, 0.02, summary, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'error_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved error analysis to {output_path}")
    plt.close()


def plot_sequence_prediction_timeline(predictions, targets, output_dir, n_sequences=5, seq_length=64):
    """Visualize sequence predictions over time."""
    print("📊 Creating sequence prediction timeline...")

    # Load vocabulary
    vocab_path = Path('data/vocabulary.json')
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Get G-code IDs for highlighting
    g_ids = set(idx for token, idx in vocab.items() if token.startswith('G') and len(token) <= 3)

    # Create figure
    fig, axes = plt.subplots(n_sequences, 1, figsize=(18, 3*n_sequences))
    if n_sequences == 1:
        axes = [axes]

    for seq_idx in range(min(n_sequences, len(predictions))):
        ax = axes[seq_idx]

        # Get actual sequence length (use minimum of requested and actual)
        actual_len = min(seq_length, len(predictions[seq_idx]))
        pred_seq = predictions[seq_idx][:actual_len].numpy()
        true_seq = targets[seq_idx][:actual_len].numpy()

        # Calculate matches
        matches = pred_seq == true_seq

        # Create timeline with actual length
        timesteps = np.arange(actual_len)

        # Plot true sequence
        ax.scatter(timesteps, true_seq, c='green', marker='o', s=50, alpha=0.6, label='Ground Truth', zorder=2)

        # Plot predictions (only errors)
        error_mask = ~matches
        if error_mask.any():
            ax.scatter(timesteps[error_mask], pred_seq[error_mask], c='red', marker='x', s=100,
                      label='Prediction Error', zorder=3)

        # Connect with lines to show divergence
        for t in range(actual_len):
            if not matches[t]:
                ax.plot([t, t], [true_seq[t], pred_seq[t]], 'r--', alpha=0.3, linewidth=1)

        # Highlight G-commands
        g_mask = np.isin(true_seq, list(g_ids))
        if g_mask.any():
            ax.scatter(timesteps[g_mask], true_seq[g_mask], facecolors='none', edgecolors='blue',
                      s=200, linewidths=2, label='G-Commands', zorder=1)

        # Calculate accuracy
        acc = matches.mean() * 100

        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Token ID', fontsize=11)
        ax.set_title(f'Sequence {seq_idx+1} - Accuracy: {acc:.1f}%', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sequence Prediction Timeline (Green=Correct, Red=Errors, Blue Circle=G-Commands)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'sequence_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved sequence timeline to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for G-code model")
    parser.add_argument('--checkpoint', type=Path, required=True, help="Model checkpoint")
    parser.add_argument('--history', type=Path, required=True, help="Training history JSON")
    parser.add_argument('--data-dir', type=Path, required=True, help="Preprocessed data directory")
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/visualizations'), help="Output directory")
    parser.add_argument('--device', type=str, default=None, help="Device (auto-detects GPU if available, or specify: cpu/cuda/mps)")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device (auto-detect GPU or use specified device)
    device = get_device(args.device)
    print_device_info(device)

    print("="*60)
    print("  G-CODE FINGERPRINTING VISUALIZATIONS")
    print("="*60)
    print()

    # 1. Plot loss curves
    train_losses, val_losses = plot_loss_curves(args.history, args.output_dir)

    # 2. Plot training summary
    plot_training_summary(args.history, args.output_dir)

    # 3. Load model and data
    print("📦 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint['config'])
    model = MM_DTAE_LSTM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Model loaded ({model.count_params(model):,} parameters)")

    # 4. Load test data
    print("📦 Loading test data...")
    test_dataset = GCodeDataset(args.data_dir / 'test_sequences.npz')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"✅ Loaded {len(test_dataset)} test samples")

    # 5. Extract embeddings
    print("🔍 Extracting embeddings...")
    embeddings, labels, gcode_texts = extract_embeddings(model, test_loader, device)

    # 6. Plot embeddings
    plot_embeddings_tsne(embeddings, labels, gcode_texts, args.output_dir)
    plot_embeddings_umap(embeddings, labels, gcode_texts, args.output_dir)

    # 7. Get predictions for detailed analysis
    print("\n🎯 Generating predictions for analysis...")
    predictions, targets, pred_gcode_texts = get_predictions(model, test_loader, device)

    # 8. NEW VISUALIZATIONS
    print("\n📊 Creating additional visualizations...")

    # 8a. Confusion Matrix
    plot_confusion_matrix(predictions, targets, args.output_dir, top_n=30)

    # 8b. Per-Token-Type Accuracy
    plot_per_token_type_accuracy(predictions, targets, args.output_dir)

    # 8c. Prediction Examples Gallery
    plot_prediction_examples(predictions, targets, pred_gcode_texts, args.output_dir, n_examples=10)

    # 8d. Token Frequency Distribution
    plot_token_frequency_distribution(targets, args.output_dir)

    # 8e. Error Analysis
    plot_error_analysis(predictions, targets, args.output_dir, top_n=20)

    # 8f. Sequence Prediction Timeline
    plot_sequence_prediction_timeline(predictions, targets, args.output_dir, n_sequences=5, seq_length=64)

    print()
    print("="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(args.output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
