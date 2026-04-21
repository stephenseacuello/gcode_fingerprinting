#!/usr/bin/env python3
"""Generate publication-quality training curves and t-SNE embedding plots.

Reads training_history.json and model checkpoints from cross-validation
experiments to produce:
  1. Per-fold training/validation loss and accuracy curves
  2. Averaged (mean ± std) learning curves across folds
  3. t-SNE visualization of 128-d fingerprint embeddings coloured by class

Usage:
    # Training curves only (fast, no GPU needed)
    python scripts/analysis/generate_training_figures.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --output-dir outputs/experiments_2026_02_25/paper/figures \
        --mode curves

    # t-SNE only (needs GPU or CPU, loads model checkpoints)
    python scripts/analysis/generate_training_figures.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --output-dir outputs/experiments_2026_02_25/paper/figures \
        --mode tsne

    # Both
    python scripts/analysis/generate_training_figures.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --output-dir outputs/experiments_2026_02_25/paper/figures \
        --mode all

    # Specific experiment config
    python scripts/analysis/generate_training_figures.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --config full_w64_s32_cv \
        --mode all
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES_9 = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket',
]

CLASS_DISPLAY = {
    'adaptive': 'Adaptive',
    'adaptive150025': 'Adaptive-HF',
    'face': 'Face',
    'face150025': 'Face-HF',
    'pocket': 'Pocket',
    'pocket150025': 'Pocket-HF',
    'damageadaptive': 'Dmg-Adaptive',
    'damageface': 'Dmg-Face',
    'damagepocket': 'Dmg-Pocket',
}

# Colour palette: normal ops (blues/greens), HF variants (teals), damage (reds)
CLASS_COLORS = {
    'adaptive':       '#1f77b4',
    'adaptive150025': '#17becf',
    'face':           '#2ca02c',
    'face150025':     '#7f7f7f',
    'pocket':         '#ff7f0e',
    'pocket150025':   '#bcbd22',
    'damageadaptive': '#d62728',
    'damageface':     '#e377c2',
    'damagepocket':   '#9467bd',
}

CLASS_MARKERS = {
    'adaptive': 'o', 'adaptive150025': 's', 'face': '^', 'face150025': 'D',
    'pocket': 'v', 'pocket150025': 'P', 'damageadaptive': 'X',
    'damageface': '*', 'damagepocket': 'h',
}

# MDPI Machines style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_encoder_experiments(experiments_dir):
    """Return list of (config_name, experiment_dir) for encoder configs."""
    exps = []
    for d in sorted(experiments_dir.iterdir()):
        if d.is_dir() and d.name.endswith('_cv'):
            # Check that at least fold_1/encoder/checkpoint exists
            if (d / 'fold_1' / 'encoder' / 'checkpoint').exists():
                exps.append((d.name, d))
    return exps


def load_training_histories(experiment_dir, n_folds=5):
    """Load training_history.json for each fold. Returns list of dicts."""
    histories = []
    for fold in range(1, n_folds + 1):
        hist_path = experiment_dir / f'fold_{fold}' / 'encoder' / 'checkpoint' / 'training_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                histories.append(json.load(f))
        else:
            print(f"  Warning: missing {hist_path}")
    return histories


def load_results_json(experiment_dir, n_folds=5):
    """Load results.json for each fold to get val-best / test-best epochs."""
    results = []
    for fold in range(1, n_folds + 1):
        res_path = experiment_dir / f'fold_{fold}' / 'encoder' / 'results.json'
        if res_path.exists():
            with open(res_path) as f:
                results.append(json.load(f))
    return results


# ── Training Curves ──────────────────────────────────────────────────────────

def generate_per_fold_curves(histories, results, config_name, output_dir):
    """Generate a 2×N grid showing loss and accuracy for each fold."""
    n_folds = len(histories)
    fig, axes = plt.subplots(2, n_folds, figsize=(4 * n_folds, 6), squeeze=False)

    for i, (hist, res) in enumerate(zip(histories, results)):
        epochs = hist['epoch']
        ckpt = res.get('checkpoint_info', {})
        val_ep = ckpt.get('val_best_epoch', None)

        # Loss
        ax = axes[0, i]
        ax.plot(epochs, hist['train_loss'], 'b-', alpha=0.8, label='Train')
        ax.plot(epochs, hist['val_loss'], 'g-', alpha=0.8, label='Val')
        if val_ep:
            ax.axvline(x=val_ep, color='green', ls='--', alpha=0.5)
        ax.set_title(f'Fold {i+1}', fontsize=10)
        if i == 0:
            ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.legend(fontsize=7)

        # Accuracy
        ax = axes[1, i]
        ax.plot(epochs, [a*100 for a in hist['train_acc']], 'b-', alpha=0.8, label='Train')
        ax.plot(epochs, [a*100 for a in hist['val_acc']], 'g-', alpha=0.8, label='Val')
        ax.plot(epochs, [a*100 for a in hist['test_acc']], 'r-', alpha=0.8, label='Test')
        if val_ep:
            ax.axvline(x=val_ep, color='green', ls='--', alpha=0.5)
        if i == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Epoch')
        ax.set_ylim([40, 105])
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Per-Fold Training Curves — {config_name}', fontsize=13)
    fig.tight_layout()
    out_path = output_dir / f'fig_curves_perfold_{config_name}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_averaged_curves(histories, config_name, output_dir):
    """Generate mean ± std learning curves across folds (publication figure)."""
    if not histories:
        return

    # Align to shortest epoch count
    min_epochs = min(len(h['epoch']) for h in histories)
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'test_acc']

    # Build arrays [n_folds, min_epochs]
    arrays = {}
    for m in metrics:
        arr = np.array([h[m][:min_epochs] for h in histories])
        arrays[m] = arr

    epochs = np.arange(1, min_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Loss ──
    for key, color, label in [('train_loss', '#1f77b4', 'Train'),
                               ('val_loss', '#2ca02c', 'Validation')]:
        mean = arrays[key].mean(axis=0)
        std = arrays[key].std(axis=0)
        ax1.plot(epochs, mean, color=color, label=label)
        ax1.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training and Validation Loss')
    ax1.legend()

    # ── Accuracy ──
    for key, color, label in [('train_acc', '#1f77b4', 'Train'),
                               ('val_acc', '#2ca02c', 'Validation'),
                               ('test_acc', '#d62728', 'Test')]:
        mean = arrays[key].mean(axis=0) * 100
        std = arrays[key].std(axis=0) * 100
        ax2.plot(epochs, mean, color=color, label=label)
        ax2.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Classification Accuracy')
    ax2.set_ylim([40, 105])
    ax2.legend()

    fig.suptitle(f'Mean ± Std Learning Curves (5-Fold CV)', fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = output_dir / f'fig_learning_curves_{config_name}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── t-SNE ────────────────────────────────────────────────────────────────────

def extract_fingerprints(experiment_dir, fold, device):
    """Load model checkpoint, run test data through it, return fingerprints + labels."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
    from miracle.model.model import MM_DTAE_LSTM, ModelConfig

    fold_dir = experiment_dir / f'fold_{fold}'
    ckpt_path = fold_dir / 'encoder' / 'checkpoint' / 'best_model.pt'
    data_dir = fold_dir / 'encoder' / 'data'

    if not ckpt_path.exists():
        print(f"  Warning: no checkpoint at {ckpt_path}")
        return None, None

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sensor_dims = ckpt['sensor_dims']
    n_ops = ckpt.get('n_operations', 9)

    config = ModelConfig(sensor_dims=sensor_dims, d_model=ckpt.get('d_model', 256))
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, n_ops)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load metadata for modality indices
    meta_path = data_dir / 'metadata.json'
    with open(meta_path) as f:
        meta = json.load(f)

    # Rebuild modality group indices from continuous_columns (format: "sensor.Channel")
    columns = meta['continuous_columns']
    group_names = ckpt['group_names']  # e.g. ['accelerometer','gyroscope',...,'electrical']

    sensor_patterns = {
        'accelerometer': {'Ax', 'Ay', 'Az'},
        'gyroscope': {'Gx', 'Gy', 'Gz'},
        'magnetometer': {'Mx', 'My', 'Mz'},
        'environmental': {'Pressure', 'Temperature', 'Proximity'},
        'color': {'ColorR', 'ColorG', 'ColorB', 'ColorA'},
        'rms': {'RMS'},
    }
    # Electrical features don't have sensor prefix (e.g. 'spindle', 'spindle_A')
    electrical_names = {
        'spindle', 'spindle_a', 'x_motor', 'y_motor', 'z_motor',
        'x_motor_a', 'y_motor_a', 'z_motor_a',
    }

    groups = {name: [] for name in group_names}
    for idx, col in enumerate(columns):
        if '.' in col:
            # Format: sensor_name.ChannelName
            channel = col.split('.', 1)[1]
            matched = False
            for gname, patterns in sensor_patterns.items():
                if channel in patterns:
                    groups[gname].append(idx)
                    matched = True
                    break
            if not matched:
                groups['electrical'].append(idx)
        else:
            # No dot — electrical feature
            groups['electrical'].append(idx)

    group_indices = [groups[n] for n in group_names]

    # Verify dims match checkpoint
    for gn, gi, expected in zip(group_names, group_indices, sensor_dims):
        if len(gi) != expected:
            print(f"  Warning: group '{gn}' has {len(gi)} cols but checkpoint expects {expected}")

    # Load test data
    test_npz = data_dir / 'test_sequences.npz'
    data = np.load(test_npz, allow_pickle=True)
    continuous = torch.from_numpy(data['continuous'].astype(np.float32)).to(device)
    labels = data['operation_type'].astype(np.int64)

    # Extract fingerprints in batches
    all_fps = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(continuous), batch_size):
            batch = continuous[start:start + batch_size]
            B, T = batch.size(0), batch.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [batch[:, :, idx] for idx in group_indices]
            out = model(mods, lengths)
            all_fps.append(out['fingerprint'].cpu().numpy())

    fingerprints = np.concatenate(all_fps, axis=0)
    return fingerprints, labels


def generate_tsne_plot(experiment_dir, config_name, output_dir, device,
                       n_folds=5, perplexity=30, seed=42):
    """Generate t-SNE scatter plot from fingerprint embeddings across folds."""
    from sklearn.manifold import TSNE

    print(f"  Extracting fingerprints from {n_folds} folds...")
    all_fps = []
    all_labels = []
    all_folds = []

    for fold in range(1, n_folds + 1):
        fps, labels = extract_fingerprints(experiment_dir, fold, device)
        if fps is not None:
            all_fps.append(fps)
            all_labels.append(labels)
            all_folds.append(np.full(len(labels), fold))
            print(f"    Fold {fold}: {len(labels)} test samples, fingerprint shape {fps.shape}")

    if not all_fps:
        print("  No fingerprints extracted, skipping t-SNE.")
        return

    fps = np.concatenate(all_fps, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    folds = np.concatenate(all_folds, axis=0)
    print(f"  Total: {len(labels)} samples, running t-SNE (perplexity={perplexity})...")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(fps)

    # ── Plot 1: All folds combined, coloured by class ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(CLASS_NAMES_9):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CLASS_COLORS[cls_name],
                   marker=CLASS_MARKERS[cls_name],
                   s=25, alpha=0.7, edgecolors='white', linewidths=0.3,
                   label=CLASS_DISPLAY[cls_name])

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE of Fingerprint Embeddings (All Test Folds)')
    ax.legend(loc='best', markerscale=1.5, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path = output_dir / f'fig_tsne_{config_name}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── Plot 2: Per-fold subplots ──
    unique_folds = sorted(set(folds))
    n = len(unique_folds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, fold_id in enumerate(unique_folds):
        ax = axes[0, i]
        fold_mask = folds == fold_id
        fold_coords = coords[fold_mask]
        fold_labels = labels[fold_mask]
        for cls_idx, cls_name in enumerate(CLASS_NAMES_9):
            mask = fold_labels == cls_idx
            if mask.sum() == 0:
                continue
            ax.scatter(fold_coords[mask, 0], fold_coords[mask, 1],
                       c=CLASS_COLORS[cls_name],
                       marker=CLASS_MARKERS[cls_name],
                       s=20, alpha=0.7, edgecolors='white', linewidths=0.2)
        ax.set_title(f'Fold {int(fold_id)}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    handles = [Line2D([0], [0], marker=CLASS_MARKERS[c], color='w',
                      markerfacecolor=CLASS_COLORS[c], markersize=8,
                      label=CLASS_DISPLAY[c]) for c in CLASS_NAMES_9]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f't-SNE per Fold — {config_name}', fontsize=13)
    fig.tight_layout()
    out_path = output_dir / f'fig_tsne_perfold_{config_name}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── Plot 3: Normal vs damage overlay ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    damage_classes = {'damageadaptive', 'damageface', 'damagepocket'}
    normal_mask = np.array([CLASS_NAMES_9[l] not in damage_classes for l in labels])
    damage_mask = ~normal_mask

    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
               c='#2196F3', s=15, alpha=0.4, label=f'Normal ({normal_mask.sum()})')
    ax.scatter(coords[damage_mask, 0], coords[damage_mask, 1],
               c='#F44336', s=30, alpha=0.8, marker='X',
               edgecolors='white', linewidths=0.3,
               label=f'Damage ({damage_mask.sum()})')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Normal vs. Damage-State Embedding Separation')
    ax.legend(markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path = output_dir / f'fig_tsne_damage_{config_name}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Save raw coordinates for reproducibility
    np.savez(output_dir / f'tsne_data_{config_name}.npz',
             coords=coords, labels=labels, folds=folds,
             class_names=CLASS_NAMES_9)
    print(f"  Saved tsne_data_{config_name}.npz")


# ── Auto-select best config ─────────────────────────────────────────────────

def auto_select_best_config(experiments_dir):
    """Find the encoder config with highest mean test accuracy across folds."""
    best_config = None
    best_acc = -1

    for config_name, exp_dir in find_encoder_experiments(experiments_dir):
        accs = []
        for fold in range(1, 6):
            res_path = exp_dir / f'fold_{fold}' / 'encoder' / 'results.json'
            if res_path.exists():
                with open(res_path) as f:
                    r = json.load(f)
                acc = r.get('val_best_test', {}).get('accuracy', 0)
                accs.append(acc)
        if accs:
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_config = (config_name, exp_dir)

    if best_config:
        print(f"Auto-selected best config: {best_config[0]} (mean test acc: {best_acc:.4f})")
    return best_config


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate training curves and t-SNE plots for paper figures')
    parser.add_argument('--experiments-dir', type=str, required=True,
                        help='Root experiments directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: <experiments-dir>/paper/figures)')
    parser.add_argument('--config', type=str, default=None,
                        help='Specific config name (e.g., full_w64_s32_cv). Default: auto-select best')
    parser.add_argument('--mode', choices=['curves', 'tsne', 'all'], default='all',
                        help='What to generate')
    parser.add_argument('--all-configs', action='store_true',
                        help='Generate curves for ALL encoder configs (not just best)')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch device (default: auto-detect)')
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiments_dir / 'paper' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.mode in ('tsne', 'all'):
        import torch
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f"Device: {device}")

    # Determine configs to process
    if args.all_configs:
        configs = find_encoder_experiments(experiments_dir)
    elif args.config:
        exp_dir = experiments_dir / args.config
        if not exp_dir.exists():
            print(f"Error: config directory not found: {exp_dir}")
            sys.exit(1)
        configs = [(args.config, exp_dir)]
    else:
        result = auto_select_best_config(experiments_dir)
        if result is None:
            print("Error: no encoder experiments found.")
            sys.exit(1)
        configs = [result]

    print(f"Processing {len(configs)} config(s): {[c[0] for c in configs]}")
    print(f"Output: {output_dir}")
    print()

    for config_name, exp_dir in configs:
        print(f"── {config_name} ──")

        if args.mode in ('curves', 'all'):
            histories = load_training_histories(exp_dir)
            results = load_results_json(exp_dir)
            if histories and results:
                generate_averaged_curves(histories, config_name, output_dir)
                generate_per_fold_curves(histories, results, config_name, output_dir)
            else:
                print(f"  Skipping curves: {len(histories)} histories, {len(results)} results")

        if args.mode in ('tsne', 'all'):
            generate_tsne_plot(exp_dir, config_name, output_dir, device,
                               perplexity=args.perplexity)

        print()

    print("Done!")


if __name__ == '__main__':
    main()
