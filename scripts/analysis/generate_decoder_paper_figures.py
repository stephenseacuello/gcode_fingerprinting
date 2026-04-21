#!/usr/bin/env python3
"""Generate 7 publication-quality figures for the decoder paper (Figures 4-10)."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from collections import defaultdict, Counter

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

BASE = 'outputs/decoder20260304'
FIG_DIR = f'{BASE}/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Class display names mapping
CLASS_DISPLAY = {
    'adaptive': 'Adaptive (air)',
    'adaptive150025': 'Adaptive (cut)',
    'damageadaptive': 'Adaptive (dmg)',
    'face': 'Face (air)',
    'face150025': 'Face (cut)',
    'damageface': 'Face (dmg)',
    'pocket': 'Pocket (air)',
    'pocket150025': 'Pocket (cut)',
    'damagepocket': 'Pocket (dmg)',
}
CLASS_ORDER = ['adaptive', 'adaptive150025', 'damageadaptive',
               'face', 'face150025', 'damageface',
               'pocket', 'pocket150025', 'damagepocket']

def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path)
        print(f'  Saved {path}')
    plt.close(fig)


def annotate_heatmap_rdylgn(ax, data, fmt='.1f', vmin=None, vmax=None, fontsize=8):
    """Annotate with adaptive text color for RdYlGn colormap."""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # RdYlGn: red at low, yellow in middle, green at high
            norm_val = (val - vmin) / (vmax - vmin + 1e-10)
            if norm_val < 0.3:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                    color=color, fontsize=fontsize, fontweight='bold')


# ============================================================================
# FIGURE 4: Fold x Modality heatmap
# ============================================================================
def figure4():
    print('Generating Figure 4: fold_modality_heatmap')
    modalities = ['no_accel', 'no_gyro', 'no_mag', 'no_pressure', 'no_temp',
                  'no_prox', 'no_color', 'no_audio', 'no_elec']
    mod_labels = ['Accel', 'Gyro', 'Mag', 'Press', 'Temp', 'Prox', 'Color', 'Audio', 'Elec']

    data = np.zeros((5, 9))
    for j, mod in enumerate(modalities):
        for i in range(5):
            path = f'{BASE}/modality_ablation/{mod}/fold_{i+1}/results/metrics.json'
            with open(path) as f:
                m = json.load(f)
            data[i, j] = m['test_metrics']['sequence_accuracy'] * 100

    fig, ax = plt.subplots(figsize=(7, 3.2))
    vmin, vmax = 0, 100
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(9))
    ax.set_xticklabels(mod_labels, rotation=45, ha='right')
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_xlabel('Removed Modality')
    ax.set_ylabel('')

    # Annotate
    annotate_heatmap_rdylgn(ax, data, fmt='.1f', vmin=vmin, vmax=vmax, fontsize=7.5)

    # Highlight 0.0% cells with a special marker
    for i in range(5):
        for j in range(9):
            if data[i, j] == 0.0:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                                 edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(j, i + 0.35, 'FAILED', ha='center', va='center',
                        color='darkred', fontsize=5.5, fontstyle='italic')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Sequence Accuracy (%)', fontsize=9)
    ax.set_title('Sequence Accuracy by Removed Modality and Fold', fontsize=11, pad=8)
    fig.tight_layout()
    save_fig(fig, 'fold_modality_heatmap')


# ============================================================================
# FIGURE 5: Y-value confusion matrix
# ============================================================================
def figure5():
    print('Generating Figure 5: y_value_confusion')

    # Collect all Y-value pairs
    y_pairs = []
    for fold in range(1, 6):
        path = f'{BASE}/v7_best_5fold_eval/fold_{fold}/results/beam_0_all_predictions.json'
        with open(path) as f:
            preds = json.load(f)
        for p in preds:
            true_toks = p['true'].split()
            pred_toks = p['pred'].split()
            true_y = [t for t in true_toks if t.startswith('NUM_Y_')]
            pred_y = [t for t in pred_toks if t.startswith('NUM_Y_')]
            # Align by position
            for ty in true_y:
                idx = true_toks.index(ty)
                if idx < len(pred_toks) and pred_toks[idx].startswith('NUM_Y_'):
                    y_pairs.append((ty, pred_toks[idx]))
                elif pred_y:
                    # Fallback: match first available
                    y_pairs.append((ty, pred_y[0]))

    # Get unique Y tokens and sort by numeric value
    def y_to_float(tok):
        num = tok.replace('NUM_Y_', '')
        return int(num) / 1000.0

    all_y = sorted(set(t for pair in y_pairs for t in pair), key=y_to_float)

    # Build confusion matrix
    y_idx = {y: i for i, y in enumerate(all_y)}
    n = len(all_y)
    cm = np.zeros((n, n), dtype=float)
    for true_y, pred_y in y_pairs:
        if true_y in y_idx and pred_y in y_idx:
            cm[y_idx[true_y], y_idx[pred_y]] += 1

    # Row-normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Shorten labels
    labels = [f'{y_to_float(y):.3f}' for y in all_y]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, cmap='Oranges', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel('Predicted Y Value')
    ax.set_ylabel('True Y Value')
    ax.set_title('Y-Axis Value Confusion Matrix (Row-Normalized)', fontsize=11, pad=8)

    # Annotate - only show values >= 0.005
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val >= 0.005:
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=5.5, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Proportion', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'y_value_confusion')


# ============================================================================
# FIGURE 6: Per-class metric heatmap
# ============================================================================
def figure6():
    print('Generating Figure 6: per_class_metric_heatmap')

    with open(f'{BASE}/paper/per_class_v7_definitive.json') as f:
        definitive = json.load(f)

    # Per-class sequence and token accuracy from definitive
    seq_acc = {}
    tok_acc = {}
    for cls in CLASS_ORDER:
        s = definitive['per_class_sequence'][cls]
        seq_acc[cls] = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        t = definitive['per_class_token'][cls]
        tok_acc[cls] = t['correct'] / t['total'] * 100 if t['total'] > 0 else 0

    # For command/param/numeric accuracy per class, parse predictions
    class_cmd = defaultdict(lambda: {'correct': 0, 'total': 0})
    class_param = defaultdict(lambda: {'correct': 0, 'total': 0})
    class_num = defaultdict(lambda: {'correct': 0, 'total': 0})

    commands = {'G0', 'G1', 'G2', 'G3', 'G53', 'M30'}
    params = {'X', 'Y', 'Z', 'F', 'R'}

    for fold in range(1, 6):
        path = f'{BASE}/v7_best_5fold_eval/fold_{fold}/results/beam_0_all_predictions.json'
        with open(path) as f:
            preds = json.load(f)
        npz = np.load(f'{BASE}/preprocessed_v7/fold_{fold}/test_sequences.npz', allow_pickle=True)
        op_names = npz['operation_type_names']

        for idx, p in enumerate(preds):
            cls = op_names[idx]
            true_toks = p['true'].split()
            pred_toks = p['pred'].split()
            max_len = max(len(true_toks), len(pred_toks))

            for pos in range(max_len):
                tt = true_toks[pos] if pos < len(true_toks) else '<PAD>'
                pt = pred_toks[pos] if pos < len(pred_toks) else '<PAD>'

                if tt in commands:
                    class_cmd[cls]['total'] += 1
                    if tt == pt:
                        class_cmd[cls]['correct'] += 1
                elif tt in params:
                    class_param[cls]['total'] += 1
                    if tt == pt:
                        class_param[cls]['correct'] += 1
                elif tt.startswith('NUM_'):
                    class_num[cls]['total'] += 1
                    if tt == pt:
                        class_num[cls]['correct'] += 1

    # Build matrix: 9 classes x 5 metrics
    metrics = ['Seq', 'Tok', 'Cmd', 'Param', 'Num']
    data = np.zeros((9, 5))
    for i, cls in enumerate(CLASS_ORDER):
        data[i, 0] = seq_acc[cls]
        data[i, 1] = tok_acc[cls]
        data[i, 2] = class_cmd[cls]['correct'] / class_cmd[cls]['total'] * 100 if class_cmd[cls]['total'] > 0 else 0
        data[i, 3] = class_param[cls]['correct'] / class_param[cls]['total'] * 100 if class_param[cls]['total'] > 0 else 0
        data[i, 4] = class_num[cls]['correct'] / class_num[cls]['total'] * 100 if class_num[cls]['total'] > 0 else 0

    row_labels = [CLASS_DISPLAY[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmin, vmax = 50, 100
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(5))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(9))
    ax.set_yticklabels(row_labels)
    ax.set_title('Per-Class Accuracy by Metric Type', fontsize=11, pad=8)

    annotate_heatmap_rdylgn(ax, data, fmt='.1f', vmin=vmin, vmax=vmax, fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'per_class_metric_heatmap')


# ============================================================================
# FIGURE 7: Window size x Fold heatmap
# ============================================================================
def figure7():
    print('Generating Figure 7: window_fold_heatmap')

    w256 = np.array([71.2, 89.4, 88.7, 89.8, 91.4])

    w128 = np.zeros(5)
    w64 = np.zeros(5)
    for i in range(5):
        with open(f'{BASE}/window_ablation/full_w128_s32/decoder/fold_{i+1}/results/metrics.json') as f:
            m = json.load(f)
        w128[i] = m['test_metrics']['sequence_accuracy'] * 100
        with open(f'{BASE}/window_ablation/full_w64_s16/decoder/fold_{i+1}/results/metrics.json') as f:
            m = json.load(f)
        w64[i] = m['test_metrics']['sequence_accuracy'] * 100

    data = np.zeros((3, 6))  # 3 rows x (5 folds + mean)
    for j in range(5):
        data[0, j] = w256[j]
        data[1, j] = w128[j]
        data[2, j] = w64[j]
    data[:, 5] = data[:, :5].mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    vmin = data.min() - 5
    vmax = data.max() + 2
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    col_labels = [f'Fold {i+1}' for i in range(5)] + ['Mean']
    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels)

    # Row labels with duration
    row_labels = ['w256 (64s)', 'w128 (32s)', 'w64 (16s)']
    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels)
    ax.set_title('Sequence Accuracy by Window Size and Fold', fontsize=11, pad=8)

    annotate_heatmap_rdylgn(ax, data, fmt='.1f', vmin=vmin, vmax=vmax, fontsize=8)

    # Draw a vertical line before Mean column
    ax.axvline(x=4.5, color='gray', linewidth=1, linestyle='-')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'window_fold_heatmap')


# ============================================================================
# FIGURE 8: Ablation x Fold heatmap
# ============================================================================
def figure8():
    print('Generating Figure 8: ablation_fold_heatmap')

    ablation_names = [
        '2A1_no_window_pos', '2A2_no_mwc_neighbors', '2A3_no_scheduled_sampling',
        '2A4_all_data_off', '2B1_no_mpe', '2B2_no_sensor_prior',
        '2B3_no_focal', '2B4_no_grammar', '2B5_no_label_smoothing',
        '2C1_shallow', '2C2_narrow', '2C3_fewer_heads'
    ]
    display_names = {
        '2A1_no_window_pos': 'No window pos.',
        '2A2_no_mwc_neighbors': 'No MWC neighbors',
        '2A3_no_scheduled_sampling': 'No sched. sampling',
        '2A4_all_data_off': 'All data aug. off',
        '2B1_no_mpe': 'No MPE',
        '2B2_no_sensor_prior': 'No sensor prior',
        '2B3_no_focal': 'No focal loss',
        '2B4_no_grammar': 'No grammar mask',
        '2B5_no_label_smoothing': 'No label smooth.',
        '2C1_shallow': 'Shallow (2 layers)',
        '2C2_narrow': 'Narrow (d=128)',
        '2C3_fewer_heads': 'Fewer heads (2)',
    }

    base_vals = np.array([71.2, 89.4, 88.7, 89.8, 91.4])

    # Load ablation data
    n_abl = len(ablation_names)
    fold_data = np.zeros((n_abl, 5))
    for i, abl in enumerate(ablation_names):
        for j in range(5):
            path = f'{BASE}/ablations/{abl}/fold_{j+1}/results/metrics.json'
            with open(path) as f:
                m = json.load(f)
            fold_data[i, j] = m['test_metrics']['sequence_accuracy'] * 100

    # Add base row
    all_data = np.vstack([base_vals.reshape(1, -1), fold_data])
    all_names = ['Base (full model)'] + [display_names[a] for a in ablation_names]

    # Add mean column
    means = all_data.mean(axis=1, keepdims=True)
    all_data_with_mean = np.hstack([all_data, means])

    # Sort by mean descending
    sort_idx = np.argsort(-all_data_with_mean[:, 5])
    all_data_with_mean = all_data_with_mean[sort_idx]
    all_names = [all_names[i] for i in sort_idx]

    n_rows = len(all_names)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmin = all_data_with_mean.min() - 3
    vmax = all_data_with_mean.max() + 2
    im = ax.imshow(all_data_with_mean, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    col_labels = [f'Fold {i+1}' for i in range(5)] + ['Mean']
    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(all_names, fontsize=7.5)
    ax.set_title('Ablation Study: Sequence Accuracy by Condition and Fold', fontsize=11, pad=8)

    annotate_heatmap_rdylgn(ax, all_data_with_mean, fmt='.1f', vmin=vmin, vmax=vmax, fontsize=6.5)

    # Highlight base row
    base_row_idx = list(all_names).index('Base (full model)')
    rect = Rectangle((-0.5, base_row_idx - 0.5), 6, 1, linewidth=2,
                     edgecolor='navy', facecolor='none')
    ax.add_patch(rect)

    # Vertical line before Mean
    ax.axvline(x=4.5, color='gray', linewidth=1, linestyle='-')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'ablation_fold_heatmap')


# ============================================================================
# FIGURE 9: Token type confusion matrix
# ============================================================================
def figure9():
    print('Generating Figure 9: token_type_confusion')

    special = {'BOS', 'EOS', 'PAD', 'UNK', 'MASK'}
    commands = {'G0', 'G1', 'G2', 'G3', 'G53', 'M30'}
    params = {'X', 'Y', 'Z', 'F', 'R'}

    def token_type(tok):
        if tok in special:
            return 'Special'
        elif tok in commands:
            return 'Command'
        elif tok in params:
            return 'Parameter'
        elif tok.startswith('NUM_'):
            return 'Numeric'
        else:
            return 'Special'  # fallback

    type_names = ['Special', 'Command', 'Parameter', 'Numeric']
    type_idx = {t: i for i, t in enumerate(type_names)}

    cm = np.zeros((4, 4), dtype=float)
    for fold in range(1, 6):
        path = f'{BASE}/v7_best_5fold_eval/fold_{fold}/results/beam_0_all_predictions.json'
        with open(path) as f:
            preds = json.load(f)
        for p in preds:
            true_toks = p['true'].split()
            pred_toks = p['pred'].split()
            max_len = max(len(true_toks), len(pred_toks))
            for pos in range(max_len):
                tt = true_toks[pos] if pos < len(true_toks) else 'PAD'
                pt = pred_toks[pos] if pos < len(pred_toks) else 'PAD'
                ti = type_idx[token_type(tt)]
                pi = type_idx[token_type(pt)]
                cm[ti, pi] += 1

    # Store raw counts for annotation
    cm_raw = cm.copy()

    # Row-normalize for color
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(type_names, rotation=30, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(type_names)
    ax.set_xlabel('Predicted Type')
    ax.set_ylabel('True Type')
    ax.set_title('Token Type Confusion Matrix', fontsize=11, pad=8)

    # Annotate with both proportion and count
    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            count = int(cm_raw[i, j])
            color = 'white' if val > 0.5 else 'black'
            if count > 0:
                ax.text(j, i, f'{val:.3f}\n({count:,})', ha='center', va='center',
                        color=color, fontsize=7.5, fontweight='bold')
            else:
                ax.text(j, i, '0', ha='center', va='center',
                        color='gray', fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Row-Normalized Proportion', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'token_type_confusion')


# ============================================================================
# FIGURE 10: Error type per class
# ============================================================================
def figure10():
    print('Generating Figure 10: error_type_per_class')

    commands = {'G0', 'G1', 'G2', 'G3', 'G53', 'M30'}
    params = {'X', 'Y', 'Z', 'F', 'R'}

    # Count errors by type per class
    class_errors = {cls: {'Command': 0, 'Parameter': 0, 'Numeric': 0}
                    for cls in CLASS_ORDER}

    for fold in range(1, 6):
        path = f'{BASE}/v7_best_5fold_eval/fold_{fold}/results/beam_0_all_predictions.json'
        with open(path) as f:
            preds = json.load(f)
        npz = np.load(f'{BASE}/preprocessed_v7/fold_{fold}/test_sequences.npz', allow_pickle=True)
        op_names = npz['operation_type_names']

        for idx, p in enumerate(preds):
            cls = op_names[idx]
            true_toks = p['true'].split()
            pred_toks = p['pred'].split()
            max_len = max(len(true_toks), len(pred_toks))

            for pos in range(max_len):
                tt = true_toks[pos] if pos < len(true_toks) else '<PAD>'
                pt = pred_toks[pos] if pos < len(pred_toks) else '<PAD>'
                if tt != pt:
                    if tt in commands:
                        class_errors[cls]['Command'] += 1
                    elif tt in params:
                        class_errors[cls]['Parameter'] += 1
                    elif tt.startswith('NUM_'):
                        class_errors[cls]['Numeric'] += 1

    # Build data: percentage of each error type per class
    error_types = ['Command', 'Parameter', 'Numeric']
    data = np.zeros((9, 3))
    for i, cls in enumerate(CLASS_ORDER):
        total_errors = sum(class_errors[cls].values())
        if total_errors > 0:
            for j, et in enumerate(error_types):
                data[i, j] = class_errors[cls][et] / total_errors * 100
        else:
            data[i, :] = 0

    row_labels = [CLASS_DISPLAY[c] for c in CLASS_ORDER]
    col_labels = ['Command\nError %', 'Parameter\nError %', 'Numeric\nError %']

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(9))
    ax.set_yticklabels(row_labels)
    ax.set_title('Error Distribution by Token Type per Class', fontsize=11, pad=8)

    # Annotate
    for i in range(9):
        for j in range(3):
            val = data[i, j]
            total_errors = sum(class_errors[CLASS_ORDER[i]].values())
            color = 'white' if val > 65 else 'black'
            if total_errors > 0:
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        color=color, fontsize=7.5, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        color='gray', fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Percentage of Errors', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'error_type_per_class')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    os.chdir('/home/seacuello/Documents/gcode_fingerprinting')
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure9()
    figure10()
    print('\nAll 7 figures generated successfully.')
