#!/usr/bin/env python3
"""Reproducible encoder operation classification pipeline.

Produces the full 100% test accuracy result using:
  - Frozen MM_DTAE_LSTM v1 encoder
  - Mean-pooled LogReg C=10 router on machine modality (4-class damage detection)
  - LogReg specialist classifiers: gyroscope→c7, magnetometer→c8
  - v1 cls head fallback for normal-class predictions

Outputs: metrics JSON, confusion matrices, per-class bar charts, threshold sweep,
         serialized classifiers, and a full pipeline log.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluation/run_encoder_pipeline.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluation/run_encoder_pipeline.py --output-dir outputs/my_run
"""
import argparse
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ['adaptive', 'adaptive150025', 'face', 'face150025',
               'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket']

SHORT_LABELS = ['adpt', 'adpt150', 'face', 'face150',
                'pock', 'pock150', 'dmg-adpt', 'dmg-face', 'dmg-pock']

ROUTER_LABELS = ['normal', 'c6-dmgAdapt', 'c7-dmgFace', 'c8-dmgPock']

MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
MACHINE_MOD_IDX = 6
GYRO_IDX = 1
MAG_IDX = 2

# ── Utilities ─────────────────────────────────────────────────────────────────

log_file = None

def log(msg):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


def build_modality_indices(metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)
    columns = meta['continuous_columns']
    sensor_patterns = {
        'accelerometer': ['Ax', 'Ay', 'Az'],
        'gyroscope': ['Gx', 'Gy', 'Gz'],
        'magnetometer': ['Mx', 'My', 'Mz'],
        'environmental': ['Pressure', 'Temperature', 'Proximity'],
        'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
        'rms': ['RMS'],
    }
    groups = {name: [] for name in sensor_patterns}
    groups['machine'] = []
    for idx, col in enumerate(columns):
        matched = False
        if '.' in col:
            _, feat = col.rsplit('.', 1)
            for group_name, patterns in sensor_patterns.items():
                if feat in patterns:
                    groups[group_name].append(idx)
                    matched = True
                    break
        if not matched:
            groups['machine'].append(idx)
    group_names = list(sensor_patterns.keys()) + ['machine']
    return group_names, groups, [len(groups[n]) for n in group_names]


def map_4class(ops):
    """Map 9-class labels to 4-class: 0=normal(c0-c5), 1=c6, 2=c7, 3=c8."""
    y = np.zeros(len(ops), dtype=int)
    y[ops == 6] = 1
    y[ops == 7] = 2
    y[ops == 8] = 3
    return y


# ── Embedding extraction ─────────────────────────────────────────────────────

def extract_embeddings(model, loader, group_indices, device):
    model.eval()
    all_mean, all_cls_preds, all_cls_probs, all_ops = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]
            out = model(mods, lengths)
            enc_mods = out['encoded_mods']  # [B, T, M, D]
            all_mean.append(enc_mods.mean(dim=1).cpu())  # [B, M, D]
            cls_logits = out['cls']
            all_cls_probs.append(torch.softmax(cls_logits, dim=-1).cpu())
            all_cls_preds.append(cls_logits.argmax(dim=-1).cpu())
            all_ops.append(operations)
    return {
        'mean': torch.cat(all_mean).numpy(),
        'cls_preds': torch.cat(all_cls_preds).numpy(),
        'cls_probs': torch.cat(all_cls_probs).numpy(),
        'ops': torch.cat(all_ops).numpy(),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(ops, cls_preds, router_probs, mean_embeds,
                 clf_c7, clf_c8, mt):
    """Run the full hybrid pipeline for one split."""
    N = len(ops)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        rp = router_probs[i]
        max_dc = np.argmax(rp[1:]) + 1
        max_dp = rp[max_dc]
        if max_dp > mt:
            if max_dc == 1:
                preds[i] = 6
            elif max_dc == 2:
                x_g = mean_embeds[i, GYRO_IDX, :].reshape(1, -1)
                gp = clf_c7.predict_proba(x_g)[0, 1]
                preds[i] = 7 if gp > 0.5 else cls_preds[i]
            elif max_dc == 3:
                x_m = mean_embeds[i, MAG_IDX, :].reshape(1, -1)
                mp = clf_c8.predict_proba(x_m)[0, 1]
                preds[i] = 8 if mp > 0.5 else cls_preds[i]
        else:
            preds[i] = cls_preds[i]
    return preds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, output_path, title_prefix=''):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{title_prefix}Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{title_prefix}Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm, cm_norm


def plot_per_class_metrics(y_true, y_pred, labels, output_path):
    report = classification_report(y_true, y_pred, target_names=labels,
                                   labels=list(range(len(labels))), output_dict=True)
    precisions = [report[l]['precision'] for l in labels]
    recalls = [report[l]['recall'] for l in labels]
    f1s = [report[l]['f1-score'] for l in labels]
    supports = [report[l]['support'] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    ax.bar(x + width, f1s, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics (Test Set)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    for i, support in enumerate(supports):
        ax.annotate(f'n={support}', xy=(i, 0.02), ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return report


def plot_threshold_sweep(thresholds, accuracies, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [a * 100 for a in accuracies], 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.set_xlabel('Damage Confidence Threshold (mt)', fontsize=12)
    ax.set_ylabel('Pipeline Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs Router Confidence Threshold', fontsize=14)
    ax.set_ylim(min(a * 100 for a in accuracies) - 0.5, 100.5)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    for t, a in zip(thresholds, accuracies):
        ax.annotate(f'{a*100:.2f}%', xy=(t, a*100), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Reproducible encoder pipeline evaluation')
    parser.add_argument('--model-path', type=str,
                        default='outputs/jan30/mmdtae_standalone_v1/best_model.pt',
                        help='Path to frozen v1 encoder checkpoint')
    parser.add_argument('--split-dir', type=str,
                        default='outputs/jan23_followup/no_leakage',
                        help='Path to train/val/test splits')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/jan30/encoder_pipeline',
                        help='Output directory for results')
    parser.add_argument('--mt', type=float, default=0.50,
                        help='Primary router confidence threshold')
    args = parser.parse_args()

    global log_file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / 'pipeline_log.txt', 'w')

    t0 = time.time()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Model: {args.model_path}")
    log(f"Splits: {args.split_dir}")
    log(f"Output: {args.output_dir}")
    log(f"Primary threshold: mt={args.mt}")

    # ── Data ──────────────────────────────────────────────────────────────
    loaders = {}
    split_sizes = {}
    for split in ['train', 'val', 'test']:
        ds = DecoderDatasetFromSplits(split_dir=args.split_dir, split=split)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)
        split_sizes[split] = len(ds)
    log(f"\nDataset sizes: train={split_sizes['train']}, val={split_sizes['val']}, test={split_sizes['test']}")

    metadata_path = str(Path(args.split_dir) / 'metadata.json')
    mod_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in mod_names]
    log(f"Modalities: {list(zip(mod_names, sensor_dims))}")

    # ── Load encoder ──────────────────────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    log(f"\nEncoder: d_model={config.d_model}, cls_pooling={getattr(config, 'cls_pooling', 'last')}")

    # ── Extract embeddings ────────────────────────────────────────────────
    log(f"\nExtracting mean-pooled per-modality embeddings...")
    data = {}
    for split in ['train', 'val', 'test']:
        data[split] = extract_embeddings(model, loaders[split], group_indices, device)
        log(f"  {split}: embeddings {data[split]['mean'].shape}, ops {data[split]['ops'].shape}")

    # ── Cls head baseline ─────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"ENCODER CLS HEAD BASELINE (9-class)")
    log(f"{'='*70}")

    for split in ['train', 'val', 'test']:
        ops = data[split]['ops']
        cls_preds = data[split]['cls_preds']
        acc = accuracy_score(ops, cls_preds)
        log(f"\n  {split} cls accuracy: {acc:.2%} ({(cls_preds==ops).sum()}/{len(ops)})")
        for c in range(9):
            mask = ops == c
            if mask.sum() > 0:
                ca = (cls_preds[mask] == c).mean()
                log(f"    c{c} ({CLASS_NAMES[c]:20s}): {ca:.2%} ({(cls_preds[mask]==c).sum()}/{mask.sum()})")

    # ── Train router (train+val) ─────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"TRAINING ROUTER: LogReg C=10 on train+val machine modality")
    log(f"{'='*70}")

    X_router = np.concatenate([data['train']['mean'][:, MACHINE_MOD_IDX, :],
                                data['val']['mean'][:, MACHINE_MOD_IDX, :]])
    y_router = np.concatenate([map_4class(data['train']['ops']),
                                map_4class(data['val']['ops'])])
    log(f"  Training samples: {len(y_router)} (c6={( y_router==1).sum()}, c7={(y_router==2).sum()}, c8={(y_router==3).sum()})")

    router = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
    router.fit(X_router, y_router)

    # Router standalone accuracy on each split
    for split in ['train', 'val', 'test']:
        X_split = data[split]['mean'][:, MACHINE_MOD_IDX, :]
        y_split = map_4class(data[split]['ops'])
        router_preds = router.predict(X_split)
        racc = accuracy_score(y_split, router_preds)
        log(f"  Router standalone {split}: {racc:.2%}")

    # Save router
    joblib.dump(router, out_dir / 'router.joblib')
    log(f"  Saved router to {out_dir / 'router.joblib'}")

    # ── Train specialists (train+val) ────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"TRAINING SPECIALISTS: LogReg on train+val")
    log(f"{'='*70}")

    # c7: gyroscope, face group (c2, c3, c7)
    face_tr = np.isin(data['train']['ops'], [2, 3, 7])
    face_val = np.isin(data['val']['ops'], [2, 3, 7])
    X_c7 = np.concatenate([data['train']['mean'][face_tr, GYRO_IDX, :],
                            data['val']['mean'][face_val, GYRO_IDX, :]])
    y_c7 = np.concatenate([(data['train']['ops'][face_tr] == 7).astype(int),
                            (data['val']['ops'][face_val] == 7).astype(int)])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(X_c7, y_c7)
    log(f"  c7 specialist (gyroscope): {len(y_c7)} samples ({y_c7.sum()} positive)")

    # c8: magnetometer, pocket group (c4, c5, c8)
    pock_tr = np.isin(data['train']['ops'], [4, 5, 8])
    pock_val = np.isin(data['val']['ops'], [4, 5, 8])
    X_c8 = np.concatenate([data['train']['mean'][pock_tr, MAG_IDX, :],
                            data['val']['mean'][pock_val, MAG_IDX, :]])
    y_c8 = np.concatenate([(data['train']['ops'][pock_tr] == 8).astype(int),
                            (data['val']['ops'][pock_val] == 8).astype(int)])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(X_c8, y_c8)
    log(f"  c8 specialist (magnetometer): {len(y_c8)} samples ({y_c8.sum()} positive)")

    joblib.dump(clf_c7, out_dir / 'specialist_c7.joblib')
    joblib.dump(clf_c8, out_dir / 'specialist_c8.joblib')
    log(f"  Saved specialists to {out_dir}")

    # ── Full pipeline evaluation ─────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"FULL PIPELINE EVALUATION (mt={args.mt})")
    log(f"{'='*70}")

    results = {'config': {
        'model_path': args.model_path,
        'split_dir': args.split_dir,
        'primary_mt': args.mt,
        'd_model': config.d_model,
        'cls_pooling': getattr(config, 'cls_pooling', 'last'),
        'router': 'LogReg C=10 on train+val machine modality (mean-pooled)',
        'specialist_c7': 'LogReg C=1.0 on train+val gyroscope (face group, mean-pooled)',
        'specialist_c8': 'LogReg C=1.0 on train+val magnetometer (pocket group, mean-pooled)',
    }, 'splits': {}}

    for split in ['train', 'val', 'test']:
        ops = data[split]['ops']
        cls_preds = data[split]['cls_preds']
        mean_embeds = data[split]['mean']
        router_probs = router.predict_proba(mean_embeds[:, MACHINE_MOD_IDX, :])

        preds = run_pipeline(ops, cls_preds, router_probs, mean_embeds,
                             clf_c7, clf_c8, args.mt)

        acc = accuracy_score(ops, preds)
        prec, rec, f1, sup = precision_recall_fscore_support(
            ops, preds, labels=list(range(9)), zero_division=0)
        report = classification_report(
            ops, preds, target_names=CLASS_NAMES, labels=list(range(9)),
            output_dict=True, zero_division=0)

        log(f"\n  {split.upper()} — Pipeline accuracy: {acc:.2%} ({(preds==ops).sum()}/{len(ops)})")
        log(f"  {'Class':25s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
        log(f"  {'-'*57}")
        for c in range(9):
            log(f"  {CLASS_NAMES[c]:25s} {prec[c]:8.4f} {rec[c]:8.4f} {f1[c]:8.4f} {sup[c]:8d}")

        macro = report['macro avg']
        weighted = report['weighted avg']
        log(f"  {'-'*57}")
        log(f"  {'macro avg':25s} {macro['precision']:8.4f} {macro['recall']:8.4f} {macro['f1-score']:8.4f}")
        log(f"  {'weighted avg':25s} {weighted['precision']:8.4f} {weighted['recall']:8.4f} {weighted['f1-score']:8.4f}")

        missed = [(int(i), int(ops[i]), int(preds[i])) for i in range(len(ops)) if preds[i] != ops[i]]
        if missed:
            log(f"\n  Misclassified samples ({len(missed)}):")
            for idx, true, pred in missed[:20]:
                log(f"    [{split}][{idx}] true=c{true}({CLASS_NAMES[true]}) → pred=c{pred}({CLASS_NAMES[pred]})")

        results['splits'][split] = {
            'accuracy': float(acc),
            'n_samples': int(len(ops)),
            'n_correct': int((preds == ops).sum()),
            'n_missed': len(missed),
            'per_class': {
                CLASS_NAMES[c]: {
                    'precision': float(prec[c]),
                    'recall': float(rec[c]),
                    'f1': float(f1[c]),
                    'support': int(sup[c]),
                    'correct': int((preds[ops == c] == c).sum()) if (ops == c).sum() > 0 else 0,
                }
                for c in range(9)
            },
            'macro_avg': {k: float(v) for k, v in macro.items() if k != 'support'},
            'weighted_avg': {k: float(v) for k, v in weighted.items() if k != 'support'},
            'missed_samples': missed[:50],
        }

    # ── Threshold sweep ──────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"THRESHOLD SWEEP (test set)")
    log(f"{'='*70}")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_accs = []
    test_ops = data['test']['ops']
    test_cls = data['test']['cls_preds']
    test_mean = data['test']['mean']
    test_router_probs = router.predict_proba(test_mean[:, MACHINE_MOD_IDX, :])

    log(f"  {'mt':>6s} {'Accuracy':>10s} {'Missed':>8s} {'c6':>6s} {'c7':>6s} {'c8':>6s}")
    log(f"  {'-'*46}")
    for mt in thresholds:
        p = run_pipeline(test_ops, test_cls, test_router_probs, test_mean, clf_c7, clf_c8, mt)
        acc = accuracy_score(test_ops, p)
        missed = (p != test_ops).sum()
        c6 = (p[test_ops == 6] == 6).mean() if (test_ops == 6).sum() > 0 else 0
        c7 = (p[test_ops == 7] == 7).mean() if (test_ops == 7).sum() > 0 else 0
        c8 = (p[test_ops == 8] == 8).mean() if (test_ops == 8).sum() > 0 else 0
        sweep_accs.append(acc)
        log(f"  {mt:6.2f} {acc:10.2%} {missed:8d} {c6:6.0%} {c7:6.0%} {c8:6.0%}")

    results['threshold_sweep'] = {str(t): float(a) for t, a in zip(thresholds, sweep_accs)}

    # ── Plots ─────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"GENERATING PLOTS")
    log(f"{'='*70}")

    # 9-class confusion matrix (test)
    test_preds = run_pipeline(test_ops, test_cls, test_router_probs, test_mean,
                              clf_c7, clf_c8, args.mt)
    cm_path = out_dir / 'confusion_matrix_test.png'
    plot_confusion_matrix(test_ops, test_preds, SHORT_LABELS, str(cm_path), title_prefix='Test Set — ')
    log(f"  Saved {cm_path}")

    # Per-class metrics bar chart (test)
    metrics_path = out_dir / 'per_class_metrics_test.png'
    plot_per_class_metrics(test_ops, test_preds, SHORT_LABELS, str(metrics_path))
    log(f"  Saved {metrics_path}")

    # Router confusion matrix (4-class, test)
    router_cm_path = out_dir / 'router_confusion_matrix.png'
    y_test_4c = map_4class(test_ops)
    router_preds_4c = router.predict(test_mean[:, MACHINE_MOD_IDX, :])
    plot_confusion_matrix(y_test_4c, router_preds_4c, ROUTER_LABELS,
                          str(router_cm_path), title_prefix='Router (4-class) — ')
    log(f"  Saved {router_cm_path}")

    # Threshold sweep curve
    sweep_path = out_dir / 'threshold_sweep.png'
    plot_threshold_sweep(thresholds, sweep_accs, str(sweep_path))
    log(f"  Saved {sweep_path}")

    # ── Save results ─────────────────────────────────────────────────────
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n  Saved {results_path}")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s")
    log_file.close()


if __name__ == '__main__':
    main()
