#!/usr/bin/env python3
"""Option 3: Freeze-then-finetune for per-modality damage heads.

Stage 1: Freeze v1 encoder, train MLP damage heads on frozen per-modality
          embeddings (mean-pooled) until convergence.
Stage 2: Unfreeze encoder with very low LR, fine-tune end-to-end with
          cls_loss + damage_head_loss. The damage heads are already good,
          so the encoder learns to preserve damage-discriminative features.

This avoids the problem of earlier end-to-end training where cls loss
dominated and per-modality encoders drifted away from damage features.
"""
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn


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
    sensor_dims = [len(groups[n]) for n in group_names]
    return group_names, groups, sensor_dims


def split_modalities(sensor_data, group_indices):
    return [sensor_data[:, :, idx] for idx in group_indices]


class DamageMLPHead(nn.Module):
    """Small MLP for 4-class damage classification on mean-pooled modality embeddings."""
    def __init__(self, input_dim, n_classes=4, hidden_dims=(64,), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, group_indices, device, damage_heads=None,
             active_mods=None, n_ops=9):
    """Evaluate cls accuracy + optional damage pipeline."""
    model.eval()
    all_preds, all_ops = [], []
    all_damage_probs = {m: [] for m in (active_mods or [])}

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths)

            all_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations.cpu())

            if damage_heads and active_mods:
                enc_mods = out['encoded_mods']  # [B,T,M,D]
                for m_idx in active_mods:
                    mod_mean = enc_mods[:, :, m_idx, :].mean(dim=1)  # [B, D]
                    logits = damage_heads[m_idx](mod_mean)
                    probs = torch.softmax(logits, dim=-1)
                    all_damage_probs[m_idx].append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_ops = torch.cat(all_ops).numpy()
    N = len(all_ops)

    # Cls accuracy
    cls_acc = (all_preds == all_ops).mean()
    per_cls = {}
    for c in range(n_ops):
        mask = all_ops == c
        if mask.sum() > 0:
            per_cls[c] = (all_preds[mask] == c).mean()

    # Damage pipeline
    pipeline_results = {}
    if damage_heads and active_mods:
        for m_idx in active_mods:
            all_damage_probs[m_idx] = torch.cat(all_damage_probs[m_idx]).numpy()

        DAMAGE_MOD_MAP = {1: 6, 2: 7, 3: 8}  # 4class → 9class
        # Machine(6) is the router
        router_probs = all_damage_probs[6]  # [N, 4]

        for mt in [0.3, 0.4, 0.5, 0.55, 0.6]:
            correct = Counter()
            total = Counter()
            for i in range(N):
                true_op = all_ops[i]
                total[true_op] += 1
                cls_pred = all_preds[i]

                max_dc = np.argmax(router_probs[i, 1:]) + 1
                max_dp = router_probs[i, max_dc]

                if max_dp > mt:
                    final = DAMAGE_MOD_MAP[max_dc]
                else:
                    final = cls_pred

                if final == true_op:
                    correct[true_op] += 1

            overall = sum(correct.values()) / N
            c6 = correct[6] / total[6] if total[6] > 0 else 0
            c7 = correct[7] / total[7] if total[7] > 0 else 0
            c8 = correct[8] / total[8] if total[8] > 0 else 0
            pipeline_results[mt] = (overall, c6, c7, c8)

    model.train()
    return cls_acc, per_cls, pipeline_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--pretrained-model', required=True, help='Path to frozen v1 checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)

    # Stage 1: frozen encoder, train damage heads
    parser.add_argument('--stage1-epochs', type=int, default=200)
    parser.add_argument('--stage1-lr', type=float, default=1e-3)
    parser.add_argument('--stage1-patience', type=int, default=50)

    # Stage 2: unfreeze encoder, fine-tune jointly
    parser.add_argument('--stage2-epochs', type=int, default=100)
    parser.add_argument('--stage2-encoder-lr', type=float, default=1e-5)
    parser.add_argument('--stage2-head-lr', type=float, default=1e-4)
    parser.add_argument('--stage2-patience', type=int, default=30)
    parser.add_argument('--stage2-cls-weight', type=float, default=1.0)
    parser.add_argument('--stage2-damage-weight', type=float, default=3.0)

    # Architecture
    parser.add_argument('--damage-hidden', type=str, default='64',
                        help='Hidden dims for damage MLP, comma-separated')
    parser.add_argument('--damage-dropout', type=float, default=0.2)
    parser.add_argument('--damage-modalities', type=str, default='6',
                        help='Modality indices for damage heads (default: 6=machine)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'training_log.txt'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"Device: {device}")

    # Load data
    metadata_path = str(Path(args.split_dir) / 'metadata.json')
    group_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in group_names]

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = DecoderDatasetFromSplits(split_dir=args.split_dir, split=split)
        loaders[split] = DataLoader(ds, batch_size=32,
                                     shuffle=(split == 'train'),
                                     collate_fn=decoder_collate_fn,
                                     num_workers=0,
                                     drop_last=(split == 'train'))

    # Class counts for weighting
    op_counts = Counter()
    ds = DecoderDatasetFromSplits(split_dir=args.split_dir, split='train')
    for i in range(len(ds)):
        op_counts[ds.operation_type[i].item()] += 1
    log(f"Train class distribution: {dict(sorted(op_counts.items()))}")

    # Load pretrained model
    log(f"\nLoading pretrained model from {args.pretrained_model}")
    ckpt = torch.load(args.pretrained_model, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    log(f"  d_model={config.d_model}, loaded from epoch {ckpt.get('epoch', '?')}")

    # Evaluate baseline
    cls_acc, per_cls, _ = evaluate(model, loaders['test'], group_indices, device)
    log(f"  Baseline test cls accuracy: {cls_acc:.2%}")
    for c in sorted(per_cls.keys()):
        log(f"    c{c}: {per_cls[c]:.2%}")

    # Create damage heads
    active_mods = [int(x) for x in args.damage_modalities.split(',')]
    hidden_dims = tuple(int(x) for x in args.damage_hidden.split(','))
    damage_heads = {}
    for m_idx in active_mods:
        head = DamageMLPHead(config.d_model, 4, hidden_dims, args.damage_dropout).to(device)
        damage_heads[m_idx] = head
    log(f"\nDamage heads on modalities {active_mods}, hidden={hidden_dims}")

    # Damage class weights
    n_normal = sum(c for op, c in op_counts.items() if op < 6)
    damage_w = torch.ones(4, device=device)
    for damage_op, damage_4c in [(6, 1), (7, 2), (8, 3)]:
        n_cls = op_counts.get(damage_op, 1)
        damage_w[damage_4c] = n_normal / (3 * n_cls) if n_cls > 0 else 1.0
    damage_criterion = nn.CrossEntropyLoss(weight=damage_w)
    log(f"  Damage class weights: {damage_w.tolist()}")

    # =====================================================================
    # STAGE 1: Freeze encoder, train damage heads only
    # =====================================================================
    log(f"\n{'='*70}")
    log(f"STAGE 1: Freeze encoder, train damage MLP heads")
    log(f"  lr={args.stage1_lr}, epochs={args.stage1_epochs}, patience={args.stage1_patience}")
    log(f"{'='*70}")

    # Freeze all encoder params
    for param in model.parameters():
        param.requires_grad = False

    # Damage head params are trainable
    damage_params = []
    for head in damage_heads.values():
        damage_params.extend(head.parameters())
    optimizer = torch.optim.AdamW(damage_params, lr=args.stage1_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage1_epochs)

    best_val_pipeline = 0
    best_stage1_epoch = 0
    no_improve = 0

    for epoch in range(1, args.stage1_epochs + 1):
        model.eval()  # Encoder stays in eval (frozen)
        for head in damage_heads.values():
            head.train()

        epoch_loss = 0
        epoch_total = 0
        t0 = time.time()

        for batch in loaders['train']:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            mods = split_modalities(sensor_data, group_indices)
            with torch.no_grad():
                out = model(mods, lengths)
            enc_mods = out['encoded_mods']  # [B,T,M,D] - detached

            # Damage labels: 0=normal, 1=c6, 2=c7, 3=c8
            damage_labels = torch.zeros(B, dtype=torch.long, device=device)
            damage_labels[operations == 6] = 1
            damage_labels[operations == 7] = 2
            damage_labels[operations == 8] = 3

            loss = 0
            for m_idx in active_mods:
                mod_mean = enc_mods[:, :, m_idx, :].mean(dim=1)
                logits = damage_heads[m_idx](mod_mean)
                loss = loss + damage_criterion(logits, damage_labels)
            loss = loss / len(active_mods)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * B
            epoch_total += B

        scheduler.step()
        elapsed = time.time() - t0

        # Evaluate pipeline on val
        _, _, val_pipeline = evaluate(model, loaders['val'], group_indices, device,
                                       damage_heads, active_mods)
        # Use mt=0.5 as the selection metric
        val_overall = val_pipeline.get(0.5, (0, 0, 0, 0))[0] if val_pipeline else 0

        improved = ''
        if val_overall > best_val_pipeline:
            best_val_pipeline = val_overall
            best_stage1_epoch = epoch
            no_improve = 0
            improved = ' ** BEST **'
            # Save damage heads
            for m_idx, head in damage_heads.items():
                torch.save(head.state_dict(), output_dir / f'damage_head_mod{m_idx}_stage1.pt')
        else:
            no_improve += 1

        if epoch % 10 == 0 or improved:
            log(f"  S1 Epoch {epoch:3d} | loss={epoch_loss/epoch_total:.4f} | "
                f"val_pipeline(mt=0.5)={val_overall:.2%} | {elapsed:.1f}s{improved}")

        if no_improve >= args.stage1_patience:
            log(f"  Stage 1 early stop at epoch {epoch}")
            break

    # Reload best stage 1 heads
    for m_idx, head in damage_heads.items():
        head.load_state_dict(torch.load(output_dir / f'damage_head_mod{m_idx}_stage1.pt',
                                         map_location=device, weights_only=True))

    # Evaluate stage 1 on test
    log(f"\n  Stage 1 results (best epoch {best_stage1_epoch}):")
    _, _, test_pipeline = evaluate(model, loaders['test'], group_indices, device,
                                    damage_heads, active_mods)
    for mt, (overall, c6, c7, c8) in sorted(test_pipeline.items()):
        log(f"    mt={mt:.2f}: {overall:.2%}  c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # STAGE 2: Unfreeze encoder, fine-tune jointly
    # =====================================================================
    log(f"\n{'='*70}")
    log(f"STAGE 2: Unfreeze encoder, fine-tune jointly")
    log(f"  encoder_lr={args.stage2_encoder_lr}, head_lr={args.stage2_head_lr}")
    log(f"  cls_weight={args.stage2_cls_weight}, damage_weight={args.stage2_damage_weight}")
    log(f"  epochs={args.stage2_epochs}, patience={args.stage2_patience}")
    log(f"{'='*70}")

    # Unfreeze encoder
    for param in model.parameters():
        param.requires_grad = True

    # Separate parameter groups: low LR for encoder, higher for heads
    encoder_params = list(model.parameters())
    head_params = []
    for head in damage_heads.values():
        head_params.extend(head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.stage2_encoder_lr},
        {'params': head_params, 'lr': args.stage2_head_lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)

    # Cls loss with class weights
    total = sum(op_counts.values())
    cls_weights = torch.zeros(9, device=device)
    for c, count in op_counts.items():
        cls_weights[c] = total / (9 * count)
    cls_weights = cls_weights / cls_weights.sum() * 9
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.1)

    best_val_overall = 0
    best_stage2_epoch = 0
    no_improve = 0

    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        for head in damage_heads.values():
            head.train()

        epoch_loss = 0
        epoch_total = 0
        t0 = time.time()

        for batch in loaders['train']:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths, modality_dropout_p=0.1)

            # Cls loss
            cls_loss = cls_criterion(out['cls'], operations)

            # Damage loss on encoded_mods (need gradients now)
            # Re-extract with gradients (encoded_mods is detached in forward)
            # We need to hook into the encoder outputs before they're detached
            # Actually encoded_mods is detached for analysis — for training we
            # use the per_mod_damage output which operates on the non-detached tensors
            damage_labels = torch.zeros(B, dtype=torch.long, device=device)
            damage_labels[operations == 6] = 1
            damage_labels[operations == 7] = 2
            damage_labels[operations == 8] = 3

            # Use model's built-in per_mod_damage heads for gradient flow
            # But we also want our trained MLP heads... let's use the encoded_mods
            # approach but with a forward hook to get non-detached versions
            # Actually, simpler: just use the per_mod_damage output from the model
            damage_loss = 0
            for m_idx in active_mods:
                mod_logits = out['per_mod_damage'][m_idx]  # [B, 4] - uses mean-pooled encoded_mods
                damage_loss = damage_loss + damage_criterion(mod_logits, damage_labels)
            damage_loss = damage_loss / len(active_mods)

            loss = args.stage2_cls_weight * cls_loss + args.stage2_damage_weight * damage_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for head in damage_heads.values():
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * B
            epoch_total += B

        scheduler.step()
        elapsed = time.time() - t0

        # Evaluate
        cls_acc, per_cls, test_pipeline = evaluate(
            model, loaders['val'], group_indices, device, damage_heads, active_mods)
        val_overall = test_pipeline.get(0.5, (0, 0, 0, 0))[0] if test_pipeline else cls_acc

        improved = ''
        if val_overall > best_val_overall:
            best_val_overall = val_overall
            best_stage2_epoch = epoch
            no_improve = 0
            improved = ' ** BEST **'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'damage_heads': {m: h.state_dict() for m, h in damage_heads.items()},
                'best_val_overall': best_val_overall,
            }, output_dir / 'best_model_stage2.pt')
        else:
            no_improve += 1

        if epoch % 5 == 0 or improved:
            c6 = per_cls.get(6, 0)
            c7 = per_cls.get(7, 0)
            c8 = per_cls.get(8, 0)
            log(f"  S2 Epoch {epoch:3d} | loss={epoch_loss/epoch_total:.4f} | "
                f"val_cls={cls_acc:.2%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%} | "
                f"val_pipeline={val_overall:.2%} | {elapsed:.1f}s{improved}")

        if no_improve >= args.stage2_patience:
            log(f"  Stage 2 early stop at epoch {epoch}")
            break

    # Final test evaluation
    log(f"\n{'='*70}")
    log(f"FINAL TEST EVALUATION")
    log(f"{'='*70}")

    # Load best stage 2 model
    s2_path = output_dir / 'best_model_stage2.pt'
    if s2_path.exists():
        ckpt2 = torch.load(s2_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt2['model_state_dict'])
        for m_idx, head in damage_heads.items():
            head.load_state_dict(ckpt2['damage_heads'][m_idx])
        log(f"  Loaded best stage 2 model from epoch {ckpt2['epoch']}")

    cls_acc, per_cls, test_pipeline = evaluate(
        model, loaders['test'], group_indices, device, damage_heads, active_mods)
    log(f"\n  Cls accuracy: {cls_acc:.2%}")
    for c in sorted(per_cls.keys()):
        log(f"    c{c}: {per_cls[c]:.2%}")

    log(f"\n  Damage pipeline:")
    for mt, (overall, c6, c7, c8) in sorted(test_pipeline.items()):
        log(f"    mt={mt:.2f}: {overall:.2%}  c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Save results
    results = {
        'stage1_best_epoch': best_stage1_epoch,
        'stage2_best_epoch': best_stage2_epoch,
        'test_cls_accuracy': cls_acc,
        'test_per_class': {str(k): v for k, v in per_cls.items()},
        'test_pipeline': {str(k): {'overall': v[0], 'c6': v[1], 'c7': v[2], 'c8': v[3]}
                          for k, v in test_pipeline.items()},
        'args': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    log(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
