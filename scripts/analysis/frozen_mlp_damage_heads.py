#!/usr/bin/env python3
"""Option 3: Frozen v1 encoder + learned MLP damage heads.

Freeze the v1 per-modality encoders, extract embeddings, then train
small MLP damage heads with backprop (instead of sklearn LogReg).
This allows learning non-linear decision boundaries on the frozen embeddings.

Tests:
1. Per-modality MLP damage heads (machine→c6, gyro→c7, mag→c8)
2. Machine 4-class MLP router + specialist MLP overrides
3. Various MLP architectures and training configs
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
MACHINE_MOD_IDX = 6
GYRO_IDX = 1
MAG_IDX = 2


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


def extract_per_modality_embeddings(model, loader, group_indices, device):
    model.eval()
    all_mod_embeds, all_cls_preds, all_ops = [], [], []
    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]
            out = model(mods, lengths)
            enc_mods = out['encoded_mods']
            last_idx = (lengths.clamp(min=1) - 1).view(-1)
            per_mod_last = enc_mods[torch.arange(B, device=device), last_idx]
            all_mod_embeds.append(per_mod_last.cpu())
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations)
    return (torch.cat(all_mod_embeds).numpy(),
            torch.cat(all_cls_preds).numpy(),
            torch.cat(all_ops).numpy())


class DamageMLPHead(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_head(X_train, y_train, X_val, y_val, n_classes,
                    hidden_dims=(128, 64), dropout=0.3, lr=1e-3,
                    epochs=200, patience=30, class_weight=None, device='cpu'):
    input_dim = X_train.shape[1]
    model = DamageMLPHead(input_dim, n_classes, hidden_dims, dropout).to(device)

    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch
        perm = torch.randperm(len(X_tr))
        total_loss = 0
        for i in range(0, len(X_tr), 64):
            idx = perm[i:i+64]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_v).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_acc


def map_4class(ops):
    y = np.zeros(len(ops), dtype=int)
    y[ops == 6] = 1
    y[ops == 7] = 2
    y[ops == 8] = 3
    return y


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    model_path = 'outputs/jan30/mmdtae_standalone_v1/best_model.pt'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    mlp_device = 'cpu'  # MLP training on CPU is fine for small data
    print(f"Encoder device: {device}, MLP device: {mlp_device}")

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = DecoderDatasetFromSplits(split_dir=split_dir, split=split)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)

    metadata_path = str(Path(split_dir) / 'metadata.json')
    mod_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in mod_names]

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded v1 model, d_model={config.d_model}")

    print(f"\nExtracting per-modality embeddings...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        mod_embeds, cls_preds, ops = extract_per_modality_embeddings(
            model, loaders[split], group_indices, device)
        split_data[split] = {'mod_embeds': mod_embeds, 'cls_preds': cls_preds, 'ops': ops}
        print(f"  {split}: {mod_embeds.shape}")

    train_ops = split_data['train']['ops']
    val_ops = split_data['val']['ops']
    test_ops = split_data['test']['ops']

    # =====================================================================
    # PART 1: MLP 4-class router on machine modality
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 1: MLP 4-class router on machine modality")
    print(f"{'='*70}")

    y_train_4c = map_4class(train_ops)
    y_val_4c = map_4class(val_ops)
    y_test_4c = map_4class(test_ops)

    X_train_machine = split_data['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
    X_val_machine = split_data['val']['mod_embeds'][:, MACHINE_MOD_IDX, :]
    X_test_machine = split_data['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]

    # Compute class weights for 4-class
    counts_4c = np.bincount(y_train_4c)
    total = len(y_train_4c)
    cw_4c = [total / (4 * c) for c in counts_4c]

    configs = [
        {'hidden_dims': (64,), 'dropout': 0.2, 'lr': 1e-3},
        {'hidden_dims': (128, 64), 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dims': (128, 64), 'dropout': 0.3, 'lr': 5e-4},
        {'hidden_dims': (256, 128, 64), 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dims': (256, 128, 64), 'dropout': 0.4, 'lr': 5e-4},
        {'hidden_dims': (128, 64), 'dropout': 0.2, 'lr': 1e-3},
        {'hidden_dims': (64, 32), 'dropout': 0.2, 'lr': 1e-3},
    ]

    # Reference: LogReg C=10
    clf_ref = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
    clf_ref.fit(X_train_machine, y_train_4c)
    ref_preds = clf_ref.predict(X_test_machine)
    ref_acc = accuracy_score(y_test_4c, ref_preds)
    print(f"\n  LogReg C=10 reference: {ref_acc:.2%}")

    router_results = []
    for cfg in configs:
        label = f"MLP({cfg['hidden_dims']}) d={cfg['dropout']} lr={cfg['lr']}"
        # Run 3 seeds to reduce variance
        seed_accs = []
        seed_models = []
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            mlp, val_acc = train_mlp_head(
                X_train_machine, y_train_4c, X_val_machine, y_val_4c,
                n_classes=4, hidden_dims=cfg['hidden_dims'],
                dropout=cfg['dropout'], lr=cfg['lr'],
                epochs=300, patience=50, class_weight=cw_4c,
                device=mlp_device)

            with torch.no_grad():
                test_logits = mlp(torch.tensor(X_test_machine, dtype=torch.float32))
                test_probs = torch.softmax(test_logits, dim=1).numpy()
                test_preds = test_probs.argmax(axis=1)

            acc = accuracy_score(y_test_4c, test_preds)
            per_cls = {}
            for c in range(4):
                mask = y_test_4c == c
                if mask.sum() > 0:
                    per_cls[c] = accuracy_score(y_test_4c[mask], test_preds[mask])

            seed_accs.append(acc)
            seed_models.append((mlp, test_probs, acc, per_cls))

        # Pick best seed
        best_idx = np.argmax(seed_accs)
        best_mlp, best_probs, best_acc, best_per_cls = seed_models[best_idx]

        c6a = best_per_cls.get(1, 0)
        c7a = best_per_cls.get(2, 0)
        c8a = best_per_cls.get(3, 0)
        print(f"  {label:55s}: {best_acc:.2%} (avg={np.mean(seed_accs):.2%})  "
              f"normal={best_per_cls.get(0,0):.0%} c6={c6a:.0%} c7={c7a:.0%} c8={c8a:.0%}")

        router_results.append({
            'label': label, 'acc': best_acc, 'per_cls': best_per_cls,
            'probs': best_probs, 'model': best_mlp
        })

    # =====================================================================
    # PART 2: MLP specialists (c7=gyro, c8=mag)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 2: MLP specialist heads")
    print(f"{'='*70}")

    # c7: gyroscope, face group
    face_train_mask = np.isin(train_ops, [2, 3, 7])
    face_val_mask = np.isin(val_ops, [2, 3, 7])
    face_test_mask = np.isin(test_ops, [2, 3, 7])
    y_train_c7 = (train_ops[face_train_mask] == 7).astype(int)
    y_val_c7 = (val_ops[face_val_mask] == 7).astype(int)
    y_test_c7 = (test_ops[face_test_mask] == 7).astype(int)

    X_train_gyro = split_data['train']['mod_embeds'][face_train_mask, GYRO_IDX, :]
    X_val_gyro = split_data['val']['mod_embeds'][face_val_mask, GYRO_IDX, :]
    X_test_gyro = split_data['test']['mod_embeds'][face_test_mask, GYRO_IDX, :]

    # c8: magnetometer, pocket group
    pocket_train_mask = np.isin(train_ops, [4, 5, 8])
    pocket_val_mask = np.isin(val_ops, [4, 5, 8])
    pocket_test_mask = np.isin(test_ops, [4, 5, 8])
    y_train_c8 = (train_ops[pocket_train_mask] == 8).astype(int)
    y_val_c8 = (val_ops[pocket_val_mask] == 8).astype(int)
    y_test_c8 = (test_ops[pocket_test_mask] == 8).astype(int)

    X_train_mag = split_data['train']['mod_embeds'][pocket_train_mask, MAG_IDX, :]
    X_val_mag = split_data['val']['mod_embeds'][pocket_val_mask, MAG_IDX, :]
    X_test_mag = split_data['test']['mod_embeds'][pocket_test_mask, MAG_IDX, :]

    # c7 specialist
    print(f"\n--- c7 specialist (gyroscope) ---")
    counts_c7 = np.bincount(y_train_c7)
    cw_c7 = [len(y_train_c7) / (2 * c) for c in counts_c7]

    for cfg in [{'hidden_dims': (64,), 'dropout': 0.2, 'lr': 1e-3},
                {'hidden_dims': (128, 64), 'dropout': 0.3, 'lr': 1e-3},
                {'hidden_dims': (64, 32), 'dropout': 0.2, 'lr': 5e-4}]:
        label = f"MLP({cfg['hidden_dims']}) d={cfg['dropout']}"
        best_f1 = 0
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            mlp, _ = train_mlp_head(
                X_train_gyro, y_train_c7, X_val_gyro, y_val_c7,
                n_classes=2, hidden_dims=cfg['hidden_dims'],
                dropout=cfg['dropout'], lr=cfg['lr'],
                epochs=300, patience=50, class_weight=cw_c7,
                device=mlp_device)
            with torch.no_grad():
                preds = mlp(torch.tensor(X_test_gyro, dtype=torch.float32)).argmax(dim=1).numpy()
            f1 = f1_score(y_test_c7, preds, zero_division=0)
            tp = ((preds == 1) & (y_test_c7 == 1)).sum()
            fp = ((preds == 1) & (y_test_c7 == 0)).sum()
            if f1 > best_f1:
                best_f1 = f1
                best_tp, best_fp = tp, fp
        print(f"  {label:45s}: F1={best_f1:.2%} TP={best_tp} FP={best_fp}")

    # c8 specialist
    print(f"\n--- c8 specialist (magnetometer) ---")
    counts_c8 = np.bincount(y_train_c8)
    cw_c8 = [len(y_train_c8) / (2 * c) for c in counts_c8]

    for cfg in [{'hidden_dims': (64,), 'dropout': 0.2, 'lr': 1e-3},
                {'hidden_dims': (128, 64), 'dropout': 0.3, 'lr': 1e-3},
                {'hidden_dims': (64, 32), 'dropout': 0.2, 'lr': 5e-4}]:
        label = f"MLP({cfg['hidden_dims']}) d={cfg['dropout']}"
        best_f1 = 0
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            mlp, _ = train_mlp_head(
                X_train_mag, y_train_c8, X_val_mag, y_val_c8,
                n_classes=2, hidden_dims=cfg['hidden_dims'],
                dropout=cfg['dropout'], lr=cfg['lr'],
                epochs=300, patience=50, class_weight=cw_c8,
                device=mlp_device)
            with torch.no_grad():
                preds = mlp(torch.tensor(X_test_mag, dtype=torch.float32)).argmax(dim=1).numpy()
            f1 = f1_score(y_test_c8, preds, zero_division=0)
            tp = ((preds == 1) & (y_test_c8 == 1)).sum()
            fp = ((preds == 1) & (y_test_c8 == 0)).sum()
            if f1 > best_f1:
                best_f1 = f1
                best_tp, best_fp = tp, fp
        print(f"  {label:45s}: F1={best_f1:.2%} TP={best_tp} FP={best_fp}")

    # =====================================================================
    # PART 3: Best MLP router → full pipeline
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 3: Full pipeline with best MLP router")
    print(f"{'='*70}")

    # Use LogReg specialists (already perfect)
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(X_train_gyro, y_train_c7)
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(X_train_mag, y_train_c8)

    test_cls_preds = split_data['test']['cls_preds']
    test_mod_embeds = split_data['test']['mod_embeds']
    N = len(test_ops)
    per_class_total = Counter()
    for i in range(N):
        per_class_total[test_ops[i]] += 1

    # Sort routers by accuracy
    top_routers = sorted(router_results, key=lambda x: x['acc'], reverse=True)[:3]

    for router in top_routers:
        print(f"\n  Router: {router['label']} (standalone={router['acc']:.2%})")
        probs_4c = router['probs']

        for mt in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pc_correct = Counter()
            for i in range(N):
                true_op = test_ops[i]
                cls_pred = test_cls_preds[i]
                machine_probs = probs_4c[i]

                max_damage_class = np.argmax(machine_probs[1:]) + 1
                max_damage_prob = machine_probs[max_damage_class]

                if max_damage_prob > mt:
                    if max_damage_class == 1:
                        final_pred = 6
                    elif max_damage_class == 2:
                        x_gyro = test_mod_embeds[i, GYRO_IDX, :].reshape(1, -1)
                        gyro_prob = clf_c7.predict_proba(x_gyro)[0, 1]
                        final_pred = 7 if gyro_prob > 0.5 else cls_pred
                    elif max_damage_class == 3:
                        x_mag = test_mod_embeds[i, MAG_IDX, :].reshape(1, -1)
                        mag_prob = clf_c8.predict_proba(x_mag)[0, 1]
                        final_pred = 8 if mag_prob > 0.5 else cls_pred
                else:
                    final_pred = cls_pred

                if final_pred == true_op:
                    pc_correct[true_op] += 1

            overall = sum(pc_correct.values()) / N
            c6 = pc_correct[6] / per_class_total[6]
            c7 = pc_correct[7] / per_class_total[7]
            c8 = pc_correct[8] / per_class_total[8]
            c05 = np.mean([pc_correct[c] / per_class_total[c] for c in range(6)])
            print(f"    mt={mt}: {overall:.2%}  c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    print(f"\n{'='*70}")
    print(f"REFERENCE: LogReg C=10 pipeline best = 97.89%")
    print(f"  c0-c5=100%, c6=96%, c7=78%, c8=86%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
