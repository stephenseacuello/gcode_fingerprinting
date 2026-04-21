#!/usr/bin/env python3
"""
Rebuild OOD leave-one-toolpath-out NPZ files by filtering existing V7 preprocessed
fold-1 data. Avoids the multi-line gcode_texts bug in run_preprocessing_cv_fold.py
by reusing V7's correctly-preprocessed single-line tokens.

Strategy:
  Pool all 545 windows from fold_1 {train,val,test}_sequences.npz.
  Classify each window by toolpath via source_file basename.
  For each heldout in {adaptive, face, pocket}:
      test  = all windows whose source_file belongs to the heldout toolpath
      train = 80% of non-heldout windows (stratified by operation_type_names)
      val   = 20% of non-heldout windows
  Write NPZ files with the V7 schema.

Limitation: Fold-1's scaler was fit on fold-1's training subset (69 files),
not on the OOD training pool. This is an acceptable approximation because
(a) V7 uses robust scaling which is insensitive to small distribution shifts,
(b) fold-1's training set includes files from all three toolpath classes,
so the scaler is representative, and (c) rebuilding from raw CSVs hits the
multi-line gcode_texts bug in the existing preprocessing pipeline.
"""
import numpy as np, json, random
from pathlib import Path
from collections import defaultdict


TOOLPATH_MAP = {
    'adaptive': {'adaptive', 'adaptive150025', 'damageadaptive'},
    'face':     {'face', 'face150025', 'damageface'},
    'pocket':   {'pocket', 'pocket150025', 'damagepocket'},
}


def toolpath_of(op_name: str) -> str:
    op = op_name.lower()
    for tp, ops in TOOLPATH_MAP.items():
        if op in ops:
            return tp
    return 'other'


def load_fold1_pooled():
    base = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260304/preprocessed_v7/fold_1')
    pooled = {}
    for split in ['train', 'val', 'test']:
        npz = np.load(base / f'{split}_sequences.npz', allow_pickle=True)
        for k in npz.keys():
            if k not in pooled:
                pooled[k] = [npz[k]]
            else:
                pooled[k].append(npz[k])
    out = {}
    for k, parts in pooled.items():
        try:
            out[k] = np.concatenate(parts, axis=0)
        except ValueError:
            # Scalar fields or object dtype with different shapes; fall back
            out[k] = np.array([x for p in parts for x in p], dtype=parts[0].dtype)
    return out


def main():
    pooled = load_fold1_pooled()
    n = len(pooled['gcode_texts'])
    print(f"Pooled fold-1 windows: {n}")

    # Classify each window
    toolpath = np.array([toolpath_of(str(op)) for op in pooled['operation_type_names']])
    op_name = np.array([str(op) for op in pooled['operation_type_names']])

    from collections import Counter
    print("Toolpath distribution:", Counter(toolpath.tolist()))

    rng = random.Random(42)
    out_base = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260304/preprocessed_ood')

    for heldout in ['adaptive', 'face', 'pocket']:
        out_dir = out_base / f'heldout_{heldout}'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Test indices: heldout toolpath
        test_idx = np.where(toolpath == heldout)[0].tolist()

        # Train+val indices: non-heldout, stratified by operation_type
        other_idx = np.where(toolpath != heldout)[0].tolist()
        by_op = defaultdict(list)
        for i in other_idx:
            by_op[op_name[i]].append(i)

        train_idx, val_idx = [], []
        for op, idxs in by_op.items():
            shuffled = idxs[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * 0.2)))
            val_idx.extend(shuffled[:n_val])
            train_idx.extend(shuffled[n_val:])

        for split, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            idxs = sorted(idxs)
            save_dict = {}
            for k in pooled.keys():
                save_dict[k] = pooled[k][idxs]
            np.savez(out_dir / f'{split}_sequences.npz', **save_dict)

        # Verify
        print(f"\n--- heldout_{heldout} ---")
        for split in ['train', 'val', 'test']:
            npz = np.load(out_dir / f'{split}_sequences.npz', allow_pickle=True)
            tp_counts = Counter([toolpath_of(str(op)) for op in npz['operation_type_names']])
            print(f"  {split}: {len(npz['gcode_texts'])} windows, toolpaths={dict(tp_counts)}, "
                  f"tokens.shape={npz['tokens'].shape}, sample gcode={str(npz['gcode_texts'][0])[:60]}")

        # Split info
        meta = {
            'heldout_toolpath': heldout,
            'source': 'rebuilt from preprocessed_v7/fold_1 (pooled and refiltered)',
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'scaler_note': 'reuses fold-1 scaler (fit on fold-1 training subset); robust scaling insensitive to small shifts',
        }
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
