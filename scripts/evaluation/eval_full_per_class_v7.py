#!/usr/bin/env python3
"""Full per-class evaluation using the training script's DataLoader for correct MWC handling."""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from collections import defaultdict, Counter
from run_decoder_quick_test import (
    DecoderQuickTestDataset, CachedDecoderDataset, SensorMultiHeadDecoder,
    load_frozen_encoder, GCodeTokenizer, decoder_collate_fn,
)
from miracle.utilities.gcode_tokenizer import TokenizerConfig
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = "/home/seacuello/Documents/gcode_fingerprinting"
VOCAB_PATH = f"{PROJECT}/data/gcode_vocab_712.json"
BEST_SEEDS = {1: 2024, 2: 123, 3: 456, 4: 2024, 5: 789}
MULTISEED_BASE = f"{PROJECT}/outputs/decoder20260304/v7_best_5fold_multiseed"
PREPROC_BASE = f"{PROJECT}/outputs/decoder20260304/preprocessed_v7"
OUTPUT_DIR = f"{PROJECT}/outputs/decoder20260304/paper"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocab/tokenizer
with open(VOCAB_PATH) as f:
    vdata = json.load(f)
vocab = vdata['vocab']
id_to_token = {v: k for k, v in vocab.items()}
BOS = vocab['BOS']; EOS = vocab['EOS']; PAD = vocab['PAD']
cfg = TokenizerConfig(**vdata['config'])
tokenizer = GCodeTokenizer(cfg, vocab=vocab)

all_predictions = []
per_class_seq = defaultdict(lambda: {'correct': 0, 'total': 0})
per_class_tok = defaultdict(lambda: {'correct': 0, 'total': 0})
fold_results = {}

for fold in range(1, 6):
    seed = BEST_SEEDS[fold]
    run_dir = f"{MULTISEED_BASE}/fold_{fold}_seed_{seed}"
    print(f"\nFold {fold} (seed={seed})")

    # Load checkpoint
    ckpt_path = f"{run_dir}/decoder_checkpoint/best_decoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['decoder_state_dict']
    max_win = sd['window_pos_embed.weight'].shape[0] if 'window_pos_embed.weight' in sd else 10

    # Build model matching the checkpoint
    decoder = SensorMultiHeadDecoder(
        vocab_size=712, d_model=384, n_heads=12, n_layers=8,
        sensor_dim=256, n_operations=9,
        memory_pos_encoding=True, use_sensor_prior=True,
        use_window_position=True, max_seq_len=16,
        max_windows_per_file=max_win,
    ).to(device)

    # Load state dict (handle grammar mask buffers)
    model_state = decoder.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_state and v.shape == model_state[k].shape}
    decoder.load_state_dict(filtered, strict=False)
    decoder.eval()

    # Load test data using the training script's dataset classes
    test_npz = f"{PREPROC_BASE}/fold_{fold}/test_sequences.npz"
    test_ds = DecoderQuickTestDataset(test_npz, tokenizer, max_token_len=16)

    # Load encoder memory and op_pred
    memory = torch.load(f"{run_dir}/encoder_memory/test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(f"{run_dir}/encoder_memory/test_op_pred.pt", map_location=device, weights_only=False)

    # Wrap with MWC (same as training)
    npz_data = np.load(test_npz, allow_pickle=True)
    source_files = [str(s) for s in npz_data['source_file']]
    op_names = [str(s) for s in npz_data['operation_type_names']]
    gcode_texts = [str(s) for s in npz_data['gcode_texts']]

    aug_ds = CachedDecoderDataset(
        test_ds, memory, op_pred,
        multi_window_context=2,
        window_dropout=0.0, training=False,
    )

    loader = DataLoader(aug_ds, batch_size=32, shuffle=False,
                       collate_fn=decoder_collate_fn, num_workers=0)

    correct = 0
    total = 0
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            inp = batch['input_tokens'].to(device)
            tgt = batch['target_tokens'].to(device)
            mem = batch['memory'].to(device)
            op = batch['op_pred'].to(device)
            pad_mask = (inp == PAD)

            extra = {}
            if 'window_index' in batch:
                extra['window_index'] = batch['window_index'].to(device)
            if 'total_windows' in batch:
                extra['total_windows'] = batch['total_windows'].to(device)

            outputs = decoder(tokens=inp, sensor_embeddings=mem, operation_type=op,
                            tgt_key_padding_mask=pad_mask, **extra)
            logits = outputs.get('legacy_logits', outputs.get('raw_legacy_logits'))
            preds = logits.argmax(-1)

            B = inp.size(0)
            for b in range(B):
                if sample_idx >= len(op_names):
                    break
                op_name = op_names[sample_idx]

                # Get non-pad target tokens (excluding EOS for comparison)
                tgt_ids = [t.item() for t in tgt[b] if t.item() != PAD]
                tgt_no_eos = [t for t in tgt_ids if t != EOS]
                pred_ids = [preds[b, j].item() for j in range(len(tgt_no_eos))]

                seq_match = tgt_no_eos == pred_ids

                tgt_str = ' '.join([id_to_token.get(t, '?') for t in tgt_no_eos])
                pred_str = ' '.join([id_to_token.get(t, '?') for t in pred_ids])

                all_predictions.append({
                    'fold': fold, 'idx': sample_idx, 'op': op_name,
                    'gcode': gcode_texts[sample_idx],
                    'target': tgt_str, 'pred': pred_str, 'match': seq_match,
                })

                per_class_seq[op_name]['total'] += 1
                if seq_match:
                    correct += 1
                    per_class_seq[op_name]['correct'] += 1

                # Token-level
                for j in range(len(tgt_no_eos)):
                    per_class_tok[op_name]['total'] += 1
                    if j < len(pred_ids) and tgt_no_eos[j] == pred_ids[j]:
                        per_class_tok[op_name]['correct'] += 1

                total += 1
                sample_idx += 1

    fold_acc = correct / total * 100
    fold_results[fold] = {'acc': fold_acc, 'correct': correct, 'total': total}
    print(f"  Seq: {correct}/{total} = {fold_acc:.1f}%")

# Summary
total_c = sum(r['correct'] for r in fold_results.values())
total_t = sum(r['total'] for r in fold_results.values())
overall = total_c / total_t * 100
print(f"\nOVERALL: {total_c}/{total_t} = {overall:.1f}%")

print("\nPER-CLASS SEQUENCE ACCURACY:")
for op in sorted(per_class_seq):
    r = per_class_seq[op]
    acc = r['correct'] / r['total'] * 100 if r['total'] else 0
    print(f"  {op:<22} {r['correct']:>4}/{r['total']:<4} = {acc:>6.1f}%")

print("\nPER-CLASS TOKEN ACCURACY:")
for op in sorted(per_class_tok):
    r = per_class_tok[op]
    acc = r['correct'] / r['total'] * 100 if r['total'] else 0
    print(f"  {op:<22} {r['correct']:>4}/{r['total']:<4} = {acc:>6.1f}%")

# Save JSON
with open(f"{OUTPUT_DIR}/per_class_v7_definitive.json", 'w') as f:
    json.dump({
        'fold_results': {str(k): v for k, v in fold_results.items()},
        'per_class_sequence': {op: dict(v) for op, v in per_class_seq.items()},
        'per_class_token': {op: dict(v) for op, v in per_class_tok.items()},
        'overall_seq': overall,
        'best_seeds': BEST_SEEDS,
        'eval_method': 'teacher_forced_with_mwc',
        'all_predictions': all_predictions,
    }, f, indent=2)

# Confusion matrix
seq_pairs = Counter()
for p in all_predictions:
    seq_pairs[(p['target'], p['pred'])] += 1

all_targets = sorted(set(p['target'] for p in all_predictions))
n = len(all_targets)
s2i = {s: i for i, s in enumerate(all_targets)}

def shorten(s):
    for prefix in ['NUM_X_','NUM_Y_','NUM_Z_','NUM_R_','NUM_F_']:
        s = s.replace(prefix, prefix[4:6])
    return s[:30] + '...' if len(s) > 30 else s

cm = np.zeros((n, n), dtype=int)
for (tgt, pred), cnt in seq_pairs.items():
    ti, pi = s2i.get(tgt), s2i.get(pred)
    if ti is not None and pi is not None:
        cm[ti, pi] += cnt

labels = [shorten(s) for s in all_targets]
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
ax.set_yticklabels(labels, fontsize=5.5)
ax.set_xlabel('Predicted Sequence', fontsize=11)
ax.set_ylabel('True Sequence', fontsize=11)
ax.set_title(f'Sequence-Level Confusion Matrix (V7 Best Config, 5-Fold, n={total_t})', fontsize=12)
for i in range(n):
    for j in range(n):
        if cm[i, j] > 0:
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=5, color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/confusion_matrix_v7.pdf", dpi=150, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/figures/confusion_matrix_v7.png", dpi=150, bbox_inches='tight')
print(f"\nSaved to {OUTPUT_DIR}/per_class_v7_definitive.json and confusion_matrix_v7.pdf")
