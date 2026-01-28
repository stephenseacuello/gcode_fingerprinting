#!/usr/bin/env python3
"""
Comprehensive evaluation: Precision/Recall/F1, BLEU, and Token Edit Distance.

Computes:
1. Per-class and macro-averaged Precision, Recall, F1 (teacher-forced)
2. BLEU-1 through BLEU-4 (autoregressive decoding)
3. Normalized token edit distance (autoregressive decoding)

Usage:
    python scripts/evaluation/evaluate_comprehensive_metrics.py
    python scripts/evaluation/evaluate_comprehensive_metrics.py --model-dir outputs/jan26/ensemble/seed_42
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits
from miracle.utilities.device import get_device


# ---------------------------------------------------------------------------
# Levenshtein edit distance (no external dependency)
# ---------------------------------------------------------------------------
def edit_distance(a, b):
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# BLEU (simple n-gram implementation, no nltk needed)
# ---------------------------------------------------------------------------
def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, hypothesis, max_n=4):
    """Compute BLEU-1..4 for a single reference/hypothesis pair."""
    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(reference, n))
        hyp_ngrams = Counter(ngrams(hypothesis, n))
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        scores[n] = clipped / total
    return scores


def corpus_bleu(references, hypotheses, max_n=4):
    """Compute corpus-level BLEU with brevity penalty."""
    import math
    clipped_counts = {n: 0 for n in range(1, max_n + 1)}
    total_counts = {n: 0 for n in range(1, max_n + 1)}
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            ref_ng = Counter(ngrams(ref, n))
            hyp_ng = Counter(ngrams(hyp, n))
            clipped_counts[n] += sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
            total_counts[n] += max(sum(hyp_ng.values()), 1)

    # Brevity penalty
    bp = math.exp(1 - ref_len / max(hyp_len, 1)) if hyp_len < ref_len else 1.0

    # Geometric mean of precisions
    log_avg = 0.0
    for n in range(1, max_n + 1):
        p = clipped_counts[n] / max(total_counts[n], 1)
        if p == 0:
            return {f'BLEU-{i}': 0.0 for i in range(1, max_n + 1)}
        log_avg += math.log(p) / max_n

    bleu_total = bp * math.exp(log_avg)

    # Individual BLEU-n (with BP)
    results = {}
    for n in range(1, max_n + 1):
        p = clipped_counts[n] / max(total_counts[n], 1)
        results[f'BLEU-{n}'] = bp * p
    results['BLEU'] = bleu_total
    return results


# ---------------------------------------------------------------------------
# Model loading (same pattern as evaluate_topk_accuracy.py)
# ---------------------------------------------------------------------------
def load_models(model_dir, vocab_path, split_dir, device):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = {}
    args_path = model_dir / 'args.json'
    if args_path.exists():
        with open(args_path) as f:
            args = json.load(f)
    elif 'args' in checkpoint:
        args = checkpoint['args']

    decomposer = TokenDecomposer(str(vocab_path))

    input_dim = 296
    metadata_path = Path(split_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 296)

    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=args.get('encoder_hidden_dim', 256),
        latent_dim=args.get('sensor_dim', 128),
        n_operations=args.get('n_operations', 9),
        n_scales=args.get('encoder_n_scales', 4),
        kernel_sizes=args.get('encoder_kernel_sizes', [3, 5, 7, 11]),
        dilations=args.get('encoder_dilations', [1, 2, 4, 8]),
        dropout=args.get('encoder_dropout', 0.3),
        lstm_layers=args.get('encoder_lstm_layers', 2),
    ).to(device)
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    encoder.eval()

    decoder = SensorMultiHeadDecoder(
        vocab_size=decomposer.vocab_size,
        d_model=args.get('d_model', 192),
        n_heads=args.get('n_heads', 8),
        n_layers=args.get('n_layers', 4),
        sensor_dim=128,
        n_operations=args.get('n_operations', 9),
        n_types=args.get('n_types', 4),
        n_commands=args.get('n_commands', 6),
        n_param_types=args.get('n_param_types', 10),
        max_int_digits=2,
        n_decimal_digits=4,
        dropout=args.get('dropout', 0.3),
        embed_dropout=args.get('embed_dropout', 0.1),
        max_seq_len=args.get('max_seq_len', 32),
    ).to(device)
    decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder.eval()

    return encoder, decoder, decomposer, args


# ---------------------------------------------------------------------------
# Greedy autoregressive decode
# ---------------------------------------------------------------------------
def greedy_decode(encoder, decoder, sensor_features, decomposer, device, max_len=32):
    """Autoregressively decode a single sample."""
    bos_id = decomposer.vocab.get('<BOS>', 1)
    eos_id = decomposer.vocab.get('<EOS>', 2)

    pooled, sensor_emb = encoder.encode(sensor_features)
    op_logits, _ = encoder.classify(pooled)
    pred_op = op_logits.argmax(-1)

    generated = [bos_id]
    for _ in range(max_len):
        input_ids = torch.tensor([generated], device=device)
        decoder_out = decoder(input_ids, sensor_emb, pred_op)
        logits = decoder_out.get('legacy_logits', decoder_out.get('logits', decoder_out.get('composed_logits')))
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if next_id == eos_id:
            break

    return generated[1:]  # strip BOS


# ---------------------------------------------------------------------------
# Classify token for per-class reporting
# ---------------------------------------------------------------------------
def classify_token(token_str):
    if token_str.startswith('<'):
        return 'special'
    if len(token_str) >= 1 and token_str[0] in ('G', 'M'):
        return 'command'
    if len(token_str) == 1 and token_str.isalpha():
        return 'parameter'
    if token_str in ('+', '-'):
        return 'sign'
    return 'numeric_or_other'


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='outputs/jan26/ensemble/seed_42')
    parser.add_argument('--vocab-path', default='data/vocabulary_4digit_full.json')
    parser.add_argument('--split-dir', default='outputs/jan23_followup/no_leakage')
    parser.add_argument('--output-dir', default='outputs/analysis/comprehensive_metrics')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("\nLoading models...")
    encoder, decoder, decomposer, model_args = load_models(
        args.model_dir, args.vocab_path, args.split_dir, device
    )

    print("\nLoading test data...")
    test_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'test', max_token_len=model_args.get('max_seq_len', 32)
    )
    print(f"  Test samples: {len(test_dataset)}")

    id_to_token = {v: k for k, v in decomposer.vocab.items()}
    pad_id = decomposer.vocab.get('<PAD>', 0)
    bos_id = decomposer.vocab.get('<BOS>', 1)
    eos_id = decomposer.vocab.get('<EOS>', 2)

    # Accumulators
    # Per-class TP/FP/FN for precision/recall/F1
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # For BLEU and edit distance (autoregressive)
    all_references = []
    all_hypotheses = []
    edit_distances = []
    normalized_edit_distances = []

    n_samples = len(test_dataset)
    print(f"\nRunning evaluation on {n_samples} samples...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Evaluating"):
            sample = test_dataset[idx]
            sensor_features = sample['sensor_features'].unsqueeze(0).to(device)
            target_tokens = sample['target_tokens'].to(device)

            # ---------------------------------------------------------------
            # Part 1: Teacher-forced precision/recall/F1
            # ---------------------------------------------------------------
            pooled, sensor_emb = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(pooled)
            pred_op = op_logits.argmax(-1)

            decoder_input = torch.cat([
                torch.tensor([bos_id], device=device),
                target_tokens[:-1]
            ]).unsqueeze(0)

            decoder_out = decoder(decoder_input, sensor_emb, pred_op)
            logits = decoder_out.get('legacy_logits', decoder_out.get('logits', decoder_out.get('composed_logits')))
            logits = logits.squeeze(0)  # [seq_len, vocab]

            preds = logits.argmax(dim=-1)  # [seq_len]
            targets = target_tokens  # [seq_len]

            for t in range(targets.size(0)):
                tgt = targets[t].item()
                pred = preds[t].item()
                if tgt == pad_id:
                    continue
                if pred == tgt:
                    tp[tgt] += 1
                else:
                    fp[pred] += 1
                    fn[tgt] += 1

            # ---------------------------------------------------------------
            # Part 2: Autoregressive decode for BLEU + edit distance
            # ---------------------------------------------------------------
            hyp_ids = greedy_decode(encoder, decoder, sensor_features, decomposer, device,
                                   max_len=target_tokens.size(0) + 5)

            # Convert to token strings, strip special tokens
            ref_strs = []
            for t in target_tokens.cpu().tolist():
                if t in (pad_id, bos_id, eos_id):
                    continue
                ref_strs.append(id_to_token.get(t, str(t)))

            hyp_strs = []
            for t in hyp_ids:
                if t in (pad_id, bos_id, eos_id):
                    continue
                hyp_strs.append(id_to_token.get(t, str(t)))

            all_references.append(ref_strs)
            all_hypotheses.append(hyp_strs)

            ed = edit_distance(ref_strs, hyp_strs)
            edit_distances.append(ed)
            ned = ed / max(len(ref_strs), 1)
            normalized_edit_distances.append(ned)

    # ===================================================================
    # Compute Precision / Recall / F1
    # ===================================================================
    print("\n" + "=" * 70)
    print("PRECISION / RECALL / F1")
    print("=" * 70)

    # Collect all token IDs that appeared
    all_ids = set(tp.keys()) | set(fp.keys()) | set(fn.keys())

    # Per-class metrics
    per_class = {}
    for tid in all_ids:
        precision = tp[tid] / max(tp[tid] + fp[tid], 1)
        recall = tp[tid] / max(tp[tid] + fn[tid], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        support = tp[tid] + fn[tid]
        per_class[tid] = {
            'token': id_to_token.get(tid, str(tid)),
            'category': classify_token(id_to_token.get(tid, '')),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }

    # Macro averages
    precisions = [v['precision'] for v in per_class.values() if v['support'] > 0]
    recalls = [v['recall'] for v in per_class.values() if v['support'] > 0]
    f1s = [v['f1'] for v in per_class.values() if v['support'] > 0]

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    # Weighted averages
    total_support = sum(v['support'] for v in per_class.values())
    w_precision = sum(v['precision'] * v['support'] for v in per_class.values()) / max(total_support, 1)
    w_recall = sum(v['recall'] * v['support'] for v in per_class.values()) / max(total_support, 1)
    w_f1 = sum(v['f1'] * v['support'] for v in per_class.values()) / max(total_support, 1)

    print(f"\n{'Averaging':>15}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 50)
    print(f"{'Macro':>15}  {macro_precision:>10.4f}  {macro_recall:>10.4f}  {macro_f1:>10.4f}")
    print(f"{'Weighted':>15}  {w_precision:>10.4f}  {w_recall:>10.4f}  {w_f1:>10.4f}")

    # Per-category averages
    categories = defaultdict(list)
    for v in per_class.values():
        if v['support'] > 0:
            categories[v['category']].append(v)

    print(f"\n{'Category':>18}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
    print("-" * 62)
    for cat in sorted(categories.keys()):
        items = categories[cat]
        cp = np.mean([v['precision'] for v in items])
        cr = np.mean([v['recall'] for v in items])
        cf = np.mean([v['f1'] for v in items])
        cs = sum(v['support'] for v in items)
        print(f"{cat:>18}  {cp:>10.4f}  {cr:>10.4f}  {cf:>10.4f}  {cs:>8}")

    # Top-10 most frequent tokens
    sorted_by_support = sorted(per_class.values(), key=lambda x: -x['support'])[:15]
    print(f"\nTop-15 tokens by frequency:")
    print(f"{'Token':>15}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
    print("-" * 58)
    for v in sorted_by_support:
        print(f"{v['token']:>15}  {v['precision']:>10.4f}  {v['recall']:>10.4f}  {v['f1']:>10.4f}  {v['support']:>8}")

    # ===================================================================
    # BLEU scores
    # ===================================================================
    print("\n" + "=" * 70)
    print("BLEU SCORES (Autoregressive Decoding)")
    print("=" * 70)

    bleu_results = corpus_bleu(all_references, all_hypotheses)
    for k, v in sorted(bleu_results.items()):
        print(f"  {k}: {v:.4f}")

    # ===================================================================
    # Edit Distance
    # ===================================================================
    print("\n" + "=" * 70)
    print("TOKEN EDIT DISTANCE (Autoregressive Decoding)")
    print("=" * 70)

    print(f"  Mean edit distance: {np.mean(edit_distances):.2f}")
    print(f"  Median edit distance: {np.median(edit_distances):.2f}")
    print(f"  Mean normalized edit distance: {np.mean(normalized_edit_distances):.4f}")
    print(f"  Median normalized edit distance: {np.median(normalized_edit_distances):.4f}")
    print(f"  Std normalized edit distance: {np.std(normalized_edit_distances):.4f}")
    print(f"  Perfect sequences (ED=0): {sum(1 for d in edit_distances if d == 0)}/{len(edit_distances)} ({sum(1 for d in edit_distances if d == 0)/len(edit_distances):.1%})")

    # ===================================================================
    # Save results
    # ===================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_dir': str(args.model_dir),
        'n_samples': n_samples,
        'precision_recall_f1': {
            'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
            'weighted': {'precision': w_precision, 'recall': w_recall, 'f1': w_f1},
            'per_category': {
                cat: {
                    'precision': float(np.mean([v['precision'] for v in items])),
                    'recall': float(np.mean([v['recall'] for v in items])),
                    'f1': float(np.mean([v['f1'] for v in items])),
                    'support': sum(v['support'] for v in items),
                }
                for cat, items in categories.items()
            },
            'top_tokens': [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in entry.items()}
                for entry in sorted_by_support
            ],
        },
        'bleu': {k: float(v) for k, v in bleu_results.items()},
        'edit_distance': {
            'mean': float(np.mean(edit_distances)),
            'median': float(np.median(edit_distances)),
            'mean_normalized': float(np.mean(normalized_edit_distances)),
            'median_normalized': float(np.median(normalized_edit_distances)),
            'std_normalized': float(np.std(normalized_edit_distances)),
            'perfect_sequences': sum(1 for d in edit_distances if d == 0),
            'total_sequences': len(edit_distances),
        },
    }

    output_path = output_dir / 'comprehensive_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
