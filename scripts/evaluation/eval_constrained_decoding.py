#!/usr/bin/env python3
"""
Evaluate constrained decoding on existing decoder checkpoints.

Uses the type head predictions to mask the legacy head logits at inference time:
- If type_head predicts COMMAND → only allow command token IDs
- If type_head predicts PARAM → only allow param token IDs
- If type_head predicts NUMERIC → only allow numeric token IDs
- If type_head predicts SPECIAL → only allow special token IDs

No retraining needed — just loads best checkpoints and re-evaluates.

Usage:
    python3 scripts/evaluation/eval_constrained_decoding.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.digit_value_head import DigitByDigitLoss
from miracle.utilities.gcode_tokenizer import GCodeTokenizer

# Import shared utilities from the training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_decoder_quick_test import (
    PAD, BOS, EOS, UNK, TYPE_SPECIAL, TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC,
    COMMANDS, PARAMS, CMD2ID, PARAM2ID,
    classify_token, build_modality_indices,
    DecoderQuickTestDataset, CachedDecoderDataset,
    load_frozen_encoder, cache_encoder_memory, decoder_collate_fn,
)


def build_type_masks(tokenizer):
    """Build boolean masks mapping each vocab token to its type."""
    vocab = tokenizer.vocab
    vocab_size = len(vocab)
    precision = tokenizer.cfg.precision

    # type_id for each token in vocab
    masks = {
        TYPE_SPECIAL: torch.zeros(vocab_size, dtype=torch.bool),
        TYPE_COMMAND: torch.zeros(vocab_size, dtype=torch.bool),
        TYPE_PARAM: torch.zeros(vocab_size, dtype=torch.bool),
        TYPE_NUMERIC: torch.zeros(vocab_size, dtype=torch.bool),
    }

    for tok_str, tok_id in vocab.items():
        type_id, _, _, _, _, _ = classify_token(tok_str, precision)
        masks[type_id][tok_id] = True

    # Print summary
    for type_id, name in [(TYPE_SPECIAL, 'SPECIAL'), (TYPE_COMMAND, 'COMMAND'),
                           (TYPE_PARAM, 'PARAM'), (TYPE_NUMERIC, 'NUMERIC')]:
        print(f"  {name}: {masks[type_id].sum().item()} tokens")

    return masks


@torch.no_grad()
def evaluate_constrained(decoder, loader, device, tokenizer, type_masks, constrained=True):
    """Evaluate with optional type-constrained decoding."""
    decoder.eval()

    token_correct = 0
    token_total = 0
    seq_match = 0
    seq_total = 0

    # Per-type tracking
    type_correct = 0
    type_total = 0
    cmd_correct = 0
    cmd_total = 0
    pt_correct = 0
    pt_total = 0
    num_correct = 0
    num_total = 0

    # Move masks to device
    device_masks = {k: v.to(device) for k, v in type_masks.items()}

    for batch in loader:
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        memory = batch['memory'].to(device)
        op_pred = batch['op_pred'].to(device)
        padding_mask = (input_tokens == PAD)
        type_targets = batch['type_targets'].to(device)
        cmd_targets = batch['command_targets'].to(device)
        pt_targets = batch['param_type_targets'].to(device)

        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op_pred,
            tgt_key_padding_mask=padding_mask,
        )

        logits = outputs.get('raw_legacy_logits', outputs.get('legacy_logits'))
        if logits is None:
            continue

        if constrained:
            # Get type predictions
            type_preds = outputs['type_logits'].argmax(-1)  # [B, L]

            # Apply type masks to legacy logits
            masked_logits = logits.clone()
            for type_id in [TYPE_SPECIAL, TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC]:
                # Where type_preds == type_id, mask out tokens not of that type
                type_match = (type_preds == type_id)  # [B, L]
                if type_match.any():
                    # Set logits of non-matching tokens to -inf
                    mask = ~device_masks[type_id]  # tokens to suppress
                    masked_logits[type_match] = masked_logits[type_match].masked_fill(
                        mask.unsqueeze(0).expand(type_match.sum(), -1), float('-inf')
                    )
            preds = masked_logits.argmax(-1)
        else:
            preds = logits.argmax(-1)

        # Token accuracy
        mask = target_tokens != PAD
        token_correct += ((preds == target_tokens) & mask).sum().item()
        token_total += mask.sum().item()

        # Sequence accuracy
        seq_match += ((preds == target_tokens) | ~mask).all(dim=1).sum().item()
        seq_total += target_tokens.size(0)

        # Type head accuracy
        type_preds_eval = outputs['type_logits'].argmax(-1)
        type_mask = type_targets >= 0
        type_correct += ((type_preds_eval == type_targets) & type_mask).sum().item()
        type_total += type_mask.sum().item()

        # Command head accuracy
        cmd_preds = outputs['command_logits'].argmax(-1)
        cmd_mask = cmd_targets >= 0
        cmd_correct += ((cmd_preds == cmd_targets) & cmd_mask).sum().item()
        cmd_total += cmd_mask.sum().item()

        # Param type head accuracy
        pt_preds = outputs['param_type_logits'].argmax(-1)
        pt_mask = pt_targets >= 0
        pt_correct += ((pt_preds == pt_targets) & pt_mask).sum().item()
        pt_total += pt_mask.sum().item()

        # Numeric token accuracy (subset of token accuracy)
        numeric_mask = (type_targets == TYPE_NUMERIC) & mask
        num_correct += ((preds == target_tokens) & numeric_mask).sum().item()
        num_total += numeric_mask.sum().item()

    return {
        'token_accuracy': token_correct / max(token_total, 1),
        'sequence_accuracy': seq_match / max(seq_total, 1),
        'type_accuracy': type_correct / max(type_total, 1),
        'command_accuracy': cmd_correct / max(cmd_total, 1),
        'param_type_accuracy': pt_correct / max(pt_total, 1),
        'numeric_accuracy': num_correct / max(num_total, 1),
        'num_numeric_tokens': num_total,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_path = Path("data/gcode_vocab_712.json")
    tokenizer = GCodeTokenizer.load(vocab_path)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab: {vocab_size} tokens")

    print("\nBuilding type masks:")
    type_masks = build_type_masks(tokenizer)

    base = Path("outputs/experiments_2026_02_25/full_w128_s32_cv")
    decoder_base = Path("outputs/decoder20260303/feat110")

    results = {'unconstrained': [], 'constrained': []}

    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        data_dir = base / f"fold_{fold}" / "preprocessed"
        ckpt_path = base / f"fold_{fold}" / "encoder" / "checkpoint" / "best_model.pt"
        decoder_ckpt_path = decoder_base / f"fold_{fold}" / "decoder_checkpoint" / "best_decoder.pt"

        # Load encoder
        encoder, enc_config, _ = load_frozen_encoder(str(ckpt_path), device)

        # Load metadata for group indices
        with open(data_dir / 'metadata.json') as f:
            metadata = json.load(f)
        columns = metadata['continuous_columns']
        _, group_indices, sensor_dims = build_modality_indices(columns)

        # Load test dataset
        # Phase-3 (decoder20260511): None auto-resolves from NPZ; V7 NPZ → ~16, V8 NPZ → larger.
        test_ds = DecoderQuickTestDataset(data_dir / 'test_sequences.npz', tokenizer, max_token_len=None)
        print(f"  Test samples: {test_ds.stats['n_samples']}")

        # Cache encoder memory for test
        memory, op_pred, cls_acc = cache_encoder_memory(encoder, test_ds, group_indices, device, batch_size=32)
        cached_ds = CachedDecoderDataset(test_ds, memory, op_pred)
        test_loader = DataLoader(cached_ds, batch_size=32, shuffle=False, collate_fn=decoder_collate_fn)

        # Load decoder
        decoder = SensorMultiHeadDecoder(
            vocab_size=vocab_size, d_model=192, sensor_dim=enc_config.d_model,
            n_operations=9, n_heads=8, n_layers=4,
            max_int_digits=2, n_decimal_digits=4, dropout=0.3, max_seq_len=16,
        )
        decoder.set_vocab(tokenizer.vocab)
        decoder = decoder.to(device)

        ckpt = torch.load(decoder_ckpt_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt['decoder_state_dict'])

        # Evaluate unconstrained
        unconstrained = evaluate_constrained(decoder, test_loader, device, tokenizer, type_masks, constrained=False)
        print(f"  Unconstrained: token={unconstrained['token_accuracy']:.1%} "
              f"seq={unconstrained['sequence_accuracy']:.1%} "
              f"numeric={unconstrained['numeric_accuracy']:.1%} ({unconstrained['num_numeric_tokens']} tokens)")

        # Evaluate constrained
        constrained = evaluate_constrained(decoder, test_loader, device, tokenizer, type_masks, constrained=True)
        print(f"  Constrained:   token={constrained['token_accuracy']:.1%} "
              f"seq={constrained['sequence_accuracy']:.1%} "
              f"numeric={constrained['numeric_accuracy']:.1%} ({constrained['num_numeric_tokens']} tokens)")

        delta_tok = constrained['token_accuracy'] - unconstrained['token_accuracy']
        delta_num = constrained['numeric_accuracy'] - unconstrained['numeric_accuracy']
        delta_seq = constrained['sequence_accuracy'] - unconstrained['sequence_accuracy']
        print(f"  Delta:         token={delta_tok:+.1%} seq={delta_seq:+.1%} numeric={delta_num:+.1%}")

        results['unconstrained'].append(unconstrained)
        results['constrained'].append(constrained)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (5-fold mean ± std)")
    print(f"{'='*60}")

    for mode in ['unconstrained', 'constrained']:
        metrics = results[mode]
        for key in ['token_accuracy', 'sequence_accuracy', 'numeric_accuracy', 'type_accuracy',
                    'command_accuracy', 'param_type_accuracy']:
            vals = [m[key] for m in metrics]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            print(f"  {mode:14s} {key:20s}: {mean:.1%} ± {std:.1%}")
        print()

    # Save results
    out_path = Path("outputs/decoder20260303/constrained_decoding_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
