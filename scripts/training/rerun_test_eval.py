#!/usr/bin/env python3
"""Rerun test evaluation on a saved checkpoint from train_sensor_multihead.py."""
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn
from miracle.training.metrics import compute_comprehensive_metrics
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory with best_model.pt and args.json')
    parser.add_argument('--device', default=None)
    cli_args = parser.parse_args()

    output_dir = Path(cli_args.output_dir)
    ckpt_path = output_dir / 'best_model.pt'
    args_path = output_dir / 'args.json'

    with open(args_path) as f:
        args = json.load(f)

    if cli_args.device:
        device = torch.device(cli_args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load vocab
    with open(args['vocab_path']) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    vocab_size = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    PAD_TOKEN_ID = vocab.get('<PAD>', 0)
    EOS_TOKEN_ID = vocab.get('<EOS>', 2)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Infer input_dim from checkpoint conv weight shape
    enc_sd = ckpt['encoder_state_dict']
    for k, v in enc_sd.items():
        if 'conv_branches.0.0.weight' in k:
            sensor_input_dim = v.shape[1]
            break
    else:
        sensor_input_dim = args.get('sensor_input_dim', 155)
    print(f"Inferred encoder input_dim: {sensor_input_dim}")

    encoder = EnhancedEncoder(
        input_dim=sensor_input_dim,
        hidden_dim=args.get('encoder_hidden_dim', 256),
        latent_dim=args.get('sensor_dim', 128),
        n_operations=args.get('n_operations', 9),
        use_multiscale=True,
        n_scales=args.get('encoder_n_scales', 4),
        use_multihead_pooling=args.get('use_multihead_pooling', False),
        pooling_n_heads=args.get('pooling_n_heads', 4),
        pooling_n_queries=args.get('pooling_n_queries', 8),
    )
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    encoder.to(device)
    encoder.eval()

    # Build decoder
    d_model = args['d_model']
    n_heads = args['n_heads']
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=args['n_layers'],
        dropout=args['dropout'],
        max_seq_len=args.get('max_seq_len', 32),
        sensor_dim=args.get('sensor_dim', 128),
        n_operations=args.get('n_operations', 9),
        n_types=args.get('n_types', 4),
        n_commands=args.get('n_commands', 6),
        n_param_types=args.get('n_param_types', 10),
    )
    dec_sd = dict(ckpt.get('decoder_state_dict', ckpt.get('model_state_dict')))
    saved_type_mask = dec_sd.pop('type_token_mask', None)
    decoder.load_state_dict(dec_sd, strict=True)
    # Disable type constraint for evaluation (it was trained with it but
    # the raw legacy_logits already reflect training; applying the mask
    # at eval with a potentially mismatched mask corrupts predictions)
    decoder.use_type_constraint = False
    decoder.to(device)
    decoder.eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Best epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best val token acc: {ckpt.get('best_val_acc', '?')}")

    # Load test set
    max_seq_len = args.get('max_seq_len', 32)
    test_dataset = DecoderDatasetFromSplits(
        split_dir=args['split_dir'], split='test', max_token_len=max_seq_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.get('batch_size', 10),
        shuffle=False, collate_fn=decoder_collate_fn, num_workers=0,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # Evaluate
    total_correct = 0
    total_tokens = 0
    seq_correct = 0
    seq_total = 0
    op_correct = 0
    op_total = 0
    all_preds = []
    all_targets = []
    all_pred_seqs = []
    all_target_seqs = []

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']

            # Encoder op acc
            if 'cls_logits' in encoder_out:
                op_preds = encoder_out['cls_logits'].argmax(-1)
                op_correct += (op_preds == operations).sum().item()
                op_total += operations.size(0)

            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)
            B, T = target_tokens.shape

            all_preds.append(preds)
            all_targets.append(target_tokens)

            for b in range(B):
                eos_positions = (target_tokens[b] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                eos_pos = eos_positions[0].item() + 1 if len(eos_positions) > 0 else T

                valid_mask = torch.zeros(T, dtype=torch.bool, device=device)
                valid_mask[:eos_pos] = True
                valid_mask = valid_mask & (target_tokens[b] != PAD_TOKEN_ID)

                total_correct += (preds[b][valid_mask] == target_tokens[b][valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

                if valid_mask.sum() > 0:
                    pred_seq = preds[b][valid_mask]
                    target_seq = target_tokens[b][valid_mask]
                    all_pred_seqs.append(pred_seq.cpu().tolist())
                    all_target_seqs.append(target_seq.cpu().tolist())
                    if (pred_seq == target_seq).all():
                        seq_correct += 1
                    seq_total += 1

    token_acc = total_correct / total_tokens if total_tokens > 0 else 0
    seq_acc = seq_correct / seq_total if seq_total > 0 else 0
    encoder_op_acc = op_correct / op_total if op_total > 0 else 0

    # Comprehensive metrics
    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    comp = compute_comprehensive_metrics(
        all_preds_t, all_targets_t,
        all_pred_seqs, all_target_seqs,
        pad_token=PAD_TOKEN_ID,
    )

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Token Accuracy:      {token_acc:.4f} ({token_acc:.2%})")
    print(f"  Sequence Accuracy:   {seq_acc:.4f} ({seq_acc:.2%})")
    print(f"  Encoder Op Accuracy: {encoder_op_acc:.4f} ({encoder_op_acc:.2%})")
    print(f"  Precision (macro):   {comp.get('precision', 0):.4f}")
    print(f"  Recall (macro):      {comp.get('recall', 0):.4f}")
    print(f"  F1 (macro):          {comp.get('f1', 0):.4f}")
    print(f"  BLEU-1:              {comp.get('bleu_1', 0):.4f}")
    print(f"  BLEU-4:              {comp.get('bleu_4', 0):.4f}")
    print(f"  BLEU:                {comp.get('bleu', 0):.4f}")
    print(f"  Exact Match:         {comp.get('exact_match', 0):.4f}")
    print(f"  Avg Edit Distance:   {comp.get('avg_edit_distance', 0):.4f}")
    print(f"  Token Accuracy (comp): {comp.get('accuracy', 0):.4f}")

    # Save results
    results = {
        'test_token_accuracy': token_acc,
        'test_sequence_accuracy': seq_acc,
        'test_encoder_op_accuracy': encoder_op_acc,
        'comprehensive': comp,
        'best_epoch': ckpt.get('epoch', None),
        'best_val_token_acc': ckpt.get('best_val_acc', None),
    }
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
