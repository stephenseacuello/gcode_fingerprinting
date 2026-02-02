#!/usr/bin/env python3
"""
Train best model configuration with multiple seeds for ensemble.
After training, learns optimal ensemble weights.
"""
import os
import sys
import json
import argparse
import platform
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Windows has issues with multiprocessing DataLoader workers
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn
from miracle.training.metrics import compute_comprehensive_metrics


def evaluate_on_test(
    decoder,
    encoder,
    data_dir: str,
    vocab_path: str,
    config: dict,
    device: str = 'cuda',
):
    """Evaluate model on test set and return detailed metrics."""
    from collections import defaultdict

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    idx_to_token = {v: k for k, v in vocab.items()}

    max_seq_len = config.get('max_seq_len', 32)
    test_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='test',
        max_token_len=max_seq_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.get('batch_size', 16), shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )

    decoder.eval()
    encoder.eval()

    total_correct = 0
    total_tokens = 0

    # Per-token-type tracking (command, param_type, digit, etc.)
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    # Per-operation tracking
    op_correct = defaultdict(int)
    op_total = defaultdict(int)

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)
            mask = target_tokens != 0

            # Overall accuracy
            correct = ((preds == target_tokens) & mask)
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Per-operation accuracy
            for i in range(operations.size(0)):
                op = operations[i].item()
                sample_mask = mask[i]
                sample_correct = correct[i]
                op_correct[op] += sample_correct.sum().item()
                op_total[op] += sample_mask.sum().item()

            # Per-token-type accuracy (based on token position patterns)
            for i in range(target_tokens.size(0)):
                for j in range(target_tokens.size(1)):
                    if mask[i, j]:
                        target_idx = target_tokens[i, j].item()
                        target_tok = idx_to_token.get(target_idx, '<UNK>')

                        # Classify token type
                        if target_tok.startswith('G') or target_tok.startswith('M'):
                            tok_type = 'command'
                        elif target_tok in ['X', 'Y', 'Z', 'E', 'F', 'I', 'J', 'R', 'S']:
                            tok_type = 'param_type'
                        elif target_tok in ['+', '-']:
                            tok_type = 'sign'
                        elif target_tok == '.':
                            tok_type = 'decimal'
                        elif target_tok.isdigit():
                            tok_type = 'digit'
                        else:
                            tok_type = 'other'

                        type_total[tok_type] += 1
                        if correct[i, j]:
                            type_correct[tok_type] += 1

    test_acc = total_correct / total_tokens if total_tokens > 0 else 0

    # Compute per-type accuracies
    type_accs = {}
    for t in type_total:
        type_accs[t] = type_correct[t] / type_total[t] if type_total[t] > 0 else 0

    # Compute per-operation accuracies
    op_accs = {}
    for op in op_total:
        op_accs[op] = op_correct[op] / op_total[op] if op_total[op] > 0 else 0

    # Comprehensive metrics: collect all preds/targets for precision/recall/F1/BLEU/ED
    all_preds_list = []
    all_targets_list = []
    pred_sequences = []
    target_sequences = []

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)
            mask = target_tokens != 0

            for i in range(preds.size(0)):
                m = mask[i]
                all_preds_list.append(preds[i][m])
                all_targets_list.append(target_tokens[i][m])
                pred_sequences.append(preds[i][m].cpu().tolist())
                target_sequences.append(target_tokens[i][m].cpu().tolist())

    all_preds_flat = torch.cat(all_preds_list)
    all_targets_flat = torch.cat(all_targets_list)
    comprehensive = compute_comprehensive_metrics(
        all_preds_flat, all_targets_flat,
        pred_sequences, target_sequences,
        pad_token=0, num_classes=len(vocab),
    )

    return {
        'test_acc': test_acc,
        'type_accs': type_accs,
        'op_accs': op_accs,
        'total_tokens': total_tokens,
        'precision': comprehensive['precision'],
        'recall': comprehensive['recall'],
        'f1': comprehensive['f1'],
        'bleu': comprehensive['bleu'],
        'bleu_1': comprehensive['bleu_1'],
        'bleu_4': comprehensive['bleu_4'],
        'avg_edit_distance': comprehensive['avg_edit_distance'],
        'exact_match': comprehensive['exact_match'],
    }


def train_single_seed(
    config: dict,
    seed: int,
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
    freeze_encoder: bool = False,
    encoder_lr_scale: float = 0.1,
    max_epochs: int = None,
    patience: int = None,
):
    """Train a single model with specified seed."""
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"{'='*60}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load metadata to get input_dim
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 155)
    else:
        input_dim = config.get('sensor_input_dim', 155)

    # Load datasets
    max_seq_len = config.get('max_seq_len', 32)
    train_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='train',
        max_token_len=max_seq_len,
    )
    val_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='val',
        max_token_len=max_seq_len,
    )

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
        pooling_n_heads=config.get('pooling_n_heads', 8),
        pooling_n_queries=config.get('pooling_n_queries', 16),
    )

    # Load pretrained encoder
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt)), strict=False)
    encoder.to(device)
    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        print(f"  Encoder FROZEN")
    else:
        encoder.train()
        print(f"  Encoder UNFROZEN (lr_scale={encoder_lr_scale})")

    # Create decoder
    d_model = config['d_model']
    n_heads = config['n_heads']
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    ffn_multiplier = config.get('ffn_multiplier', 4)
    d_ff = d_model * ffn_multiplier

    decoder = SensorMultiHeadDecoder(
        sensor_dim=128,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config['n_layers'],
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_ff,
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config.get('embed_dropout', 0.1),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_sensor_prior=config.get('use_sensor_prior', True),
        sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
    )

    if hasattr(decoder, 'set_vocab'):
        decoder.set_vocab(vocab)
    decoder.to(device)

    # Optimizer
    lr = config['learning_rate']
    wd = config['weight_decay']
    betas = (config.get('beta1', 0.9), config.get('beta2', 0.999))
    if freeze_encoder:
        optimizer = AdamW(decoder.parameters(), lr=lr, weight_decay=wd, betas=betas)
    else:
        optimizer = AdamW([
            {'params': decoder.parameters(), 'lr': lr},
            {'params': encoder.parameters(), 'lr': lr * encoder_lr_scale},
        ], weight_decay=wd, betas=betas)

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('restart_period', 20),
    )

    # Loss
    label_smoothing = config.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # Training loop
    if max_epochs is None:
        max_epochs = config.get('max_epochs', 200)
    if patience is None:
        patience = config.get('patience', 50)
    best_val_acc = 0.0
    patience_counter = 0

    seed_output_dir = Path(output_dir) / f'seed_{seed}'
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        # Training
        decoder.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            if freeze_encoder:
                with torch.no_grad():
                    encoder_out = encoder(sensor_data)
                    sensor_memory = encoder_out['memory']
            else:
                encoder_out = encoder(sensor_data)
                sensor_memory = encoder_out['memory']

            optimizer.zero_grad()
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            # Encoder operation classification loss
            if not freeze_encoder and 'cls_logits' in encoder_out:
                op_loss_weight = config.get('op_loss_weight', 1.0)
                op_cls_loss = nn.CrossEntropyLoss()(encoder_out['cls_logits'], operations)
                loss = loss + op_loss_weight * op_cls_loss

            loss.backward()
            all_params = list(decoder.parameters())
            if not freeze_encoder:
                all_params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, config.get('grad_clip', 1.0))
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = target_tokens != 0
            train_correct += ((preds == target_tokens) & mask).sum().item()
            train_total += mask.sum().item()

        scheduler.step()

        # Validation
        decoder.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                sensor_data = batch['sensor_features'].to(device)
                operations = batch['operation_type'].to(device)
                input_tokens = batch['input_tokens'].to(device)
                target_tokens = batch['target_tokens'].to(device)

                encoder_out = encoder(sensor_data)
                sensor_memory = encoder_out['memory']
                outputs = decoder(
                    tokens=input_tokens,
                    sensor_embeddings=sensor_memory,
                    operation_type=operations,
                )

                logits = outputs['legacy_logits']
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                mask = target_tokens != 0
                val_correct += ((preds == target_tokens) & mask).sum().item()
                val_total += mask.sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch}:  Token Acc: train={train_acc:.2%}  val={val_acc:.2%}")

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'seed': seed,
            }, seed_output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Best Val Token Acc: {best_val_acc:.2%}")

    # Evaluate on test set
    print(f"  Evaluating on test set...")
    test_results = evaluate_on_test(
        decoder=decoder,
        encoder=encoder,
        data_dir=data_dir,
        vocab_path=vocab_path,
        config=config,
        device=device,
    )
    test_acc = test_results['test_acc']
    print(f"  Test Token Acc: {test_acc:.2%}")
    print(f"  Per-type accuracies:")
    for tok_type, acc in sorted(test_results['type_accs'].items()):
        print(f"    {tok_type}: {acc:.2%}")

    # Save final results
    with open(seed_output_dir / 'results.json', 'w') as f:
        json.dump({
            'seed': seed,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_precision': test_results.get('precision', 0),
            'test_recall': test_results.get('recall', 0),
            'test_f1': test_results.get('f1', 0),
            'test_bleu': test_results.get('bleu', 0),
            'test_bleu_1': test_results.get('bleu_1', 0),
            'test_bleu_4': test_results.get('bleu_4', 0),
            'test_avg_edit_distance': test_results.get('avg_edit_distance', 0),
            'test_exact_match': test_results.get('exact_match', 0),
            'test_type_accs': test_results['type_accs'],
            'test_op_accs': {str(k): v for k, v in test_results['op_accs'].items()},
            'final_epoch': epoch,
            'freeze_encoder': freeze_encoder,
            'config': config,
        }, f, indent=2)

    return best_val_acc, test_acc, seed_output_dir / 'best_model.pt'


def learn_ensemble_weights(
    model_paths: list,
    config: dict,
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
):
    """Learn optimal ensemble weights on validation set."""
    print(f"\n{'='*60}")
    print("Learning ensemble weights")
    print(f"{'='*60}")

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load metadata to get input_dim
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        input_dim = metadata.get('n_continuous_features', 155)
    else:
        input_dim = config.get('sensor_input_dim', 155)

    # Load validation data
    max_seq_len = config.get('max_seq_len', 32)
    val_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='val',
        max_token_len=max_seq_len,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get('batch_size', 16), shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS
    )

    # Load encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
        pooling_n_heads=config.get('pooling_n_heads', 8),
        pooling_n_queries=config.get('pooling_n_queries', 16),
    )
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt)), strict=False)
    encoder.to(device)
    encoder.eval()

    # Load all decoders
    decoders = []
    for path in model_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)

        d_model = config['d_model']
        n_heads = config['n_heads']
        if d_model % n_heads != 0:
            for nh in [32, 24, 16, 8]:
                if d_model % nh == 0:
                    n_heads = nh
                    break

        decoder = SensorMultiHeadDecoder(
            sensor_dim=128,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=config['n_layers'],
            vocab_size=len(vocab),
            n_operations=9,
            n_types=4,
            n_commands=6,
            n_param_types=10,
            d_ff=d_model * config.get('ffn_multiplier', 4),
            dropout=config['dropout'],
            max_seq_len=max_seq_len,
            n_decimal_digits=4,
            max_int_digits=2,
            embed_dropout=config.get('embed_dropout', 0.1),
            drop_path_rate=config.get('drop_path_rate', 0.1),
            use_sensor_prior=config.get('use_sensor_prior', True),
            sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
        )
        decoder.load_state_dict(ckpt['model_state_dict'], strict=False)
        decoder.to(device)
        decoder.eval()
        decoders.append(decoder)

    # Learnable weights
    n_models = len(decoders)
    weights = nn.Parameter(torch.ones(n_models, device=device) / n_models)
    optimizer = torch.optim.Adam([weights], lr=0.01)

    # Train weights
    print(f"  Training weights for {n_models} models...")
    for weight_epoch in range(100):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for batch in val_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            with torch.no_grad():
                encoder_out = encoder(sensor_data)
                sensor_memory = encoder_out['memory']

            # Get logits from all models
            all_logits = []
            with torch.no_grad():
                for decoder in decoders:
                    outputs = decoder(
                        tokens=input_tokens,
                        sensor_embeddings=sensor_memory,
                        operation_type=operations,
                    )
                    all_logits.append(outputs['legacy_logits'])

            # Stack and apply softmax weights
            stacked = torch.stack(all_logits, dim=0)  # [n_models, B, L, V]
            softmax_weights = torch.softmax(weights, dim=0)
            ensemble_logits = (stacked * softmax_weights.view(-1, 1, 1, 1)).sum(dim=0)

            # Loss
            loss = nn.functional.cross_entropy(
                ensemble_logits.reshape(-1, ensemble_logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = ensemble_logits.argmax(dim=-1)
            mask = target_tokens != 0
            total_correct += ((preds == target_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()

        if weight_epoch % 20 == 0:
            acc = total_correct / total_tokens if total_tokens > 0 else 0
            print(f"    Weight epoch {weight_epoch}: loss={total_loss/len(val_loader):.4f}, acc={acc:.4f}")

    # Final weights
    final_weights = torch.softmax(weights, dim=0).detach().cpu().numpy()
    print(f"  Final weights: {final_weights}")

    # Final accuracy
    acc = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"  Ensemble val_acc: {acc:.4f}")

    # Save ensemble
    output_path = Path(output_dir)
    ensemble_data = {
        'model_paths': [str(p) for p in model_paths],
        'weights': final_weights.tolist(),
        'ensemble_val_acc': acc,
        'config': config,
    }
    with open(output_path / 'ensemble_weights.json', 'w') as f:
        json.dump(ensemble_data, f, indent=2)

    return final_weights, acc


def main():
    parser = argparse.ArgumentParser(description='Train ensemble with multiple seeds')
    parser.add_argument('--config', type=str, required=True, help='Path to best config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data splits')
    parser.add_argument('--encoder-path', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated seeds')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detected if not set)')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder (default: unfrozen)')
    parser.add_argument('--encoder-lr-scale', type=float, default=0.1, help='Encoder LR multiplier (default: 0.1)')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override max epochs from config')
    parser.add_argument('--patience', type=int, default=None, help='Override patience from config')
    parser.add_argument('--wandb-project', type=str, default='gcode-jan30-ensemble', help='WandB project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Device: {args.device}")

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f"Training with seeds: {seeds}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each seed (skip if model already exists)
    model_paths = []
    results = []
    for seed in seeds:
        expected_path = output_dir / f'seed_{seed}' / 'best_model.pt'
        results_path = output_dir / f'seed_{seed}' / 'results.json'
        if expected_path.exists():
            print(f"\n{'='*60}")
            print(f"Seed {seed}: Model already exists, skipping training")
            print(f"{'='*60}")
            # Load existing results
            ckpt = torch.load(expected_path, map_location='cpu', weights_only=False)
            val_acc = ckpt.get('val_acc', ckpt.get('best_val_acc', 0.0))
            # Try to load test_acc from results.json
            test_acc = 0.0
            if results_path.exists():
                with open(results_path) as f:
                    saved_results = json.load(f)
                test_acc = saved_results.get('test_acc', 0.0)
            print(f"  Loaded model with val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
            model_paths.append(str(expected_path))
            results.append({'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc})
        else:
            val_acc, test_acc, model_path = train_single_seed(
                config=config,
                seed=seed,
                data_dir=args.data_dir,
                encoder_path=args.encoder_path,
                vocab_path=args.vocab_path,
                output_dir=str(output_dir),
                device=args.device,
                freeze_encoder=args.freeze_encoder,
                encoder_lr_scale=args.encoder_lr_scale,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            model_paths.append(model_path)
            results.append({'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc})

    # Learn ensemble weights
    weights, ensemble_acc = learn_ensemble_weights(
        model_paths=model_paths,
        config=config,
        data_dir=args.data_dir,
        encoder_path=args.encoder_path,
        vocab_path=args.vocab_path,
        output_dir=str(output_dir),
        device=args.device,
    )

    # Save summary
    best_individual_val = max(r['val_acc'] for r in results)
    best_individual_test = max(r['test_acc'] for r in results)
    avg_test_acc = sum(r['test_acc'] for r in results) / len(results)

    summary = {
        'seeds': seeds,
        'individual_results': results,
        'ensemble_weights': weights.tolist(),
        'ensemble_val_acc': ensemble_acc,
        'best_individual_val_acc': best_individual_val,
        'best_individual_test_acc': best_individual_test,
        'avg_test_acc': avg_test_acc,
    }
    with open(output_dir / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Individual results:")
    for r in results:
        print(f"  Seed {r['seed']}: val={r['val_acc']:.2%}  test={r['test_acc']:.2%}")
    print(f"\nBest individual (val):  {best_individual_val:.2%}")
    print(f"Best individual (test): {best_individual_test:.2%}")
    print(f"Average test:           {avg_test_acc:.2%}")
    print(f"Ensemble (val):         {ensemble_acc:.2%}")
    print(f"Improvement:            {ensemble_acc - best_individual_val:+.2%}")


if __name__ == '__main__':
    main()
