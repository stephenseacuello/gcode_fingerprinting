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


def train_single_seed(
    config: dict,
    seed: int,
    data_dir: str,
    encoder_path: str,
    vocab_path: str,
    output_dir: str,
    device: str = 'cuda',
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
        encoder.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    encoder.to(device)
    encoder.eval()  # Freeze encoder

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
    optimizer = AdamW(
        decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('restart_period', 20),
    )

    # Loss
    label_smoothing = config.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # Training loop
    max_epochs = config.get('max_epochs', 200)
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

            with torch.no_grad():
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.get('grad_clip', 1.0))
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
            print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

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

    print(f"  Best val_acc: {best_val_acc:.4f}")

    # Save final results
    with open(seed_output_dir / 'results.json', 'w') as f:
        json.dump({
            'seed': seed,
            'best_val_acc': best_val_acc,
            'final_epoch': epoch,
            'config': config,
        }, f, indent=2)

    return best_val_acc, seed_output_dir / 'best_model.pt'


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
        encoder.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
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
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f"Training with seeds: {seeds}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each seed
    model_paths = []
    results = []
    for seed in seeds:
        val_acc, model_path = train_single_seed(
            config=config,
            seed=seed,
            data_dir=args.data_dir,
            encoder_path=args.encoder_path,
            vocab_path=args.vocab_path,
            output_dir=str(output_dir),
            device=args.device,
        )
        model_paths.append(model_path)
        results.append({'seed': seed, 'val_acc': val_acc})

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
    summary = {
        'seeds': seeds,
        'individual_results': results,
        'ensemble_weights': weights.tolist(),
        'ensemble_val_acc': ensemble_acc,
        'best_individual_acc': max(r['val_acc'] for r in results),
    }
    with open(output_dir / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best individual: {summary['best_individual_acc']:.4f}")
    print(f"Ensemble:        {ensemble_acc:.4f}")
    print(f"Improvement:     {ensemble_acc - summary['best_individual_acc']:.4f}")


if __name__ == '__main__':
    main()
