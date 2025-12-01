"""
Training script for G-code prediction model.

Uses the full MM_DTAE_LSTM model with:
- Multi-task losses (G-code prediction, reconstruction, fingerprinting)
- Comprehensive metrics tracking
- Checkpointing and logging
"""
import os
import platform

# Enable MPS fallback for Mac (required for transformer padding masks)
# This must be set before importing torch
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.training.losses import MultiTaskLoss
from miracle.training.metrics import (
    compute_classification_metrics,
    compute_perplexity,
    MetricsTracker,
    compute_top_k_accuracy,
)
from miracle.utilities.device import get_device, print_device_info

__all__ = ["Trainer"]


def convert_to_python_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.

    Args:
        obj: Any object that may contain NumPy types

    Returns:
        Object with all NumPy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class Trainer:
    """Trainer for G-code prediction model with scheduled sampling support."""

    def __init__(
        self,
        model: MM_DTAE_LSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        device: str = None,  # Auto-detects GPU if available
        output_dir: Path = Path('outputs/training'),
        scheduled_sampling: bool = False,
        sampling_strategy: str = 'linear',
        total_epochs: int = 100,
        use_wandb: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: float = 1.0,
        warmup_epochs: int = 0,
    ):
        # Auto-detect device if not specified
        if device is None or isinstance(device, str):
            device = get_device(device)
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Scheduled sampling parameters
        self.scheduled_sampling = scheduled_sampling
        self.sampling_strategy = sampling_strategy
        self.total_epochs = total_epochs

        # W&B tracking
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: W&B requested but not available. Continuing without W&B logging.")

        # Metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # Best model tracking
        self.best_val_loss = float('inf')
        self.epoch = 0

    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """
        Calculate teacher forcing ratio based on current epoch.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Teacher forcing ratio (1.0 = always use ground truth, 0.0 = always use predictions)
        """
        if not self.scheduled_sampling:
            return 1.0  # Always use teacher forcing if scheduled sampling is disabled

        progress = epoch / self.total_epochs

        if self.sampling_strategy == 'linear':
            # Linearly decay from 1.0 to 0.5
            return max(0.5, 1.0 - 0.5 * progress)

        elif self.sampling_strategy == 'exponential':
            # Exponential decay
            import math
            k = 5.0  # decay rate
            return max(0.5, math.exp(-k * progress))

        elif self.sampling_strategy == 'inverse_sigmoid':
            # Inverse sigmoid decay (slow at first, then rapid, then slow)
            import math
            k = 10.0
            return max(0.5, k / (k + math.exp(progress * k)))

        else:
            return 1.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with optional scheduled sampling."""
        self.model.train()
        self.train_metrics.reset()

        # Get teacher forcing ratio for this epoch
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train] TF={teacher_forcing_ratio:.2f}")

        for batch in pbar:
            # Move to device
            continuous = batch['continuous'].to(self.device)  # [B, T, D_cont]
            categorical = batch['categorical'].float().to(self.device)  # [B, T, D_cat] - convert to float
            tokens = batch['tokens'].to(self.device)  # [B, max_token_len]
            lengths = batch['lengths'].to(self.device)  # [B]

            B, T, D_cont = continuous.shape
            D_cat = categorical.size(-1)

            # Prepare modality inputs
            mods = [continuous, categorical]  # List of [B, T, D_m]

            # Scheduled sampling: decide whether to use ground truth or model predictions
            use_teacher_forcing = (not self.scheduled_sampling) or (np.random.random() < teacher_forcing_ratio)

            if use_teacher_forcing:
                # Standard teacher forcing: use ground truth tokens
                gcode_in = tokens
            else:
                # Use model's own predictions (scheduled sampling)
                with torch.no_grad():
                    # First get memory from encoder
                    temp_outputs = self.model(mods=mods, lengths=lengths, gcode_in=None, modality_dropout_p=0.0)
                    memory = temp_outputs['memory']

                    # Generate tokens autoregressively
                    bos_id = 1  # <SOS> token
                    generated = self.model.gcode_head.generate(memory=memory, max_len=tokens.size(1), bos_id=bos_id)

                    # Use generated tokens as input (but still compute loss against ground truth)
                    gcode_in = generated

            # Forward pass
            outputs = self.model(
                mods=mods,
                lengths=lengths,
                gcode_in=gcode_in,
                modality_dropout_p=0.1,  # Apply modality dropout
            )

            # Prepare targets (focus on G-code token prediction)
            targets = {
                'gcode_tokens': tokens,
                # Skip cls and recon losses - focus on token prediction
            }

            # Compute loss
            loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.optimizer.step()

            # Step scheduler if using OneCycleLR (steps per batch)
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Track metrics
            self.train_metrics.update(loss_dict, count=B)

            # Update progress bar
            pbar.set_postfix(self.train_metrics.get_latest())

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        self.val_metrics.reset()

        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]"):
            # Move to device
            continuous = batch['continuous'].to(self.device)
            categorical = batch['categorical'].float().to(self.device)  # Convert to float
            tokens = batch['tokens'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            B = continuous.size(0)

            # Prepare modality inputs
            mods = [continuous, categorical]

            # Forward pass
            outputs = self.model(
                mods=mods,
                lengths=lengths,
                gcode_in=tokens,
                modality_dropout_p=0.0,  # No dropout during validation
            )

            # Prepare targets (focus on G-code token prediction)
            targets = {
                'gcode_tokens': tokens,
                # Skip cls and recon losses
            }

            # Compute loss
            loss, loss_dict = self.criterion(outputs, targets)

            # Track metrics
            self.val_metrics.update(loss_dict, count=B)

            # Collect predictions for metrics
            if 'gcode_logits' in outputs:
                preds = torch.argmax(outputs['gcode_logits'], dim=-1)  # [B, T]
                all_predictions.append(preds.cpu())
                all_targets.append(tokens.cpu())

        # Compute additional metrics
        metrics = self.val_metrics.compute()

        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Token-level accuracy
            token_metrics = compute_classification_metrics(
                all_predictions.flatten(),
                all_targets.flatten(),
                ignore_index=0,  # Ignore PAD
            )
            metrics.update({f"token_{k}": v for k, v in token_metrics.items()})

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.to_config_dict(),
        }

        # Save latest
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model to {best_path}")

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        if self.use_wandb:
            print(f"W&B logging: Enabled")
        print(f"{'='*60}\n")

        history = {
            'train': [],
            'val': [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history['train'].append(train_metrics)

            # Validate
            val_metrics = self.validate()
            history['val'].append(val_metrics)

            # Log to W&B
            if self.use_wandb:
                wandb_metrics = {
                    'epoch': epoch + 1,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                }
                # Add training metrics with 'train/' prefix
                for k, v in train_metrics.items():
                    wandb_metrics[f'train/{k}'] = v
                # Add validation metrics with 'val/' prefix
                for k, v in val_metrics.items():
                    wandb_metrics[f'val/{k}'] = v
                wandb.log(wandb_metrics)

            # Print summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - {self.train_metrics}")
            print(f"  Val   - {self.val_metrics}")

            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']

            self.save_checkpoint(is_best=is_best)

            # Step scheduler (except OneCycleLR which steps per batch)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the metric
                    self.scheduler.step(val_metrics['total'])
                else:
                    # Other schedulers step by epoch
                    self.scheduler.step()

        # Save training history (convert NumPy types to Python types)
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(convert_to_python_types(history), f, indent=2)

        # Log best model to W&B
        if self.use_wandb:
            wandb.log({'best_val_loss': self.best_val_loss})
            # Save best model as artifact
            best_checkpoint_path = self.output_dir / 'checkpoint_best.pt'
            if best_checkpoint_path.exists():
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(str(best_checkpoint_path))
                wandb.log_artifact(artifact)
                print("Logged best model to W&B artifacts")

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        return history


def main():
    parser = argparse.ArgumentParser(description="Train G-code prediction model")
    parser.add_argument('--data-dir', type=Path, required=True, help="Preprocessed data directory")
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/training'), help="Output directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--device', type=str, default=None, help="Device (auto-detects GPU if available, or specify: cpu/cuda/mps)")
    parser.add_argument('--d-model', type=int, default=128, help="Model hidden dimension")
    parser.add_argument('--lstm-layers', type=int, default=2, help="LSTM layers")

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
                        help="Optimizer type")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="Weight decay (L2 regularization)")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum (for SGD)")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")

    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['cosine', 'step', 'plateau', 'exponential', 'onecycle'],
                        help="LR scheduler type (None to disable)")
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument('--scheduler-step-size', type=int, default=10,
                        help="Step size for StepLR scheduler")
    parser.add_argument('--scheduler-gamma', type=float, default=0.1,
                        help="Gamma for StepLR/ExponentialLR scheduler")
    # NOTE: warmup_epochs parameter exists but warmup is not yet implemented here
    # For LR warmup, use OneCycleLR scheduler (includes automatic warmup + cosine decay)
    # Available via train_sweep.py with scheduler='onecycle'
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help="Number of warmup epochs (NOT YET IMPLEMENTED - use OneCycleLR instead)")

    # Scheduled sampling arguments
    parser.add_argument('--scheduled-sampling', action='store_true',
                        help="Enable scheduled sampling (gradually reduce teacher forcing)")
    parser.add_argument('--sampling-strategy', type=str, default='linear',
                        choices=['linear', 'exponential', 'inverse_sigmoid'],
                        help="Decay strategy for teacher forcing ratio")

    # Class weights arguments (for handling class imbalance)
    parser.add_argument('--use-class-weights', action='store_true',
                        help="Enable class weights for handling imbalanced vocabulary")
    parser.add_argument('--class-weights-path', type=Path, default=None,
                        help="Path to precomputed class weights (.pt file)")
    parser.add_argument('--class-weight-alpha', type=float, default=2.0,
                        help="Alpha scaling factor for class weight computation")
    parser.add_argument('--use-focal-loss', action='store_true',
                        help="Use focal loss instead of cross-entropy for G-code prediction")
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help="Gamma parameter for focal loss (higher = more focus on hard examples)")
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help="Label smoothing factor for G-code prediction (0.0-0.2 recommended)")

    # W&B arguments
    parser.add_argument('--use-wandb', action='store_true',
                        help="Enable Weights & Biases logging")
    parser.add_argument('--wandb-project', type=str, default='gcode-fingerprinting',
                        help="W&B project name")
    parser.add_argument('--wandb-name', type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=[],
                        help="W&B tags for organizing runs")

    args = parser.parse_args()

    # Setup device (auto-detect GPU or use specified device)
    device = get_device(args.device)
    print_device_info(device)

    # Update args.device to actual device for later use
    args.device = str(device)

    # Load datasets
    print("Loading datasets...")
    train_dataset = GCodeDataset(args.data_dir / 'train_sequences.npz')
    val_dataset = GCodeDataset(args.data_dir / 'val_sequences.npz')

    # Load metadata
    metadata_path = args.data_dir / 'train_sequences_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Continuous features: {metadata['n_continuous_features']}")
    print(f"Categorical features: {metadata['n_categorical_features']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load or compute class weights (if enabled)
    class_weights = None
    if args.use_class_weights:
        print("\nLoading class weights for imbalanced vocabulary...")
        if args.class_weights_path and args.class_weights_path.exists():
            # Load precomputed weights
            class_weights = torch.load(args.class_weights_path, map_location='cpu', weights_only=True)
            print(f"✅ Loaded class weights from: {args.class_weights_path}")
        else:
            # Load weights from default location
            default_path = args.data_dir / 'class_weights.pt'
            if default_path.exists():
                class_weights = torch.load(default_path, map_location='cpu', weights_only=True)
                print(f"✅ Loaded class weights from: {default_path}")
            else:
                print(f"⚠️  Class weights file not found. Expected at: {default_path}")
                print("   Generate weights using: python train_with_class_weights.py")
                print("   Continuing without class weights...")
                args.use_class_weights = False

        if class_weights is not None:
            # Scale weights by alpha
            class_weights = class_weights * args.class_weight_alpha / 2.0
            class_weights = class_weights.to(device)
            print(f"   Min weight: {class_weights.min():.4f}")
            print(f"   Max weight: {class_weights.max():.4f}")
            print(f"   Mean weight: {class_weights.mean():.4f}")
            print(f"   Alpha scaling: {args.class_weight_alpha}")

    # Create model
    print("\nCreating model...")
    config = ModelConfig(
        sensor_dims=[metadata['n_continuous_features'], metadata['n_categorical_features']],
        d_model=args.d_model,
        lstm_layers=args.lstm_layers,
        gcode_vocab=metadata['vocab_size'],
        n_heads=4,
        fp_dim=128,
    )
    model = MM_DTAE_LSTM(config)
    print(f"Model parameters: {model.count_params(model):,}")

    # Create loss
    criterion = MultiTaskLoss(
        vocab_size=metadata['vocab_size'],
        pad_token_id=0,
        adaptive=False,  # Disable adaptive weighting to prevent NaN (use fixed weights)
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    # Print loss configuration
    loss_info = []
    if args.use_class_weights and class_weights is not None:
        loss_info.append("class weights")
    if args.use_focal_loss:
        loss_info.append(f"focal loss (γ={args.focal_gamma})")
    if args.label_smoothing > 0:
        loss_info.append(f"label smoothing ({args.label_smoothing})")
    if loss_info:
        print(f"Loss configuration: {', '.join(loss_info)}")
    else:
        print("Loss configuration: standard cross-entropy")

    # Create optimizer
    print(f"\nCreating {args.optimizer.upper()} optimizer...")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Create learning rate scheduler
    scheduler = None
    if args.scheduler:
        print(f"Creating {args.scheduler} LR scheduler...")
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
            )
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=args.scheduler_patience,
                factor=args.scheduler_gamma, verbose=True
            )
        elif args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.scheduler_gamma
            )
        elif args.scheduler == 'onecycle':
            steps_per_epoch = len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
            )
        else:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"gcode_model_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        training_args = vars(args).copy()
        for key, value in training_args.items():
            if isinstance(value, Path):
                training_args[key] = str(value)

        json.dump({
            'model_config': config.__dict__,
            'training_args': training_args,
            'metadata': metadata,
        }, f, indent=2)

    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        # Generate run name if not provided
        run_name = args.wandb_name
        if run_name is None:
            run_name = f"d{args.d_model}_lstm{args.lstm_layers}_lr{args.lr}_bs{args.batch_size}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            config={
                # Model architecture
                'd_model': args.d_model,
                'lstm_layers': args.lstm_layers,
                'n_heads': config.n_heads,
                'fp_dim': config.fp_dim,
                # Training hyperparameters
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'device': args.device,
                # Optimizer
                'optimizer': args.optimizer,
                'weight_decay': args.weight_decay,
                'momentum': args.momentum if args.optimizer == 'sgd' else None,
                'grad_clip': args.grad_clip,
                # Scheduler
                'scheduler': args.scheduler,
                'scheduler_patience': args.scheduler_patience if args.scheduler == 'plateau' else None,
                'scheduler_step_size': args.scheduler_step_size if args.scheduler == 'step' else None,
                'scheduler_gamma': args.scheduler_gamma,
                'warmup_epochs': args.warmup_epochs,
                # Scheduled sampling
                'scheduled_sampling': args.scheduled_sampling,
                'sampling_strategy': args.sampling_strategy,
                # Data info
                'vocab_size': metadata['vocab_size'],
                'n_continuous_features': metadata['n_continuous_features'],
                'n_categorical_features': metadata['n_categorical_features'],
            }
        )
        print(f"W&B run: {wandb.run.name}")
        print(f"W&B URL: {wandb.run.url}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        output_dir=output_dir,
        scheduled_sampling=args.scheduled_sampling,
        sampling_strategy=args.sampling_strategy,
        total_epochs=args.epochs,
        use_wandb=args.use_wandb,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
    )

    history = trainer.train(num_epochs=args.epochs)

    # Finish W&B run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\n✅ Training complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
