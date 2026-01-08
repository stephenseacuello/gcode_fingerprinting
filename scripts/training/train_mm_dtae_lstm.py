#!/usr/bin/env python3
"""
Train MM-DTAE_LSTM: Multi-Modal Denoising Temporal Auto-Encoder with LSTM.

Key Features:
1. Bi-directional LSTM for temporal modeling (captures long-range dependencies)
2. Denoising autoencoder for reconstruction (robust feature learning)
3. Multi-task classification head (9-class operation type)
4. Designed for scalability (handles more classes/data well)

Advantages over MLP:
- Temporal modeling: Explicitly captures time-series patterns
- Reconstruction loss: Learns more robust representations
- Scalability: Better suited for adding new classes

MPS-compatible: Uses conditional pin_memory.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# MM-DTAE_LSTM Model Architecture
# ============================================================================

class MM_DTAE_LSTM(nn.Module):
    """
    Multi-Modal Denoising Temporal Auto-Encoder with LSTM.

    Architecture:
    1. Input projection: sensor features → hidden_dim
    2. Encoder LSTM: Bi-directional, captures temporal patterns
    3. Bottleneck: Compressed representation (for classification)
    4. Decoder LSTM: Reconstructs original sequence
    5. Classification head: Multi-task outputs

    Loss = reconstruction_loss + classification_loss
    """

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_classes: int = 9,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        noise_factor: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.noise_factor = noise_factor
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Encoder LSTM (bi-directional for richer representations)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Bottleneck (compressed representation)
        encoder_output_dim = hidden_dim * self.num_directions
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Decoder projection
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Decoder LSTM (uni-directional for reconstruction)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Classification head (from bottleneck)
        # Uses mean pooling over time of bottleneck features
        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

        # Temporal attention for classification (optional, better than mean pooling)
        self.temporal_attention = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=1)
        )

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for denoising autoencoder."""
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        # Input projection
        x = self.input_proj(x)  # [B, T, hidden_dim]

        # LSTM encoding
        encoded, (h_n, c_n) = self.encoder_lstm(x)  # [B, T, hidden_dim*num_directions]

        # Bottleneck
        latent = self.bottleneck(encoded)  # [B, T, latent_dim]

        return latent, (h_n, c_n)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        # Decoder projection
        x = self.decoder_proj(latent)  # [B, T, hidden_dim]

        # LSTM decoding
        decoded, _ = self.decoder_lstm(x)  # [B, T, hidden_dim]

        # Reconstruction
        reconstruction = self.reconstruction_head(decoded)  # [B, T, input_dim]

        return reconstruction

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """Classify from latent representation using temporal attention."""
        # Temporal attention pooling
        attention_weights = self.temporal_attention(latent)  # [B, T, 1]
        pooled = (latent * attention_weights).sum(dim=1)  # [B, latent_dim]

        # Classification
        logits = self.classification_head(pooled)  # [B, n_classes]

        return logits, attention_weights

    def forward(self, x: torch.Tensor, add_noise: bool = True):
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, input_dim]
            add_noise: Whether to add noise for denoising (training only)

        Returns:
            reconstruction: Reconstructed input [B, T, input_dim]
            class_logits: Classification logits [B, n_classes]
            latent: Latent representation [B, T, latent_dim]
            attention_weights: Temporal attention weights [B, T, 1]
        """
        # Optionally add noise
        noisy_x = self.add_noise(x) if add_noise else x

        # Encode
        latent, _ = self.encode(noisy_x)

        # Decode (reconstruct original, not noisy)
        reconstruction = self.decode(latent)

        # Classify
        class_logits, attention_weights = self.classify(latent)

        return reconstruction, class_logits, latent, attention_weights

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for downstream tasks."""
        self.eval()
        with torch.no_grad():
            latent, _ = self.encode(x)
            attention_weights = self.temporal_attention(latent)
            pooled = (latent * attention_weights).sum(dim=1)
        return pooled


# ============================================================================
# Dataset
# ============================================================================

class SensorDataset(Dataset):
    """Dataset for sensor-based classification with reconstruction target."""

    def __init__(self, continuous: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.continuous = torch.from_numpy(continuous).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.continuous[idx]
        y = self.labels[idx]

        if self.augment:
            # Jittering
            x = x + torch.randn_like(x) * 0.01
            # Random time shift (circular)
            shift = torch.randint(0, x.size(0), (1,)).item()
            x = torch.roll(x, shifts=shift, dims=0)

        return x, y


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion_cls, criterion_rec,
                device, rec_weight=0.3):
    """Train one epoch with combined loss."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        reconstruction, class_logits, latent, attention = model(x, add_noise=True)

        # Classification loss
        cls_loss = criterion_cls(class_logits, y)

        # Reconstruction loss
        rec_loss = criterion_rec(reconstruction, x)

        # Combined loss
        loss = cls_loss + rec_weight * rec_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_cls_loss += cls_loss.item() * x.size(0)
        total_rec_loss += rec_loss.item() * x.size(0)

        pred = class_logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'rec_loss': total_rec_loss / total,
        'accuracy': correct / total
    }


def evaluate(model, loader, criterion_cls, criterion_rec, device,
             rec_weight=0.3, class_names=None):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    all_preds = []
    all_labels = []
    all_attention = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass (no noise during evaluation)
            reconstruction, class_logits, latent, attention = model(x, add_noise=False)

            # Losses
            cls_loss = criterion_cls(class_logits, y)
            rec_loss = criterion_rec(reconstruction, x)
            loss = cls_loss + rec_weight * rec_loss

            total_loss += loss.item() * x.size(0)
            total_cls_loss += cls_loss.item() * x.size(0)
            total_rec_loss += rec_loss.item() * x.size(0)

            all_preds.extend(class_logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_attention.append(attention.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Per-class recall
    per_class_recall = {}
    for c in np.unique(all_labels):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = (all_preds[mask] == c).mean()
            name = class_names[c] if class_names else str(c)
            per_class_recall[name] = recall

    return {
        'loss': total_loss / len(all_labels),
        'cls_loss': total_cls_loss / len(all_labels),
        'rec_loss': total_rec_loss / len(all_labels),
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class_recall': per_class_recall
    }


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MM-DTAE_LSTM Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction_comparison(model, loader, device, output_path, n_samples=3):
    """Plot original vs reconstructed sequences."""
    model.eval()

    # Get a batch
    x, y = next(iter(loader))
    x = x[:n_samples].to(device)

    with torch.no_grad():
        reconstruction, _, _, _ = model(x, add_noise=False)

    # Plot
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))

    x_np = x.cpu().numpy()
    rec_np = reconstruction.cpu().numpy()

    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(x_np[i].T, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'Original (Sample {i+1})')
        axes[i, 0].set_ylabel('Feature')

        # Reconstruction
        axes[i, 1].imshow(rec_np[i].T, aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed (Sample {i+1})')

    axes[-1, 0].set_xlabel('Time Step')
    axes[-1, 1].set_xlabel('Time Step')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MM-DTAE_LSTM model')
    parser.add_argument('--split-dir', type=str, default='outputs/grouped_splits_4digit',
                        help='Directory with grouped splits')
    parser.add_argument('--output-dir', type=str, default='outputs/mm_dtae_lstm',
                        help='Output directory')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--noise-factor', type=float, default=0.1,
                        help='Noise factor for denoising autoencoder')
    parser.add_argument('--bidirectional', action='store_true', default=True)

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--rec-weight', type=float, default=0.3,
                        help='Weight for reconstruction loss')
    parser.add_argument('--augment', action='store_true', help='Enable augmentation')
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Logging
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='gcode-mm-dtae-lstm')
    parser.add_argument('--run-name', type=str, default=None)

    args = parser.parse_args()

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MPS-compatible: Disable pin_memory on MPS
    use_pin_memory = device.type == 'cuda'

    # Load data
    data_dir = Path(args.split_dir)
    train_data = np.load(data_dir / 'train_sequences.npz', allow_pickle=True)
    val_data = np.load(data_dir / 'val_sequences.npz', allow_pickle=True)
    test_data = np.load(data_dir / 'test_sequences.npz', allow_pickle=True)

    # Get class names - handle missing operation_type_names key
    if 'operation_type_names' in train_data.files:
        class_names = list(np.unique(train_data['operation_type_names']))
    else:
        # Infer from operation_type - use default operation names
        DEFAULT_OPERATION_NAMES = [
            "adaptive",        # 0
            "adaptive150025",  # 1
            "face",            # 2
            "face150025",      # 3
            "pocket",          # 4
            "pocket150025",    # 5
            "damageadaptive",  # 6
            "damageface",      # 7
            "damagepocket",    # 8
        ]
        unique_ops = np.unique(train_data['operation_type'])
        n_unique = len(unique_ops)
        if n_unique <= len(DEFAULT_OPERATION_NAMES):
            class_names = DEFAULT_OPERATION_NAMES[:n_unique]
        else:
            class_names = [f"Op-{i}" for i in range(n_unique)]
        print(f"Note: operation_type_names not found, using default names")
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    # Create datasets
    train_dataset = SensorDataset(
        train_data['continuous'],
        train_data['operation_type'],
        augment=args.augment
    )
    val_dataset = SensorDataset(val_data['continuous'], val_data['operation_type'])
    test_dataset = SensorDataset(test_data['continuous'], test_data['operation_type'])

    # Create dataloaders with MPS-compatible pin_memory
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             pin_memory=use_pin_memory)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Input dimension
    input_dim = train_data['continuous'].shape[-1]  # 155
    seq_len = train_data['continuous'].shape[1]
    print(f"Input dim: {input_dim}, Seq len: {seq_len}")

    # Create model
    model = MM_DTAE_LSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_classes=n_classes,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        noise_factor=args.noise_factor
    ).to(device)

    print(f"\nModel: MM-DTAE_LSTM")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Latent dim: {args.latent_dim}")
    print(f"  - LSTM layers: {args.num_lstm_layers}")
    print(f"  - Bidirectional: {args.bidirectional}")
    print(f"  - Noise factor: {args.noise_factor}")

    # Class weights for imbalanced data
    class_counts = Counter(train_data['operation_type'])
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / class_counts[i] for i in range(n_classes)]).float().to(device)
    class_weights = class_weights / class_weights.sum() * n_classes

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_rec = nn.MSELoss()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        run_name = args.run_name or f"mm-dtae-lstm-{args.hidden_dim}h-{args.latent_dim}l"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print("TRAINING MM-DTAE_LSTM")
    print(f"{'='*60}")

    for epoch in range(args.max_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_rec,
            device, rec_weight=args.rec_weight
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion_cls, criterion_rec,
            device, rec_weight=args.rec_weight, class_names=class_names
        )

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scheduler.step()

        # Log
        print(f"Epoch {epoch+1}/{args.max_epochs}: "
              f"Train Loss={train_metrics['loss']:.4f} (cls={train_metrics['cls_loss']:.4f}, rec={train_metrics['rec_loss']:.4f}), "
              f"Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}")

        if args.use_wandb:
            log_dict = {
                'train/loss': train_metrics['loss'],
                'train/cls_loss': train_metrics['cls_loss'],
                'train/rec_loss': train_metrics['rec_loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/cls_loss': val_metrics['cls_loss'],
                'val/rec_loss': val_metrics['rec_loss'],
                'val/accuracy': val_metrics['accuracy'],
                'lr': scheduler.get_last_lr()[0],
            }
            for name, recall in val_metrics['per_class_recall'].items():
                log_dict[f'val/recall_{name}'] = recall
            wandb.log(log_dict)

        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_metrics['accuracy'],
                'args': vars(args),
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(
        model, test_loader, criterion_cls, criterion_rec,
        device, rec_weight=args.rec_weight, class_names=class_names
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - MM-DTAE_LSTM")
    print(f"{'='*60}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Val/Test Gap: {(best_val_acc - test_metrics['accuracy'])*100:.1f}%")
    print(f"\nReconstruction Loss (Test): {test_metrics['rec_loss']:.4f}")
    print(f"\nPer-class Test Recall:")
    for name, recall in test_metrics['per_class_recall'].items():
        print(f"  {name}: {recall:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    present_classes = sorted(set(test_metrics['labels']) | set(test_metrics['predictions']))
    present_names = [class_names[i] for i in present_classes]
    print(classification_report(
        test_metrics['labels'], test_metrics['predictions'],
        labels=present_classes, target_names=present_names, zero_division=0
    ))

    # Save visualizations
    plot_confusion_matrix(
        test_metrics['labels'], test_metrics['predictions'],
        class_names, output_dir / 'confusion_matrix.png'
    )

    plot_reconstruction_comparison(
        model, test_loader, device, output_dir / 'reconstruction_comparison.png'
    )

    # Save results
    results = {
        'model': 'MM-DTAE_LSTM',
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_metrics['accuracy']),
        'val_test_gap': float(best_val_acc - test_metrics['accuracy']),
        'test_rec_loss': float(test_metrics['rec_loss']),
        'per_class_recall': {k: float(v) for k, v in test_metrics['per_class_recall'].items()},
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'args': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if args.use_wandb:
        wandb.log({
            'test/accuracy': test_metrics['accuracy'],
            'test/rec_loss': test_metrics['rec_loss'],
            'test/val_test_gap': best_val_acc - test_metrics['accuracy'],
        })
        for name, recall in test_metrics['per_class_recall'].items():
            wandb.log({f'test/recall_{name}': recall})
        wandb.finish()

    print(f"\nResults saved to {output_dir}")
    print(f"\nModel advantages for scalability:")
    print("  1. LSTM captures temporal patterns (better than mean pooling)")
    print("  2. Reconstruction loss learns robust features")
    print("  3. Latent space can be used for anomaly detection")
    print("  4. Easy to add new classes with minimal retraining")


if __name__ == '__main__':
    main()
