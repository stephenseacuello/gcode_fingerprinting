#!/usr/bin/env python3
"""
Test trained MM-LSTM-DAE model on temporally-shuffled data.

This shows whether MM-LSTM-DAE is robust to temporal disruption compared to XGBoost.

Expected outcome:
- If MM-LSTM-DAE maintains performance on shuffled data → robust temporal features
- If MM-LSTM-DAE degrades significantly → relies on specific temporal order

Usage:
    python romesh_changes/test_mmlstm_on_shuffled.py \
        --data-dir outputs/experiments/file_level_split/preprocessed \
        --model-dir outputs/experiments/file_level_split/encoder \
        --output-dir outputs/experiments/mmlstm_shuffled_test
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig


OPERATION_NAMES = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive',
    'damageface', 'damagepocket'
]


def load_npz_data(npz_path):
    """Load preprocessed data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    continuous = data['continuous']  # Shape: (N, T, F)
    categorical = data['categorical']  # Shape: (N, T, C)
    labels = data['operation_type']  # Operation type labels

    return continuous, categorical, labels


def shuffle_temporal_dimension(X, seed=42):
    """Shuffle timesteps within each window."""
    rng = np.random.RandomState(seed)
    X_shuffled = X.copy()

    for i in range(len(X_shuffled)):
        perm = rng.permutation(X_shuffled.shape[1])
        X_shuffled[i] = X_shuffled[i, perm, :]

    return X_shuffled


def load_model(model_dir, device):
    """Load trained MM-LSTM-DAE model."""
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        # Try to infer from checkpoint
        checkpoint_path = model_dir / 'best_model_test.pt'
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / 'best_model_val.pt'

        if not checkpoint_path.exists():
            raise ValueError(f"No checkpoint found in {model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Infer config from checkpoint
        encoder_state = checkpoint['encoder_state_dict']
        classifier_state = checkpoint['classifier_state_dict']

        # Get dimensions from state dict
        d_continuous = encoder_state['continuous_embedding.weight'].shape[1]
        d_categorical = encoder_state['categorical_embedding.weight'].shape[0]
        n_classes = classifier_state['fc.weight'].shape[0]
        d_model = encoder_state['continuous_embedding.weight'].shape[0]

        config = {
            'd_continuous': d_continuous,
            'd_categorical': d_categorical,
            'n_classes': n_classes,
            'd_model': d_model,
            'dropout': 0.2,
            'modality_dropout': 0.1
        }
    else:
        with open(config_path) as f:
            config = json.load(f)

    # Create model
    encoder = MMLSTMDAE(
        d_continuous=config['d_continuous'],
        d_categorical=config['d_categorical'],
        d_model=config.get('d_model', 256),
        dropout=config.get('dropout', 0.2),
        modality_dropout=config.get('modality_dropout', 0.1)
    )

    classifier = DirectClassifier(
        d_input=config.get('d_model', 256),
        n_classes=config['n_classes']
    )

    # Load checkpoint
    checkpoint_path = model_dir / 'best_model_test.pt'
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / 'best_model_val.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    encoder.to(device)
    classifier.to(device)
    encoder.eval()
    classifier.eval()

    return encoder, classifier


def evaluate_model(encoder, classifier, dataloader, device):
    """Evaluate model and return predictions and metrics."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for continuous, categorical, labels in dataloader:
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            # Forward pass
            encoding = encoder(continuous, categorical)
            logits = classifier(encoding)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    accuracy = np.mean(all_preds == all_labels)

    # Per-class accuracy
    per_class = {}
    for i, class_name in enumerate(OPERATION_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(all_preds[mask] == all_labels[mask])
            per_class[class_name] = float(class_acc)

    return {
        'accuracy': float(accuracy),
        'per_class': per_class,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test MM-LSTM-DAE on temporally-shuffled data"
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("MM-LSTM-DAE TEMPORAL SHUFFLE TEST")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    X_continuous, X_categorical, y = load_npz_data(args.data_dir / 'test_sequences.npz')
    print(f"  Continuous: {X_continuous.shape}")
    print(f"  Categorical: {X_categorical.shape}")
    print(f"  Labels: {y.shape}")

    # Load trained model
    print(f"\nLoading trained model from: {args.model_dir}")
    encoder, classifier = load_model(args.model_dir, device)
    print("  ✓ Model loaded successfully")

    # Test 1: Original data (temporal order preserved)
    print("\n" + "="*80)
    print("TEST 1: ORIGINAL DATA (Temporal Order Preserved)")
    print("="*80)

    dataset_original = TensorDataset(
        torch.FloatTensor(X_continuous),
        torch.LongTensor(X_categorical),
        torch.LongTensor(y)
    )
    dataloader_original = DataLoader(
        dataset_original,
        batch_size=args.batch_size,
        shuffle=False
    )

    results_original = evaluate_model(encoder, classifier, dataloader_original, device)
    print(f"\nOverall Accuracy: {results_original['accuracy']*100:.2f}%")
    print("\nPer-class accuracy:")
    for op, acc in results_original['per_class'].items():
        print(f"  {op:<20} {acc*100:>6.1f}%")

    # Test 2: Shuffled data (temporal order destroyed)
    print("\n" + "="*80)
    print("TEST 2: SHUFFLED DATA (Temporal Order Destroyed)")
    print("="*80)
    print("\nShuffling temporal dimension...")

    X_continuous_shuffled = shuffle_temporal_dimension(X_continuous, seed=args.seed)
    X_categorical_shuffled = shuffle_temporal_dimension(X_categorical, seed=args.seed)

    dataset_shuffled = TensorDataset(
        torch.FloatTensor(X_continuous_shuffled),
        torch.LongTensor(X_categorical_shuffled),
        torch.LongTensor(y)
    )
    dataloader_shuffled = DataLoader(
        dataset_shuffled,
        batch_size=args.batch_size,
        shuffle=False
    )

    results_shuffled = evaluate_model(encoder, classifier, dataloader_shuffled, device)
    print(f"\nOverall Accuracy: {results_shuffled['accuracy']*100:.2f}%")
    print("\nPer-class accuracy:")
    for op, acc in results_shuffled['per_class'].items():
        print(f"  {op:<20} {acc*100:>6.1f}%")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<25} {'Original':>15} {'Shuffled':>15} {'Drop':>15}")
    print("-"*80)
    print(f"{'Overall Accuracy':<25} {results_original['accuracy']*100:>14.2f}% "
          f"{results_shuffled['accuracy']*100:>14.2f}% "
          f"{(results_original['accuracy'] - results_shuffled['accuracy'])*100:>+14.2f}%")

    print("\nPer-class comparison:")
    print("-"*80)
    print(f"{'Operation':<25} {'Original':>15} {'Shuffled':>15} {'Drop':>15}")
    print("-"*80)

    for op in OPERATION_NAMES:
        if op in results_original['per_class'] and op in results_shuffled['per_class']:
            orig = results_original['per_class'][op] * 100
            shuf = results_shuffled['per_class'][op] * 100
            drop = orig - shuf
            print(f"{op:<25} {orig:>14.1f}% {shuf:>14.1f}% {drop:>+14.1f}%")

    # Save results
    comparison = {
        'original': {
            'accuracy': results_original['accuracy'],
            'per_class': results_original['per_class']
        },
        'shuffled': {
            'accuracy': results_shuffled['accuracy'],
            'per_class': results_shuffled['per_class']
        },
        'drop_percent': float((results_original['accuracy'] - results_shuffled['accuracy']) * 100)
    }

    output_path = args.output_dir / 'mmlstm_shuffled_results.json'
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
