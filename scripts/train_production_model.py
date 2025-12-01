#!/usr/bin/env python3
"""
End-to-End Production Model Training Script

This script performs the complete workflow:
1. Extract and prepare vocabulary (668 tokens)
2. Preprocess raw data
3. Train model with best hyperparameters from sweep
4. Evaluate and save final model

Best Hyperparameters (from sweep ab0ypky2):
- command_weight: 15.0
- hidden_dim: 128
- num_layers: 2
- num_heads: 4
- learning_rate: 0.001
- batch_size: 16
- label_smoothing: 0.0
- max_epochs: 15
"""

import json
import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def step1_extract_vocabulary():
    """Extract 668-token vocabulary from nested structure."""
    print("="*80)
    print("STEP 1: Extracting Vocabulary (668 tokens)")
    print("="*80)

    vocab_file = project_root / "data" / "vocabulary.json"
    output_file = project_root / "data" / "vocab_production.json"

    # Load nested vocabulary
    with open(vocab_file) as f:
        data = json.load(f)

    # Extract the actual vocab
    if 'vocab' in data:
        vocab = data['vocab']
    else:
        vocab = data

    print(f"  Loaded vocabulary: {len(vocab)} tokens")
    print(f"  Sample tokens: {list(vocab.keys())[:10]}")

    # Save flat vocabulary
    with open(output_file, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"  ✓ Saved to: {output_file}")
    print(f"  Vocabulary size: {len(vocab)} tokens\n")

    return output_file

def step2_preprocess_data(vocab_path):
    """Run data preprocessing with full vocabulary."""
    print("="*80)
    print("STEP 2: Preprocessing Data")
    print("="*80)

    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "processed_production"

    # Use original nested vocabulary file for preprocessing
    # (tokenizer expects nested structure with 'config' and 'vocab' keys)
    original_vocab = project_root / "data" / "vocabulary.json"

    # Use the preprocessing module
    preprocess_script = project_root / "src" / "miracle" / "dataset" / "preprocessing.py"

    cmd = [
        str(project_root / ".venv" / "bin" / "python"),
        str(preprocess_script),
        "--data-dir", str(data_dir),
        "--vocab-path", str(original_vocab),
        "--output-dir", str(output_dir),
        "--window-size", "100",
        "--stride", "50"
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={
        **subprocess.os.environ,
        "PYTHONPATH": str(project_root / "src")
    })

    if result.returncode != 0:
        print(f"\n  ⚠️  Preprocessing may have encountered issues")
    else:
        print(f"  ✓ Preprocessing complete")

    print(f"  Output directory: {output_dir}\n")

    return output_dir

def step3_train_model(data_dir, vocab_path):
    """Train model with best hyperparameters from sweep."""
    print("="*80)
    print("STEP 3: Training Production Model")
    print("="*80)
    print("\nBest Hyperparameters (from sweep ab0ypky2):")
    print("  - command_weight: 15.0")
    print("  - hidden_dim: 128")
    print("  - num_layers: 2")
    print("  - num_heads: 4")
    print("  - learning_rate: 0.001")
    print("  - batch_size: 16")
    print("  - label_smoothing: 0.0")
    print("  - max_epochs: 15\n")

    output_dir = project_root / "outputs" / "production_model"
    train_script = project_root / "scripts" / "train_multihead.py"

    # Use original vocabulary file for training
    original_vocab = project_root / "data" / "vocabulary.json"

    cmd = [
        str(project_root / ".venv" / "bin" / "python"),
        str(train_script),
        "--data-dir", str(data_dir),
        "--vocab-path", str(original_vocab),
        "--output-dir", str(output_dir),
        # Best hyperparameters
        "--command_weight", "15.0",
        "--hidden_dim", "128",
        "--num_layers", "2",
        "--num_heads", "4",
        "--learning_rate", "0.001",
        "--batch_size", "16",
        "--label_smoothing", "0.0",
        "--max-epochs", "15",
        # Additional settings
        "--use-wandb",
        "--wandb-project", "gcode-fingerprinting"
    ]

    print(f"  Running training...")
    print(f"  Command: {' '.join(cmd[:4])} [+ hyperparameters]\n")

    result = subprocess.run(cmd, env={
        **subprocess.os.environ,
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTHONPATH": str(project_root / "src")
    })

    if result.returncode == 0:
        print(f"\n  ✓ Training complete!")
        print(f"  Model saved to: {output_dir}")
    else:
        print(f"\n  ⚠️  Training encountered issues (exit code: {result.returncode})")

    return output_dir

def main():
    print("\n" + "="*80)
    print("PRODUCTION MODEL TRAINING PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("  1. Extract 668-token vocabulary")
    print("  2. Preprocess all data")
    print("  3. Train model with best hyperparameters")
    print("  4. Save production-ready checkpoint\n")

    input("Press Enter to continue or Ctrl+C to cancel...")
    print()

    # Step 1: Vocabulary
    vocab_path = step1_extract_vocabulary()

    # Step 2: Preprocessing
    data_dir = step2_preprocess_data(vocab_path)

    # Step 3: Training
    model_dir = step3_train_model(data_dir, vocab_path)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nVocabulary: {vocab_path}")
    print(f"Processed Data: {data_dir}")
    print(f"Trained Model: {model_dir}")
    print(f"\nBest checkpoint: {model_dir}/checkpoint_best.pt")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
