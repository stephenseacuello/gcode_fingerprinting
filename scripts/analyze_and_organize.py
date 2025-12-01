#!/usr/bin/env python3
"""
Analyze training results and organize model outputs.

This script:
1. Scans all model checkpoints in outputs/
2. Ranks them by validation accuracy
3. Analyzes data for class imbalance
4. Suggests improvements
"""

import json
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


def analyze_checkpoints():
    """Find and rank all model checkpoints."""
    print("\n" + "="*80)
    print("🔍 ANALYZING MODEL CHECKPOINTS")
    print("="*80)

    checkpoints = []

    # Scan outputs directory
    output_dirs = [
        Path('outputs/wandb_sweeps'),
        Path('outputs/training'),
        Path('outputs/final_model'),
    ]

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        for checkpoint_path in output_dir.rglob('checkpoint_*.pt'):
            try:
                # Load checkpoint metadata
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                metrics = {
                    'path': str(checkpoint_path),
                    'name': checkpoint_path.parent.name,
                    'epoch': ckpt.get('epoch', 0),
                    'val_loss': ckpt.get('val_loss', float('inf')),
                    'val_acc': ckpt.get('val_acc', 0.0),
                    'train_loss': ckpt.get('train_loss', float('inf')),
                    'train_acc': ckpt.get('train_acc', 0.0),
                    'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime),
                }

                # Check if it's a best checkpoint
                if 'best' in checkpoint_path.name:
                    metrics['is_best'] = True
                else:
                    metrics['is_best'] = False

                checkpoints.append(metrics)

            except Exception as e:
                print(f"⚠️  Could not load {checkpoint_path.name}: {e}")

    if not checkpoints:
        print("❌ No checkpoints found!")
        return None

    # Create DataFrame and sort by validation accuracy
    df = pd.DataFrame(checkpoints)
    df = df.sort_values('val_acc', ascending=False)

    print(f"\n✅ Found {len(df)} checkpoints\n")

    # Display top 10
    print("🏆 TOP 10 MODELS (by validation accuracy):")
    print("-" * 80)
    for idx, row in df.head(10).iterrows():
        best_marker = "⭐" if row['is_best'] else "  "
        print(f"{best_marker} {row['name']:<40} Val Acc: {row['val_acc']*100:5.2f}%  Val Loss: {row['val_loss']:.4f}")

    # Save full ranking
    ranking_path = Path('outputs/model_ranking.csv')
    df.to_csv(ranking_path, index=False)
    print(f"\n💾 Full ranking saved to: {ranking_path}")

    return df


def analyze_vocabulary():
    """Analyze vocabulary for class imbalance."""
    print("\n" + "="*80)
    print("📊 ANALYZING VOCABULARY & CLASS DISTRIBUTION")
    print("="*80)

    vocab_path = Path('data/vocabulary.json')
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found at {vocab_path}")
        return None

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    # Try both possible vocabulary structures
    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))

    if not vocab:
        print("❌ Vocabulary is empty!")
        return None

    # Categorize tokens
    g_commands = []
    m_commands = []
    numeric_params = []
    special_tokens = []
    axis_params = []

    for token in vocab.keys():
        if isinstance(token, int):  # Skip if numeric ID
            continue
        if token.startswith('G'):
            g_commands.append(token)
        elif token.startswith('M'):
            m_commands.append(token)
        elif token.startswith('NUM_'):
            numeric_params.append(token)
        elif token in ['PAD', 'BOS', 'EOS', 'UNK', 'MASK']:
            special_tokens.append(token)
        elif token in ['X', 'Y', 'Z', 'A', 'B', 'C', 'I', 'J', 'K', 'F', 'S', 'R', 'P', 'Q', 'E']:
            axis_params.append(token)

    total_tokens = len(g_commands) + len(m_commands) + len(numeric_params) + len(special_tokens) + len(axis_params)

    print(f"\n📈 Token Distribution:")
    print(f"  G-commands:      {len(g_commands):4d} tokens  ({len(g_commands)/max(total_tokens,1)*100:.1f}%)")
    print(f"  M-commands:      {len(m_commands):4d} tokens  ({len(m_commands)/max(total_tokens,1)*100:.1f}%)")
    print(f"  Numeric params:  {len(numeric_params):4d} tokens  ({len(numeric_params)/max(total_tokens,1)*100:.1f}%)")
    print(f"  Axis params:     {len(axis_params):4d} tokens  ({len(axis_params)/max(total_tokens,1)*100:.1f}%)")
    print(f"  Special tokens:  {len(special_tokens):4d} tokens  ({len(special_tokens)/max(total_tokens,1)*100:.1f}%)")
    print(f"  TOTAL:           {len(vocab):4d} entries ({total_tokens} categorized tokens)")

    print(f"\n🎯 G-commands found: {', '.join(sorted(g_commands)[:20])}")
    if len(g_commands) > 20:
        print(f"     ... and {len(g_commands)-20} more")

    # Check for severe imbalance
    if len(numeric_params) > len(g_commands) * 10:
        print(f"\n⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print(f"   Numeric tokens outnumber G-commands by {len(numeric_params)/max(len(g_commands),1):.1f}x")
        print(f"   This explains why model predicts numbers instead of commands!")

    return {
        'g_commands': g_commands,
        'm_commands': m_commands,
        'numeric_params': numeric_params,
        'vocab_size': len(vocab)
    }


def analyze_training_data():
    """Analyze training data distribution."""
    print("\n" + "="*80)
    print("📂 ANALYZING TRAINING DATA")
    print("="*80)

    # Find aligned CSV files
    data_dir = Path('data')
    csv_files = list(data_dir.glob('*_aligned.csv'))

    if not csv_files:
        print("❌ No aligned CSV files found")
        return None

    print(f"\n✅ Found {len(csv_files)} aligned CSV files")

    all_gcodes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'gcode_string' in df.columns:
                gcodes = df['gcode_string'].dropna().tolist()
                all_gcodes.extend(gcodes)
                print(f"  {csv_file.name:<40} {len(df):6d} rows, {len(gcodes):6d} G-codes")
        except Exception as e:
            print(f"⚠️  Could not read {csv_file.name}: {e}")

    if not all_gcodes:
        print("\n❌ No G-code data found in CSVs")
        return None

    # Analyze distribution
    gcode_counts = Counter(all_gcodes)

    print(f"\n📊 G-code Distribution:")
    print(f"  Total samples:     {len(all_gcodes):,}")
    print(f"  Unique G-codes:    {len(gcode_counts):,}")
    print(f"  Most common (top 20):")

    for gcode, count in gcode_counts.most_common(20):
        percentage = count / len(all_gcodes) * 100
        print(f"    {gcode:<30} {count:6d} samples ({percentage:5.2f}%)")

    # Check for extreme imbalance
    top_10_pct = sum(count for _, count in gcode_counts.most_common(10)) / len(all_gcodes) * 100

    if top_10_pct > 80:
        print(f"\n⚠️  EXTREME DATA IMBALANCE!")
        print(f"   Top 10 G-codes account for {top_10_pct:.1f}% of all samples")
        print(f"   Model will heavily bias toward these commands!")

    return gcode_counts


def suggest_improvements(vocab_info, data_stats, model_df):
    """Suggest specific improvements based on analysis."""
    print("\n" + "="*80)
    print("💡 RECOMMENDED IMPROVEMENTS")
    print("="*80)

    suggestions = []

    # 1. Check class imbalance
    if vocab_info and len(vocab_info['numeric_params']) > len(vocab_info['g_commands']) * 5:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Severe vocabulary imbalance (too many numeric tokens)',
            'solution': [
                '1. Use class weights in loss function',
                '2. Oversample rare G-commands during training',
                '3. Add focal loss to focus on hard examples',
                '4. Use label smoothing to prevent overconfidence on numbers'
            ]
        })

    # 2. Check model confidence
    if model_df is not None and model_df['val_acc'].max() < 0.3:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Very low validation accuracy (<30%)',
            'solution': [
                '1. Increase model capacity (more layers, hidden dimensions)',
                '2. Train for more epochs (current models may be undertrained)',
                '3. Lower learning rate for better convergence',
                '4. Add more training data (data augmentation)'
            ]
        })

    # 3. Repetitive generation
    suggestions.append({
        'priority': 'HIGH',
        'issue': 'Autoregressive generation stuck in loops',
        'solution': [
            '1. Increase temperature (currently 0.1 → try 0.5-1.0)',
            '2. Use nucleus sampling (top-p) with p=0.9',
            '3. Add repetition penalty',
            '4. Implement proper EOS token handling',
            '5. Use diverse beam search instead of regular beam search'
        ]
    })

    # 4. Training optimization
    suggestions.append({
        'priority': 'MEDIUM',
        'issue': 'Hyperparameter tuning needed',
        'solution': [
            '1. Try larger batch sizes (16-32 instead of 8)',
            '2. Use cosine annealing learning rate schedule',
            '3. Add gradient clipping (max_norm=1.0)',
            '4. Increase dropout to 0.2-0.3 for regularization'
        ]
    })

    # Print suggestions
    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. [{sugg['priority']}] {sugg['issue']}")
        print("   Solutions:")
        for sol in sugg['solution']:
            print(f"     {sol}")

    return suggestions


def create_improvement_sweep():
    """Create a new sweep configuration with suggested improvements."""
    sweep_config = {
        'name': 'gcode-improved-sweep',
        'method': 'bayes',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            # Architecture
            'hidden_dim': {'values': [256, 384, 512]},
            'num_layers': {'values': [3, 4, 5]},
            'num_heads': {'values': [4, 6, 8]},

            # Training
            'batch_size': {'values': [16, 24, 32]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
            'weight_decay': {'values': [0.0, 0.01, 0.05]},

            # Regularization
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'label_smoothing': {'values': [0.0, 0.05, 0.1]},

            # Class imbalance handling
            'use_class_weights': {'values': [True]},
            'focal_loss_gamma': {'values': [0.0, 1.0, 2.0]},

            # Generation
            'temperature': {'values': [0.5, 0.7, 1.0]},
            'top_p': {'values': [0.9, 0.95]},
        }
    }

    sweep_path = Path('sweeps/improved_sweep.yaml')
    sweep_path.parent.mkdir(exist_ok=True)

    import yaml
    with open(sweep_path, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)

    print(f"\n💾 Improved sweep config saved to: {sweep_path}")
    print(f"   Run with: wandb sweep {sweep_path}")

    return sweep_path


def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("🔬 G-CODE FINGERPRINTING - MODEL ANALYSIS & ORGANIZATION")
    print("="*80)

    # 1. Analyze checkpoints
    model_df = analyze_checkpoints()

    # 2. Analyze vocabulary
    vocab_info = analyze_vocabulary()

    # 3. Analyze training data
    data_stats = analyze_training_data()

    # 4. Generate suggestions
    suggestions = suggest_improvements(vocab_info, data_stats, model_df)

    # 5. Create improved sweep config
    create_improvement_sweep()

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review outputs/model_ranking.csv to find your best model")
    print("  2. Try the suggested improvements above")
    print("  3. Run the improved sweep: wandb sweep sweeps/improved_sweep.yaml")
    print("  4. Monitor results on wandb dashboard")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
