#!/usr/bin/env python3
"""Compare three W&B experiments and provide analysis."""

import wandb
import sys

def fetch_sweep_best(entity: str, project: str, sweep_id: str):
    """Fetch best run from a sweep."""
    api = wandb.Api()
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs

        # Filter finished runs
        finished_runs = [r for r in runs if r.state == 'finished']

        if not finished_runs:
            return None

        # Sort by val/param_type_acc
        best_run = max(finished_runs,
                      key=lambda r: r.summary.get('val/param_type_acc', 0))

        return {
            'type': 'sweep',
            'project': project,
            'sweep_id': sweep_id,
            'num_runs': len(finished_runs),
            'best_run_id': best_run.id,
            'best_run_name': best_run.name,
            'val_param_type_acc': best_run.summary.get('val/param_type_acc', 0),
            'val_command_acc': best_run.summary.get('val/command_acc', 0),
            'test_param_type_acc': best_run.summary.get('test/param_type_acc', 0),
            'test_command_acc': best_run.summary.get('test/command_acc', 0),
            'val_loss': best_run.summary.get('val/loss', 0),
            'config': {k: v for k, v in best_run.config.items() if not k.startswith('_')},
            'epochs': best_run.summary.get('epoch', 0),
            'best_epoch': best_run.summary.get('best_epoch', 0),
        }
    except Exception as e:
        print(f"Error fetching sweep {sweep_id}: {e}")
        return None

def fetch_run(entity: str, project: str, run_id: str):
    """Fetch single run data."""
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")

        return {
            'type': 'run',
            'project': project,
            'run_id': run_id,
            'run_name': run.name,
            'state': run.state,
            'val_param_type_acc': run.summary.get('val/param_type_acc', 0),
            'val_command_acc': run.summary.get('val/command_acc', 0),
            'test_param_type_acc': run.summary.get('test/param_type_acc', 0),
            'test_command_acc': run.summary.get('test/command_acc', 0),
            'val_loss': run.summary.get('val/loss', 0),
            'config': {k: v for k, v in run.config.items() if not k.startswith('_')},
            'epochs': run.summary.get('epoch', 0),
            'best_epoch': run.summary.get('best_epoch', 0),
        }
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None

def print_experiment(data, name):
    """Print experiment details."""
    if not data:
        print(f"\n{name}: NO DATA AVAILABLE")
        return

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Project: {data['project']}")

    if data['type'] == 'sweep':
        print(f"Sweep ID: {data['sweep_id']}")
        print(f"Completed runs: {data['num_runs']}")
        print(f"Best run: {data['best_run_name']} ({data['best_run_id']})")
    else:
        print(f"Run ID: {data['run_id']}")
        print(f"Status: {data['state']}")

    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Validation Param Type Acc: {data['val_param_type_acc']:.4f}")
    print(f"  Test Param Type Acc:       {data['test_param_type_acc']:.4f}")
    print(f"  Validation Command Acc:    {data['val_command_acc']:.4f}")
    print(f"  Test Command Acc:          {data['test_command_acc']:.4f}")
    print(f"  Validation Loss:           {data['val_loss']:.4f}")
    print(f"  Generalization Gap:        {data['val_param_type_acc'] - data['test_param_type_acc']:.4f}")

    print(f"\n⚙️  HYPERPARAMETERS:")
    config = data['config']
    important_params = ['hidden_dim', 'num_heads', 'num_layers', 'learning_rate',
                       'batch_size', 'dropout', 'weight_decay', 'lr_scheduler',
                       'warmup_epochs', 'num_epochs']
    for key in important_params:
        if key in config:
            print(f"  {key:20s}: {config[key]}")

def compare_and_advise(exp1, exp2, exp3):
    """Compare experiments and provide recommendations."""
    experiments = [
        ("Sweep 83bwwuca (gcode-fingerprinting)", exp1),
        ("Sweep chw5jqaj (gcode-fingerprinting-2)", exp2),
        ("Run kae3w55d (g-code-fingerprinting)", exp3),
    ]

    # Filter out None
    experiments = [(name, data) for name, data in experiments if data is not None]

    if not experiments:
        print("\n❌ No valid experiments to compare")
        return

    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*80}")

    # Sort by test accuracy
    experiments.sort(key=lambda x: x[1]['test_param_type_acc'], reverse=True)

    print("\n🏆 PERFORMANCE RANKING (by Test Param Type Accuracy):")
    for i, (name, data) in enumerate(experiments, 1):
        gap = data['val_param_type_acc'] - data['test_param_type_acc']
        print(f"\n{i}. {name}")
        print(f"   Test Accuracy:  {data['test_param_type_acc']:.4f}")
        print(f"   Val Accuracy:   {data['val_param_type_acc']:.4f}")
        print(f"   Gen Gap:        {gap:.4f} {'⚠️  overfitting' if gap > 0.03 else '✅ good'}")

    best_name, best_data = experiments[0]

    print(f"\n💡 KEY INSIGHTS:")
    print(f"\n1. BEST PERFORMER: {best_name}")
    print(f"   - Achieved {best_data['test_param_type_acc']:.4f} test accuracy")

    # Analyze generalization
    print(f"\n2. GENERALIZATION ANALYSIS:")
    for name, data in experiments:
        gap = data['val_param_type_acc'] - data['test_param_type_acc']
        if gap > 0.05:
            print(f"   ⚠️  {name}: Significant overfitting (gap: {gap:.4f})")
        elif gap > 0.03:
            print(f"   ⚡ {name}: Mild overfitting (gap: {gap:.4f})")
        else:
            print(f"   ✅ {name}: Good generalization (gap: {gap:.4f})")

    # Compare configurations
    print(f"\n3. CONFIGURATION COMPARISON:")

    # Architecture
    print(f"\n   Model Architecture:")
    for name, data in experiments:
        cfg = data['config']
        print(f"   {name}:")
        print(f"      Hidden: {cfg.get('hidden_dim', 'N/A')}, "
              f"Layers: {cfg.get('num_layers', 'N/A')}, "
              f"Heads: {cfg.get('num_heads', 'N/A')}")

    # Training config
    print(f"\n   Training Configuration:")
    for name, data in experiments:
        cfg = data['config']
        print(f"   {name}:")
        print(f"      LR: {cfg.get('learning_rate', 'N/A')}, "
              f"Batch: {cfg.get('batch_size', 'N/A')}, "
              f"Dropout: {cfg.get('dropout', 'N/A')}")
        print(f"      Scheduler: {cfg.get('lr_scheduler', 'N/A')}, "
              f"Warmup: {cfg.get('warmup_epochs', 'N/A')}, "
              f"WD: {cfg.get('weight_decay', 'N/A')}")

    print(f"\n4. RECOMMENDATIONS:")

    # Best config summary
    best_cfg = best_data['config']
    print(f"\n   ✅ Use configuration from {best_name}:")
    print(f"      - Hidden dim: {best_cfg.get('hidden_dim')}")
    print(f"      - Num layers: {best_cfg.get('num_layers')}")
    print(f"      - Num heads: {best_cfg.get('num_heads')}")
    print(f"      - Learning rate: {best_cfg.get('learning_rate')}")
    print(f"      - Batch size: {best_cfg.get('batch_size')}")
    print(f"      - Dropout: {best_cfg.get('dropout')}")
    print(f"      - Weight decay: {best_cfg.get('weight_decay')}")

    # Specific advice based on patterns
    print(f"\n   📋 Action Items:")

    # Check for overfitting
    best_gap = best_data['val_param_type_acc'] - best_data['test_param_type_acc']
    if best_gap > 0.03:
        print(f"      - Consider increasing dropout or weight decay to reduce overfitting")
        print(f"      - Try data augmentation techniques")

    # Compare learning rates
    lrs = [data['config'].get('learning_rate') for _, data in experiments]
    if len(set(lrs)) > 1:
        print(f"      - Learning rate appears important: best is {best_cfg.get('learning_rate')}")

    # Model size
    if data['type'] == 'sweep':
        print(f"      - Continue sweep to explore more hyperparameter combinations")

    print(f"\n   🎯 Next Steps:")
    print(f"      1. Use the best configuration as your baseline")
    print(f"      2. Fine-tune around the best hyperparameters")
    print(f"      3. Consider ensemble methods if multiple configs perform well")
    print(f"      4. Analyze error patterns to understand failure modes")

def main():
    entity = "seacuello-university-of-rhode-island"

    print("Fetching W&B experiment data...")
    print("This may take a moment...\n")

    # Fetch all three experiments
    exp1 = fetch_sweep_best(entity, "gcode-fingerprinting", "83bwwuca")
    exp2 = fetch_sweep_best(entity, "gcode-fingerprinting-2", "chw5jqaj")
    exp3 = fetch_run(entity, "g-code-fingerprinting", "kae3w55d")

    # Print individual results
    print_experiment(exp1, "EXPERIMENT 1: Sweep 83bwwuca (Current)")
    print_experiment(exp2, "EXPERIMENT 2: Sweep chw5jqaj")
    print_experiment(exp3, "EXPERIMENT 3: Run kae3w55d")

    # Compare and provide advice
    compare_and_advise(exp1, exp2, exp3)

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
