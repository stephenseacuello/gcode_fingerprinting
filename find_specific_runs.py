#!/usr/bin/env python3
"""Find specific runs by ID."""

import wandb

def search_run(entity, run_id):
    """Search for a run across projects."""
    api = wandb.Api(timeout=30)

    projects = ["gcode-fingerprinting", "gcode-fingerprinting-2"]

    for project in projects:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            return (project, run)
        except:
            continue

    return None, None

def get_run_details(run):
    """Extract run details."""
    if not run:
        return None

    return {
        'id': run.id,
        'name': run.name,
        'state': run.state,
        'val_param_type_acc': run.summary.get('val/param_type_acc', 0),
        'val_command_acc': run.summary.get('val/command_acc', 0),
        'test_param_type_acc': run.summary.get('test/param_type_acc', 0),
        'test_command_acc': run.summary.get('test/command_acc', 0),
        'val_loss': run.summary.get('val/loss', 0),
        'test_loss': run.summary.get('test/loss', 0),
        'config': {k: v for k, v in run.config.items() if not k.startswith('_')},
        'epochs': run.summary.get('epoch', 0),
    }

def print_run(data, title):
    """Print run information."""
    if not data:
        print(f"\n{title}: NOT FOUND")
        return

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Run: {data['name']} ({data['id']})")
    print(f"State: {data['state']}")

    print(f"\n📊 METRICS:")
    print(f"  Val Param Type Acc:  {data['val_param_type_acc']:.4f}")
    print(f"  Test Param Type Acc: {data['test_param_type_acc']:.4f}")
    print(f"  Val Command Acc:     {data['val_command_acc']:.4f}")
    print(f"  Test Command Acc:    {data['test_command_acc']:.4f}")

    print(f"\n⚙️  CONFIG:")
    cfg = data['config']
    for key in ['hidden_dim', 'num_heads', 'num_layers', 'learning_rate',
                'batch_size', 'dropout', 'weight_decay']:
        if key in cfg:
            print(f"  {key:15s}: {cfg[key]}")

def main():
    entity = "seacuello-university-of-rhode-island"

    # Search for the three runs
    print("Searching for specific runs...")

    # Run from sweep 83bwwuca
    print("\n1. Getting best run from sweep 83bwwuca...")
    api = wandb.Api(timeout=30)
    try:
        sweep = api.sweep(f"{entity}/gcode-fingerprinting/83bwwuca")
        runs = [r for r in sweep.runs if r.state == 'finished']
        if runs:
            best_run = max(runs, key=lambda r: r.summary.get('val/param_type_acc', 0))
            run1_data = get_run_details(best_run)
            print(f"   Found: {best_run.name} ({best_run.id})")
        else:
            run1_data = None
            print("   No finished runs yet")
    except Exception as e:
        print(f"   Error: {e}")
        run1_data = None

    # Run chw5jqaj
    print("\n2. Searching for run chw5jqaj...")
    project2, run2 = search_run(entity, "chw5jqaj")
    if run2:
        print(f"   Found in project: {project2}")
        run2_data = get_run_details(run2)
    else:
        run2_data = None
        print("   Not found")

    # Run kae3w55d
    print("\n3. Searching for run kae3w55d...")
    project3, run3 = search_run(entity, "kae3w55d")
    if run3:
        print(f"   Found in project: {project3}")
        run3_data = get_run_details(run3)
    else:
        run3_data = None
        print("   Not found - please verify the run ID")

    # Print details
    print_run(run1_data, "EXPERIMENT 1: Best from Sweep 83bwwuca")
    print_run(run2_data, "EXPERIMENT 2: Run chw5jqaj")
    print_run(run3_data, "EXPERIMENT 3: Run kae3w55d")

    # Compare if we have data
    valid_runs = [(d, i) for i, d in enumerate([run1_data, run2_data, run3_data], 1) if d]

    if len(valid_runs) < 2:
        print("\n⚠️  Not enough valid runs to compare")
        return

    print(f"\n{'='*80}")
    print("COMPARISON & ANALYSIS")
    print(f"{'='*80}")

    print("\n📊 Performance Summary:")
    for data, idx in valid_runs:
        print(f"\nExperiment {idx}: {data['name']}")
        print(f"  Val Acc:  {data['val_param_type_acc']:.4f}")
        print(f"  Test Acc: {data['test_param_type_acc']:.4f}")
        if data['test_param_type_acc'] > 0:
            gap = data['val_param_type_acc'] - data['test_param_type_acc']
            print(f"  Gap:      {gap:.4f}")
        else:
            print(f"  Gap:      N/A (test not run)")

    # Check if any have test results
    has_test = any(d['test_param_type_acc'] > 0 for d, _ in valid_runs)

    if not has_test:
        print("\n⚠️  WARNING: No test results available yet!")
        print("   All experiments show test_acc = 0.0000")
        print("   This means:")
        print("   - Training is still in progress, OR")
        print("   - Test evaluation hasn't been run yet")
        print("\n   Based on VALIDATION results only:")

    # Sort by val accuracy if no test results
    metric = 'test_param_type_acc' if has_test else 'val_param_type_acc'
    valid_runs.sort(key=lambda x: x[0][metric], reverse=True)

    print(f"\n🏆 Ranking (by {'test' if has_test else 'validation'} accuracy):")
    for rank, (data, idx) in enumerate(valid_runs, 1):
        acc = data[metric]
        print(f"  {rank}. Experiment {idx}: {data['name']} - {acc:.4f}")

    # Best config
    best_data, best_idx = valid_runs[0]
    print(f"\n💡 BEST CONFIGURATION (Experiment {best_idx}):")
    cfg = best_data['config']
    print(f"   Hidden dim:   {cfg.get('hidden_dim', 'N/A')}")
    print(f"   Num layers:   {cfg.get('num_layers', 'N/A')}")
    print(f"   Num heads:    {cfg.get('num_heads', 'N/A')}")
    print(f"   Learning rate: {cfg.get('learning_rate', 'N/A')}")
    print(f"   Batch size:   {cfg.get('batch_size', 'N/A')}")
    print(f"   Dropout:      {cfg.get('dropout', 'N/A')}")
    print(f"   Weight decay: {cfg.get('weight_decay', 'N/A')}")

    print(f"\n📋 RECOMMENDATIONS:")
    if not has_test:
        print(f"   1. ⚠️  WAIT for test results before making final conclusions")
        print(f"   2. Monitor the sweep - more runs may still be in progress")
        print(f"   3. Once test results are available, check for overfitting")
    else:
        best_gap = best_data['val_param_type_acc'] - best_data['test_param_type_acc']
        if best_gap > 0.05:
            print(f"   1. ⚠️  Significant overfitting detected (gap: {best_gap:.4f})")
            print(f"   2. Consider increasing dropout or weight decay")
            print(f"   3. Try data augmentation")
        else:
            print(f"   1. ✅ Good generalization (gap: {best_gap:.4f})")
            print(f"   2. Use this configuration as baseline")

    print(f"   4. Continue hyperparameter sweep to find better configurations")
    print(f"   5. Analyze error patterns in predictions")

if __name__ == "__main__":
    main()
