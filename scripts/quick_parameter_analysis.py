#!/usr/bin/env python3
"""
Quick Parameter Analysis from W&B Metrics

Analyzes parameter prediction performance from the best sweep run metrics
without needing to load checkpoints.
"""

import wandb
import json
from pathlib import Path

def main():
    # Initialize W&B API
    api = wandb.Api()

    # Best run from sweep analysis
    project = "gcode-fingerprinting"
    entity = "seacuello-university-of-rhode-island"
    sweep_id = "wj8tc5br"

    print("="*80)
    print("PARAMETER PREDICTION ANALYSIS FROM BEST SWEEP RUN")
    print("="*80)

    # Get sweep and find best run
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs

    # Find run with best overall accuracy
    best_run = None
    best_acc = 0

    for run in runs:
        if run.state == "finished":
            summary = run.summary
            overall_acc = summary.get('val/overall_acc', 0)
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_run = run

    if best_run is None:
        print("No completed runs found!")
        return

    print(f"\nBest Run: {best_run.name}")
    print(f"Overall Accuracy: {best_acc*100:.2f}%\n")

    # Extract key metrics
    summary = best_run.summary
    config = best_run.config

    # Current Performance
    print("📊 CURRENT PERFORMANCE:")
    print(f"  Type Accuracy:        {summary.get('val/type_acc', 0)*100:.2f}%")
    print(f"  Command Accuracy:     {summary.get('val/command_acc', 0)*100:.2f}%")
    print(f"  Param Type Accuracy:  {summary.get('val/param_type_acc', 0)*100:.2f}%  ⬅ TARGET: 95%+")
    print(f"  Param Value Accuracy: {summary.get('val/param_value_acc', 0)*100:.2f}%  ⬅ TARGET: 98%+")
    print(f"  Operation Accuracy:   {summary.get('val/operation_acc', 0)*100:.2f}%")

    # Gap Analysis
    param_type_acc = summary.get('val/param_type_acc', 0) * 100
    param_value_acc = summary.get('val/param_value_acc', 0) * 100

    param_type_gap = 95.0 - param_type_acc
    param_value_gap = 98.0 - param_value_acc

    print(f"\n📏 GAP TO TARGET:")
    print(f"  Param Type Gap:  {param_type_gap:.2f}pp (percentage points)")
    print(f"  Param Value Gap: {param_value_gap:.2f}pp (percentage points)")

    # Current Configuration
    print(f"\n⚙️  CURRENT CONFIGURATION:")
    print(f"  command_weight:     {config.get('command_weight', 'N/A')}")
    print(f"  label_smoothing:    {config.get('label_smoothing', 'N/A')}")
    print(f"  hidden_dim:         {config.get('hidden_dim', 'N/A')}")
    print(f"  learning_rate:      {config.get('learning_rate', 'N/A')}")
    print(f"  param_type_weight:  {config.get('param_type_weight', 'NOT SET - using default 1.0')}")
    print(f"  param_value_weight: {config.get('param_value_weight', 'NOT SET - using default 1.0')}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print()

    if param_type_gap > 0:
        print(f"1. PARAM TYPE OPTIMIZATION (Gap: {param_type_gap:.2f}pp)")
        print(f"   Current: {param_type_acc:.2f}% → Target: 95.0%")
        print(f"   Recommendation: Add param_type_weight parameter")
        print(f"   Suggested sweep: param_type_weight in [5.0, 10.0, 15.0, 20.0]")
        print(f"   Expected impact: +2-5pp improvement")
        print()

    if param_value_gap > 0:
        print(f"2. PARAM VALUE OPTIMIZATION (Gap: {param_value_gap:.2f}pp)")
        print(f"   Current: {param_value_acc:.2f}% → Target: 98.0%")
        print(f"   Recommendation: Add param_value_weight parameter")
        print(f"   Suggested sweep: param_value_weight in [5.0, 10.0, 15.0, 20.0]")
        print(f"   Expected impact: +1-2pp improvement")
        print()

    if param_type_gap <= 0 and param_value_gap <= 0:
        print(f"✅ Both targets achieved! Ready for production training.")
        print()
    else:
        print(f"3. COMBINED PARAMETER SWEEP")
        print(f"   Create targeted sweep exploring:")
        print(f"   - param_type_weight: [5.0, 10.0, 15.0]")
        print(f"   - param_value_weight: [5.0, 10.0, 15.0]")
        print(f"   - Keep best base config: command_weight={config.get('command_weight')}, hidden_dim={config.get('hidden_dim')}")
        print(f"   - Total runs: 3 × 3 = 9 configurations")
        print(f"   - Estimated time: ~3 hours (9 runs × 20min each)")
        print()

    # Save analysis
    output_dir = Path("outputs/parameter_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        "best_run": best_run.name,
        "current_performance": {
            "overall_acc": float(best_acc),
            "type_acc": float(summary.get('val/type_acc', 0)),
            "command_acc": float(summary.get('val/command_acc', 0)),
            "param_type_acc": float(summary.get('val/param_type_acc', 0)),
            "param_value_acc": float(summary.get('val/param_value_acc', 0)),
            "operation_acc": float(summary.get('val/operation_acc', 0))
        },
        "gaps": {
            "param_type_gap_pp": float(param_type_gap),
            "param_value_gap_pp": float(param_value_gap)
        },
        "current_config": dict(config),
        "recommendations": {
            "param_type_weight": [5.0, 10.0, 15.0, 20.0] if param_type_gap > 0 else None,
            "param_value_weight": [5.0, 10.0, 15.0, 20.0] if param_value_gap > 0 else None
        }
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"✅ Analysis saved to {output_dir}/analysis_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
