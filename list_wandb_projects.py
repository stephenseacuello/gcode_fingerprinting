#!/usr/bin/env python3
"""List all W&B projects and recent runs to help locate experiments."""

import wandb

def main():
    api = wandb.Api()
    entity = "seacuello-university-of-rhode-island"

    print("Fetching your W&B projects...\n")

    # Try to list projects
    try:
        # Get runs from different project names
        project_names = [
            "gcode-fingerprinting",
            "gcode-fingerprinting-2",
            "g-code-fingerprinting",
            "gcode_fingerprinting",
        ]

        for project in project_names:
            print(f"\n{'='*80}")
            print(f"PROJECT: {project}")
            print(f"{'='*80}")

            try:
                runs = api.runs(f"{entity}/{project}", per_page=10)
                run_list = list(runs)

                if not run_list:
                    print("  No runs found")
                    continue

                print(f"  Found {len(run_list)} recent runs:\n")

                for run in run_list[:10]:
                    val_acc = run.summary.get('val/param_type_acc', 0)
                    test_acc = run.summary.get('test/param_type_acc', 0)
                    print(f"  • {run.id} ({run.name})")
                    print(f"    State: {run.state}")
                    print(f"    Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

                    # Check if part of a sweep
                    if run.sweep:
                        print(f"    Sweep: {run.sweep.id}")

                # List sweeps
                print(f"\n  Sweeps in {project}:")
                try:
                    sweeps = api.sweeps(f"{entity}/{project}")
                    sweep_list = list(sweeps)
                    if sweep_list:
                        for sweep in sweep_list[:5]:
                            print(f"    • {sweep.id} - {sweep.name} ({sweep.state})")
                    else:
                        print("    No sweeps found")
                except Exception as e:
                    print(f"    Error fetching sweeps: {e}")

            except Exception as e:
                print(f"  Error accessing project: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
