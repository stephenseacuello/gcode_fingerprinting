"""Master pipeline orchestrator for the toolpath signature study."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_phase(phase_num):
    """Run a specific phase by number."""
    start = time.time()
    print(f"\n{'#' * 70}")
    print(f"# Starting Phase {phase_num}")
    print(f"{'#' * 70}\n")

    if phase_num == 1:
        from toolpath_signature.phase1_preprocess import run_phase1
        run_phase1()
    elif phase_num == 2:
        from toolpath_signature.phase2_feature_isolation import run_phase2
        run_phase2()
    elif phase_num == 3:
        from toolpath_signature.phase3_dim_reduction import run_phase3
        run_phase3()
    elif phase_num == 4:
        from toolpath_signature.phase4_classification import run_phase4
        run_phase4()
    elif phase_num == 5:
        from toolpath_signature.phase5_ablation import run_phase5
        run_phase5()
    elif phase_num == 6:
        from toolpath_signature.phase6_generalization import run_phase6
        run_phase6()
    elif phase_num == 7:
        from toolpath_signature.phase7_figures import run_phase7
        run_phase7()
    else:
        print(f"Unknown phase: {phase_num}")
        return

    elapsed = time.time() - start
    print(f"\nPhase {phase_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Toolpath Signature Pipeline")
    parser.add_argument("--phase", nargs="+", type=int, default=list(range(1, 8)),
                       help="Phase numbers to run (1-7). Default: all.")
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 70)
    print("TOOLPATH SIGNATURE PIPELINE")
    print(f"Phases to run: {args.phase}")
    print("=" * 70)

    for phase_num in sorted(args.phase):
        run_phase(phase_num)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete. Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
