#!/bin/bash
# ── POST_SWEEP_PLAN Launcher ──
#
# Runs all decoder baselines (1A-1D) and ablations (2A-2C) across 5 folds.
# Baselines are fast (no GPU training for 1A/1B/1D, ~5min for 1C).
# Ablations need GPU (~30 min each × 12 conditions × 5 folds).
#
# Usage:
#   # Everything:
#   bash scripts/experiments/launch_post_sweep.sh
#
#   # Baselines only (fast, ~30 min):
#   bash scripts/experiments/launch_post_sweep.sh --baselines-only
#
#   # Ablations only:
#   bash scripts/experiments/launch_post_sweep.sh --ablations-only
#
#   # Specific folds:
#   FOLDS="1 2" bash scripts/experiments/launch_post_sweep.sh
#
# Author: Claude Code
# Date: March 2026

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=${GPU_ID:-0}
FOLDS=${FOLDS:-"1 2 3 4 5"}
RUN_BASELINES=1
RUN_ABLATIONS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --baselines-only) RUN_ABLATIONS=0; shift ;;
        --ablations-only) RUN_BASELINES=0; shift ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --dry-run) export DRY_RUN=1; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "================================================================"
echo "POST_SWEEP_PLAN Launcher — $TIMESTAMP"
echo "GPU: $GPU_ID, Folds: $FOLDS"
echo "Baselines: $RUN_BASELINES, Ablations: $RUN_ABLATIONS"
echo "================================================================"

# ── Phase 1: Baselines (fast) ──
if [[ "$RUN_BASELINES" == "1" ]]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "PHASE 1: Baselines (1A-1D) × 5 folds"
    echo "════════════════════════════════════════"

    for fold in $FOLDS; do
        echo ""
        echo "── Fold $fold ──"
        CUDA_VISIBLE_DEVICES=$GPU_ID /home/seacuello/Documents/venv/bin/python scripts/evaluation/run_decoder_baselines.py \
            --encoder_config v7_f110_w256_s64 \
            --fold "$fold" \
            --vocab data/gcode_vocab_712.json \
            --output_dir outputs/decoder20260304/baselines \
            --baselines 1A 1B 1C 1D \
            --device auto
    done

    # Aggregate baseline results across folds
    echo ""
    echo "── Aggregating baseline results ──"
    python3 - <<'PYEOF'
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

base = Path("outputs/decoder20260304/baselines")
all_results = defaultdict(lambda: defaultdict(list))

for fold_dir in sorted(base.glob("fold_*")):
    results_path = fold_dir / "baseline_results.json"
    if results_path.exists():
        fold = fold_dir.name.replace("fold_", "")
        results = json.load(open(results_path))
        for name, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_results[name][metric].append(value)

print(f"\n{'Baseline':<25} {'Seq':>10} {'Tok':>10} {'Cmd':>10} {'Num':>10}")
print("-" * 70)
for name in sorted(all_results.keys()):
    m = all_results[name]
    seq = np.array(m.get('sequence_accuracy', [0]))
    tok = np.array(m.get('token_accuracy', [0]))
    cmd = np.array(m.get('command_accuracy', [0]))
    num = np.array(m.get('numeric_accuracy', [0]))
    print(f"{name:<25} {seq.mean():.1%}±{seq.std():.1%} {tok.mean():.1%}±{tok.std():.1%} "
          f"{cmd.mean():.1%}±{cmd.std():.1%} {num.mean():.1%}±{num.std():.1%}")

# Save
summary = {}
for name, metrics in all_results.items():
    summary[name] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v}
                     for k, v in metrics.items() if isinstance(v[0] if v else None, (int, float))}
with open(base / "baseline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {base / 'baseline_summary.json'}")
PYEOF
fi

# ── Phase 2: Ablations (slow, GPU-intensive) ──
if [[ "$RUN_ABLATIONS" == "1" ]]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "PHASE 2: Ablations (2A-2C) × 5 folds"
    echo "════════════════════════════════════════"

    export GPU_ID FOLDS
    export MAX_PARALLEL=1  # Sequential to avoid GPU OOM
    bash scripts/experiments/run_decoder_ablations.sh
fi

echo ""
echo "================================================================"
echo "POST_SWEEP_PLAN COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
