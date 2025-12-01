#!/bin/bash
# Cleanup and organization script for G-code fingerprinting project
# Usage: bash cleanup.sh [--archive] [--dry-run]

set -e

DRY_RUN=false
ARCHIVE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --archive)
            ARCHIVE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash cleanup.sh [--archive] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "G-Code Fingerprinting Project Cleanup"
echo "============================================"
echo "Dry run: $DRY_RUN"
echo "Archive old files: $ARCHIVE"
echo ""

# Function to execute or print command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $1"
    else
        eval "$1"
    fi
}

# ============================================
# 1. Clean Python cache
# ============================================
echo "→ Cleaning Python cache files..."
run_cmd "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
run_cmd "find . -type f -name '*.pyc' -delete 2>/dev/null || true"
run_cmd "find . -type f -name '*.pyo' -delete 2>/dev/null || true"
run_cmd "find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true"
echo "✓ Python cache cleaned"
echo ""

# ============================================
# 2. Clean temporary files
# ============================================
echo "→ Cleaning temporary files..."
run_cmd "find . -type f -name '.DS_Store' -delete 2>/dev/null || true"
run_cmd "find . -type f -name '*.log' -exec rm -f {} + 2>/dev/null || true"
run_cmd "find . -type f -name '*~' -delete 2>/dev/null || true"
echo "✓ Temporary files cleaned"
echo ""

# ============================================
# 3. Archive deprecated scripts (if requested)
# ============================================
if [ "$ARCHIVE" = true ]; then
    echo "→ Archiving deprecated Phase 1 scripts..."

    ARCHIVE_DIR="archive/phase1_experiments"
    run_cmd "mkdir -p $ARCHIVE_DIR"

    # List of deprecated scripts
    DEPRECATED_SCRIPTS=(
        "train_with_class_weights.py"
        "verify_class_weights.py"
        "test_phase1_improvements.py"
    )

    for script in "${DEPRECATED_SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            echo "  Archiving: $script"
            run_cmd "mv $script $ARCHIVE_DIR/"
        fi
    done

    echo "✓ Deprecated scripts archived to $ARCHIVE_DIR"
    echo ""
fi

# ============================================
# 4. Organize outputs
# ============================================
echo "→ Checking output directories..."
OUTPUT_DIRS=(
    "outputs/processed_data"
    "outputs/processed_v2"
    "outputs/baseline_v2"
    "outputs/augmented_v2"
    "outputs/multihead_v2"
    "outputs/multihead_aug_v2"
    "outputs/wandb_sweeps"
    "outputs/visualizations"
    "outputs/evaluation"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  ✓ $dir ($size)"
    else
        echo "  ✗ $dir (not found)"
    fi
done
echo ""

# ============================================
# 5. Clean wandb runs (keep metadata)
# ============================================
echo "→ Cleaning W&B cache..."
if [ -d "wandb" ]; then
    # Only clean run cache, keep metadata
    run_cmd "find wandb -type d -name 'run-*' -exec rm -rf {}/files/media {} + 2>/dev/null || true"
    echo "✓ W&B media cache cleaned (metadata preserved)"
else
    echo "  ✗ No wandb directory found"
fi
echo ""

# ============================================
# 6. Report disk usage
# ============================================
echo "============================================"
echo "Disk Usage Summary"
echo "============================================"
echo ""

if [ -d "outputs" ]; then
    echo "Output directories:"
    du -sh outputs/* 2>/dev/null | sort -h
    echo ""
fi

if [ -d "data" ]; then
    echo "Data directory:"
    du -sh data 2>/dev/null
    echo ""
fi

if [ -d ".venv" ]; then
    echo "Virtual environment:"
    du -sh .venv 2>/dev/null
    echo ""
fi

if [ -d "wandb" ]; then
    echo "W&B logs:"
    du -sh wandb 2>/dev/null
    echo ""
fi

echo "Total project size:"
du -sh . 2>/dev/null
echo ""

# ============================================
# 7. List checkpoints
# ============================================
echo "============================================"
echo "Available Checkpoints"
echo "============================================"
echo ""

find outputs -name "checkpoint_best.pt" 2>/dev/null | while read -r checkpoint; do
    size=$(du -h "$checkpoint" | cut -f1)
    echo "  $checkpoint ($size)"
done
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Cleanup Complete"
echo "============================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a dry run. No files were modified."
    echo "Run without --dry-run to apply changes."
fi

echo ""
echo "Next steps:"
echo "1. Review [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md)"
echo "2. Check training progress: wandb.ai/<your-username>/gcode-fingerprinting"
echo "3. Run evaluation: python test_evaluation.py --checkpoint <path>"
