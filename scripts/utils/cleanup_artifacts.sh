#!/bin/bash
# Cleanup script for G-code fingerprinting project artifacts
# Safe cleanup that preserves important files

set -e  # Exit on error

echo "=========================================="
echo "G-code Fingerprinting Project Cleanup"
echo "=========================================="
echo

# Navigate to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo

# 1. Backup current state
echo "1. Creating backup list..."
find wandb -type d -name "run-*" | head -20 > .cleanup_keep_runs.txt
echo "   Saved list of 20 most recent runs to .cleanup_keep_runs.txt"
echo

# 2. Clean old WandB runs (keep 20 most recent)
echo "2. Cleaning old WandB runs..."
if [ -d "wandb" ]; then
    TOTAL_RUNS=$(find wandb -type d -name "run-*" | wc -l)
    echo "   Total runs found: $TOTAL_RUNS"

    if [ $TOTAL_RUNS -gt 20 ]; then
        echo "   Removing runs older than the 20 most recent..."
        find wandb -type d -name "run-*" -print0 | \
            xargs -0 ls -td | \
            tail -n +21 | \
            xargs rm -rf

        REMAINING_RUNS=$(find wandb -type d -name "run-*" | wc -l)
        echo "   ✓ Kept $REMAINING_RUNS most recent runs"
        echo "   ✓ Removed $((TOTAL_RUNS - REMAINING_RUNS)) old runs"
    else
        echo "   ✓ Only $TOTAL_RUNS runs, keeping all"
    fi
else
    echo "   No wandb directory found"
fi
echo

# 3. Clean empty output directories
echo "3. Removing empty output directories..."
if [ -d "outputs" ]; then
    EMPTY_DIRS=$(find outputs -type d -empty | wc -l)
    if [ $EMPTY_DIRS -gt 0 ]; then
        find outputs -type d -empty -delete
        echo "   ✓ Removed $EMPTY_DIRS empty directories"
    else
        echo "   ✓ No empty directories found"
    fi
else
    echo "   No outputs directory found"
fi
echo

# 4. Clean Python cache
echo "4. Removing Python cache files..."
PYCACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ $PYCACHE_DIRS -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "   ✓ Removed $PYCACHE_DIRS __pycache__ directories"
else
    echo "   ✓ No __pycache__ directories found"
fi

# Also remove .pyc files
PYC_FILES=$(find . -name "*.pyc" 2>/dev/null | wc -l)
if [ $PYC_FILES -gt 0 ]; then
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "   ✓ Removed $PYC_FILES .pyc files"
fi
echo

# 5. Clean old checkpoints (keep best and latest 5)
echo "5. Cleaning old checkpoints..."
if [ -d "outputs" ]; then
    # Find all epoch checkpoints, keep only the 5 most recent per directory
    for dir in outputs/*/; do
        if [ -d "$dir" ]; then
            EPOCH_CKPTS=$(find "$dir" -name "checkpoint_epoch_*.pt" -type f 2>/dev/null | wc -l)
            if [ $EPOCH_CKPTS -gt 5 ]; then
                echo "   Cleaning $dir..."
                find "$dir" -name "checkpoint_epoch_*.pt" -type f | \
                    sort -V | \
                    head -n -5 | \
                    xargs rm -f
                echo "     ✓ Kept 5 most recent epoch checkpoints"
            fi
        fi
    done
    echo "   ✓ Checkpoint cleanup complete"
else
    echo "   No outputs directory found"
fi
echo

# 6. Remove specific temporary directories
echo "6. Removing known temporary directories..."
TEMP_DIRS=(
    "outputs/phase1_retrain"
    "outputs/phase1_focal_loss"
    "outputs/phase1_strong_weights"
    "outputs/test_run"
)

REMOVED_COUNT=0
for dir in "${TEMP_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "   ✓ Removed $dir"
        ((REMOVED_COUNT++))
    fi
done

if [ $REMOVED_COUNT -gt 0 ]; then
    echo "   ✓ Removed $REMOVED_COUNT temporary directories"
else
    echo "   ✓ No temporary directories to remove"
fi
echo

# 7. Archive old vocabulary
echo "7. Archiving old vocabulary..."
if [ -f "data/gcode_vocab.json" ]; then
    if [ ! -f "data/gcode_vocab_v1_668tokens.json.bak" ]; then
        cp data/gcode_vocab.json data/gcode_vocab_v1_668tokens.json.bak
        echo "   ✓ Backed up data/gcode_vocab.json → data/gcode_vocab_v1_668tokens.json.bak"
    else
        echo "   ✓ Backup already exists"
    fi
else
    echo "   No vocabulary file to backup"
fi
echo

# 8. Summary
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo

echo "Current disk usage:"
echo "  wandb:   $(du -sh wandb 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  outputs: $(du -sh outputs 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  data:    $(du -sh data 2>/dev/null | cut -f1 || echo 'N/A')"
echo

echo "Important files preserved:"
echo "  ✓ data/*.csv (raw data)"
echo "  ✓ outputs/processed_data/ (preprocessed sequences)"
echo "  ✓ Recent W&B runs (20 most recent)"
echo "  ✓ checkpoint_best.pt files"
echo "  ✓ All source code"
echo

echo "✅ Cleanup complete!"
echo
