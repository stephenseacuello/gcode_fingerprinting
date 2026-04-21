#!/usr/bin/env bash
# run_5fold.sh — preprocess + train + eval all 6 tokenizers across 5 folds.
# Total wall time: ~10 minutes on a single GPU. Logs go to design_space_results/5fold_log/.
set -euo pipefail

TOKENIZERS=(domain_bpe bytelevel_bpe wordpiece char_level flat_per_value gpt4_compact hierarchical)
FOLDS=(1 2 3 4 5)

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
fi
cd "$REPO_ROOT"
LOG_DIR=outputs/decoder20260303/grammarpaper/design_space_results/5fold_log
mkdir -p "$LOG_DIR"

START=$(date +%s)
for fold in "${FOLDS[@]}"; do
  for tok in "${TOKENIZERS[@]}"; do
    log="$LOG_DIR/fold${fold}_${tok}.log"
    echo "[$(date +%H:%M:%S)] fold=$fold tok=$tok"
    {
      python3 scripts/experiments/grammar_design_space/preprocess_alt.py \
        --tokenizer "$tok" --fold "$fold" --max-len 64
      python3 scripts/experiments/grammar_design_space/train_alt.py \
        --tokenizer "$tok" --fold "$fold" --epochs 80 --max-token-len 64
    } > "$log" 2>&1
  done
  # Eval all tokenizers for this fold in one shot
  echo "[$(date +%H:%M:%S)] eval fold=$fold"
  python3 scripts/experiments/grammar_design_space/eval_alt.py --fold "$fold" \
    > "$LOG_DIR/fold${fold}_eval.log" 2>&1
done
END=$(date +%s)
echo "Done in $((END - START))s"
