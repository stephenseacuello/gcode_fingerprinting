#!/bin/bash
# launch_groups.sh -- Called by Phase 0 after cache_inference + generate_attacks complete.
# Creates tmux windows with parallel experiment groups.
set -euo pipefail

PROJECT="/home/seacuello/Documents/gcode_fingerprinting"
BASE="outputs/anomaly20260319"
SCRIPTS="scripts/anomaly"
LOG="$BASE/logs"
SESSION="anomaly"

cd "$PROJECT"
mkdir -p "$LOG"

# Helper: wait until a file exists, polling every 10 seconds
wait_for_file() {
    local target="$1"
    local label="$2"
    echo "[$label] Waiting for $target ..."
    while [ ! -f "$target" ]; do
        sleep 10
    done
    echo "[$label] Found $target -- proceeding."
}

# ============================================================
# groupA window: Exps 1, 2, 3, 4 (CPU, no dependencies)
# ============================================================
tmux new-window -t "$SESSION" -n "groupA"

# Pane 0: Exp 1 -- NLL Scoring (CORE)
tmux send-keys -t "$SESSION:groupA" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp01_nll_scoring.py 2>&1 | tee $LOG/exp01_nll_scoring.log" C-m

# Pane 1: Exp 2 -- Calibration
tmux split-window -t "$SESSION:groupA" -h
tmux send-keys -t "$SESSION:groupA.1" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp02_calibration.py 2>&1 | tee $LOG/exp02_calibration.log" C-m

# Pane 2: Exp 3 -- Grammar Violation Partition
tmux split-window -t "$SESSION:groupA" -v
tmux send-keys -t "$SESSION:groupA.2" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp03_grammar.py 2>&1 | tee $LOG/exp03_grammar_violation.log" C-m

# Pane 3: Exp 4 -- Multi-Head Disagreement
tmux select-pane -t "$SESSION:groupA.0"
tmux split-window -t "$SESSION:groupA.0" -v
tmux send-keys -t "$SESSION:groupA.3" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp04_disagreement.py 2>&1 | tee $LOG/exp04_multihead_disagreement.log" C-m

tmux select-layout -t "$SESSION:groupA" tiled

# ============================================================
# groupA2 window: Exps 11, 13, 14 (CPU, no dependencies)
# ============================================================
tmux new-window -t "$SESSION" -n "groupA2"

# Pane 0: Exp 11 -- Embedding Distance
tmux send-keys -t "$SESSION:groupA2" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp11_embedding.py 2>&1 | tee $LOG/exp11_embedding_distance.log" C-m

# Pane 1: Exp 13 -- NLL vs Argmax Head-to-Head
tmux split-window -t "$SESSION:groupA2" -h
tmux send-keys -t "$SESSION:groupA2.1" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp13_nll_vs_argmax.py 2>&1 | tee $LOG/exp13_nll_vs_argmax.log" C-m

# Pane 2: Exp 14 -- Encoder DTAE vs Decoder NLL
tmux split-window -t "$SESSION:groupA2" -v
tmux send-keys -t "$SESSION:groupA2.2" \
  "cd $PROJECT && python3 $SCRIPTS/run_exp14_encoder_vs_decoder.py 2>&1 | tee $LOG/exp14_dtae_vs_nll.log" C-m

tmux select-layout -t "$SESSION:groupA2" tiled

# ============================================================
# groupB window: Exps 7, 8, 12, 18 (CPU, depend on Exp 1)
# Each pane polls for exp01 aggregate_results.json before starting.
# ============================================================
EXP01_RESULT="$BASE/exp01_nll_scoring/aggregate_results.json"

tmux new-window -t "$SESSION" -n "groupB"

# Pane 0: Exp 7 -- Cross-Condition Transfer
tmux send-keys -t "$SESSION:groupB" \
  "cd $PROJECT && \
   $(declare -f wait_for_file) && \
   wait_for_file '$EXP01_RESULT' 'Exp07' && \
   python3 $SCRIPTS/run_exp07_cross_condition.py 2>&1 | tee $LOG/exp07_cross_condition.log" C-m

# Pane 1: Exp 8 -- Sliding Window Ensemble
tmux split-window -t "$SESSION:groupB" -h
tmux send-keys -t "$SESSION:groupB.1" \
  "cd $PROJECT && \
   while [ ! -f '$EXP01_RESULT' ]; do sleep 10; done && \
   echo '[Exp08] exp01 results found -- proceeding.' && \
   python3 $SCRIPTS/run_exp08_ensemble.py 2>&1 | tee $LOG/exp08_sliding_window.log" C-m

# Pane 2: Exp 12 -- Graded Synthetic Injection
tmux split-window -t "$SESSION:groupB" -v
tmux send-keys -t "$SESSION:groupB.2" \
  "cd $PROJECT && \
   while [ ! -f '$EXP01_RESULT' ]; do sleep 10; done && \
   echo '[Exp12] exp01 results found -- proceeding.' && \
   python3 $SCRIPTS/run_exp12_graded_injection.py 2>&1 | tee $LOG/exp12_graded_injection.log" C-m

# Pane 3: Exp 18 -- CUSUM Temporal Detection
tmux select-pane -t "$SESSION:groupB.0"
tmux split-window -t "$SESSION:groupB.0" -v
tmux send-keys -t "$SESSION:groupB.3" \
  "cd $PROJECT && \
   while [ ! -f '$EXP01_RESULT' ]; do sleep 10; done && \
   echo '[Exp18] exp01 results found -- proceeding.' && \
   python3 $SCRIPTS/run_exp18_cusum.py 2>&1 | tee $LOG/exp18_cusum_temporal.log" C-m

tmux select-layout -t "$SESSION:groupB" tiled

# ============================================================
# gpu0 window: Exp 6 -- Sensor Failure (GPU 0)
# ============================================================
tmux new-window -t "$SESSION" -n "gpu0"

tmux send-keys -t "$SESSION:gpu0" \
  "cd $PROJECT && \
   CUDA_VISIBLE_DEVICES=0 python3 $SCRIPTS/run_exp06_sensor_failure.py 2>&1 | tee $LOG/exp06_sensor_failure.log" C-m

# ============================================================
# gpu1 window: Exps 5 -> 15 -> 16 -> 10 (sequential, GPU 1)
# ============================================================
tmux new-window -t "$SESSION" -n "gpu1"

tmux send-keys -t "$SESSION:gpu1" \
  "cd $PROJECT && \
   CUDA_VISIBLE_DEVICES=1 python3 $SCRIPTS/run_exp05_modality_ablation.py 2>&1 | tee $LOG/exp05_sensor_ablation.log && \
   CUDA_VISIBLE_DEVICES=1 python3 $SCRIPTS/run_exp15_position_confound.py 2>&1 | tee $LOG/exp15_window_position.log && \
   CUDA_VISIBLE_DEVICES=1 python3 $SCRIPTS/run_exp16_feature_config.py 2>&1 | tee $LOG/exp16_feature_config.log && \
   CUDA_VISIBLE_DEVICES=1 python3 $SCRIPTS/run_exp10_loco.py 2>&1 | tee $LOG/exp10_loco.log && \
   echo '=== gpu1 chain DONE ==='" C-m

# ============================================================
# postanalysis window: Exps 9, 17 + figures + summary
# Waits for ALL experiment aggregate_results.json files.
# ============================================================
tmux new-window -t "$SESSION" -n "postanalysis"

tmux send-keys -t "$SESSION:postanalysis" \
  "cd $PROJECT && \
   echo '=== Post-Analysis: waiting for all experiments ===' && \
   EXPS='exp01_nll_scoring exp02_calibration exp03_grammar exp04_disagreement \
         exp05_modality_ablation exp06_sensor_failure exp07_cross_condition exp08_ensemble \
         exp10_loco exp11_embedding exp12_graded_injection exp13_nll_vs_argmax \
         exp14_encoder_vs_decoder exp15_position_confound exp16_feature_config exp18_cusum' && \
   for exp in \$EXPS; do \
     while [ ! -f \"$BASE/\$exp/aggregate_results.json\" ]; do \
       sleep 15; \
     done; \
     echo \"[postanalysis] \$exp complete.\"; \
   done && \
   echo '=== All prerequisite experiments done ===' && \
   python3 $SCRIPTS/run_exp09_error_taxonomy.py 2>&1 | tee $LOG/exp09_error_taxonomy.log && \
   python3 $SCRIPTS/run_exp17_cost_benefit.py 2>&1 | tee $LOG/exp17_cost_benefit.log && \
   python3 $SCRIPTS/generate_all_figures.py 2>&1 | tee $LOG/generate_all_figures.log && \
   python3 $SCRIPTS/update_summary.py 2>&1 | tee $LOG/update_summary.log && \
   echo '=== ALL EXPERIMENTS COMPLETE ==='" C-m

# ============================================================
# monitor window: watch last line of each log file
# ============================================================
tmux new-window -t "$SESSION" -n "monitor"

tmux send-keys -t "$SESSION:monitor" \
  "watch -n 5 'echo \"=== Anomaly Experiment Monitor ===\"; echo; \
   for f in $LOG/*.log; do \
     name=\$(basename \$f .log); \
     if [ -f \"\$f\" ]; then \
       last=\$(tail -1 \"\$f\" 2>/dev/null || echo \"(empty)\"); \
       printf \"%-35s %s\\n\" \"\$name\" \"\$last\"; \
     fi; \
   done; echo; \
   echo \"--- Completed experiments ---\"; \
   for d in $BASE/exp*/; do \
     if [ -f \"\${d}aggregate_results.json\" ]; then \
       echo \"  DONE: \$(basename \$d)\"; \
     fi; \
   done'" C-m

echo "=== launch_groups.sh: All windows created ==="
