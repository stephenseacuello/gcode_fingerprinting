# Decoder Paper

Single active paper file: `decoder_paper.tex`. No other versions — regenerate MDPI formatting at submission time.

## Compile
```bash
pdflatex decoder_paper.tex && bibtex decoder_paper && pdflatex decoder_paper.tex && pdflatex decoder_paper.tex
```

## Rules
- Never fabricate numbers. Every claim must trace to a verified data source.
- The headline metric is **86.1% mean cross-fold sequence accuracy** (5 folds, best of 5 seeds per fold). Do not round to 86.0 — the multi-seed retrained result is 86.1.
- All reported accuracies are **teacher-forced** (beam_width=0), matching the training script's eval. Greedy autoregressive decode produces different results.
- Y-axis is **lateral step-over position**, NOT depth. Z is depth.
- Machine name: **Bantam Tools Explorer™ CNC Milling Machine**.
- The encoder paper is under review at MDPI Machines. The grammar paper is in preparation. Neither is published yet.

## Key paths
- Paper source: `outputs/decoder20260304/paper/decoder_paper.tex`
- V7 multi-seed checkpoints: `outputs/decoder20260304/v7_best_5fold_multiseed/fold_{1-5}_seed_{best}/`
- Best seeds: fold1=2024, fold2=123, fold3=456, fold4=2024, fold5=789
- Ablation results: `outputs/decoder20260304/ablations/`
- Baselines: `outputs/decoder20260304/baselines/baseline_summary.json`
- Modality ablation: `outputs/decoder20260304/modality_ablation/{no_accel,...}/`
- Preprocessed V7 data: `outputs/decoder20260304/preprocessed_v7/fold_{1-5}/`
- Vocab: `data/gcode_vocab_712.json` (712 tokens)
- Per-class predictions: `outputs/decoder20260304/v7_best_5fold_eval/fold_{1-5}/results/beam_0_all_predictions.json`
- ANOVA: `outputs/decoder20260304/paper/anova_ablation_results.json`

## Known issues
- Modality ablation bug: `romesh_changes/run_preprocessing_cv_fold.py` produces multi-line gcode_texts per window. Fixed by channel-stripping V7 preprocessed data into `preprocessed_fixed/`. Only affects modality ablation, not the main results.
- The evaluate() function in `run_decoder_quick_test.py` was patched to output all predictions (was hardcoded to 20). Keep this change.
- Fold 1 is always the hardest (~71%). This is an encoder representation issue, not a decoder bug.
