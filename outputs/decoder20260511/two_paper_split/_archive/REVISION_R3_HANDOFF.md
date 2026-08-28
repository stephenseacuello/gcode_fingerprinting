# Revision Round 3 — handoff (round-2 referee blockers)

Round-2 verdicts: Paper A = 3x Minor (trivial). Paper B = 2x Minor + 1x Major (one real root cause).
Authoritative data files (all regenerated/verified this round):
- `audit/source_vs_tokenized_entropy.json` — per-axis SOURCE(4-dec) vs TOKENIZED(axis-precision) slot entropy + three regimes
- `audit/patched_numeric_analysis.json` — patched accuracy bathtub, per-axis MAE, corr r=-0.83 (n=30)
- `audit/raw_gcode_source_entropy.json` — corrected decoder-free source apparatus (round encoding)

## PAPER A — 3 trivial blockers + 3 minor polish (decoder_paper/)
B1 (ML, blocking): seq2seq baseline is misdescribed as "per-row"; the released configs are FULL-WINDOW.
  Change "same per-row target formulation" -> "same full-window target formulation" in the seq2seq Baseline
  paragraph AND in tables/seq2seq_baseline.tex caption comment. (Comparison is already correct full-vs-full;
  prose only.)
B2 (CNC, blocking): Fig numeric_error_histogram.pdf shows pre-patch MAE annotations contradicting patched
  Table 5. Regenerate it from checkpoints/full_window_5fold_patched/fold_*/results/predictions.npz (patched
  per-axis |pred-target| in inches; annotate exact-match fraction). Patched MAE: X 0.248/Y 0.205/Z 0.009/R 0.021.
B3 (OT, blocking): one residual "Forensic-grade" heading (~line 909). Change to "Forensic-support". Grep the
  whole tex for any other residual "forensic-grade" (case-insensitive) and fix all.
Minor: (a) regenerate or DELETE orphan tables/seq2seq_baseline_per_fold.tex (its per-fold values contradict
  the released metrics JSONs; it is not \input anywhere). (b) "+0.223" -> "+0.222" in Appendix encoder-spec.
  (c) add a one-line footnote that per-class modal frequencies use the first-command-position basis
  (class_conditional_modal.json) vs all-positions (confound_class_prior.json).
Optional correctness: where A says the ten-thousandths is "structurally ~0", add that the mechanism is the
  V8 tokenizer's 0.001 quantization grid for X/Y/Z (not a 3-decimal corpus). Keep minimal; do not regress A.
RECOMPILE clean (0 undefined refs/cites).

## PAPER B — reconcile source vs tokenized (the Major root cause) + 2 Minor (toolpath_entropy/)

### ROOT CAUSE (all 3 referees): the decoder-free apparatus contradicted the "3-decimal/structural-zero"
claim. TRUTH (now established): the corpus is **4-decimal** (raw X/Y/Z e.g. X3.3121; 11,173 4-dec vs 1,456
3-dec) and the **V8 tokenizer quantizes X/Y/Z to a 0.001 grid** (precision X/Y/Z=0.001, R=0.0001, F=1.0).
So:
- SOURCE entropy (what the CAM file holds): X/Y/Z are ~uniform at 10^-2, 10^-3 AND 10^-4 (all ~3.2-3.3 bits).
- TOKENIZED target (what the model can represent): X/Y/Z 10^-4 -> 0 bits (discarded by the 0.001 grid);
  R retains 10^-4 (0.0001 grid, H=2.51).
- The 10^-4 "structural zero" for X/Y/Z is a **tokenizer representation choice, NOT a source or sensor limit.**
- The genuine information-theoretic recoverability limit is at the **hundredths/thousandths** (~3.3 bits,
  SOURCE == TOKENIZED, genuinely unrecoverable from the 4 Hz stream).

### Authoritative per-axis slot entropy (audit/source_vs_tokenized_entropy.json), [10^1,10^0,10^-1,10^-2,10^-3,10^-4]:
- X (prec 0.001): source [0,1.72,3.21,3.31,3.30,3.31] ; tokenized [0,1.72,3.21,3.31,3.30,0.00]
- Y (prec 0.001): source [0,1.68,3.20,3.31,3.32,3.31] ; tokenized [...,3.32,0.00]
- Z (prec 0.001): source [0,0,0.64,3.23,3.31,3.31] ; tokenized [...,3.31,0.00]
- R (prec 0.0001): source == tokenized [0,0.09,0.67,2.38,2.47,2.51]  (all 4 decimals retained)
- F (prec 1.0): integer/units only.

### REQUIRED Paper B fixes
D-MAJOR (reconcile): rewrite the fine-digit narrative to the two-object story above. Present BOTH source and
  tokenized per-(axis,slot) entropy (a small table); the ONLY divergence is X/Y/Z 10^-4 (3.31 -> 0). State
  the limit is at hundredths/thousandths; the 10^-4 for X/Y/Z is tokenizer quantization (cite the 0.001 grid),
  while R retains 10^-4. This converts the contradiction into a precise, defensible mechanism and is the
  decoder-free apparatus's strongest demonstration.
D1 (Table 1 / tab:raw_vs_npz off-by-one): repopulate columns with the correct slot->decimal mapping using
  the authoritative numbers above; make the slot index explicit (slot 3 = 10^-2 hundredths, slot 4 = 10^-3
  thousandths, slot 5 = 10^-4 ten-thousandths). Verify every column.
D2 (three regimes, single authoritative source): use audit/source_vs_tokenized_entropy.json
  three_regime_thousandths_X (SOURCE): **face 2.10 / adaptive 3.31 / pocket 3.32 bits at 10^-3 (thousandths)**.
  Reconcile abstract / Sec 3 / Sec 5 / Sec 6 / Conclusion to these exact numbers (replace the prior 1.64 and
  any 2.34). Facing is the low-entropy (coordinate-repeat) regime; adaptive/pocket sit at the ceiling.
D3 (entropy<->accuracy): use TOKENIZED entropy paired with patched accuracy (both the recoverable object) ->
  Pearson r=-0.83 / Spearman -0.91 (n=30), pooled -0.89 (n=6); from patched_numeric_analysis.json. State the
  relation is over the tokenizer-representable digits.
D4 ("peak" wording): replace "peaks at the thousandths" with "plateaus near the uniform ceiling across the
  hundredths-thousandths (~3.3 bits)".
D5 (k-spectrum Friedman): use the exact Monte-Carlo p-values from the audit JSON: **p=0.0018 (AR token),
  p=0.019 (AR sequence)** (not the asymptotic chi^2).
D6 (stale figures): regenerate from the corrected apparatus/JSONs where feasible; else replace the in-text
  claim with the corrected table and add the figure to a clearly-captioned "regenerated; corrected values
  tabulated" note. Targets: bits_recoverable_per_axis, per_slot_per_opclass_entropy_heatmap,
  per_axis_per_slot_per_opclass_entropy_heatmap, quantization_heatmap, digit_value_histograms,
  entropy_accuracy_all_heads (54-cell). The corrected raw apparatus (raw_gcode_source_entropy.json) has
  per-op-class per-axis per-slot entropy to drive the heatmaps.
D7 (citations): verify Miller 1955 + Paninski 2003 bibliographic details (web search); resolve the Zenodo TODO
  or leave a single explicit proof-stage placeholder.
RECOMPILE clean (0 undefined refs/cites).
