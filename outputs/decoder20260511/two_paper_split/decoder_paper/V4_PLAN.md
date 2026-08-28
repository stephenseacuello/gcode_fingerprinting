# decoder_paper_v4 Plan — Figure-First Self-Contained Edition (2026-08-05)

**Goal:** single self-contained paper (no supplementary), ≤55pp, with the bulk of the supplement's figures integrated. Response to feedback: "too wordy and long-winded" — v4 pays the page budget with prose, not figures.

**Base:** v3 (all 94 supplementary pointers already resolved; zero S-references). v2/v3/supplement untouched; v4 is a new file.

## Ranking (cuts go bottom-up; figures rank ABOVE corroborating prose)

1. **Untouchable:** headline claims + numbers, hedges/scope qualifiers (one canonical statement each), core tables (headline, per-class command, per-axis, data-coverage, ablations, nonumeric, sensor, LOMO, tamper), threat-model definitions, Limitations content.
2. **Restore (Tier-1 figures):** five_fold_spread, per_axis_recoverability, per_digit bathtub, per_op_class, loco_per_holdout, sensor_ablation_bars, lomo_modality_bars, tamper ROC pair, mode_collapse_heatmap, confusion_matrix_command.
3. **Restore (Tier-2 figures):** sensor/G-code overlay (dataset illustration), encoder_memory_pca, sign/digit confusion pair, lomo_nested heatmap, reliability_diagram, failure plate.
4. **Keep compressed:** appendix protocol/repro unique numbers.
5. **Cut (in order):** (a) prose that narrates what a restored figure now shows (rule 2.1 — the figure+caption is the carrier); (b) caveat restatements beyond canonical home + Conclusion; (c) defensive meta-commentary ("we report this because…", "must not be conflated", reading instructions); (d) multi-clause qualifier chains → one qualifier per claim (rule 2.2); (e) enumerated artifact-path prose; (f) Tier-3 figures stay artifact-cited (learning curves, t-SNE, per-file fingerprint, seed-variance, K-spectrum curve, per-class bar charts, numeric histograms).

## Procedure (scientific-writing rules govern every edit)

1. Seed v4 from v3; restore Tier-1+2 figure blocks verbatim from the supplement/draft archives; flip each artifact-citation sentence back to a `\ref`.
2. Measure (expect ~63pp). Concision pass per section: target −15–20% words; claim-invariant on numbers/hedges/cites; sentences one job; paragraphs Theme→Evidence→Reasoning→Connection; figure-described detail collapses to claim + ref.
3. Verify: cite-set identical to v3; numeric set lossless (documented label-only exceptions); zero undefined refs; zero S-references; compile clean.
4. Fit to ≤55pp; if short, cuts continue bottom-up per the ranking above — figures are cut LAST, and only Tier-2.
