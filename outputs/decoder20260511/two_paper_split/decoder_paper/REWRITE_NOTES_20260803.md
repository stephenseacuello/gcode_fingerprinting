# Decoder Paper Rewrite — 2026-08-03

Two-pass rewrite under the `scientific-writing` rules (Romesh Prasad's method, 2026-07-29). Pre-rewrite state is committed in `2b6203f` (branch `bundle-backup-2026-06-17`); every change below is git-recoverable. Nothing is committed — commit when ready.

## Files changed

- `latex/decoder_paper_v2.tex` + `.pdf` — main paper, **57 → 49 pages**, 0 errors, 0 undefined refs
- `latex/supplementary_materials.tex` + `.pdf` — 56 → 62 pages (received relocated exhibits as new final section S39)
- `tables/headline_5fold.tex` — footnote pointer updated (bathtub figure now in S39)

## Pass 1 — claim-invariant prose rewrite (all 10 sections)

Each section rewritten by a dedicated agent, then adversarially verified (mechanical set-equality on every number, `\cite`, `\ref`/`\label`, plus claim-by-claim semantic audit; fix loops until pass). What changed: 85 → 9 triple-dash chains (survivors are captions/table markers/appendix), multi-claim sentences split to one job each, labeling openers and filler cut, paragraphs restructured to Theme → Evidence → Reasoning → Connection, banned-word audit applied (kept only where evidence-backed or where removal would weaken a certified claim). Figure/table environments and captions were kept verbatim in this pass.

## Pass 2 — deduplication + length cut (56 → 49 pages)

**Dedup cuts** (rule 2.1; each cut item keeps a canonical home): VMC platform contrast (canonical: §Experimental Setup "Platform characterization"; trimmed from Intro scope statement and Limitations), sampling-rate/77.7% aliasing treatment (canonical: §Methodological Limitations), verbatim-overlap paragraph (canonical numbers kept, LOCO detail pointed to §Results), tamper operating-point numbers restated in prose beside their own table, K=10/ρ operating point (stated 3× → canonical in threat-model security-numbers ¶), K-spectrum story (canonical: §Evaluation Limitations), companion-paper division of labor (canonical: §Related Work + appendix), "cannot rank modalities" (stated 3× → canonical in RQ3 finding ¶), metadata-anchor mechanism (canonical: RQ2), appendix pointer sections merged 6 → 2 with body-duplicated statistics dropped, data-availability paragraph deduped against MDPI `\dataavailability` field.

**Relocations to Supplementary S39** ("Relocated Main-Text Exhibits") — content preserved verbatim, pointer left in body, all supporting statistics remain stated in body prose/tables: `tab:anova`, `tab:bootstrap_ci`, `tab:seq2seq_baseline`, `tab:per_class_param_type`, `tab:lit_comparison` (positioning table), `fig:five_fold_spread`, `fig:encoder_memory_pca`, `fig:sensor_ablation`, `fig:lomo_modality`, `fig:per_axis`, `fig:loco_per_holdout`, `fig:per_digit_position_curve` (bathtub), `alg:forward` (pseudocode; FSM spec was already S35). All remaining body floats are load-bearing (headline, per-class command, per-axis, data-coverage, ablations, nonumeric, sensor/LOMO tables, tamper table, system overview).

**Layout-only**: float separations tightened, `\bibsep` 0pt, abbreviations converted from two fixed minipages to MDPI run-in style (same content), algorithm baselinestretch 1.1 → 1.0 (moot after relocation).

**Verification (final vs. pre-rewrite)**: citation set identical in both directions (bibliography byte-comparable); numeric-value set — nothing lost (every value absent from the main body is present in supplementary S39); the only new numeric token is "39" (section references). Warning profile matched baseline before relocations; final compile has zero errors and zero undefined references.

## Flagged for author adjudication (evidence-category A/B/C/D unclear — left unchanged)

1. **Intro**: "substantially harder signal environment" — comparative supported by cited mechanisms, intensifier unquantified. "sensor-to-G-code mapping is comparatively low-noise" (AM) — attributed to citations but not established as their finding. "mapping is many-to-one … broadly similar sensor signatures" — physical reasoning stated as fact, uncited.
2. **Related Work**: "achieved strong results" (CNN monitoring) and "high agreement" (Al Faruque) — degree qualifiers resting entirely on cited sources.
3. **Problem Formulation**: "plus an end-of-sequence token" — ambiguous attachment (loss vs. target sequence). "substantially lower accuracies under AR" — backed by measured drops but no number in the sentence.
4. **Method**: grammar-mask "improves both calibration and exact-match accuracy" — no in-section measurement or pointer; scheduled sampling "reduces exposure bias" stated as fact (cite attaches to the regime, not the effect); auxiliary-head "two purposes" design rationale unmeasured.
5. **Experimental Setup**: "well within standard plant-IT storage tiers" (uncited); "factory-rated sensitivities" (untraceable in-section).
6. **Limitations**: "Label set scale" cross-reference points to its own subsection, which does not report the promised per-fold distinct-line counts — possibly a misdirected pointer.
7. **Discussion**: motion-sign asymmetry attributed to sensor-separability here vs. the prior-driven account in Results §Per-Axis — the two accounts should be reconciled. "Replay is detectable indirectly (argued, not benchmarked)" self-labels between C and D.
8. **Conclusion**: item 2 groups 4-Hz aliasing under "decoder-side mechanisms" while §Methodological Limitations frames it as a physical sensing-stack limit.

## Pre-existing inconsistencies noticed (not fixed — content decisions)

- §Evaluation Limitations says a beam-search variant "has not been evaluated," while the S10 diagnostics pointer cites a beam-width sensitivity study (S10.4, "gap not closed by beam-3/5").
- The supplementary's own section numbering reaches S38 while it contains 37 `\section` commands (an internal counter skip); hand-written S-numbers in the body were verified to still resolve, and the new section lands at S39.
