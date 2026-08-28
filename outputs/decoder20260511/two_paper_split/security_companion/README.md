# Security Companion Paper (Paper F of the bundle) — draft assembled 2026-08-13

**Title:** Sensor-Side Tamper Detection for CNC Machining: Threat Model, Illustrative
Operating Points, and Structural Limits of a G-Code Recoverability Monitor
**Venue target:** MDPI Sensors (bundle default; a security venue, e.g. MDPI JCP, is an option).
**State:** 12pp, 0 errors, 0 undefined refs, abstract 198 words, 23 refs, 4 figures, 1 table.

This is the "companion security study (in preparation)" cited by decoder_paper_v5 (36pp) and
decoder_paper_v5_conf_ieee (10pp). It makes those citations true.

## Provenance (all quantitative content is certified material, claim-invariant)
- Threat Model section: decoder_paper_v4.tex §Threat Model (verbatim, refs adapted).
- Detectability map + operating points + tamper table + ROC pair: v4 §Threat Model Detectability.
- Extended alert-budget / aggregation / prior-dominance control (incl. the class-conditional-null
  footnote) + usage patterns: supplementary_materials.tex §app:tamper-detail + §app:deployment-detail
  (the richest certified versions).
- Aggregation-curves + autocorrelation figures: supplement §Tamper-Detection Diagnostics.
- NEW prose (written under the scientific-writing rules): abstract, Introduction, Related Work
  reframing, §Recoverability Foundation (summary of the companion decoder's four load-bearing
  results, all numbers cited to it), Limitations consolidation, Conclusion.
- New bib entry `decoder_paper` (companion, in preparation) added to the copied decoder_references.bib.
- 34 certified numbers verified present in the compiled PDF (incl. MITRE codes, all operating points).

## Not covered here (by design)
Physically-executed attacks and sensor spoofing remain the scoped-but-unbuilt physical-attack
paper (outputs/physical_attack_paper_20260617/PROPOSAL.md); this paper's Limitations point to it.

## Compile
cd latex && pdflatex security_companion && bibtex security_companion && pdflatex x2

## Inline review pass (2026-08-13): CLEAN after 1 fix
- Mechanical: all \ref targets resolve AND point where the adapted text promises (the v4→companion
  ref adaptation introduced no misdirections); zero leaked decoder-internal section refs; zero
  unearned intensifiers in the new prose.
- Numeric (artifact-verified): rho_honest 0.421, population s.d. 0.218 ≈ paper's "0.42 ± 0.22"
  (matches the decoder bundle's population-s.d. convention); K=10 majority @ rho_empirical
  with-metadata FPR 0.098 / TPR 0.966 and baseline 0.184 / 0.94 (exact); modal-substitution control
  TPR 0.841±0.022 / FPR 0.391±0.028 vs real 0.92 / 0.302 (exact); ECE 0.0501→0.0197 (exact).
- FIX: the ECE 0.050→0.020 sentence cited \cite{decoder_paper}, but v5 CUT its calibration section —
  re-pointed to \texttt{audit/calibration_cross_fold.json} (verified to hold the numbers) and added
  it + class_conditional_modal.json to the Data Availability artifact list.
- Layout: Intro p1 / Related p2 / Threat Model p3 / Foundation+Detectability p4 / Operating Points p5
  / Alert Budget p6 / Adaptive p7 / Deployment p8 / Limitations p9 / Conclusion p10 / refs p11-12.
- FINAL: 12pp, 0 errors, 0 undefined, abstract 198w. Review depth = single thorough referee +
  mechanical/artifact verification; a multi-agent panel stamp needs the spend limit raised.
