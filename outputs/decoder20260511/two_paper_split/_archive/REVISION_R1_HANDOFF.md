# Revision Round 1 — handoff (patched numbers + corrected narrative + referee fix-lists)

Authoritative patched data: `outputs/decoder20260511/audit/patched_numeric_analysis.json`
Patched checkpoints: `checkpoints/full_window_5fold_patched/` (released `full_window_5fold` untouched).

## A. Patched-retrain results (BOTH papers)

**Headline is robust to the float-epsilon patch** (5-fold TF, patched vs released):
token 0.774 vs 0.781; numeric 0.570 vs 0.585; command 0.891 vs 0.888; type 0.983 vs 0.984;
param_type 0.993 vs 0.993; sequence 0.029 vs 0.026. All drift <1.5 pp → the categorical/headline
claims are unaffected; report this as a robustness confirmation, NOT a number change.

**Corpus-weighted target-correction rate** (target_tokens identical between runs; clean control):
**32.5%** of numeric positions had >=1 digit slot corrected; **30.1%** were terminal-zero 9->0 fixes.
(Replaces the paper's "23.6% of terminal zeros" — the true figure is ~30% of ALL numeric positions.)

## B. The corrected numeric story (mainly Paper B; also Paper A numeric tier)

The float-epsilon bug corrupted BOTH the entropy axis and the accuracy axis at the finest digits, and
**mislocated the precision floor by one decimal place.** Corrected (round-encoding, entropy and accuracy
now consistent):

| slot | source H (bits) | recovery acc |
|---|---|---|
| 10^1 tens | 0.007 | 1.00 |
| 10^0 units | 1.69 | 0.92 |
| 10^-1 tenths | 3.08 | 0.75 |
| 10^-2 hundredths | 3.30 | 0.55 |
| **10^-3 thousandths** | **3.29** | **0.42**  <- TRUE precision floor (finest digit CAM emits) |
| 10^-4 ten-thousandths | 0.31 | 0.99  <- structurally ~0 (corpus uses 3-decimal coords) |

**Entropy <-> accuracy anticorrelation is now STRONGER and encoding-consistent:**
pooled n=6 Pearson **-0.89** (Spearman -0.94); per-(axis,slot) n=30 Pearson **-0.83** (Spearman -0.91).
(Pre-patch fixed value was r=-0.68 with a known encoding inconsistency between the two axes.)

**Per-(axis,slot) corrected entropy / patched accuracy:**
- X: H [0, 1.72, 3.11, 3.29, 3.24, 0] ; acc [1.00, 0.90, 0.71, 0.47, 0.43, 1.00]
- Y: H [0, 1.61, 3.11, 3.29, 3.29, 0] ; acc [1.00, 0.91, 0.67, 0.42, 0.38, 1.00]
- Z: H [0, 0, 0.71, 3.27, 3.31, 0] ; acc [1.00, 1.00, 0.98, 0.91, 0.27, 1.00]  (recoverable through hundredths; cutting-depth steps)
- R: H [0, 0, 1.17, 1.97, 2.33, 1.99] ; acc [1.00, 1.00, 0.89, 0.89, 0.88, 0.91]  (discrete arc-radius vocab; recoverable across all slots, incl. 10^-4)
- F: integer/units only.

**Per-axis value MAE (patched, inches):** X 0.248, Y 0.205, Z 0.009, R 0.021, F 1.11
(released-paper values were X 0.277, Y 0.179, Z 0.042, R 0.018; Z MAE improves to 0.009).

**Three operation-class regimes SURVIVE but at the THOUSANDTHS (10^-3), not the ten-thousandths.**
Corrected per-op-family X-axis entropy at 10^-3: **face 1.64 / adaptive 3.25 / pocket 3.27 bits**
(coordinate-repeat in facing is real, just at the correct slot). Slot-5 (10^-4) is ~0 entropy for ALL
op-classes (adaptive 1.00 / face 1.00 / pocket 0.97 accuracy) — the previous slot-5 three-regime story
(adaptive 3.3 / face 1.7) and the "96.5% of face X from 4 overruns AT slot-5" were float-epsilon artifacts.

### REQUIRED recast for Paper B (do not just swap numbers — change the claim):
1. Move the "near-uniform, information-theoretically unrecoverable fine digit" claim from the
   ten-thousandths (10^-4) to the **thousandths (10^-3)** = the finest digit the CAM postprocessor emits.
   State explicitly the corpus uses 3-decimal coordinates, so 10^-4 is structurally ~0 (deterministic),
   recovered at 0.99 — NOT an unrecoverable slot.
2. Recast the three operation-class regimes at the thousandths (face 1.64 vs adaptive/pocket ~3.25 bits).
3. The bathtub "upturn" at 10^-4 is genuine low-entropy structure (always 0), not an artifact and not
   "fine recovery" — describe precisely.
4. Section 8 (float-epsilon walk-back): UPGRADE from "we disclose a pre-patch artifact" to "we corrected
   the target encoding, retrained 5-fold, and re-derived the entropy/accuracy with consistent encoding;
   the corrected correlation is r=-0.83 (n=30); the prior slot-5 structure was an encoding artifact now
   removed." This converts the IT referee's deepest objection into a resolved strength.
5. Report per-cell symbol counts n and add a plug-in entropy bias note (Miller-Madow) per the IT referee.

## C. Paper A — referee fix-list (2 Major + 1 Minor; MDPI Sensors)
1. (ML, major) Reframe the seq2seq "+77 pp command lift": the headline command head is a non-AR classifier
   on encoder memory (TF/AR-invariant by construction); the baseline 0.122 is command read off AR-generated
   tokens. State this is the value of routing categorical prediction OFF the token stream, NOT
   autoregressive-decoder superiority. Remove/qualify "+77 pp architectural" framing.
2. (ML, major) n=5: demote all four-significant-figure p-values from 5-point tests to 1 sig fig or
   effect-size+per-fold scatter; explicitly state fold-1 command is at/below its class prior, so the
   +22 pp increment is a 4/5-fold effect; stop citing BH-FDR/Holm "survival" as confirmatory.
3. (ML/CNC, major) Lead abstract+contributions with the +22 pp class-prior-controlled increment
   (as an UPPER bound) and the LOCO open-vocab floor (0.213), not 0.888. Bound the increment against the
   class-conditional modal (~0.83) where possible, shrinking the genuine sensor-driven margin.
4. (ML, major) AR mode-collapse: reserve "recovery" for the categorical heads; state up front that
   end-to-end generative reconstruction is at floor (AR seq exact-match 0.026); retitle/abstract wording
   from generative "reconstruction/recovery" to "per-field categorical recoverability."
5. (CNC, major) NUMERIC TIER NOW PATCHED: replace per-axis MAE + per-digit-slot numbers with the patched
   values (Section B above) and state they were re-derived on a float-epsilon-corrected retrain; remove the
   pre-patch caveat. Per-axis MAE: X 0.248 / Y 0.205 / Z 0.009 / R 0.021.
6. (CNC, major) Sign asymmetry is prior-driven: lead with Z as the only sensor-bearing sign axis
   (H_Z=1.04 bits), state Y/X sign is prior-deterministic; cite companion Paper B for sign entropies.
7. (OT, moderate) Tamper FPR is the recoverability-complement relabelled (no physical tamper; ~0 pp over
   class-conditional modal) — move operating points to an explicitly "illustrative, prior-dominated, not a
   validated detector" subsection; say so in the abstract.
8. (OT, moderate) "forensic-grade" -> "forensic-support"/"post-hoc investigative aid" throughout.
9. (CNC, moderate) Add a per-modality PSD/SNR characterization note of the 4 Hz envelope vs raw (or
   explicitly scope it as a stated gap); annotate the "what transfers" table's categorical-recovery row as
   mechanism-uncertain across platforms.

## D. Paper B — referee fix-list (2 substantive Major; MDPI Entropy)
1. (IT, major) Fano/rate-distortion is name-dropped, not instantiated: you measure H(X), not H(X|obs).
   Either (a) DOWNGRADE "limit/bound" language to an empirical "source-entropy-correlated recoverability
   ordering" (reserve "limit" for the 0-bit slots and the sub-Nyquist floor), or (b) instantiate a real
   conditional-entropy/MI bound. RECOMMENDED: do (a) cleanly + add the corrected encoding-consistent
   correlation (r=-0.83, n=30) which is now the honest empirical core; note the within-head vs cross-head
   structure (the negative relation is cross-head; command head is near-ceiling/flat).
2. (IT, major) Entropy bias: add Miller-Madow (or note plug-in downward bias ~ (K-1)/(2n ln2)); REPORT
   per-cell symbol counts n; flag the cells with n<30.
3. (IT, major) "Two recoverers land at the same ceiling" is contradicted on the numeric head (seq2seq beats
   the decoder by ~17 pp TF on numeric). Restrict the convergence claim to the CATEGORICAL command head;
   do not claim architecture-invariance on the numeric head.
4. (IT/side-channel, moderate->major) Float-epsilon: see Section B.4 — this is now a RESOLVED strength.
5. (side-channel, major) Replace ALL TODO stub CAM citations with real references (trochoidal/adaptive HSM,
   pocketing, postprocessor coordinate emission, ISO 14649, ISO 230-2). Do NOT fabricate; use established
   textbooks/standards with correct details, verify via web search if unsure.
6. (side-channel, major) Reduce figure/result duplication with companion Paper A (cross-cite instead of
   re-plot shared assets).
7. (side-channel, moderate) Per-head 5-fold entropy provenance inconsistency: B1 confirmed 5-fold ~= fold-1
   (cosmetic); state the 5-fold mean+/-sd and remove the internal inconsistency.
8. (side-channel, moderate) Abstract overstates what is patched-clean: now you CAN say the retrain is done
   and the headline is robust; update accordingly.
9. (all) Single-platform/n=5 scope: keep explicit; frame cross-CAM/second-corpus as registered next step.
