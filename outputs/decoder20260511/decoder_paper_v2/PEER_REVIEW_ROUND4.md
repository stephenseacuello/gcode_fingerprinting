# Simulated Peer Review — Round 4 (post-K-spectrum + formal-stats revision)

**Manuscript:** Structured Multi-Head Decoding for Per-Field CNC G-Code
Recoverability from Multi-Modal Sensor Embeddings · **Venue:** MDPI *Sensors* ·
**77 pp single PDF** (commit 296a3d9) · **Generated:** 2026-05-25

## Status

Round 3.5 (2026-05-19) declared convergence at 67 pp. Since then **10
commits** of new content have landed:
- §sec:abl-nonumeric "Vocabulary-cardinality spectrum" paragraph (K-spectrum 4-point sweep)
- New table `tab:k_spectrum` and figure `fig:k_spectrum`
- New inferential layer: RM-ANOVA + Friedman + Holm-adjusted pairwise t
- Surfacing into Abstract, Conclusion item (2a), §sec:abl-vocab
- Float-epsilon encoder fix + supplementary merge (already reviewed in R3)

Round 4 is a stress-test of the new K-spectrum content only. Round 3.5's
remaining polish items are not re-litigated here.

## SYNTHESIS — three referees, all Major Revision, one dominant finding

| Referee | Focus | Verdict | Headline |
|---|---|---|---|
| R4-A | Stats / inferential methodology | Major Revision | Asymptotic Friedman should be exact; fold-5 "outlier" claim is empirically wrong; n=5 d_z biased upward |
| R4-B | CNC / mfg security domain | Major Revision | K=335 "default" recommendation **widens** the threat-model bypass; tamper-detection tradeoff undisclosed |
| R4-C | ML methods / training-config fairness | Major Revision | "Sweet spot" framing conflates K-cardinality with SS-schedule; paper's own line 860 names SS=0.5 instability as a leading alternative |

**All three converge on this central issue:** the "K=335 structural sweet spot"
claim, as currently framed in the Abstract and Conclusion, is undersupported.
The headline-vs-smaller-K comparison crosses a deliberate methodology boundary
(SS=0.5/dw=1.0 vs SS=0/dw=0). The body acknowledges this in
`sec:abl-nonumeric` and the table footnote — but the caveat is not propagated
to the load-bearing sections (Abstract ~L64, Conclusion ~L1442). Two
independent referees (R4-B, R4-C) note that the leading alternative
explanation — that the +9 pp AR gain is *entirely* the SS=0 schedule change,
not a vocabulary effect — is consistent with the data and with the paper's
own statement at L860–866 that SS=0.5 destabilizes under collapsed
vocabularies.

**Resolution path:** either (i) **run the K=2,418/SS=0/dw=0 control**
(~10 hr compute) to disentangle the confound, or (ii) **downgrade the
sweet-spot framing** in the Abstract and Conclusion to "the AR per-token
improvement is robust across smaller K under one methodology; the K=335
maximum vs. the headline cannot be cleanly attributed to vocabulary
cardinality vs. training schedule on present data."

---

## Blocking findings (consensus)

| # | Finding | Source | Severity |
|---|---|---|---|
| **B1** | "$K\!\approx\!335$ structural sweet spot" in Abstract (L64) and Conclusion item 2a (L1442) drops the methodology confound that the body and table footnote properly disclose. As-is, an Abstract-only reader is misled. | R4-A #6, R4-B #B1, R4-C #B1+B2 | All-3 consensus, blocking |
| **B2** | The deployment recommendation in `email_to_sodhi_v4.md:120–124` ("default to 2-digit bucketing") **widens** the adaptive-adversary bypass (K=335 collapses the ~0.001–0.01" regime where fine-precision tampering hides). The threat-model implication is not in the paper. | R4-B #B3 | Security-critical, must be stated or recommendation withdrawn |
| **B3** | The paired-t comparisons (K=335/69/24 vs K=2418) are testing a **two-factor change** (cardinality + SS/dw schedule) but reported as if testing vocabulary cardinality alone. The Friedman omnibus likewise tests a condition effect, not a K effect. | R4-A #6, R4-C #B3 | Inferential overclaim |
| **B4** | Asymptotic Friedman χ²(3)=9.24, p=0.026 in §sec:abl-nonumeric, Abstract, and Conclusion. With n=5 the χ² approximation is outside its validity zone; **exact Friedman p = 0.0167** (computable from `audit/k_spectrum_compare.json`). Re-running the exact test is a 30-line script. | R4-A #1 | Methodology, fixable in one paragraph |
| **B5** | The "fold-5 outlier" explanation at L924–926 for the K=69 null result is empirically wrong. Grubbs G=1.57 < G_crit=1.72 (fold 5 is not a statistical outlier). Leave-one-fold-out: removing fold-5 still leaves K=69 t(3)=1.54, p=0.22. The actual driver of the K=69 null is fold-4 (paired diff −0.045). | R4-A #2 | Verifiable misstatement |
| **B6** | The K-spectrum finding is not registered in §sec:lim-eval or §sec:lim-method. No corpus-generalization caveat (single Bantam Tools mill, 120 files), no methodology-confound caveat, no threat-model implication. | R4-B #B2 | Limitations gap |

---

## Major issues (per-referee, non-overlapping)

### R4-A (statistics)

- **M3-A.** RM-ANOVA F(3,12)=2.87 reported without sphericity check or Greenhouse-Geisser correction. At n=5 Mauchly's test is uninterpretable; the standard remedy is GG-corrected df + explicit note that Friedman is the primary inferential statement.
- **M4-A.** Cohen's $d_z$ values (1.84, 1.59, 0.84) uncorrected for small-sample bias. Hedges' $J\!=\!0.80$ gives corrected $g_z = 1.47, 1.27, 0.67$. Conclusion's "$d_z\!>\!1.5$" doesn't survive: K=24's $g_z\!=\!1.27$.
- **M5-A.** Holm family of 3 contrasts on struct_token_acc only — but seq_exact also Holm-corrected privately in JSON. Either justify hierarchical testing (omnibus → pairwise) or report joint Holm across the 6 tests. Currently reads as cherry-picked.
- **M7-A.** No bootstrap 95% CI on the K-spectrum effects, inconsistent with the rest of the paper (Table 6 has 10k-bootstrap CIs).

### R4-B (CNC / security)

- **M1-B.** Coherence with the entropy-floor story. Two adjacent Abstract claims pull opposite directions: "fine-precision recovery unattainable regardless of decoder" (entropy floor) AND "K=335 preserves per-position lexical scaffolding" (architecturally collapses fine precision). Consistent only on adaptive HSM rows; facing/pocket regimes (where the model *does* recover coordinate repeats) lose strict signal under K=335.
- **M2-B.** Tormach/HAAS generalization (asserted in the supervisor email) has no evidence on present corpus. "Would-change-our-mind" clause needed: a second-corpus replication is the minimum bar.
- **M3-B.** `tab:k_spectrum` is single-metric ("structural recovery"). Missing tradeoff axes: inference latency, training time, **numeric-tamper detectability per K** (the bypass-widening of B2 needs to be quantified, not only narrated).
- **M4-B.** Reproducibility recipe is incomplete. A practitioner who reads §sec:abl-nonumeric to retrain at K=335 needs: precise definition of "two-digit bucketing" (first 2 slots? first 2 sig digits? integer + tenths?); exact raw-float → bucket mapping (currently buried in `data/gcode_vocab_v8_b2.json`); unambiguous "Design-B style" config in body text.

### R4-C (ML methods)

- **M1-C.** Under matched methodology, K ∈ {24, 69, 335} gives AR token 0.405, 0.415, 0.429 — within one σ. The "U-shape" interpretation is an artifact of including the SS=0.5 K=2418 point. Drop it and the spectrum is essentially flat.
- **M2-C.** "Preserves per-position lexical scaffolding" mechanism is asserted, not evidenced. Alternatives not addressed: easier optimization landscape at moderate K, smoothed loss surface, per-digit prior alignment.
- **M3-C.** `fig:k_spectrum` plots all 4 K on one continuous curve. The K=2418 point should be visually distinguished (open marker, dashed connector, separate legend "SS=0.5 reference").
- **M4-C.** §sec:abl-vocab keeps the legacy K=451 single-fold pilot and bolts on a forward-ref to K=335. Two K's for ostensibly the same condition will confuse readers. Either fold the pilot into the spectrum or explicitly note K=335 supersedes K=451.
- **M5-C.** No prior-literature citation for vocab sweeps as a standard ML diagnostic (BPE merges, subword regularization).
- **M6-C.** "$K\!=\!2{,}418$/SS=0 control remains as future work" is correctly placed but understated — it is the only thing that resolves B1. Promote.

---

## Minor issues (combined)

- **m1.** Abstract sentence at L64 packs spectrum + entropy floor + sweet-spot into one 90-word clause. Split.
- **m2.** Conclusion item 2a repeats body stats verbatim. Trim to qualitative claim + section ref.
- **m3.** "$+0.111$, $+0.098$, $+0.088$~pp" — unit ambiguity. Are these proportions or percentage points? Suggest "$+11.1$, $+9.8$, $+8.8$~pp".
- **m4.** Table 9 footnote duplicates the caption caveat — keep one.
- **m5.** Figure 14 caption says K=335 "exceeds the headline" without disclosing the methodology distinction in the caption itself.
- **m6.** L924 "outlier" framing reads defensively as a post-hoc explanation. If retained, move to footnote.

---

## Prioritized action items

### Tier 1 — must do before submission (blocking)

| # | Item | Effort | Resolves |
|---|---|---|---|
| **T1.1** | **Reframe the K=335 claim in Abstract (L64) and Conclusion item 2a (L1442)**. Drop "sweet spot" or qualify it with "within Design-B methodology"; explicitly state the SS/dw confound. | 15 min (text) | B1 |
| **T1.2** | **Withdraw or qualify the supervisor-email recommendation** in `email_to_sodhi_v4.md`. The "default to 2-digit bucketing" line is a deployment claim the paper doesn't support and that *widens* the threat-model bypass. | 5 min (text) | B2 |
| **T1.3** | **Add a `\paragraph{K-spectrum methodology confound and threat-model scope.}` to §sec:lim-eval** stating: SS/dw confound, single-corpus risk, threat-model bypass-widening under K=335. | 20 min (text) | B1, B2, B6 |
| **T1.4** | **Re-run Friedman as exact**, update Abstract / §sec:abl-nonumeric / Conclusion with exact p=0.017. | 15 min (script + replace) | B4 |
| **T1.5** | **Fix the "fold-5 outlier" misstatement at L924–926.** Either drop the post-hoc explanation or replace with the correct one (fold-4 −0.045 paired diff). | 10 min (text) | B5 |
| **T1.6** | **Re-frame the pairwise stats as "condition contrasts" not "vocabulary contrasts"** in §sec:abl-nonumeric and Conclusion. | 10 min (text) | B3 |

### Tier 2 — strongly recommended (substantive)

| # | Item | Effort | Resolves |
|---|---|---|---|
| **T2.1** | Add Hedges' $g_z$ alongside $d_z$; update Conclusion's "$d_z>1.5$" sentence. | 20 min | M4-A |
| **T2.2** | Add a sphericity / GG-correction note (or explicit "Friedman is primary") to the K-spectrum paragraph. | 15 min | M3-A |
| **T2.3** | Add bootstrap 95% CIs to `tab:k_spectrum` for consistency with Table 6. | 30 min (script + table edit) | M7-A |
| **T2.4** | Resolve the K=451 ↔ K=335 confusion in §sec:abl-vocab. | 10 min | M4-C |
| **T2.5** | Re-render `fig:k_spectrum` with K=2418 visually distinguished as the methodologically distinct point. | 20 min (figure script edit) | M3-C |
| **T2.6** | Add "preserves per-position lexical scaffolding" hedging ("we hypothesize…, alternatives include…"). | 5 min | M2-C |
| **T2.7** | Cite vocabulary-sweep precedent (e.g. Sennrich et al. 2016 BPE, Kudo 2018 subword regularization). | 5 min | M5-C |

### Tier 3 — would close the issue cleanly (optional but high-leverage)

| # | Item | Effort | Resolves |
|---|---|---|---|
| **T3.1** | **Run K=2,418 under Design-B methodology** (SS=0, dw=0, 5 folds, full-window). Eliminates B1, B3, M1-C in one experiment. Compute: ~10 hr GPU time. Result is the cleanest resolution and converts "Major Revision" → "Accept (minor)" in a Round 5. | 10 hr GPU + ~30 min analysis | B1, B3, M1-C |

### Tier 4 — internal cohesion (R4 minors)

m1–m6 above; total effort ~30 min.

---

## Recommended sequencing

1. **Day 1 (text-only, ~2 hr):** T1.1, T1.2, T1.3, T1.5, T1.6, T2.4, T2.6, T2.7, m1–m6. This brings the paper out of "Major Revision" on the framing issue without re-analysis.
2. **Day 1 also (script work, ~1 hr):** T1.4, T2.1, T2.2, T2.3, T2.5. Closes the stats issues.
3. **Optional but recommended:** T3.1 — kicks off the K=2418/SS=0 control. If it ships clean, the framing in T1.1 can be re-strengthened in Round 5.

---

## Quoted closing verdicts

- **R4-A (stats):** "Sound experimental work undercut by promotional framing in three high-visibility positions (Abstract, Conclusion, supervisor email). The fixes are entirely textual and require no additional experiments — though the K=2,418-under-Design-B control would convert 'Major Revision' into 'Accept (minor)' in a follow-up round."

- **R4-B (security):** "Round 3.5's declaration that 'no Round 4 needed' was premature with respect to this security-domain angle: the K-spectrum content was added after Round 3.5 and was not stress-tested against the threat model. It is now."

- **R4-C (ML):** "A real Reviewer 2 will cite line 860–866 against the Abstract and ask why the paper does not consider its own admission about SS instability as the leading alternative."

---

## Bottom line

The paper at R3.5 was substantively complete. The post-R3.5 K-spectrum work
**added a strong new experiment but mis-framed its load-bearing conclusion**.
All three independent referees flag the same headline issue (methodology
confound + over-claim in Abstract/Conclusion), plus disjoint supporting
critiques on stats, security, and ML mechanism. The fix is mostly textual
(~3 hours of edits) plus optionally one ~10-hr GPU control to fully close
the question. **Round 5 is recommended after the Tier-1 fixes land.** If the
K=2418/SS=0 control is run, Round 5 can be a fast convergence check; if not,
the downgraded framing remains the final state and is still defensible.
