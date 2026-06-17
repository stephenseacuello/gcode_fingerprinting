# Two-Paper Split Plan — decoder_paper_v2 → Paper A (decoder) + Paper B (toolpath entropy)

*Synthesized 2026-06-13 from a 12-agent deep read (9 readers over the 80pp paper + 35pp
supplementary + entropy scripts + 6 peer-review rounds + bib + lab notes, then 3 independent
split-architect proposals), cross-checked against the source firsthand.*

---

## 0. The decision and the sequence

**Split, sequenced A-first.** The current paper is already *dual-headline*: its abstract is literally
structured "First / Second / Third," where First is the decoder and Second+Third are the source-entropy
theory + audit-leverage corollary. Contribution (6) and the §5.1 Results opener carry the same theory.
That is *why* it reads like two papers — the entropy work was promoted to co-equal billing on top of a
decoder paper that was otherwise complete.

All three architects converge on the same sequence, and it matches your goal exactly:

- **Paper A is "complete and done" minus an excision** — it is at simulated Accept(minor); splitting
  *removes* material, it does not require new experiments. Ship it first; it bankrolls B at zero risk.
- **Paper B is "close to done but needs more testing/data"** — the substance and ~16 figures already
  exist, but it needs a short de-risking pass (mostly cheap, no-GPU) + venue-gating citation work + an
  honest framing pivot before it can stand alone.

**The one non-negotiable constraint** (from the referee record): referees already flagged the
encoder/grammar/decoder 3-way split as borderline **salami-slicing (R2-M4)**, forgiving it *only* because
each slice is a self-contained characterization with no "in-preparation" dependencies. A 4th slice off the
**same single-platform corpus** must clear a higher bar. Paper B is defensible *only* if it is reframed
as a **recoverability LIMIT**, not a relocated entropy section. See §4.

---

## 1. Paper A — the decoder / recoverability paper

**Working title (unchanged):** *Structured Multi-Head Decoding for Per-Field G-Code Recoverability from
Multi-Modal Sensor Embeddings on a Desktop CNC Milling Platform: A Single-Platform, Cross-Validated
Characterization Study.*

**Thesis it keeps:** a frozen-encoder six-head Transformer decoder recovers **categorical** G-code
structure near a sensor-side ceiling (cmd 0.888, +22 pp over the op-class prior, TF=AR invariant) while
the **autoregressive token/numeric** stream collapses by exposure-bias mode-collapse to the corpus-modal
facing toolpath (0.78→0.21 token, 0.59→0.10 numeric) — a *forensic-grade post-hoc audit* capability, not a
real-time IDS. This is A's citable identity and it stands without the entropy account.

**Venue:** MDPI *Sensors* (incumbent — the simulated review was a Sensors review; sensor-fusion +
structured-decoding fits). Fallback: IEEE T-II / J. Manufacturing Systems if a fresh submission is fine.

**Effort: LOW — ~1–2 weeks of editing, no new experiments.**

### Cut list (what leaves A for B)
| Unit (current location) | Action |
|---|---|
| Abstract "Second" (source-entropy theory) + "Third" (audit-leverage) clauses (L74) | **Delete**; rewrite to two contributions; keep the 77.7% / ~89% / +22 pp empirical tail but excise "below the entropy floor" phrasing |
| Intro contribution (6) "unified source-entropy theory" (L117) | Collapse to a one-line forward-pointer to B; renumber |
| §2.3 "Information-theoretic framing: lineage and scope" paragraph (L153) | Move to B (lifts out near-intact as B's related-work spine); demote to 1 sentence in A. `brown1992englishentropy`, `feder1994relations`, `cover_thomas` leave A's bib with it |
| §5.1 "Unified Source-Entropy Recoverability Across Heads" (L480–532) | **Move entirely to B** — with Figs `entropy_accuracy_all_heads`, `per_head_opclass_entropy_heatmap`, `per_position_token_entropy`, `entropy_per_toolpath`, `per_slot_per_opclass_entropy_heatmap`, `per_axis_per_slot_per_opclass_entropy_heatmap`, and Table `per_head_entropy` |
| §5.5 "Numeric Digit Head Decomposition" entropy machinery (L627–701) | Move bathtub-as-inverse-entropy, r=−0.96/r=−0.78, slot-5 three-regime, bits-recoverable to B. **A keeps only** the bare digit accuracies (0.585 TF / 0.104 AR) + a one-line "explained by source entropy (Paper B)" pointer. Figs `bits_recoverable_per_axis`, `entropy_vs_accuracy_overlay`, `entropy_accuracy_scatter`, `digit_value_histograms`, `quantization_heatmap`, `per_axis_bathtub_overlay` + Table `per_digit_position` move to B. *(A may retain `per_digit_position_curve` as a plain decoder-behavior plot.)* |
| §5.4 per-axis **sign-entropy** paragraph (H_X=0.38 / H_Y=0.08 / H_Z=1.04) | Move the *explanation* to B; **A keeps** has-axis/sign/MAE accuracies and states the asymmetry as an observation citing B |
| §6.4 "Audit leverage by toolpath entropy" paragraph | Move to B; A leaves a one-sentence stub. **A duplicates the three sign-entropy numbers as cited facts** so §6.4's security sign-flip argument keeps its backbone (3 numbers cited-with-attribution ≠ self-plagiarism) |
| §7 "Pooled numeric accuracy averages memorization" + "Entropy bound is in-distribution" limitations; Future Direction (c) per-axis entropy-rate | Move to B |
| App J "Numeric Target Encoding Artifacts" (float-epsilon, L1357+) + supp §S12 + Table `float_epsilon_audit` | Move to B (carries the load-bearing pre-patch caveat) |
| Supp §S7 "Per-Op-Class Slot-5 Stratification" + Fig `slot5_by_op_class` + Table `gcode_examples` + Fig `per_file_fingerprint` | Move to B (S7 + S12 must travel together) |
| §8 Conclusion finding (2) "three coupled mechanisms" | Rewrite to two decoder mechanisms (exposure-bias + 4 Hz aliasing); cite B for the digit-entropy floor |
| App D "Relationship to Companion Contributions" | **Update** (not delete) to a 4-way division naming B as a *published* companion |

### A's one independent open item: the K-spectrum
The vocabulary-cardinality K-spectrum left **Round-4 at Major Revision** (a scheduled-sampling/decay-weight
confound; the "K=335 sweet spot" claim is not defensible). Cleanest resolution: **move the whole K-spectrum
to B** as a vocabulary-cardinality-vs-source-entropy study — it is half-entropy anyway, and this *removes
A's only unresolved blocker*. If A keeps any vocabulary note, keep **only** the matched-methodology T3.1
(K=2418 strict-max) result with the sweet-spot claim withdrawn.

### Also housekeeping (independent of the split)
Delete the dead `\if0…\fi` inline replicability copy and the duplicate K-spectrum figure inclusion in the
supplementary; optionally drop the explicitly-cuttable `sensor_gcode_overlay` figure.

---

## 2. Paper B — the CNC toolpath source-entropy paper

**Working title:** *An Information-Theoretic Recoverability Limit for Low-Rate Sensor Reconstruction of CNC
Toolpath Coordinates: Source Entropy, Quantization Structure, and the 4 Hz Sub-Nyquist Floor.*

**Thesis (the framing that makes it non-salami):** per-field recoverability of a CNC instruction stream
from low-rate (4 Hz) multi-modal sensing is bounded by **two coupled limits** —
1. the **Shannon source entropy** of the toolpath-coordinate corpus (per-axis, per-digit-slot,
   per-operation-class), which stratifies into three CAM-driven regimes (near-deterministic facing/drilling,
   intermediate pocketing, near-uniform adaptive-HSM at the log₂10 ≈ 3.32-bit ceiling), and
2. a **4 Hz sub-Nyquist physical floor** where 77.7% of G-code blocks occupy ≤1 sample period.

Together these set a Fano-style ceiling any learned recoverer can approach but not exceed — demonstrated by
the inverse entropy↔recovery relationship and **validated** by a from-scratch seq2seq baseline hitting the
same ceiling. **The decoder is an existence proof, not the subject.** B owns the LIMIT; A owns the DECODER.

**Venue:** MDPI *Entropy* (information theory of a structured source; keeps the MDPI relationship).
Alternatives: MDPI *Information* (lighter); IEEE TIFS / *Computers & Security* if leaning into the
audit-leverage corollary; J. Manufacturing Systems / CIRP JMST if manufacturing-framed. **Caveat:** a
pure-IT venue will *gate* on the Fano/rate-distortion grounding AND the CAM-generation mechanism — those
bibliography gaps are venue-blocking, not optional.

**Effort: MEDIUM–HIGH — ~3–8 weeks.** Drafting is fast (assets exist); the gating work is the de-risking
experiments + the citation/CAM scholarship. A general "limit" claim needs a second corpus (months — see §3).

### Proposed section outline
1. **Intro** — the in-principle recoverability question, generalized beyond one decoder; cite A (published)
   as the empirical instantiation, not the result being re-analyzed.
2. **Related Work** — lift A's §2.3 lineage paragraph; **expand** with the missing IT canon (Shannon 1951
   prediction-of-English, Fano 1961, Cover & Thomas entropy/AEP/rate-distortion ch., Lloyd–Max quantization,
   Nyquist) **and the entirely-missing CAM/toolpath-generation block** (trochoidal/adaptive-HSM, pocketing /
   Held-Voronoi, postprocessor coordinate emission, ISO 14649 STEP-NC, ISO 230-2 repeatability).
3. **The Source: CNC G-code as a structured symbol stream** — self-contained corpus + V8 vocab +
   per-(axis,slot) quantization spec (cite encoder_paper + A; describe, don't re-derive).
4. **Per-field source entropy (decoder-free)** — entropy on **raw CAM g-code** + per-(axis,slot) and
   per-(head,op-class) maps across **all 5 folds**.
5. **Three operation-class regimes** — slot-5 stratification (adaptive ~3.26–3.30 bits; facing ~1.55–1.71
   bits driven by 4 X-overrun coords = 96.5% of face X; mixed pocket ~2.58–3.22 bits), with the CAM-generation
   mechanistic explanation.
6. **The physical floor: 4 Hz sub-Nyquist aliasing** — own the `gcode_row_durations` audit (77.7% ≤1 sample).
7. **The recoverability ceiling + empirical confirmation** — **lead with cell-level n=30 r=−0.78**; demote
   pooled n=6 r=−0.96 to illustrative (bivariate normality uncheckable at n=6); 54-cell r=−0.55/ρ=−0.66;
   seq2seq 0.877≈0.888 shows the ceiling is sensor-side not model-side.
8. **Float-epsilon encoding artifact + honest walk-back** — App J promoted to a full self-demolition section.
9. **Vocabulary-cardinality and the entropy floor** — the K-spectrum, reframed correctly (sweet-spot
   withdrawn; T3.1-matched; exact Friedman p=0.0167; Hedges g_z).
10. **Audit-leverage corollary** — low-entropy ops auditable, high-entropy adaptive blind; complements
    file-hash signing (scoped as corollary, here on B's *own* limit).
11. **Limitations** — single platform/material, n=5, closed-vocab in-distribution, no S/I/J/arcs, pre-patch
    caveat (until retrain lands), single-corpus generality.
12. **Conclusion** + 13. self-contained appendices (corpus/vocab/quantization spec; 4 Hz pipeline; entropy
    estimation + small-n inference; reproducibility manifest).

### Figures/tables that move to B
Figs: `entropy_accuracy_all_heads`, `entropy_accuracy_scatter`, `entropy_vs_accuracy_overlay`,
`entropy_per_toolpath`, `per_position_token_entropy`, `per_head_opclass_entropy_heatmap`,
`per_slot_per_opclass_entropy_heatmap`, `per_axis_per_slot_per_opclass_entropy_heatmap`,
`digit_value_histograms`, `slot5_by_op_class`, `quantization_heatmap`, `bits_recoverable_per_axis`,
`per_axis_bathtub_overlay`, `per_file_fingerprint`, `gcode_line_example`, `k_spectrum`.
Tables: `per_head_entropy`, `per_digit_position`, `float_epsilon_audit` (A3), `gcode_examples`, `k_spectrum`.

### What B still needs — the "more testing / data"
**Cheap, no-GPU (days) — do these FIRST, they gate credibility:**
1. **Raw-CAM-text entropy script** *(new, salami-defeating)* — parse `data_clean/*.gcode` + `vocab_corpus.gcode`
   and compute per-(axis,slot,op-class) Shannon entropy **directly from source strings**, before any
   tokenization/decoder target-encoding. This is the single change that converts B from "re-analysis of A's
   predictions" into an independent empirical apparatus. *(Raw files verified present.)*
2. **All-folds entropy recompute** — `per_head_source_entropy.json`, `per_head_per_opclass_entropy.json`,
   `per_position_token_entropy.json` are confirmed **fold-1-train-only** (n_windows=303). Re-run
   `compute_per_head_per_opclass_entropy.py` over `fold_{1..5}/train_sequences.npz` and aggregate. *(Note:
   `digit_entropy.json` — the headline bathtub/slot/op-class numeric story — is **already** 5-fold test, so
   this recompute is narrower than the raw architect reports implied.)*
3. **Fix the broken accuracy axis** — `per_op_class_v8.json` errors on every fold (`op_names len ≠
   predictions`; same alignment-bug class noted in `feedback_data_alignment.md`); `generate_entropy_accuracy_scatter.py`
   silently falls back to 6 hardcoded TF constants. Repair the op_names↔predictions join so the 54-cell
   scatter uses **live** per-(head,op-class) accuracy.

**One GPU experiment (~10–15 GPU-h):**
4. **Patched-encoder 5-fold retrain** — `digit_value_head.py` is *already* patched in source (round-to-scaled-int,
   L289), but the **released checkpoint weights were trained pre-patch**, so slot-3/4/5 numbers inherit the
   23.6%-terminal-zero→digit-9 artifact. Retrain 5-fold (~2.5 h/fold), verify with
   `verify_digit_encoding_fix.py` (4.05%→0.000%), regenerate affected figures. Without this, §8's slot-5
   claim stays "bounded by a known encoding artifact."

**Stats rigor:** lead with n=30 r=−0.78; exact Friedman (p=0.0167) not asymptotic χ²; bias-corrected Hedges
g_z; fix the empirically-wrong "fold-5 outlier" claim (Grubbs G=1.57<1.72 — the real driver is **fold-4**,
diff −0.045).

**Citation scholarship (venue-gating, ~1–2 weeks)** — the IT canon + the entirely-missing CAM literature
above. This is the largest writing lift and the most likely rejection trigger if skipped.

**Generality — the long pole (optional but strengthening):** the summer **Tormach DOE** corpus (spindle
3/6/9k rpm × depth 0/0.10/0.25/0.50/1.00 mm × Al6061/steel1018/Delrin × tools 3/6/9.5 mm) tests whether the
three-regime decomposition is platform-specific or a general property of CAM-generated toolpaths, and adds
the arcs / S / I / J coverage the present corpus lacks. Even **re-exporting the same parts through a second
CAM postprocessor** (entropy-only, no machining) partially closes it. **If unavailable before submission, B
must scope the limit explicitly to this corpus** rather than claim generality (answers R2-M3 / R4-M2-B).

**Housekeeping:** commit the untracked scripts B's pipeline depends on —
`compute_per_head_per_opclass_entropy.py`, `plot_per_head_opclass_entropy.py`,
`generate_entropy_accuracy_scatter.py`, `shrink_decoder_metrics.py`.

---

## 3. Shared assets — handle once, cross-cite, never "in prep"

| Asset | Ownership / handling |
|---|---|
| 120-file Bantam corpus + 9 op-classes | A = "the sensor dataset"; B = "the CAM-authored G-code source" (+ raw `data_clean/*.gcode`). ~1 paragraph each, cross-cited, cite encoder_paper for provenance |
| V8 vocab + per-(axis,slot) quantization map | Full bucket-formula lives in **B** (entropy-relevant); A cites B + grammar_paper |
| 4 Hz envelope pipeline + 77.7% sub-Nyquist audit | **B owns it** as the physical floor (B §6); A cites B for the aliasing mechanism behind its numeric collapse. Same JSON, two framings, not duplicated |
| Frozen MM-DTAE-LSTM encoder + representation-reuse bound | **A owns** (decoder conditioning, App B); B cites encoder_paper + A's §4.5 probe |
| TF-vs-AR distinction (§3.4) | **A owns**; B must state its entropy↔accuracy analysis is a **TF upper bound** (under AR the digit head is at floor 0.10) or it overclaims deployment recoverability |
| seq2seq baseline | **A owns** (architecture AR-lift); B cites the 0.877≈0.888 TF result as model-independence evidence. Run once |
| Closed-vocab regime + coverage-repair + LOCO 0.213 floor | Shared caveat; each paper re-discloses independently. A owns App A |
| Per-axis sign-entropy numbers (0.38 / 0.08 / 1.04) | **B owns** the derivation/table; **A duplicates the 3 numbers as cited facts** in §6.4 |
| Replicability infra (Zenodo DOI, SHA256, env pins, stats grid) | Shared release; B writes its own short manifest for B-specific scripts, cross-cites A |

---

## 4. Salami defense — the four non-negotiables for B

1. **Distinct contribution.** B = a decoder-agnostic information-theoretic **+ physical LIMIT**,
   generalizable beyond this decoder — *not* relocated Fig 14 / Tables 7–8 (R3 called the 3-regime
   decomposition A's *strongest* contribution, so merely moving it is textbook salami).
2. **Independent empirical apparatus.** Compute entropy on **raw CAM g-code** (zero decoder). This answers
   "stripped of companions, what is independently novel?" with "a source-entropy limit that exists whether or
   not anyone builds a decoder."
3. **Self-contained, no "in-prep" dependencies.** B carries its own corpus/vocab/pipeline appendices, its own
   CAM related-work, its own Fano/Shannon/rate-distortion grounding. Cite A as **published** (sequence A
   first); A cites B as published.
4. **Does not hollow A.** A *retains* the categorical-vs-numeric factorization + exposure-bias mode-collapse
   as the phenomenon it diagnoses, and cites B for the *why*. If B takes the explanation AND leaves A with
   only the observation, A re-triggers R2-M4.

Plus the **honesty bar** all three referees credited: lead with the defensible n=30 r=−0.78, disclose the
pre-patch caveat + fold-1 provenance until recomputed + the closed-vocab/single-platform scope, and frame the
absence of prior G-code-entropy work as documented due-diligence, not asserted novelty.

---

## 5. Execution order

**Track 1 — ship A (no experiments, ~1–2 weeks):**
- A1. Clone `decoder_paper_v2/` → `two_paper_split/decoder_paper/`.
- A2. Apply the §1 cut list; rewrite abstract → two contributions; renumber intro contributions; rewrite
  Conclusion finding-2; insert forward-pointer stubs at each load-bearing hand-off; keep the 3 cited
  sign-entropy numbers in §6.4.
- A3. Strip A's bib (`brown1992englishentropy`, `feder1994relations`, `cover_thomas` → B); update App D to the
  4-way division naming B; resolve the K-spectrum (relocate to B, or keep T3.1-matched only).
- A4. Delete dead `\if0` block + duplicate K-spectrum figure. Recompile; confirm **no dangling `\ref`/`\cite`**
  to moved labels. → **Submit A to MDPI Sensors.**

**Track 2 — build B (runs in parallel; de-risk before drafting):**
- B0. (no-GPU) Write + commit the raw-CAM-text entropy script (§2 item 1).
- B1. (no-GPU) All-folds entropy recompute (item 2); re-render `per_head_opclass_entropy_heatmap`.
- B2. (no-GPU) Fix `per_op_class_v8.json` alignment (item 3); regenerate `entropy_accuracy_all_heads`.
- B3. (~10–15 GPU-h) Patched-encoder 5-fold retrain (item 4); update `digit_entropy.json` float_epsilon_audit
  + bathtub.
- B4. Assemble B's body in `two_paper_split/toolpath_entropy/` from the moved figures/tables + the new
  raw-corpus, 4 Hz-floor, and empirical-confirmation sections; promote App J to the §8 self-demolition.
- B5. Build B's bibliography (IT canon + CAM/toolpath-generation); apply the stats fixes.
- B6. Generality: schedule the Tormach second-corpus entropy run if data is available; else scope B to
  single-corpus explicitly.
- B7. Reciprocal A↔B citations; independent shared-limitation disclosure in both. Recompile; verify every B
  number traces to a regenerated all-folds JSON.

**The main fork to decide:** ship B as an explicitly **single-platform** study soon after A (~3–8 wks), or
hold for the **Tormach second corpus** to claim a general limit (months, gated on the summer campaign).
Recommendation: ship A now; build B to single-platform-submittable; add the second corpus only if it lands in
time, otherwise scope down and register cross-CAM generality as the next step.

---

## 6. Effort summary

| | Effort | Gated by |
|---|---|---|
| **Paper A → submittable** | ~1–2 weeks | Editing only (excision + recompile). No experiments |
| **Paper B → single-platform submittable** | ~3–8 weeks | B0–B2 (days, no GPU) + B3 (½ day compute) + B5 citation scholarship (~1–2 wks, the long writing pole) |
| **Paper B → general "limit" claim** | + months | Tormach second corpus (B6) |
