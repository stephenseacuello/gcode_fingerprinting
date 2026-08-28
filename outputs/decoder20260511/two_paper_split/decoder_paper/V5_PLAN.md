# decoder_paper_v5 — Lean Core, Aggressive Cut (2026-08-11)

**Frame (user, editor-mode):** conference-discipline cutting; "more papers was the plan." v5 = the focused **per-field recoverability** paper (~18–22pp). Security/tamper apparatus → deferred to a companion security paper (motivation kept, detector experiments removed). Appendices → pointers. Figures 16→8. Romesh's rules at maximum: every sentence serves the one argument; cut cut cut.

**The one argument v5 makes:** sensors recover the *categorical* structure of executing G-code (command/axis/sign) but not its *numeric/generative* content; the recoverable signal is a bounded, class-prior-controlled increment that collapses out-of-distribution; the multi-modal stack is densely redundant.

## DEFER → companion security paper (removed from v5, 1-para pointer kept)
- Problem Formulation §Threat Model (5 attack classes, MITRE/NIST framing)
- Discussion §Threat Model Detectability (tamper table, 2 ROC figures, alert-budget, operating points, adaptive adversary, audit-leverage)
- Related Work §Manufacturing Cybersecurity → compress to 2 sentences + companion pointer
- Intro: keep tamper detection as *motivation*; drop SOC/alert/operating-posture detail
- Conclusion finding #4 (tamper) → replaced by companion pointer
- ~26 cross-refs to sec:*-threat-model / tab:tamper → neutralized to "companion security paper"

## DEFER → artifacts / future (removed, artifact-cited)
- Results: Test-Time Sensor-Noise Sensitivity, Calibration, Output-Position Failure, Failure-Mode Analysis, Noise Augmentation → 1 sentence each or gone
- Seq2Seq Baseline → compressed to 3 sentences (scoping result kept, detail deferred)
- No-Numeric decomposition, K-spectrum → compressed
- Appendices: Reproducibility (→ 1 para + repo), Extended Protocols, Statistical Exhibits → removed (inline artifact cites)

## FIGURES kept (8 images, 5 float envs): system_overview; sensor_gcode_overlay; [per_axis + per_digit bathtub] pair; [per_op_class + loco_per_holdout] pair; [sensor_ablation + lomo_modality] pair.
CUT→artifact: encoder_memory_pca, five_fold_spread, confusion_matrix_command, mode_collapse_heatmap, lomo_nested_heatmap, reliability_diagram, failure_case_examples, confusion_sign/digit, tamper_detection_roc, tamper_roc_scatter.

## Procedure
1. (me, scripted) structural surgery on v5 copy: delete deferred sections, add pointers, cut figures, neutralize dangling refs; compile after each batch.
2. (agents, parallel) aggressive prose tightening −20–30% on retained sections, claim-invariant on kept numbers.
3. assemble, compile, verify, deliver v5. Target ≤22pp.

## v5 EXECUTED (2026-08-11): 57 -> 36pp, 0 err, 0 undefined, claim-invariant on all headline numbers
**Structural surgery (me):** deferred both threat-model sections -> companion pointers (-2440w); deleted 5 Results subsecs (noise-sensitivity, calibration, output-position, failure-mode, noise-aug); deleted Reproducibility/Extended-Protocols/Stats appendices -> 1 repro para (-2568w); figures 16->8 (kept: system_overview, sensor_gcode_overlay, [per_axis+bathtub], [per_op+loco], [sensor_ablation+lomo]); neutralized ~25 dangling refs -> audit/*.json citations. Mode-collapse numbers (60-80%/20%/0.20->0.17) moved from cut caption into prose. New ablations_summary_v5.tex (appendix ref neutralized).
**Aggressive tightening (5 parallel agents, all verified claim-invariant):** Intro+Related -25%, Problem+Method -25%, Setup+Results1 -20%, Results2 -25%, Disc+Lim+Concl -26%. Cybersecurity Related subsec -> 2 sentences + companion pointer; FSM/hp-sweep/no-numeric/k-spectrum/seq2seq compressed. Body -24% (~-3950w). Abstract 200w. Bib 76->68 refs (uncited security refs auto-dropped).
**Page map:** Intro p1, Related p3, Problem p5, Method p7, Setup p11, Results p13-24 (12pp, the core+6 figs), Discussion p25, Limitations p27, Conclusion p30, appendices+bib p31-36.
**36pp is the aggressive-but-safe floor.** Reaching the 18-22 target needs CONTENT decisions: defer RQ2 (metadata/target-mode) and/or RQ3 (sensor-modality) ablations to future papers (-4-6pp), cut figures 8->4-5 (-2-3pp), or trim core reported numbers. All cross into "removing results" -> author call.

## v5_conference — TWO-COLUMN IEEE, 10 pages (2026-08-13)
User: "make a decoder_v5_conference that is 10 pages." Chose two-column IEEEtran path.
- **Stage 1 (MDPI single-column cut):** 36pp -> structural surgery -> 30pp -> terse-rewrite workflow (6 regions, hard word budgets, adversarial verify) -> body 11,918 -> 5,066 words, figs 8->5 (2 wide+2 col+pair), tables 6->3, 0 err. Floored at ~16pp because mdpi.cls fixed overhead (42-ref verbose bib ~3.5pp + mandatory journal back-matter); an empty mdpi paper is ~5pp. All 42 refs cited (no dead weight). => single-column cannot reach 10pp with content intact.
- **Stage 2 (reformat to genuine conference format):** IEEEtran.cls/bst V1.8b downloaded from CTAN into latex/ (not installed system-wide). Scripted transform conf/to_ieee.py: MDPI frontmatter->IEEE (\title/\IEEEauthorblock/abstract/IEEEkeywords/\maketitle); back-matter MDPI macros dropped except Data Availability (trimmed) + Acknowledgment (one-liner); \appendix->\appendices; all \cite (46, no \citet/p) work with IEEEtran.bst directly. Floats: system_overview + numeric-tier-pair -> figure* (full width); lomo/loco -> column figure @0.72\linewidth; 3 tables -> table*[!t] _ieee variants (\resizebox\textwidth). Fixed: removed leftover MDPI \let\item\olditem (\olditem undef in IEEEtran); broke 6-term loss eq into aligned (was 75pt overfull); trimmed grammar-mask citation pile 6->4 (dropped synchromesh,lmql — cited nowhere else; claim still supported by picard/outlines/xgrammar/geng2023gcd) => bib 42->40, pulled last 2 refs onto p10.
- **RESULT: decoder_paper_v5_conf_ieee.tex/.pdf = exactly 10 pages, 0 err, 0 undefined, 40 refs, 5 figures, 3 tables.** All 19 core headline numbers verified present in final PDF (0.8875/0.888/0.7811/0.2148/0.5850/0.1038/0.0263/0.213/0.795/0.572/0.904/0.9838/0.9931/0.9888/96.3/77.7/2418/+22/31.8). Reproduces in-repo (latex/ carries IEEEtran.cls+.bst + tables/*_ieee.tex). Page map: Intro p1 | Related/Problem/Method p2 | Setup p4 | Results p5-8 | Discussion p8 | Lim/Concl/Appendices p9 | DataAvail/Ack/bib p10.
- Files in repo latex/: decoder_paper_v5_conf_ieee.{tex,pdf,bbl}, IEEEtran.cls, IEEEtran.bst; tables/{headline_5fold,per_axis_recoverability,sensor_ablation}_ieee.tex. Transform script: scratchpad/conf/to_ieee.py. NOT git-committed.

## Review pass + fix batch (2026-08-13): v5 + v5_conf_ieee
**Panel:** 10-agent workflow (9 referee lenses + verify) — spend limit killed 9/10; the surviving conf-ML referee returned 1 blocker + 6 major + 7 minor; ALL 14 findings verified inline against tex/v2/artifacts (all confirmed or confirmed-with-nuance), plus my carry-over analysis on v5 found 1 NEW v5 blocker (V5-A). Full advisory in session; re-run the full 9-lens panel after the spend-limit reset (v5 writing/CNC/numeric sweeps never ran).
**Fixes applied (both editions unless noted):**
- F1 (conf BLOCKER): per-row vs full-window incoherence — contribution 1 retitled "Per-row and full-window supervised targets", §III-C + Setup now state the headline trains on the FULL-WINDOW variant (per-row = supervision innovation + preferred AR deployment mode). Imported from v5:197/459/611.
- V5-A (v5 BLOCKER): dangling tamper operating points — L555 command-swap clause now cites \texttt{audit/threat\_model\_tamper\_loco.json} (FPR 0.519≈0.52, sign-flip TPR 0.138, feed-edit TPR 0.0 verified); L611 K=10 clause → "alert-budget operating points in the companion security study (in preparation)".
- F2 (conf): SS provenance — "the headline configuration uses scheduled sampling p_max=0.5" (was implying TF default).
- F3: restored the v2:316 bridge "categorical heads read this hidden state but do not consume the autoregressive token stream" in BOTH Methods (lost in v5 tightening).
- F4 (conf): 0.795-vs-0.8875 position-subset reconciliation imported (compressed v5:596); abstract now "(0.795 vs. 0.572 at command-token positions)".
- F5 (conf): contribution 4 dropped "sequence-classifier prior, vocabulary precision" (nowhere in conf; both covered in v5).
- F6: all "companion security study" refs → "(in preparation)"; RQ2/seq2seq "(Companion Study)" → "(Extended Version)" = the journal edition; appendix four-contribution enumeration extended with the security companion + extended-version sentence; dropped "illustrative and prior-dominated" characterization of absent results.
- F7: abstract "Coordinate reconstruction" → "Generative coordinate reconstruction" (both; body was already qualified).
- F8: created audit/nonneural_baselines.json via NEW scripts/analysis/nonneural_baselines_export.py — merges xgboost_baseline_v8.json (HGB; the JSON's "XGBoost" label is stale, script uses sklearn HistGradientBoostingClassifier) + encoder_probe_v8.json, plus a SEEDED MLP command-probe retrain for the missing macro-F1: acc 0.7745 (orig 0.7736 — validity anchor PASS), macro-F1 0.2482 ≈ paper's "≈0.24" CONFIRMED. Fixes the broken release promise in v3/v4/v5/conf with zero tex edits.
- F9: stats-policy sentence no longer claims Mann–Whitney/Wilcoxon "primary" (never used as such); now effect-sizes-first + "Wilcoxon checks accompany paired tests where reported". v5's "Welch or Mann–Whitney undefined" statement kept (legitimate).
- F10: abstract tamper sentence → deferral ("operating points are deferred to a companion security study (in preparation)").
- F11: single-fold-pilot policy scoped to exempt LOCO (finite-population descriptive) + LOCO fold-1 = covariate outlier conditionality clause (both).
- F13: "V1 violations" → "command-state (V1) violations" (both).
- F14: "zero on one fold" → "at or below its class prior on one fold" (artifact: fold-1 delta = −0.016, decoder 0.552 < prior 0.568).
**Page control (conf):** fixes added ~150 words → 11pp; reclaimed via layout-only: system_overview 0.8→0.65\linewidth, lomo/loco 0.72→0.58, numeric pair subfigs 0.44/0.52→0.41/0.48, the 3 _ieee tables \resizebox 1.0→0.85\textwidth, Repro-appendix/Data-Availability boilerplate dedup (rule 2.1).
**FINAL: v5 = 36pp, conf_ieee = 10pp; both 0 errors / 0 undefined / 0 overfull>20pt; all 17 core numbers verified present in both PDFs.** NOT committed.

## Inline certification pass (2026-08-13, second attempt): CLEAN
The 9-lens re-cert workflow died 9/9 on the spend limit (2nd time; ~916k subagent tokens burned before dying).
Ran the full pass INLINE instead: mechanical sweeps (banned-intensifier scan — all hits quantified/negated/technical;
hedge-set symmetry v5-vs-conf incl. emph-tolerant "upper bound" 3=3; zero journal-isms in conf; caption metrics)
+ complete v5 read (748 lines) under the ML/CNC/writing/integrity/numeric lenses + conf regression-check of all
16 fix landings (wrap-tolerant PDF greps; earlier zeros were pdftotext line-wrap artifacts).
**Findings: 3 minors, all fixed:** (1) v5 §per-row called full-window "(ablation)" without saying the headline
trains on it — now states it, matching the fixed conf (xver consistency); (2) v5 contribution 1 same asymmetry —
clause added; (3) v5 L656 misdirected "(Section disc-modality)" pointer on non-AR variants — dropped.
Everything else held: probe/channel/aliasing/confound/LOMO/LOSO/K-spectrum numbers all match artifacts & anchors;
n=5 posture honest (descriptive p, MDE, TOST-inconclusive disclosed); G-code semantics correct; beam-search
contradiction from REWRITE_NOTES is resolved in v5 (beam-3/5 <1pp stated with numbers).
**FINAL: v5 36pp / conf_ieee 10pp, 0 err / 0 undef, core numbers verified. Effectively certified at the
depth one thorough referee + mechanical verification can provide; a true multi-agent 5/5 panel stamp still
requires the spend limit raised (claude.ai/settings/usage).**

## Panel round 3 (2026-08-13/14): 6-review panel findings ALL FIXED (48-edit batch)
Spend limit killed the 9-lens panel twice more (19+14 agents; only 6 reviews survived round 1:
v5-writing, conf-pc, conf-writing, conf-xver, sec-writing, sec-integrity — ALL minor_revision, 0 blockers).
Their ~45 findings were verified inline against artifacts and ALL fixed:
- v5 (17): stale Friedman exact p (0.0098/0.029 -> artifact 0.0018/0.019); dead artifact cites
  (k_spectrum_5fold -> k_spectrum_compare; seq2seq_baseline_5fold -> seq2seq_baseline/fold_* dirs);
  "~1,500 windows/fold" -> ~550 (metadata 303/110/132); "0.572 unconditional modal" mislabel -> train- vs
  test-derived class-modal framing (abstract/intro/discussion/conclusion, both editions); nonumeric caption
  p<0.02 -> p=0.024 + attribution aligned with the T3.1 downgrade; numeric-ceiling mechanism rescoped to
  per-row deployment mode (2 sites); power-analysis supports basis-labeled; LOCO type drop -9 -> -10pp;
  +/-0.022~pp unit slip; "unmeasured" reuse -> "bounded"; T=1.77 dangling clause dropped; FSM counts ->
  grammar_violation_audit artifact + "affecting 31%"; 2 overlong sentences split; conclusion targets
  "per-row and full-window"; affiliation harmonized; motion-block taxonomy; non-neural 0.232/0.248 + fold-1
  disclosure.
- conf (14): head-mechanism bridge harmonized at the metrics section; fold-1 covariate support restored in
  Limitations; null-taxonomy fix; "sits at its own prior" -> "at or below"; leakage-cap phrasing; RQ prefixes
  stripped (orphaned numbering); modality-name mapping; V8/LOSO/6x6 glosses; G2-0.82 + per-op-class ->
  extended-version pointers; reuse bounded; motion blocks; non-neural range + disclosure.
- security_companion (12): "near-chance" overstatement rescoped (prior-driven + LOCO collapse) in
  abstract/results/conclusion; K=10 TPR-side rho_tamper=0.72 assumption DISCLOSED (artifact-verified);
  replay argument restored inline (labeled interpretation); metadata-rows artifact
  threat_model_tamper_AR_5fold_FSM_shortcuts.json cited in caption + Data Availability; population-s.d.
  label; FPR resolution 1/549=1.8e-3 (honest-pair denominator); (iii)-(iv)/(v) pointer; mitigations
  forward-pointer; contribution-3 scope; feed-edit vs not-evaluable reconciliation; NIST 10^2 as labeled
  assumption; rho=0.6 caption + artifact-roadmap phrasing; abstract "with metadata" qualifier; affiliation.
- tables (5): nonumeric caption; per_axis x2 "prior draft" revision-leak; sensor_ablation x2 stale
  future-work footnote -> LOMO section.
Page control: conf 11pp spill reclaimed via layout + NEW decoder_references_ieee.bib (author lists of the
two software mega-entries truncated to "and others", IEEE-conventional; v5's MDPI bib untouched).
**FINAL: v5 36pp / conf_ieee 10pp / security_companion 12pp; all 0 err / 0 undef; abstracts 199/199/200
words; core numbers verified in all three PDFs.** Panel lenses still never run: v5-ml, v5-cnc, sec-security,
sec-ml + independent verify + B-E audits (three spend-limit kills; retry when the limit genuinely clears).

## Panel round 4 (2026-08-14): the four never-run lenses — DEEPEST FINDINGS OF THE PROCESS, all verified + fixed
All 4 reviews completed (v5-ml, v5-cnc, sec-security, sec-ml; verifiers + B-E audits died on session limit).
Every serious finding INDEPENDENTLY VERIFIED before fixing (~45 more edits, all 3 papers + 3 table files):
**F1 BLOCKER (code-verified): "TF/AR-invariant categorical heads" was never a measurement.** evaluate() in
run_decoder_quick_test.py computes command/type/param/sign metrics from the TEACHER-FORCED forward pass in
BOTH regimes (only token/seq/numeric switch to the beam decode); ar_aggregate.json fold-1 command is
byte-identical to 16 digits between beam-0 and beam-1 (0.7762271414821944). The heads read the decoder hidden
state = a function of the conditioning token stream. FIX (all 3 papers + both headline-table captions):
"reference-conditioned (teacher-forced) evaluation ... free-running autoregressive categorical accuracy is
not separately measured"; seq2seq +77pp gap reframed as conditioning-regime difference; deployment claims
re-anchored on audit-mode conditioning against the controller log.
**F2 MAJOR (reproduced exactly): the +22pp increment's null is history-mismatched.** The 0.795 is TF-measured
with ground-truth within-window history; the 0.572/0.576 nulls are history-free. A sensor-free per-class
Markov-1 null WITH history = 0.6619±0.0042 (my reproduction == referee's to 3 decimals) -> history-matched
increment ≈+13pp. NEW artifact audit/markov_command_null.json + scripts/analysis/markov_command_null.py;
abstract/intro/discussion/conclusion now state both bounds; UPPER-bound mechanisms extended with
"ground-truth-history conditioning"; target-mode 0.499->0.888 contrast annotated with the history component.
**Security overhaul:** "ROC-AUC 0.80/0.86 [Hanley-McNeil CI]" was mechanically BA=(1+TPR-FPR)/2 on a BINARY
single-threshold flag -> all rows now BA, CI dropped, no-swept-ROC disclosed; internally-contradicted ROC
subfigure (fig sign AUC 0.5 vs table 0.68; non-FSM/broken-detokenizer provenance) REMOVED; unsupported
2e-3/TPR-0.2-0.3 deployment point replaced with the single measured point + scored-detector future-work;
alert-budget UNIT FIX (FPR is per-WINDOW: ~1,800 windows/shift @16s stride -> ~400 alerts/shift/machine
(LOCO ~930), several-fold over budget per machine — was "1.15e5 row decisions / 2.5e4 alerts / 2-3 orders");
footnote premise inverted (detector is FIRST-command-only per threat_model_tamper_injection.py -> 0.811
first-position modal IS the matched null; modal-substitution control operationalizes it); MITRE (iii)/(iv)
swapped (replay->T0855, log-misreport->T0856); metadata-manipulation adversary added as class (iv);
"invisible by construction" scoped to fine-grained in-envelope edits (gross feed-edit TPR 0.22-0.53 in-dist);
Y-and-X prior vs Z sensor fix; fig-2b caption matched to shipped per-fold bars; class_conditional_modal_5fold
+ markov artifacts added to Data Availability.
**v5-cnc:** 150025 gloss corrected to observed F13.3/F40.0 vs F7.3/F22.0 + constant S12000 (was "150 mm/min
@ 25,000 RPM" — contradicted by corpus); bucket formula floor->round (worked example requires it);
value-recovery vs change-detection conflation fixed (2 sites); y_bed__4/spindle2 recommendation demoted to
illustrative non-significant ordering; single-pole anti-aliasing acknowledged; feed range 7.3-40 in/min (was
"~10-100"); n=91-of-120 per-file coverage disclosed; abstract mode-collapse/aliasing mechanisms separated.
**v5-ml minors:** jackknife claim corrected (mean stays >=+0.18 after removing any two folds); LOCO
Welch/MWU artifact acknowledged as descriptive corroboration; selection-bias disclosure unified
(fold-1-composite-best, consulted fold-1 test); 31.8/beam/latency provenance status disclosed; T=1.77 pruned
earlier; nonumeric caption p=0.024 + component-bounding note.
NOT FIXED (author adjudication, pre-existing memory flag): per-axis sign entropies 0.38/0.08/1.04 match
neither released entropy artifact basis (0.35/0.06/0.99 binary; 0.23/0.11/1.10 3-way) — must be reconciled
JOINTLY with Paper B which prints the same trio.
**FINAL: v5 36pp / conf 10pp / security 12pp; all 0 err / 0 undef; abstracts 198/198/199w; corrected claims
verified present + stale claims verified absent in all three PDFs. IEEE bib now et-al-truncates all >3-author
entries (conference convention; MDPI bib untouched). STILL PENDING: adversarial verifiers + B-E sizing audits
(4 spend/session-limit kills); optional gold fix = actually measure free-running AR categorical accuracy
(feed generated streams to the heads, 5 checkpoints, GPU).**

## Round 5 (2026-08-14): regression check CLEAN-up + B-E audits + THE MEASUREMENT
**Regression panel (7 agents, all completed):** found the round-4 batch had missed the TABLE LAYER — the
retracted invariance claim survived verbatim in ablations_summary_v5 caption (BLOCKER) + both headline-table
footnotes ("categorical signal is robust to autoregressive deployment"); plus 2 splice artifacts of my own
edits (abstract non-sequitur "three heads reach command accuracy"; "+22pp over that null" antecedent broken
by the inserted Markov sentence) and a genuine pre-existing falsehood ("categorical-head accuracy is
unaffected by target mode" vs the paper's own 0.499-vs-0.888). ALL FIXED (~25 more edits): tables now say
single reference-conditioned value with AR column marked '--'; abstract sentence repaired; antecedent named
("over the class-prior null"); when-to-use states the 0.499-vs-0.888 per-row cost honestly; stale ROC/AUC/
TPR/ECE/MCE/TCB/NLL abbrevs pruned; security: 4 adversary classes counted+titled, mitigation (b)/(c)
de-ROC'd, K-row->K-window unit sweep, Foundation gets the Markov bound + test-derived-null naming,
detectability map gets the reference-conditioned qualifier; stale decoder_paper_v5_conference.tex marked
SUPERSEDED. All three papers recompile 0 err/0 undef (36pp/10pp/12pp; abstracts 194/194/199).
**B-E sizing audits (finally ran):** B toolpath_entropy = moderate-rewrite (~1 day; em-dash saturation,
'genuine(ly)' x10, draft-history narration in submission text, 6x repetition of the structural-zero claim);
C grammar = moderate-rewrite (~1 day; 450-WORD ABSTRACT vs MDPI 200, mega-sentences, 'unsurprisingly,
enormous', BPE story told 5-6x); D toolpath_signature = moderate-rewrite (~1 day; 'dramatically' x4,
headline blocks repeated 5x, editorial self-ranking 'the paper's strongest result', internal framing
contradiction re trivial separability); E anomaly_detection = LIGHT-PASS (4-6h; abstract re-voice + ~50-70
sentence edits; overclaim '"any" discrepancy' vs own A3 0.621).
**THE MEASUREMENT (in progress):** --ar_categorical flag added to run_decoder_quick_test.py (backward-
compatible; categorical heads scored from a 2nd forward pass conditioned on the GENERATED stream);
launcher scripts/experiments/ar_categorical_freerunning.sh; fold-1 RESULT: free-running command 0.392 /
type 0.504 / param-type 0.495 (vs reference-conditioned 0.776/0.98/0.99 on the same fold) with token/seq
reproducing the released AR values exactly (validity anchor PASS) -> the categorical heads COLLAPSE under
free-running conditioning; the round-4 reframing was right and will upgrade from "unmeasured" to measured
numbers when folds 2-5 land.

## MEASUREMENT COMPLETE + INTEGRATED (2026-08-14): free-running categorical accuracy
5-fold result (audit/ar_categorical_freerunning.json; greedy AR + FSM on released checkpoints, generated-
stream conditioning): **command 0.483±0.049** (vs 0.8875 reference-conditioned), type 0.551±0.087 (vs
0.9838), param-type 0.533±0.046 (vs 0.9931). Validity anchors: sequence 0.0263±0.0185 matches released
EXACTLY; token 0.212 ≈ released AR 0.215. The old "TF/AR-invariance" is now MEASURED FALSE — the heads
collapse free-running into the same ~0.48 band as the modality-zeroing collapse (garbage conditioning ≈
garbage sensors). NEW positive fact integrated: free-running headline command (0.483) still exceeds the
from-scratch baseline's AR-derived command (0.122) by +36pp, partially rehabilitating the architecture claim
with a like-for-like measured comparison. All "unmeasured" placeholders in v5/conf/security + both headline
footnotes + ablations footnote replaced with the measured numbers. FINAL: v5 36pp / conf 10pp / sec 12pp,
0 err/0 undef, abstracts 197/197/199, zero 'unmeasured' remaining. The decoder-thread science is now fully
measured, honestly framed, and artifact-backed end to end. NEXT FRONT: B-E rewrites (audit verdicts: E
light-pass 4-6h first, then B/C/D moderate ~1 day each).

## CAPSTONE: cross-paper consistency review + fix cascade (2026-08-25)
9-agent cross-paper review of the 7-doc bundle (A_v5, A_conf_ieee, F_sec, B, C, D, E) found the bundle
contradicted itself on corpus physical ground truth. The ENCODER PAPER (outputs/experiments_2026_02_25/
paper/sensors_v4.tex) + CAM headers (data_clean/face.gcode: BANTAM TOOLS EXPLORER, G20 inch, M3 S12000)
adjudicated: standard=air-cuts, 150025=active cuts in UHMW-PE @0.635mm, damage=damaged-SPINDLE air-cuts
(drive band removed -> runout); tool 6.35mm 2-flute HSS; ~400Hz tooth-passing @ commanded 12k RPM;
d_model=256; encoder 96.3% test; f98 = drop proximity+pressure (repo exp16 confirms); corpus is G20 INCH.
FIX CASCADE (all applied, all verified present):
- A_v5: corpus 3-conditions account (L92/334/336/636), slot-5->slot-4 retraction w/ pointer to B,
  finest-digit = tokenizer-retained (x2), sign-entropy 3-way basis tag, machinable-wax remnant (L636).
- A_conf: corpus fixes mirrored (L58/L174), finest-digit x2.
- F_sec: corpus line, ~60 rows/window, facing-overruns only.
- B: L310 invariance -> reference-conditioned + measured 0.483; seq2seq table AR Main cells -> MEASURED
  0.483/0.533, deltas +0.361/+0.319; caption +36/+32pp story; footnote 48%/89% framing; tamper pointer
  -> security companion (in preparation).
- A's own seq2seq table (input by SUPP): same row/caption/footnote fixes ("non-autoregressive classifier
  on encoder memory, TF/AR-invariant by construction" claim REFUTED by the 0.888->0.483 measurement and
  removed); supp §(2) narration rewritten to the measured story.
- E: class table 16/15/19/19/20/17 + 4/5/5 (sums 120), tool 1/4" HSS, 400Hz@12k, f98 def, encoder 96.3%,
  d_model 256 cascade (4 sites), six-identical-boards account, 09 attribution ("earlier analytical
  treatment... companion has since dropped that framing"), 07 V8 reference-conditioned caption tag,
  bib decoder title -> v5 title.
- C: MPU-9250/50Hz -> Nano 33 BLE Sense/LSM9DS1/4Hz; nine classes / three operating conditions;
  encoder_paper cite added (bib entry + 2 cites); FULL mm->inch cascade (precision table + caption G20
  note, worked example 0.1755 in, 0.00025in/6.4um within p/2 bound, Experiment E Hausdorff table+prose
  in inches, centimetre-scale phrase, 1277 recap); TF labels on downstream 94.7/84.4 (caption + §).
- D: damaged-TOOL -> damaged-SPINDLE AIR-CUT correction across abstract/intro/results/discussion/
  conclusion; conflation (spindle condition x cutting engagement) flagged at every interpretation site;
  "validates tool-condition monitoring" claims withdrawn; clean air-vs-air comparison named as future work.
- Headline tables v3+ieee: stale "not measured" footnote clause removed (self-contradiction fixed).
- Bibs: stale grammar title -> "Formalizing G-Code for Language Models..." in 4 bibs; stale decoder title
  -> v5 title in E's bib.
COMPILES (all 0 err / 0 undef refs): v5 36pp, conf 10pp, supp 63pp (needs decoder_paper_v2.aux present
for xr-hyper; rebuilt), B 28pp, C 38pp, D 43pp, E 33pp, F 12pp.
Author-confirmation items -> two_paper_split/AUTHOR_DOSSIER_20260825.md (machine model/specs, sensor
link, 8.6M vs 5.1M, 79,563 vs 79,345, D clean air-vs-air follow-up, C TF-label sanity check, DOIs).
Legacy editions (v2/v3/v4, old MDPI conference, v7_legacy tables) intentionally NOT corrected.

## CAPSTONE VERIFICATION ROUND (2026-08-25, same day)
8-agent adversarial verify pass over the fix cascade found 33 real defects; ALL FIXED:
- 8.6M encoder count REFUTED (checkpoint measures 27.5M total / paper-citable 5.1M active) and
  removed from v5/conf/supp/appendix; totals 39.9M->36.4M; VRAM point estimate -> <500MB bound.
- B's per_head_entropy table was STILL asserting AR=TF with starred categorical values -> AR column
  now carries measured 0.551/0.483/0.533, sign '--', dagger footnote; caption+comment rewritten.
- Supp residue: TF-AR-invariant claims (L161/L574), machinable-wax x3 (L1060/L1191), unqualified
  0.89 (L1159), algorithm comment "do not consume g-hat" -> all corrected; p_F mm/min -> in/min.
- ieee bib: 4 author-field splice artifacts from the et-al truncation (bibtex was silently dropping
  title/venue/year on vaswani/tsanousa/pedregosa/focal_loss) -> repaired, bibtex 0 errors.
- C: 2nd mm site (R p=0.0001), Experiment-B narration (0.25um "exactly half" -> 0.00025in~p/4,
  within p/2), table_b caption mm->in, G21 listing -> G20, 50Hz latency -> hypothetical framing,
  2nd feed-rate-settings site + Heaps' x2 -> operating conditions, Sense->Sense Lite, abstract TF
  label, Desktop CNC -> generic lowercase, Range-column clamp note.
- E: "perfect classification" -> 96.3%, channel arithmetic 18-24 -> 17/board (102+8=110) x2,
  damage "(5 files each)" -> 4-5, machine name genericized x3, appendix 140,000-bin illustration
  -> 5,500 bins at 0.001in, s_hybrid A4 routing S_rank->S_mean (matches results, x2 incl. hybrid-max),
  6-and-10-classes clarified, Tool damage -> Spindle damage rows + LOCO footnote; PLUS my own
  sweep: 21 delta-mm attack-magnitude labels -> inches (generator adds delta to native inch values;
  scripts/anomaly/generate_attacks.py mislabels mm), impossible X-125.45 example -> X3.2750.
- F: finest-digit CAM->tokenizer-retains fix (was missed in F), feed-rate/spindle-speed evaluability
  split (F evaluable at thin support), 9 [BA] row tags + duplicate BA formula removed.
- A_v5: damaged-tool classes -> damaged-spindle; 55-air-cuts -> 69 (55 standard + 14 damaged-spindle,
  51 engage material); headline table comments updated (both editions).
- Conf regression: honest bib entries pushed it to 11pp -> caption vspace trims + 2 concision edits
  -> back to 10pp.
FINAL COMPILES (0 err / 0 undef): v5 36pp, conf 10pp, supp 63pp, B 28pp, C 38pp, D 43pp, E 33pp,
F 12pp. Dossier item 3 (8.6M) resolved-in-repo; dossier updated.

## FIGURE-LEVEL RESIDUE ROUND (2026-08-25, after PDF-text sweep)
pdftotext sweep of all 8 rendered PDFs (figures included) caught what tex-level greps cannot:
- A's system_overview.pdf figure had "8.6M params, FROZEN" + "~39.9M total parameters" baked in.
  Source found (outputs/decoder20260511/decoder_arch.svg, exact 0.75 px->pt provenance); patched svg
  (5.1M active / 36.4M), regenerated via cairosvg (1200x1335pt preserved), visually verified, installed.
- E's architecture_overview.pdf had "z in R^{L x 128}" (glyph-encoded, unpatachable; generator lost).
  REDRAWN as scripts/anomaly/_gen_architecture_fig.py (new maintainable source), content-identical
  with L x 256; visually verified, installed.
- E's roc_curves.pdf + score_distributions.pdf titles/labels said "delta X = 0.1 mm" -> patched
  _gen_paper_figures.py (3 sites), regenerated from cached attacks+logits, installed.
- E's detection_vs_delta.pdf x-axis said "delta (mm)" -> patched run_exp12_graded_injection.py,
  regenerated from cached logits, installed.
- E intro "shifting a coordinate by 0.1~mm" -> 0.1~in (the tilde form dodged the earlier sweep).
- Script hygiene: generate_attacks.py JSON keys/log strings + exp12 log label mm -> in.
False positives verified OK: "damaged tooling" in F/Supp = attack consequence, not corpus damage;
"+76.6 pp" mentions = the corrected conflation framing; C's G21 hits = generic RS-274D definitions;
E "867" = appendix token ids. FINAL: all 8 compile 0 err/0 undef (36/10/63/28/38/43/33/12pp); PDF-level
sweeps clean for every refuted phrase.

## RE-CERTIFICATION COHERENCE ROUND (2026-08-28)
Deterministic checks: all 7 abstracts <=200 words (D exactly 200); every session-introduced number in ONE
canonical form bundle-wide (0.483±0.049 x11, 0.533±0.046 x7, 0.551±0.087 x4; gaps only +0.361/+0.319);
air-cut partition (55+14=69 air / 51 active) consistent everywhere.
7-agent referee coherence review of the changed material found 40 defects (argument dangles, orphaned
premises, scope drift) — 39 applied, 1 moot. Highlights:
- B:310 "closing the numeric gap requires... not changing the recoverer" was refuted by its own corrected
  paragraph (18.4pp recoverer effect) -> rescoped; TF qualifiers added to B's intro/contribution/conclusion
  convergence claims; 0.483 de-pluralized to command-head; tamper-section duplicate disclaimer removed;
  seq2seq caption provenance completed (token/seq/numeric-AR = released; numeric-TF = patched).
- Supp: security bullets' unqualified 0.8875/0.9931 got reference-conditioned qualifiers (the last
  security-context site); §(2) heading retitled to the conflation framing; b=round(|v|*10^{1/p_a}) formula
  typo -> round(|v|/p_a); algorithm gained explicit H definition; "Five components" count dropped;
  raw (sec:...) label -> \ref.
- A_v5: abstract ellipsis + "where"-attachment fixed; +77.9pp param-type operands now quoted (0.214/0.993);
  283/216 mechanism-sentence dedup; L636 sentence split; macro-recall promise scoped to precision+F1
  (both headline tables); both seq2seq footnotes: "few pp" -> 16.9/18.4pp, "narrows" -> "+77pp to +36pp".
- D: intro (the one unflagged enumeration site) got contrast class + coupling caveat; abstract/intro
  "same programs" -> "same strategies"; discussion "both contribute" -> "confounded, cannot be attributed
  to spindle condition alone"; dataset summary now documents the 14 damage runs' provenance; related-work
  "tool-condition classification" hedged; 61% importance given its top-20 basis; deconfounded air-vs-air
  comparison added as future-work item (4).
- E: THE DELTA GRID DEFINITION (04:67) was still in mm -> in + travel-limit caveat; 07:125 50mm -> in;
  calibration caption contradicted its own table (degrades -> marginal reduction, V8-only degradation);
  08:64 "within active-cutting"/"standard and damage" -> active-vs-damage transfer only; 03:19 "across
  all 9 classes" de-implied; 03:26 garbled list + dup ending fixed; 07:211 head-vocab claim corrected;
  appendix X-125.45 examples (x2) -> X3.2750; conclusion+intro calibration claims V8-qualified.
- C: WordPiece "2.6um" was the last mm-era gloss (0.0026 in = 64um) -> fixed; abstract TF label rescoped
  over both numbers; contribution bullet TF-labeled.
- F: threat-model pointer narrowed (spindle-speed not evaluable); "V8" designator glossed; facing
  garden-path fixed; audit/ path prefix; staccato merge.
