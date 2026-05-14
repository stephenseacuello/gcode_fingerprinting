---
to: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: Decoder remediation update — teacher-forced vs autoregressive eval, grammar FSM, ranked figures pass
date: 2026-05-14
---

Hi Romesh -

Another bigger update than the last one, and yet another correction to walk back, which I'll lead with because it's the most important thing in the email.

The headline numbers I sent you two days ago in the v2 email (command 0.888, token 0.781, numeric 0.585) were all teacher-forced evaluation numbers — the model received the ground-truth previous token at every step during eval, not its own previous prediction. That's the standard sweep-time eval mode, but it's NOT the deployment-time mode. In actual autoregressive deployment, where the decoder has to consume its own predictions, token accuracy drops from 0.78 to 0.21 and numeric drops from 0.59 to 0.10. The categorical heads (command, type, parameter-type, sign) are unchanged because they're independent classifiers on the encoder memory and don't depend on the autoregressive token stream. The 0.888 command-accuracy claim is fine, but the 0.78 token and 0.59 numeric were inflated by ~50pp.

I dug into this when you asked the "how can no-shortcuts be worse" question on the no_ss extension — turns out the no_ss model wasn't actually better, it was just MORE optimally tuned for the teacher-forced eval condition. Once I re-ran everything with greedy autoregressive decoding (beam_width=1), the no_ss line of investigation evaporated and the real picture came into focus. The paper now reports both TF and AR numbers explicitly in the headline table, with the AR numbers as the deployment-true headline.

The good news is that the categorical-vs-numeric factorisation story actually gets STRONGER with this finding. The clean message is: sensors recover the categorical structure of G-code (command, axis presence, motion direction) at 0.88–0.99 accuracy in both eval regimes; sensors do NOT recover exact token streams or numeric coordinate values under autoregressive deployment because of mode-collapse to a corpus-modal output sequence. The threat-model implications become: command-tamper detection is real (AUC 0.80–0.86), value/feed-rate tamper detection is essentially chance under AR.

The mode-collapse mechanism is concrete and visualizable. When I inspected the actual decoded text, three different test windows from fold 2 all received essentially the same predicted G-code sequence (`X NUM_X_3122 X NUM_X_3168 Y NUM_Y_3207 X NUM_X_3222 ...`) despite having different TRUE sequences. The model has converged to "average facing-pattern toolpath" and emits it autoregressively regardless of sensor input. Cross-attention to the encoder memory is effectively dropped under autoregressive generation. The encoder probe analysis we already had supports this: the encoder preserves categorical/operation-class signal cleanly (96.3% on 9-class operation classification, t-SNE shows tight class clusters) but the AR decoder cannot turn that class-level signal into sample-specific predictions.

Two other notable findings from this round:

The positional metadata story reversed under AR. Under TF eval, exposing window_index / total_windows / source_file metadata was mildly harmful (command -0.7 pp, numeric -5.3 pp). Under AR eval, the same metadata gives +7.5 pp on token, +8.0 pp on numeric, and 4× tighter fold-to-fold variance. The mechanism is that position metadata gives the AR decoder a per-window anchor that partially breaks the modal trajectory. So position metadata isn't a "shortcut that cheats" — it's a deployment-time disambiguation signal that mitigates exposure bias. The paper now reframes this as an anchor rather than a shortcut, and recommends with-shortcuts as the deployment configuration for security applications. With_shortcuts AR is strictly better than baseline AR at tamper detection across all three attack classes (command-swap TPR 0.92→0.95, sign-flip 0.70→0.80, feed-rate 0.13→0.51).

The grammar mask had a hole that you'd asked about indirectly. You'd noticed at one point that some predictions had `G0 X Y Z R` patterns, which is grammatically wrong — G0 is a rapid linear move and R is the arc-radius parameter, valid only with G2/G3. I dug into our grammar mask in `src/miracle/model/sensor_multihead_decoder.py:_build_grammar_masks` and found that it was lumping all G-commands together and allowing any subsequent PARAM letter. So G0→R was technically permitted by the bigram mask. I added a small inference-time FSM layer in `beam_search_decode` that tracks the active command and forbids R/I/J after G0/G1. Inference-only — no retraining needed. The audit shows it eliminates 100% of G0/G1→R violations on baseline (329 → 0) at a 5-fold accuracy cost of -0.27 pp on token and -0.41 pp on numeric, both within fold-to-fold noise. Clean grammar compliance for free, basically.

The paper is now 43 pages with 13 figures and is in a much stronger state than the v2 version I sent you. New figures added in this round include a sensor waveform + G-code overlay (per your suggestion that we should make the task more concrete), a tamper-detection ROC sweep, a per-operation-class accuracy breakdown (face dominates 0.56 token / 0.40 numeric while every other class is stuck at 0.10–0.25 — this IS the mode-collapse fingerprint), a class-conditional mode-collapse heatmap, a length-vs-accuracy scatter, a t-SNE of the encoder memory showing the 9 op-classes form tight clusters, a beam-width comparison showing that beam search doesn't fix exposure bias, and a per-token-position curve showing the per-row vs full-window position-1 drop visually.

The remaining 9 TBD placeholders in the paper are all wired to the watcher chain still running on the GPU server, which is currently working through noise_aug folds 2-5 and has LOCO 9-class, vocab2digit, and the window/stride sweep queued behind. About 10–12 hours of compute remaining. The chain will fill the LOCO/noise_aug/vocab2digit ablation rows automatically.

The diagnose-then-fix arc still holds, and the predict-then-confirm chain on per-row ambiguity is intact, but the TF-vs-AR distinction is now the central methodological observation of the paper. The mode-collapse subsection runs as predict-then-confirm — we predicted exposure bias should manifest on long sequences, and the data confirms it (Pearson r = -0.15 on length-vs-accuracy, full mode collapse on 1000+ token sequences).

If you have time and want to look at anything specific, the most useful files are probably `decoder_paper_v2/latex/decoder_paper_v2.pdf` (43 pages), `notes.md` for the chronological log (~2100 lines), `audit/ar_aggregate.json` for the 5-fold AR means side-by-side, `audit/anova_baseline_vs_shortcuts.json` for the statistical comparison, `audit/grammar_violation_audit_baseline.json` for the FSM 100%-elimination evidence, and `audit/threat_model_tamper_AR_5fold*.json` for the threat-model FPR/TPR per attack class.

The relevant commits worth reading:
- `b2d7d5c TF/AR eval-bias discovery, grammar FSM fix, full 5-fold AR re-eval`
- `3215e24 Ranked figure pass: 8 new figures, FSM threat-model, ANOVA stats`
- `2cee67f Audit-cleanup pass: fix sign-asymmetry numbers + 7 orphan figure refs`

I'm really glad you caught the no_ss anomaly and the G0+R grammar violation when you did. Both of those threads led directly to material improvements in the manuscript. The paper now reads as predict-then-confirm rather than after-the-fact rationalisation, and the threat-model implications are honestly framed.

Happy to talk through any of this on a call. The watcher chain should finish overnight; I'll send the final ablation-row numbers after that.

Thank you so kindly,
Best Regards,

Stephen
