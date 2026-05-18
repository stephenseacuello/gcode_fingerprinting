---
to: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>, Manbir Singh Sodhi <sodhi@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: G-Code Decoder Follow-Up
date: 2026-05-18
---

Hi Romesh / Dr. Sodhi -

This is the follow-up from the April 28 meeting on the G-Code Decoder. I worked through the code and the model with each of your points as a checklist.

Overall, the silent failure was real, and fixing it forced the paper out of a shaky full-reconstruction claim into a precise, honestly-bounded result — the categorical structure of a G-code command (which command, which axis, which direction) is recoverable from the sensors and survives real deployment, while exact coordinates, feed rate, and spindle are not, with three things we tried to push past that documented as clean negatives.

1. The silent 16-token failure is confirmed. Your diagnosis held up exactly. The audit reproduced it: max_token_length defaulted to 16 and was applied across the whole sensor window instead of per G-code line, so the training targets were silently truncated — the model was supervised on the front of each window and the rest was thrown away, just as you showed on the screen share. That one bug is the thread everything else hangs from.

2. Fixing it forced the per-row vs full-window question. Repairing the truncation meant deciding how a target is emitted at all, which is exactly the choice you raised. I built both: per-row (one training pair per G-code line that fires in the window) and a corrected full-window mode (the whole multi-line sequence), and compared them head-to-head in the paper. Full-window recovers command identity at 0.888; per-row is materially worse on the autoregressive heads, because every row in a 64-second window shares one encoder summary, so the decoder genuinely cannot tell which of ~60 rows it is being asked to emit — the within-window ambiguity we predicted.

3. With truncation fixed, is it even using the sensors? The test we raised was the right one: if a simple lookup beats the model, the model is taking a shortcut. So I built that lookup explicitly — an (operation-type × window-index) table with zero sensor input predicts the held-out G-code line at 0.80 test accuracy. That is the shortcut, measured. The window-position and file-identifier metadata are now removed from the default model, and the metadata-versus-sensor ANOVA confirms the categorical heads do not need them.

4. The shortcut's root cause is the one you named in the meeting: repetition. Your point — that ~20 repeated experiments per operation push the same G-code through the pipeline — is correct, and I quantified its effect: 97.6% of the distinct test G-code lines (99.0% of test line-instances) appear verbatim in that fold's training set. The 5-fold headline numbers are therefore a closed-vocabulary classification result, and the paper now says so up front. The honest open-vocabulary test — holding out an entire operation class so the model has never seen that toolpath family — collapses command accuracy from 0.888 to 0.213 (significant, p < 1e-5). That collapse is the shortcut, isolated and measured. I also implemented your noise-augmentation idea across 5 folds; it is a clean negative result (command ~0.86 → ~0.27), which we report rather than bury.

5. So we stopped claiming full reconstruction and characterized what actually survives. This is the pivot you both converged on in the meeting, and it is now the paper's thesis: a clean categorical-versus-numeric split, reported under both teacher-forced sweep evaluation (TF) and deployment-true autoregressive decoding (AR) — the distinction the whole reframing rests on.

| Prediction head | TF (sweep) | AR (deployment) | Recoverable? |
|---|---|---|---|
| Type (SPECIAL/CMD/PARAM/NUM) | 0.984 ± 0.009 | 0.984 (invariant) | yes |
| Parameter-type / axis identity | 0.993 ± 0.004 | 0.993 (invariant) | yes |
| Motion-sign / direction | 0.989 ± 0.006 | 0.989 (invariant) | yes |
| Motion command (G0/G1/G2/G3…) | 0.888 ± 0.056 | 0.888 (invariant) | yes |
| Token stream (full line) | 0.781 ± 0.022 | 0.215 ± 0.069 | no (collapses) |
| Numeric digit value | 0.585 ± 0.033 | 0.104 ± 0.032 | no (collapses) |
| Exact full sequence | 0.026 | ~0.02 | no |

The four categorical heads are identical under TF and AR because they read the encoder memory directly and never touch the generated token stream — that invariance is what makes the recoverable fields trustworthy at deployment. The token and numeric heads look strong at sweep time and then collapse under real autoregressive decoding. Drilling into your specific question, Romesh — which fields of a command survive:

| Axis / field | Axis present? | Direction (sign) | Coordinate value (MAE) |
|---|---|---|---|
| X | 1.000 | 0.942 | 0.28 (weak) |
| Y | 1.000 | 0.997 | 0.18 (weak) |
| Z | 1.000 | 0.938 | 0.042 (good) |
| R (arc radius) | 0.994 | 1.000 | 0.018 (good) |
| F (feed rate) | thin support | — | not recoverable |
| S (spindle) / I,J | ~zero support | — | not in this corpus |

So: which axis is moving, in which direction, under which command — that is recoverable and survives deployment; exact coordinate values, feed rate, and spindle are not, on the present 4 Hz Bantam stack, and the paper says exactly why (the autoregressive sequence head, not the encoder). On not being fooled by aggregates, Romesh: macro-F1 tracks the accuracies rather than hiding a class-imbalance problem (command macro-F1 0.879 against accuracy 0.888 — the small gap is just the G53/M30 long tail), and the full per-class precision/recall/F1, confusion matrices, and bootstrap CIs are in the manuscript.

Two meeting ideas we tried that did not pan out — both reported as clean negatives, because each sharpens the same conclusion rather than weakening it:

6. Sensor importance — an honest reversal. Dr. Sodhi, in the meeting we leaned on the prior belief that the gyroscope mattered most. The proper 5-fold sensor ablation does not support a ranking at all: zeroing any single modality collapses command accuracy by a uniform ~40-44 pp, and the ANOVA across the seven modalities is not significant (p = 0.998). The frozen encoder fuses the modalities too tightly for inference-time zeroing to attribute signal to any one of them; answering "which sensors can we drop" properly requires retraining the encoder without each modality, which is summer / next-paper work.

7. The pattern-aware decoder. Dr. Sodhi, I implemented your suggestion to inform the decoder of repeated toolpath patterns — a sequence-classifier head that biases the token distribution toward a predicted whole line. On the corrected pipeline it is a negative result (command 0.776 → 0.536): it commits to a whole-line guess too early and compounds the exposure-bias problem. Usefully, it reinforces that the bottleneck is the sequence head, not a missing line prior.

8. The summer DOE is deliberately still open. The paper only forward-references it — one appendix sentence and a few roadmap pointers — framed purely as future work, which is the right call since we have not actually designed it together. I have a one-page strawman (factors, a fractional sample, a pilot → main → extension path tied to Tim's hire and the Tormach arrival) that I'll bring as a discussion starter, not a finished plan. I deliberately left run-count numbers out of the paper so we set the real target together.

Happy to walk through any of this during our weekly Tuesday 230PM meeting (and/or any other time that works for you both), and to fold the summer-DOE design into that same discussion.

Thank you so kindly,
Best Regards,

Stephen
