---
to: Manbir Singh Sodhi <sodhi@uri.edu>
cc: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: G-Code Decoder — follow-up meeting recap (both action items resolved, plus a follow-on finding)
date: 2026-05-24
---

Hi Dr. Sodhi —

Unfortunately, Read.ai didn't save the meeting notes or I would have shared a
condensed version and transcript. These have been very helpful to me in
tracking action items and executing them timely.

Overall two action items came out of the meeting — both are now done, and
I've also added a follow-on experiment (item 3 below) that sharpens the
takeaway.

**1. Further analysis of why numeric token accuracy is poor (done).**

The decoder reads back the commands, axes, and motion directions accurately,
but misses the exact coordinate values. To see why, we broke each coordinate
into its individual digits and measured how well the model recovers each one:

| Coordinate digit | How varied it is in the data | Model accuracy |
|---|---|---|
| Tens place        | fixed — always 0      | 100% |
| Ones place        | mostly 0 or 1         | 75%  |
| Tenths            | highly varied         | 62%  |
| Hundredths        | highly varied         | 63%  |
| Thousandths       | highly varied         | 70%  |
| Ten-thousandths   | mostly a trailing 0   | 86%  |

These accuracies are measured teacher-forced — at each step the model is given
the correct previous tokens, the standard development-time setup. The model's
accuracy on a digit closely tracks how varied that digit is in the data — the
correlation is −0.96 (p = 0.002). The whole-number digits barely vary (the
parts are small, so these are almost always 0 or 1) and the model recovers
them well. The fine fractional digits in the middle are highly varied — they
are largely a by-product of how the CAM software computes the toolpath —
and that is where recovery falls off. (The ten-thousandths digit scores high
again only because it is usually just a trailing zero, an easy constant to
predict — not fine-resolution sensing.)

So the result is sharper than "the model is weak at numbers": the numeric
limit is set by how much information is in the data, not by model tuning. The
decoder is best understood as a coarse-resolution recoverer — whole-inch
positions, cutting depths, and arc radii come back well; exact fine
coordinates do not. The paper now states precisely which fields are
recoverable. Our hope is understanding the numeric limitations may guide us
to developing more generalizable architectures that scale to Tormach and
HAAS.

**2. Run token and sequence prediction without numeric token (done).**

Romesh's question was: *if the model didn't have to predict the numbers at
all, would it get everything else right?* I retrained the model 5-fold with
every coordinate value replaced by a single placeholder token — so the model
is freed from predicting numbers entirely. Here's what we found under
realistic (autoregressive) decoding, scored on the structural skeleton
(commands and axis letters), 5-fold averages:

| Model | Per-token correct | Whole lines exactly right |
|---|---|---|
| Current (predicts numbers)   | 32% | 6% |
| No-numbers (retrained)       | 41% | 7% |
| Difference                   | **+9 percentage points** (5/5 folds positive, paired t = 3.56, p = 0.024) | essentially unchanged |

**Short answer: removing the numbers helps a little, but does NOT rescue
end-to-end recovery.** The +9 percentage points on per-token accuracy is the
*wrong-number-misaligns-the-rest-of-the-line* effect — real, repeatable,
eliminated by the retraining. But the rate of getting *whole lines exactly
right* stays the same.

The reason: the failure has two causes. (1) Wrong numbers misalign the tokens
around them — removing the numbers fixes this, and that's the +9 percentage
points. (2) The model independently slips into a generic repetitive
`X-something Y-something Z-something` output regardless of what the sensors
actually say. That repetition is in the *commands and axes themselves*, not
in the numbers, and removing the numbers doesn't fix it — which is why
whole-line recovery stays flat.

**3. Follow-on: a four-point sweep across numeric-vocabulary size (new).**

Since the "no-numbers" version is a single extreme endpoint (zero numeric
distinctions), I also ran two intermediate points — 1-digit and 2-digit
bucketed numerics (K = 69 and K = 335 numeric tokens vs. the current 2,418
or the placeholder's 24). This separates "the model improved because numbers
are gone" from "the model improved because the numeric vocabulary is just
smaller."

| Numeric vocabulary | Per-token correct (AR) | Whole lines exactly right (TF) |
|---|---|---|
| K = 2,418 (current, 4-digit) | 32% | 21% |
| K = 335 (2-digit buckets)    | **43%** | **29%** |
| K = 69 (1-digit buckets)     | 41% | 25% |
| K = 24 (placeholder)         | 41% | 7%  |

Two takeaways:

- The +9-pp per-token gain is **not a placeholder artifact** — it's a
  vocabulary-cardinality effect that holds across the spectrum (statistical
  test across all four K values: Friedman χ² = 9.24, p = 0.026; pairwise
  K = 335 vs. K = 2,418 paired t = 4.12, p = 0.015, multiple-comparison-
  adjusted p = 0.044). This rules out the explanation that "removing numbers"
  is what fixed the per-token alignment problem; it's the cardinality of the
  numeric vocabulary, not its presence/absence.
- **K = 335 is a structural sweet spot.** It beats the current K = 2,418
  configuration on *both* per-token accuracy *and* whole-line accuracy,
  while the placeholder (K = 24) collapses whole-line accuracy entirely. An
  intermediate vocabulary preserves enough per-position lexical scaffolding
  to keep whole-line recovery, while substantially reducing the numeric
  misalignment that hurts per-token accuracy.

**Bottom line for the original question:** numbers are a measurable
contributor (~9 pp on per-token, statistically robust across folds and
across vocabulary sizes), but they are *not the cause* of the end-to-end
collapse. The bigger problem — the model losing the plot under realistic
decoding — is in the command and axis stream itself, and reducing or
removing the numbers doesn't fix it. **However**, the K = 335 finding
suggests a concrete practical move: a smaller numeric vocabulary appears
to be a strict improvement on both per-token and whole-line accuracy, so
the next architecture iteration should default to 2-digit bucketing
unless we discover a downstream reason to keep finer numeric resolution.

All three findings are now in the paper:
- Item 1: Section "Numeric Decomposition" (entropy analysis, r = −0.96).
- Item 2: Section "No-Numeric Retraining: Decomposing the Autoregressive
  Collapse."
- Item 3: Same section, "Vocabulary-cardinality spectrum" paragraph + new
  Table and Figure.

Thank you so kindly,
Best Regards,

Stephen
