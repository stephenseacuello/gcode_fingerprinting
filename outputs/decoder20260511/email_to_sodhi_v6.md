---
to: Manbir Singh Sodhi <sodhi@uri.edu>
cc: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: G-Code Decoder — follow-up meeting recap (both action items resolved, plus the methodology-confound control)
date: 2026-05-26
---

Hi Dr. Sodhi —

This is the final update on the G-code decoder follow-up. Both meeting
action items are closed, and the methodology-matched control I flagged
in my last email has finished — it sharpens the takeaway and overturns
my earlier "default to 2-digit bucketing" suggestion.

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

The model's accuracy on a digit closely tracks how varied that digit is in
the data — correlation −0.96 (p = 0.002). The fine fractional digits in the
middle are highly varied (a by-product of CAM toolpath calculation) and
that is where recovery falls off. So the numeric ceiling is set by how
much information is in the data, not by model tuning. The paper now
states precisely which fields are recoverable.

**2. Run token and sequence prediction without numeric token (done).**

I retrained the model 5-fold with every coordinate value replaced by a
single placeholder token. Under realistic autoregressive decoding, scored
on commands and axis letters:

| Model | Per-token correct | Whole lines exactly right |
|---|---|---|
| Headline (predicts numbers)  | 32% | 6% |
| No-numbers (retrained)       | 41% | 7% |
| Difference                   | **+9 pp** (5/5 folds positive, paired t = 3.56, p = 0.024) | essentially unchanged |

**Removing the numbers helps a little, but does NOT rescue end-to-end
recovery.** The +9 pp on per-token accuracy is the
*wrong-number-misaligns-the-rest-of-the-line* effect. But the rate of
getting whole lines exactly right stays the same — the decoder
independently converges on a corpus-modal output sequence regardless of
the sensor input. That mode-collapse is in the commands and axes
themselves, not the numbers.

**3. Follow-on: the vocabulary-cardinality sweep + matched-methodology control (final result).**

In my last email I told you about a four-point sweep across K = 24, 69,
335, 2,418 and that K = 335 was the highest performer of the sweep. I
flagged that the K = 2,418 row used our original training schedule
(scheduled sampling 0.5, digit-head loss 1.0) while the smaller-K rows
used the schedule I'd discovered while debugging the no-numbers run
(SS = 0, dw = 0), and that I was running a methodology-matched
K = 2,418 control to separate the vocabulary effect from the schedule
effect. That control has finished. Here is the full five-variant
picture (AR = autoregressive decoding, the deployment-relevant regime):

| Variant | Training schedule | AR per-token correct | AR whole-line exact |
|---|---|---|---|
| K = 2,418 headline       | SS=0.5, dw=1.0 | 32% | 6% |
| K = 2,418 matched (T3.1) | SS=0,   dw=0   | **47%** | **13%** |
| K = 335 (2-digit)        | SS=0,   dw=0   | 43% | 8% |
| K = 69 (1-digit)         | SS=0,   dw=0   | 42% | 5% |
| K = 24 (placeholder)     | SS=0,   dw=0   | 41% | 7% |

**The story I gave you in the last email needs an important correction.**
Under matched methodology, K = 2,418 with SS = 0 / dw = 0 is the
highest-performing variant on every metric — not K = 335. K = 335 falls
4 percentage points behind matched K = 2,418 on per-token accuracy
(paired t = −0.85, p = 0.44 across folds: no statistical advantage and
the point estimate goes the wrong way). The +11 pp K = 335 advantage I
reported over the headline was almost entirely the schedule change, not
the vocabulary change. Within the matched SS = 0 / dw = 0 family,
larger K is mildly better; the smaller-K alternatives sacrifice
4–6 pp on per-token AR accuracy AND collapse numeric precision below
~0.01 inch (which is the resolution regime where adversarial
coordinate substitutions of operational interest live, so smaller K
would also widen the threat-model bypass).

**Bottom line, corrected from the last email:** I would NOT default to
2-digit bucketing as the deployment architecture. The deployment-recommended
configuration is the **existing K = 2,418 numeric vocabulary trained
under SS = 0 / dw = 0**, which gives a clean +15 pp per-token AR lift
(0.32 → 0.47) and a +7 pp whole-line AR lift (0.06 → 0.13) over the
current headline (paired t = 7.7, p = 0.006 after multiple-comparison
correction). This is the cleanest single-knob change for the next
architecture iteration.

The honest scoping qualifiers in the paper are unchanged from before:
~22 pp of the headline command accuracy is genuine sensor-conditioned
signal (the other ~57 pp is the operation-class prior the frozen
encoder already encodes); the model collapses to ~21% command
accuracy on operation classes never seen in training (open-vocabulary
test); and numeric/feed-rate substitutions are undetectable at 4 Hz
by construction. The new K = 2,418 / SS = 0 / dw = 0 finding moves the
in-distribution AR ceiling upward but does not change those three
scope statements.

All three findings are in the paper:
- Item 1: Section "Numeric Decomposition" (entropy analysis, r = −0.96).
- Item 2: Section "No-Numeric Retraining: Decomposing the Autoregressive
  Collapse."
- Item 3: Same section, "Vocabulary-cardinality spectrum and
  matched-methodology control" paragraph + updated five-row Table 9 and
  updated Figure 14. The Limitations section now reports the
  methodology confound as resolved.

Thank you so kindly,
Best Regards,

Stephen
