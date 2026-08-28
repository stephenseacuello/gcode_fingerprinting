# Paper C writing-rules pass — 2026-08-18

Claim-invariant prose pass (paper-rewrite procedure). Pre-rewrite state in git (2b6203f);
baseline 38pp/0/0. 6 line-range chunks: background/method/tail by agents (method verified PASS;
background+tail verifiers died -> independent mechanical set-equality PASS); design+validation
written by agents that died AFTER writing (files recovered from scratch, mechanically verified;
one added self-\ref in design reverted); front (abstract+intro+related) written INLINE.

## Headline change: the ~450-word abstract rebuilt at 200 words
All headline claims/numbers kept (97%, 19 tokens, <=0.002% OOV, BPE 0.00±0.00% -> byte-level
97.1%, KL 0.005±0.003 nats, 94.7/84.4/96/91.1). Abstract-only cuts verified body-backed before
removal: 0.025/4.81/18-65 OOV band, 552 samples, 96.98-98.31 band, the slim-trainer caveat
(sec:exp_c referenced 8x in body), the TLP framing sentence, the "principled choice" interpretive
sentence.

## Other changes (prose only; floats/listings/definitions byte-verbatim)
Four-application mega-paragraph split into one-job sentences; ~40 em-dash constructions
converted; editorial voice removed ("unsurprisingly, enormous", "The honest read", "deserves a
closer look", "worth interpreting carefully", "honest narrowing"); announcing openers replaced;
the ~450-word Limitations paragraph split by topic; BPE-story dedup (canonical telling in
Design-Space Case 2, cross-references elsewhere); "fundamentally different" intensifier dropped.

## Final: 37pp (baseline 38 — the abstract compression saved a page), 0 errors, 0 undefined,
abstract 200 words. NOT committed.
