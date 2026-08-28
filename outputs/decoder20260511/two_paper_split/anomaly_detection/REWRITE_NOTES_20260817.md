# Paper E writing-rules light pass — 2026-08-17

Claim-invariant prose pass under the scientific-writing rules (paper-rewrite procedure).
Pre-rewrite state committed in git (2b6203f bundle backup); baseline 33pp/0 err/0 undef.

## Process
5 grouped rewriter agents (front+abstract / methods / setup+results / ablation+discussion / tail)
+ 5 adversarial verifiers + independent mechanical set-equality check (cites/refs/labels/numerics
per file; float environments byte-compared). All 12 section files pass mechanically.

## What changed (prose only; every number/cite/ref/hedge preserved)
- Abstract fully re-voiced: 200 words, no em-dash asides, no quoted hypotheticals; "Existing
  sensor-based monitoring" scope restored after a verifier catch; "pass visual inspection"
  (matches the cited body claim).
- Audit items all resolved: "Results preview:" opener gone; "argues decisively" -> "argue
  against"; "detecting any discrepancy" -> "detecting discrepancies" (INTENDED weakening —
  the universal contradicted the paper's own A3 0.621); "remarkably stable" and "dramatic
  improvement" removed from the two flagged captions (the only caption edits); scare quotes
  removed; the 5x-repeated sensor-only-baseline argument reduced to one canonical statement
  per file; multi-job sentences split throughout.
- Interpretation-labeling (audit-driven, documented): mechanism claims stated as fact now read
  "We attribute..." (07:30 multi-line context, 08:37 graceful degradation, 09:42 NLL variance,
  appendix variability) — category-D statements now labeled as such per rule 3/4.
- Verifier catches fixed before assembly: dropped "Existing" (blocking), "pass inspection"
  strengthening, an added \ref in 07 (reverted), "cannot rely" categorical -> evidential,
  conclusion "best" superlative removed, 02 "statistically normal" hedge restored.

## Final: 33pp (= baseline), 0 errors, 0 undefined, abstract 200 words. NOT committed.
