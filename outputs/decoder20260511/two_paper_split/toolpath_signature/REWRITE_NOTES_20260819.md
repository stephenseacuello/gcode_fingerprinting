# Paper D writing-rules pass — 2026-08-19

Claim-invariant prose pass (paper-rewrite procedure). Pre-rewrite state in git (2b6203f);
baseline 43pp/0/0. 4 grouped rewriters + 4 adversarial verifiers (all 8 completed) +
independent mechanical set-equality check.

## What changed (prose only; floats/captions verbatim)
- Abstract rebuilt at 199 words (from 268): all headline numbers kept (96.7/81.8/100/99.0/90,
  98.5/0.99, 0.1%/74%, 37%, 784, 96%, 1.0x); "confirming" strength RESTORED after a verifier
  caught the rewrite's weakening to "indicating" (inconsistent with the intro); the closing
  "robust normalization strategy" summary sentence cut ("robust" unmeasured blanket; the
  template claim survives) — flagged for author confirmation.
- Sanctioned dedup of the two ~5x-repeated headline blocks: intro/discussion/conclusion
  compressed to headline numbers + Section~\ref{sec:results} cross-references; every dropped
  sub-detail (81.8, 90, 79.9, 90.0, 0.99, 0.97, 0.1, 74, 784) verified to retain 2-20 canonical
  occurrences in results.tex (most also in the abstract).
- Editorial self-ranking sentences deleted ("the paper's strongest empirical result", "most
  process-relevant result", "strongest practical argument", "most important methodological
  result") — findings now stated directly; flagged for author awareness.
- methodology "meaningful classification challenge" aligned with the paper's own
  trivial-separability finding; the intro's three-tests-in-one-sentence chain split;
  "dramatically"/"striking"/"rigorous" intensifiers removed; magnetometer offset explanation
  and PCA/overfitting mechanisms relabeled as interpretation ("consistent with"); "flips
  entirely" -> "shifts almost entirely" (more accurate given the 0.1% retention); "severely
  limits" RESTORED (justified by the adjacent 400 Hz tooth-passing evidence).

## Final: 43pp (= baseline), 0 errors, 0 undefined, abstract 199 words. NOT committed.
