# v4 Fix Plan — Address All Findings by Cutting, Not Adding (2026-08-10)

Design principle: **every fix is a deletion or a one-word swap. Zero prose added.** This serves the "no expansive writing" constraint and continues the wordiness reduction. Net effect is ~−40 words. All fixes are claim-invariant except where marked (CLAIM DECISION).

## Review status
Full lens sweep done (ML-methods referee + inline CNC/security/editor/consistency + scripted number audit). 0 undefined refs; prose/table/caption numbers agree; platform figures consistent. Complete finding set below; the 4 round-2 consistency fixes (F1–F4) are already applied.

---

## TIER A — mechanical, claim-invariant (safe to apply now)

| # | Finding | Fix (cut/swap) | Location |
|---|---|---|---|
| A1 | Abstract 214 w > MDPI 200-w limit | Cut ~14 w of connective phrasing (e.g. drop "reported"/"intended"; merge two clauses). No number/claim removed. | `\abstract{}` |
| A2 | Latency 30–80 ms (body) vs 30–60 ms (sourced appendix) | Swap body **"30--80"→"30--60"** to match the appendix's source-verbatim figure | Sec Eval-Limits, l.898 |
| A3 | "well within standard plant-IT storage tiers" — uncited editorial aside | **Delete the clause** (the kB/s and GB/shift numbers stay) | Sec Preprocessing |
| A4 | grammar-mask "a reduction that improves both calibration and exact-match accuracy" — uncited effect claim | **Delete the clause** (the 2418→31.8 token reduction fact stays) | Sec Grammar |

## TIER B — deletion resolves the issue, but is a CLAIM DECISION (confirm first)

| # | Finding | Proposed minimal fix | Why it's a decision |
|---|---|---|---|
| B1 | Sign-asymmetry tension: Discussion says ±X/±Z "less separable... plausibly sensor-board layout"; Results says asymmetry is "prior-driven, not a sensor-separability ranking" | **Delete the Discussion sentence.** Results' prior-driven account is the one the paper defends; the Discussion sentence is speculation the writing rules already flag | Removes a hypothesis you may want to keep as future-work framing |
| B2 | F5 attestation trust gap: "load-time attestation receipt" is controller-generated while the controller is untrusted | **Delete the parenthetical** "(e.g., via a controller-side load-time attestation receipt~\cite{zonouz2014plc} bound to the operator session)". The assumption then reads as an external precondition, no contradiction | Security-co-author call; alternative is to *name* the hardware root-of-trust as trusted (adds words) |
| B3 | F6 NIST attribution: "≤10² per analyst per shift implicit in NIST SP 800-61 Rev.~2" — source states no such number | Swap **"implicit in NIST SP~800-61~Rev.~2"→"as a working assumption"** and move the `\cite{nist2012sp80061}` to the incident-handling clause | Changes how a cited source is characterized |

## TIER C — evidence-category flags (Romesh meeting; batch cut)

~20 items from `REWRITE_NOTES_20260803.md` §flags — mostly unsupported intensifiers and uncited physical reasoning. All fixes are word deletions/softenings under scientific-writing rule 2.2 (e.g. "substantially harder"→"harder"; "high agreement"→state the cited metric or drop). Each is a claim-strength decision, so batch them for the co-author pass rather than pre-applying. Representative: Intro "substantially harder signal environment"; RelWork "achieved strong results" / "high agreement"; Method aux-head "two purposes"; Setup "factory-rated sensitivities".

---

## Recommended order
1. **Apply Tier A now** (4 edits, ~−20 w, fixes the only hard blocker: the 200-w abstract limit).
2. **Confirm Tier B** (3 edits, all deletions/swaps, ~−20 w) — I'll apply on your yes; B2 wants Romesh/security eyes.
3. **Tier C** at the co-author meeting — hand Romesh the flag list; apply his calls in one batch.

Nothing here adds a sentence. Post-fix v4 stays at 57 pp (all cuts) with the abstract compliant.
