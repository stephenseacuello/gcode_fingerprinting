# Summer DOE — One-Page Strawman (discussion starter, not a final design)

*For the Monday working session. This is a starting point to react to and
revise together — nothing here is locked.*

## Why a new dataset (the problem this fixes)

The current Bantam corpus has only ~22 distinct G-code lines per fold and
the same tool path repeats across the ~20 files in each operation class.
Consequence (measured): a position-only lookup with **zero sensor input**
predicts ~80% of the held-out lines, and ~97% of test lines appear
verbatim in training. The model can shortcut instead of reading the
sensors. A genuinely novel toolpath (leave-one-class-out) collapses
command accuracy 0.89 → 0.21. The cure is a dataset with many *distinct*
single-line experiments where position can't be memorized.

## Proposed factors (levels open for discussion)

| Factor | Candidate levels | Why it's in |
|---|---|---|
| Motion type | straight / arc-CW / arc-CCW | recover G0/G1 vs G2/G3 |
| Direction | X± / Y± / XY-diagonal | test sign discrimination |
| Feed rate | a few levels spanning the usable range | F is absent from current data — blocks any feed-recovery claim |
| Spindle speed | low / mid / high | strong audio + current signature |
| Depth of cut | shallow → deep (+ 0 for air cut) | Stephen's specific recoverability target |
| Material | aluminum / steel / Delrin | different chip-formation regimes |
| Tool diameter | small / mid / large | modulates force/vibration |
| Air cut | yes / no | negative control: spindle on, no material |

Full factorial is ~10–20k runs (too many). Intent: a **fractional /
randomized-sweep design** that keeps good per-level coverage at a
tractable run count — the actual N is a Monday decision, not assumed.

## Proposed scale & sequencing (open)

- **Pilot (~50 runs)** — week 1 after the Tormach lands. Validate the
  recording pipeline, sensor↔G-code time alignment, and tool wear under
  steel before committing to scale.
- **Main set (a few hundred runs)** — once the pilot is clean. This is
  the set the manuscript would be re-evaluated against.
- **Extension** — only if residual shortcut persists after the main set.

## Open questions for the session (this is the real agenda)

1. **Run count** — what total is realistic given Tormach time + Tim's
   hours? (Spec doc currently says ~200; paper appendix says 188 — we
   should just pick the real number together.)
2. **Single-line vs short toolpaths** — pure single-line G-code for
   clean signatures, or also a few short multi-line paths?
3. **Tool imaging** — do we image the tool every run (Stephen's ask)?
   Cost: slows throughput; benefit: tool-state ground truth.
4. **Sample rate** — the 4 Hz Bantam rate is a known bottleneck; what
   rate does the Tormach stack give us, and does that change the
   factor design?
5. **Replication** — how many repeats per condition for a real
   variance decomposition (the current corpus has none)?
6. **Air-cut coverage** — how many air cuts as the negative control?

## What this unblocks in the paper

Feed rate, spindle, depth-of-cut, and arc parameters are currently
"not recoverable" *only because the data doesn't vary them*. A DOE that
varies them is what converts those from "not evaluable" to an actual
recoverability result, and breaks the closed-vocabulary ceiling that
bounds every headline number in the current paper.
