# G-code Fingerprinting Project

Research codebase for CNC side-channel analysis (encoder + decoder models reconstructing G-code from sensor data), plus a 5-paper publication bundle in `outputs/decoder20260511/two_paper_split/`:

- **A — decoder_paper/** (MDPI Sensors, main + supplementary) — flagship
- **B — toolpath_entropy/** (Entropy)
- **C — grammar_paper/** (MDPI Sensors)
- **D — toolpath_signature/** (MDPI Sensors)
- **E — anomaly_detection/** (MDPI Sensors)

Paper sources are backed up in git (`bundle-backup-*` branches). Never delete paper files; commit only when asked.

## Scientific writing rules (mandatory)

Whenever you write or edit prose in any paper (LaTeX body, abstract, captions, responses to reviewers), load the `scientific-writing` skill first and follow it. When rewriting a full paper or section, use the `paper-rewrite` skill for the procedure. Core principles, always in force:

1. **Every sentence has one job** and must contribute to the scientific argument. No filler, no academic-sounding padding, no announcing what a sentence is about before saying it.
2. **Claim only what the evidence supports.** No "significant / robust / novel / effective / clearly / substantially" unless measured, demonstrated, or statistically justified. If a reviewer could challenge a word as opinion, remove or justify it.
3. **Never blur the four evidence categories**: (A) literature claim — needs a reference; (B) observed result — supported by our data; (C) derived result — reproducible from our methodology; (D) interpretation — explicitly framed as such and grounded in B/C plus literature.
4. **Assumptions are labeled as assumptions.** Plausible explanations are never stated as facts.
5. **Paragraphs**: one central purpose; Theme → Evidence → Reasoning → Connection. **Sections**: a logical progression that builds the argument (Problem → Context → Evidence → Gap → Approach → Result → Implication), not a collection of correct-but-disconnected paragraphs.
6. **Never change technical meaning to simplify prose.** Preserve precise terminology, all numbers, citations, cross-references, and hedges. A prose rewrite must be claim-invariant.

These rules come from Romesh Prasad's writing method (2026-07-29) and override default writing style.
