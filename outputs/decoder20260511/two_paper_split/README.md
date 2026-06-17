# Two-paper split of decoder_paper_v2

Source of record: `../decoder_paper_v2/` (80pp main + 35pp supplementary, simulated-review Accept-minor).
That paper is currently **dual-headline**: (1) per-field G-code recoverability via the multi-head
decoder, and (2) a unified *source-entropy theory* of CNC toolpath coordinates. The entropy material
has been promoted into the abstract, the contribution list (item 6), and the §5.1 Results opener — which
is why it reads like two papers. This folder splits them.

## `decoder_paper/` — Paper A (recoverability / decoder)
The decoder contribution as a clean, self-contained characterization study. Target: complete and done.

## `toolpath_entropy/` — Paper B (CNC toolpath source-entropy)
The information-theoretic analysis of what is recoverable from CNC toolpath coordinates and why.
Target: close to done; may need additional testing/data to stand alone (and to clear the
salami-slicing concern referees already raised about the encoder/grammar/decoder split).

The detailed, executable split plan (section/figure/table allocation, cut list, missing experiments,
venue targets) is being finalized — see the plan delivered in-session.
