# Author Dossier — facts requiring author confirmation (2026-08-25)

Compiled at the close of the cross-paper consistency capstone. Each item below is a
physical or provenance fact that could NOT be resolved from the repository, the data,
or the encoder paper, and therefore needs the authors' (lab notebook / hardware) memory.
Everything that *could* be resolved from evidence has already been fixed in the sources;
this list is what remains.

## 1. Machine model and specifications (HIGHEST PRIORITY)
The CAM header in `data_clean/face.gcode` says **"BANTAM TOOLS EXPLORER"**, and papers
A/D/F name the Explorer. But the bundle still carries three incompatible spec sets:

| Claim | Paper(s) | Value on disk |
|---|---|---|
| Work envelope | C (2 sites: scope §, limitations §) | ~140 × 102 × 60 mm |
| Work envelope | D, E | 140 × 114 × 38 mm |
| Max spindle speed | D, E | 26,000 RPM |
| Machine name | E (experimental setup) | "Bantam Tools Desktop CNC Milling Machine" |
| Positioning resolution | E | 0.003 mm |

140 × 114 × 38 mm equals 5.5" × 4.5" × 1.5", which matches the *Desktop CNC Milling
Machine* published spec, not obviously the Explorer's. **Action**: confirm the actual
machine model used for the campaign and align envelope, spindle max, and resolution in
C (both sites) and E to its spec sheet. Product names were genericized to "Bantam Tools
desktop CNC mill" in C and E during the verification round (removing the SKU contradiction
with the CAM header), but the spec numbers were deliberately left untouched — only facts
with in-repo evidence were changed.
Note the commanded spindle speed in the corpus is 12,000 RPM (M3 S12000, CAM header) —
already consistent everywhere after the fix cascade; only the machine's *maximum* is open.

## 2. Sensor-board data link
Whether the six Nano 33 BLE Sense Lite boards streamed over USB serial, BLE, or a
shielded/wired link is stated differently across papers and is not derivable from the
repo. **Action**: confirm the link and align (matters for the EMI/robustness discussion
in D and the cost/deployment table in E).

## 3. Encoder parameter count — RESOLVED in-repo, one confirmation left
Counting the checkpoint's tensor bytes directly (no unpickle) gives **~27.5 M params
total** (f98 fold-4 `best_model.pt`, 206 tensors, 110 MB float32) — matching neither
8.6 M nor 5.1 M. The encoder paper's citable figure is **≈5.1 M active** (Table
tab:params, d_model=256). The 8.6 M figure matched no artifact and was removed from
A_v5 (4 sites), A_conf (3 sites), the supplementary, and the replicability appendix;
system totals recomputed (31.28 M decoder + 5.1 M active ≈ 36.4 M, was 39.9 M), and
the VRAM point estimate replaced by the supported <500 MB bound. **Action**: confirm
the 5.1 M active count corresponds to the deployed inference graph (the checkpoint's
remaining ~22 M are denoising/reconstruction components unused at classification).

## 4. Token count: 79,563 vs 79,345
Two nearby values of the corpus token count n appear in different documents' methods
prose. Likely a filtered-vs-unfiltered difference. **Action**: confirm which count
corresponds to which filtering step and label both explicitly where they appear.

## 5. Paper D damage experiment — recommended follow-up (science flag)
Now correctly described as: 51 normal *active* runs vs 14 *damaged-spindle air-cut*
runs, with the conflation (spindle condition × cutting engagement) flagged in the
abstract/results/discussion/conclusion. Two follow-ups only the authors can do:
- **Clean comparison**: damaged-spindle air-cuts (14) vs normal air-cuts of the same
  programs (55 available). This isolates spindle condition. If RF accuracy stays high,
  D's tool-condition-monitoring reading is rescued; if it drops to chance, the 98.5%
  result was mostly engagement detection.
- **Program identity**: confirm whether the damage runs executed the *standard*
  (air-cut) program variants or the *150025* (active) variants — the encoder paper's
  class naming implies standard, which means feed/depth words differ from the active
  runs D compares against. The prose now says "same toolpath strategies" (safe), not
  "identical programs" (unverified).

## 6. C downstream decoder: teacher-forced label
The 94.7% token / 84.4% sequence exact-match downstream results in C are now labeled
teacher-forced (capstone verifier's determination from the experiment script).
**Action**: authors sanity-check that the v7_best_5fold eval logged those numbers under
gold-prefix conditioning, since the label is now printed in the caption and §Results.

## 7. Zenodo DOIs + co-author reads
Standing items from earlier rounds, unchanged: mint DOIs for the audit artifacts
(`ar_categorical_freerunning.json`, `markov_command_null.json`, `nonneural_baselines.json`)
before submission; co-author read of the five certified papers plus the two decoder
editions after this capstone's corrections.
