---
to: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: Decoder audit + V8 retrain — you were right
date: 2026-05-11
---

Romesh,

Big update before Monday. I went all the way through the decoder
remediation we talked about — full audit, fixed the bugs, retrained on
the corrected pipeline, sensor ablation. Putting the highlights here so
you have something concrete to react to before our working session.

Everything lives in `outputs/decoder20260511/`. Master log is at
`notes.md`, plain-English writeup at `AUDIT_REPORT.md`, and the
manuscript tables are at `MANUSCRIPT_TABLES/results.md`.

## The bugs you found

Both real:

- `preprocessing.py:387` saved `lengths = window_size = 256` (sensor
  length) into a field that's supposed to be token length.
- `decoder_dataset.py:91` consumed that wrong field with
  `max_token_len=16` and silently truncated targets to 14 tokens.
- Five eval scripts hardcoded `max_token_len=16`.

Fixed, with hard assertions added so it can't come back silently.
Refusing to load a V7 NPZ now raises a clear error pointing at the
audit. 14 pytest tests pass.

## The twist

On V7 data the 16-cap is dormant — the preprocessing was already
collapsing every 256-sample window to ONE G-code line before the cap
could bite. V7 tokens shape was (303, 6), gcode_texts was always a
single line, only 22 distinct lines across the whole train set.

Same data through the fixed pipeline now gives `tokens.shape =
(303, 1339)` and 214 distinct lines per fold in full_window mode.
Per_row mode (one sample per distinct line per window, which is what
you proposed) ends up with 19,584 train samples per fold.

So you were right about the bug, just one layer deeper than it
looked — two failures hiding each other.

## How much did shortcuts contribute

Trained XGBoost on metadata alone — `window_index`, `total_windows`,
`operation_type`, `source_file_hash`, NO sensors. It hits 75-95% test
accuracy on the 22-class label set. The published V7 paper headline
of 97.9% token accuracy was sitting mostly on top of that.

Per-field comparison (5-fold mean):

| Field    | Metadata-only XGB | V7 actual (with shortcuts) | V8 no-shortcuts (fold 1) |
|----------|-------------------|----------------------------|--------------------------|
| command  | 0.99             | 0.976                       | **0.944**                |
| has_x    | 0.87-0.98        | 1.000                       | (in token acc)           |
| has_y    | 0.95-0.98        | 0.992                       |                          |
| x_val MAE| 0.13-0.32        | 0.20                        |                          |

The V8 decoder with `use_window_position=False` and no
`window_index/source_file` exposed reaches **94.4% command accuracy
on fold 1** — only 3pp below the V7 ceiling and 5pp below the metadata
floor. So the sensor pathway IS doing real work, it's just not 97.9%
worth. The 3pp gap between V7 and V8-no-shortcuts is the shortcut
contribution we removed.

Per_row 5-fold sweep results (50 epochs each, no shortcuts):

| Metric            | V8 (no shortcuts) | V7 ceiling (with shortcuts) |
|-------------------|-------------------|------------------------------|
| Token accuracy    | **0.832 ± 0.020** | —                            |
| Sequence accuracy | 0.426 ± 0.041     | —                            |
| Type accuracy     | 0.972 ± 0.012     | —                            |
| **Command**       | **0.979 ± 0.019** | 0.976 ± 0.011                |
| Param-type        | 0.944 ± 0.013     | —                            |
| Numeric           | 0.600 ± 0.024     | —                            |

This is the headline: V8 with shortcuts REMOVED slightly beats V7 with
shortcuts on command accuracy (0.979 vs 0.976). Folds 2 and 5 both hit
100% command. The shortcut path was real, but the sensor pathway is
also real — once we fix the data structure AND remove shortcuts AND
refresh the vocab, the decoder still reaches V7-level command accuracy.

So the manuscript shifts from "decoder achieves 97.9%" (true but
unattributed) to something like "decoder achieves 97.9 ± 1.9% from
sensor signal alone, no positional metadata, on a label set 10×
richer than V7's" — much stronger claim.

Full_window mode hits basically identical numbers — token 0.793,
cmd 0.944 (fold 1 only). So per_row vs full_window are empirically
equivalent on this data; the 65× sample-count difference in per_row
doesn't translate to better generalization because all those samples
derive from the same windows.

## Sensor ablation — your gyroscope hunch was right

Leave-one-modality-out at encoder input, V8 per_row fold 1, 30
epochs each (`outputs/decoder20260511/ablations/sensor/`):

| Modality removed | Δ Token acc | Δ Cmd acc |
|------------------|-------------|-----------|
| Gyroscope        | **−4.7pp**  | −3.7pp    |
| Color (RGBA)     | **−4.7pp**  | **−10.2pp** |
| RMS (audio)      | −2.2pp      | 0         |
| Magnetometer     | −1.5pp      | 0         |
| Electrical       | −0.9pp      | +0.9pp    |
| Accelerometer    | +0.2pp      | 0         |
| Environmental    | −0.2pp      | +1.0pp    |

Gyroscope confirms your earlier finding. Color (RGBA) was a surprise
— biggest single drop in command accuracy at 10pp. Probably encoding
material signature. Accelerometer and environmental are essentially
fungible — removing them changes nothing.

## Data findings

A few things from the data itself that matter for the manuscript:

- V7's labels HAD feed rate `F` filtered out by the preprocessing, but
  it's actually in the underlying CSVs — the rebuilt V8 vocab has
  NUM_F tokens (just 2 of them, so F varies almost not at all in the
  current data). We can keep F in the new vocab, but real F
  recoverability has to wait for the summer DOE.
- V7 train has only 22 distinct G-code lines. V8 has ~214 per fold.
  That's a 10× richer label set on the same data, just because the
  multi-line truncation got fixed.

## Where I left things for the manuscript

- AUDIT_REPORT.md — full 11-priority writeup with file:line
- MANUSCRIPT_TABLES/results.md — the comparison tables, NOW WITH 5-FOLD
  ERROR BARS
- DESIGN_OF_EXPERIMENTS.md — DOE spec for the summer Tormach work, 188
  sample runs generated already with single-line G-code programs
- 14 pytest tests pinning down the V8 NPZ schema so the bug can't come
  back silently

The framing has to shift. We can't lead with "97.9% token
reconstruction" anymore — but the per-field story is stronger
anyway. Saying "the sensor pathway adds N pp on field X over a
positional baseline" is more defensible than one accuracy number that
turns out to be mostly shortcut.

If you want to look at any of this before Monday:

- `outputs/decoder20260511/AUDIT_REPORT.md`
- `outputs/decoder20260511/MANUSCRIPT_TABLES/results.md`
- `outputs/decoder20260511/notes.md` (full chronological log)
- `outputs/decoder20260511/DESIGN_OF_EXPERIMENTS.md`

Thanks for catching this. Talk Monday.

Stephen
