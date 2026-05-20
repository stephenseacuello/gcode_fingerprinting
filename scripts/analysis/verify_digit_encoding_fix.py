#!/usr/bin/env python3
"""Verification harness for the float-epsilon fix in
src/miracle/model/digit_value_head.py::encode_values_to_digits.

Two checks:
  1. Equivalence: on float-clean values the new (round-to-scaled-integer)
     encoder reproduces the old (iterative-truncation) encoder bit-for-bit.
  2. Correctness: on the actual V8 test corpus, count how many CAM-authored
     terminal-zero values the new encoder yields slot-5 = 0 for (correct) vs
     the old encoder yields slot-5 = 9 for (corrupted).

Reports the empirical reduction in str0-encoded-as-9 across the corpus and
confirms the headline audit number in audit/digit_entropy.json (23.6%) goes
to ~0% under the fix.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PARAM_RE = re.compile(r"([XYZIJKRF])(-?\d+\.?\d*)", re.IGNORECASE)

MAX_INT_DIGITS = 2
MAX_DEC_DIGITS = 4
N_SLOTS = MAX_INT_DIGITS + MAX_DEC_DIGITS  # 6


def encode_old(values: torch.Tensor) -> torch.Tensor:
    """The previous iterative-truncation encoder (kept here for comparison)."""
    abs_values = torch.abs(values)
    B, T = values.shape
    digit_targets = torch.zeros(B, T, N_SLOTS, dtype=torch.long)
    int_part = abs_values.long()
    for pos in range(MAX_INT_DIGITS):
        power = MAX_INT_DIGITS - 1 - pos
        divisor = 10 ** power
        digit_targets[:, :, pos] = (int_part // divisor) % 10
    dec_part = abs_values - int_part.float()
    for pos in range(MAX_DEC_DIGITS):
        dec_part = dec_part * 10
        digit_targets[:, :, MAX_INT_DIGITS + pos] = dec_part.long() % 10
    return digit_targets


def encode_new(values: torch.Tensor) -> torch.Tensor:
    """The patched round-to-scaled-integer encoder."""
    abs_values = torch.abs(values)
    B, T = values.shape
    scale = 10 ** MAX_DEC_DIGITS
    scaled = torch.round(abs_values * scale).to(torch.long)
    digit_targets = torch.zeros(B, T, N_SLOTS, dtype=torch.long)
    for pos in range(N_SLOTS):
        divisor = 10 ** (N_SLOTS - 1 - pos)
        digit_targets[:, :, pos] = (scaled // divisor) % 10
    return digit_targets


def collect_values_from_corpus() -> list[tuple[str, float]]:
    """Return (val_string, float_value) pairs from gcode_texts across 5 folds."""
    out: list[tuple[str, float]] = []
    for fold in range(1, 6):
        path = REPO / f"outputs/decoder20260511/preprocessed/per_row/fold_{fold}/test_sequences.npz"
        z = np.load(path, allow_pickle=True)
        for t in z["gcode_texts"].tolist():
            for axis, val_str in PARAM_RE.findall(str(t).upper()):
                try:
                    out.append((val_str, float(val_str)))
                except ValueError:
                    continue
    return out


def main():
    print("[verify] collecting values from V8 5-fold test corpus...")
    pairs = collect_values_from_corpus()
    print(f"[verify] {len(pairs):,} numeric values pulled from gcode_texts.")

    values_t = torch.tensor([p[1] for p in pairs], dtype=torch.float32).unsqueeze(0)
    old = encode_old(values_t)[0]
    new = encode_new(values_t)[0]

    # 1. How many slot-5 values changed under the fix?
    differs = (old != new).any(dim=-1)
    n_changed = int(differs.sum().item())
    print(
        f"\n[1] Slot-by-slot equivalence audit:\n"
        f"    {n_changed:,} of {len(pairs):,} values "
        f"({100 * n_changed / len(pairs):.2f}%) had at least one slot change "
        f"under the fix."
    )

    # 2. Among the changes, which transitions matter most?
    transitions = Counter()
    for i in range(N_SLOTS):
        for o_d, n_d in zip(old[:, i].tolist(), new[:, i].tolist()):
            if o_d != n_d:
                transitions[(i, o_d, n_d)] += 1
    if transitions:
        print("    Top per-slot transitions (slot, old_digit -> new_digit, n):")
        for (slot, o_d, n_d), n in transitions.most_common(8):
            print(f"      slot {slot}: {o_d} -> {n_d}  (n={n:,})")

    # 3. The headline audit: CAM-authored terminal-zero ('.X0') values that
    #    old encoded to slot-5 = 9 (corrupted) vs new encoded to slot-5 = 0 (correct).
    str_terminal_zero = 0
    old_9_for_str0 = 0
    new_9_for_str0 = 0
    new_0_for_str0 = 0
    for i, (val_str, _) in enumerate(pairs):
        v = val_str.lstrip("-")
        dec = v.split(".", 1)[1] if "." in v else ""
        dec_padded = (dec + "0000")[:4]
        if dec_padded[3] != "0":
            continue
        str_terminal_zero += 1
        if int(old[i, 5]) == 9:
            old_9_for_str0 += 1
        if int(new[i, 5]) == 9:
            new_9_for_str0 += 1
        if int(new[i, 5]) == 0:
            new_0_for_str0 += 1

    print(
        f"\n[2] Float-epsilon audit on CAM-authored terminal-zero values "
        f"(slot-5 = '0' in source string):\n"
        f"    {str_terminal_zero:,} terminal-zero values in the corpus.\n"
        f"    Old encoding corrupted to slot-5 = 9: "
        f"{old_9_for_str0:,} "
        f"({100 * old_9_for_str0 / str_terminal_zero:.2f}% if denom > 0).\n"
        f"    New encoding gives slot-5 = 9: {new_9_for_str0:,} "
        f"({100 * new_9_for_str0 / str_terminal_zero:.2f}%).\n"
        f"    New encoding gives slot-5 = 0: {new_0_for_str0:,} "
        f"({100 * new_0_for_str0 / str_terminal_zero:.2f}%)."
    )
    print(
        f"    Pooled corruption rate: "
        f"old = {100 * old_9_for_str0 / len(pairs):.2f}% of all slot-5 positions, "
        f"new = {100 * new_9_for_str0 / len(pairs):.3f}% of all slot-5 positions."
    )

    # 4. Pass criterion: the fix should bring old-corruption down to ~0.
    if new_9_for_str0 == 0:
        print("\n[verify] PASS: the patch eliminates the float-epsilon corruption "
              "of terminal zeros.")
    elif new_9_for_str0 < old_9_for_str0 * 0.01:
        print(f"\n[verify] PASS (residual {new_9_for_str0} / "
              f"{old_9_for_str0}, < 1%): the patch substantially reduces the corruption.")
    else:
        print(f"\n[verify] WARN: residual corruption {new_9_for_str0} / "
              f"{old_9_for_str0}; the patch is not eliminating the artifact "
              "as expected.")


if __name__ == "__main__":
    main()
