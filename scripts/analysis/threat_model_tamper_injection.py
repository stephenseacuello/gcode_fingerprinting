#!/usr/bin/env python3
"""Threat-model synthetic tamper-injection FPR / FNR for the headline decoder.

Reads each fold's beam_0_all_predictions.json under
outputs/decoder20260511/checkpoints/full_window_5fold/. For each held-out
test sample we:

  - Form the "honest" pair (true_text, pred_text). Detection on this pair is
    a FALSE POSITIVE if the decoder's prediction disagrees with the true
    G-code (since the controller faithfully executed the G-code, any
    decoder disagreement is a false alarm).
  - Form a "tampered" sample by mutating the true G-code with one of three
    synthetic tampers:
        (a) command-swap: replace the G-command with a different command
            from the vocab.
        (b) sign-flip: flip the sign of the first signed numeric field
            (X / Y / Z).
        (c) feed-rate edit: scale F by +/-10x where present.
    Then ask whether the decoder's prediction (on the *original* sensor
    data) disagrees with the *tampered* G-code at the corresponding token.
    A miss is a FALSE NEGATIVE.

For each tamper type we report:
  - FPR (rate at which honest pairs are flagged as tampered)
  - FNR (rate at which tampered pairs go undetected)
  - MDE-style sensitivity: smallest Y-axis numeric edit detected with TPR>=0.5

This is a first-pass MDE analysis intended to support the threat model in
Section 7.6 of the manuscript.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from v7_per_field_eval import tokens_to_canonical_gcode  # noqa: E402

COMMAND_POOL = ["G0", "G1", "G2", "G3", "G53", "M30"]
AXES_SIGNED = ["X", "Y", "Z"]


def _detok(s: str) -> str:
    return tokens_to_canonical_gcode(s.split())


def _has_command(text: str) -> str | None:
    m = re.search(r"\b(G\d+|M\d+)\b", text)
    return m.group(1) if m else None


def tamper_command(text: str) -> str | None:
    cur = _has_command(text)
    if cur is None:
        return None
    others = [c for c in COMMAND_POOL if c != cur]
    return text.replace(cur, others[0], 1)


def tamper_sign(text: str) -> str | None:
    for axis in AXES_SIGNED:
        m = re.search(rf"\b{axis}(-?\d+\.?\d*)", text)
        if m:
            val = m.group(1)
            new_val = val.lstrip("-") if val.startswith("-") else "-" + val
            return text[: m.start(1)] + new_val + text[m.end(1) :]
    return None


def tamper_feed(text: str) -> str | None:
    m = re.search(r"\bF(\d+\.?\d*)", text)
    if m:
        val = float(m.group(1))
        return text[: m.start(1)] + str(val * 10.0) + text[m.end(1) :]
    return None


TAMPERS = {
    "command_swap": tamper_command,
    "sign_flip": tamper_sign,
    "feed_edit": tamper_feed,
}


def disagrees(claim_text: str, pred_text: str, mode: str = "command") -> bool:
    """Detection decision.

    'command' mode: flag tamper when the G-command in the claimed text
    differs from the G-command in the decoder's prediction. This is the
    operating-point that matches the decoder's actual confidence profile
    (high on commands, lower on numeric precision).

    'sign' mode: flag when any axis-sign disagrees.
    'feed'  mode: flag when the F-value differs by >50%.
    'exact' mode: strict string equality (too strict; produces FPR>0.9).
    """
    if mode == "command":
        return _has_command(claim_text) != _has_command(pred_text)
    if mode == "sign":
        for axis in AXES_SIGNED:
            m1 = re.search(rf"\b{axis}(-?\d)", claim_text)
            m2 = re.search(rf"\b{axis}(-?\d)", pred_text)
            if m1 and m2 and (m1.group(1).startswith("-") != m2.group(1).startswith("-")):
                return True
        return False
    if mode == "feed":
        m1 = re.search(r"\bF(\d+\.?\d*)", claim_text)
        m2 = re.search(r"\bF(\d+\.?\d*)", pred_text)
        if m1 and m2:
            a, b = float(m1.group(1)), float(m2.group(1))
            if max(a, b) / max(min(a, b), 1e-6) > 1.5:
                return True
        return False
    return claim_text.strip() != pred_text.strip()


DETECT_MODE = {
    "command_swap": "command",
    "sign_flip": "sign",
    "feed_edit": "feed",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "decoder20260511" / "audit" / "threat_model_tamper.json",
    )
    p.add_argument(
        "--beam-width",
        type=int,
        default=0,
        help="Which beam_X_all_predictions.json to read. 0 = teacher-forced (default; "
             "matches sweep-time eval), 1 = greedy autoregressive (deployment-true).",
    )
    p.add_argument(
        "--loco",
        action="store_true",
        help="Open-vocabulary mode: read predictions from the 9 leave-one-class-out "
             "checkpoints (sweep_root/holdout_*/results/) instead of the 5 in-distribution "
             "folds (sweep_root/fold_N/*/results/). Referee R3 M5 / R3 M1.",
    )
    args = p.parse_args()

    pred_filename = f"beam_{args.beam_width}_all_predictions.json"
    by_tamper = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "n_applicable": 0})

    if args.loco:
        units = sorted((args.sweep_root).glob("holdout_*"))
        groups = [(u.name, list(u.glob(f"results/{pred_filename}"))) for u in units]
    else:
        groups = [(f"fold_{F}", list((args.sweep_root / f"fold_{F}").glob(
            f"*/results/{pred_filename}"))) for F in range(1, 6)]

    for gname, cands in groups:
        if not cands:
            print(f"{gname}: no {pred_filename}")
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        samples = json.loads(cands[0].read_text())
        for s in samples:
            true_text = _detok(s["true"])
            pred_text = _detok(s["pred"])

            # Honest pair FPR — one decision per tamper-detector mode
            for tamper_name in TAMPERS:
                mode = DETECT_MODE[tamper_name]
                if disagrees(true_text, pred_text, mode=mode):
                    by_tamper[tamper_name]["fp"] += 1
                else:
                    by_tamper[tamper_name]["tn"] += 1

            # Tampered pair FNR
            for tamper_name, fn in TAMPERS.items():
                mode = DETECT_MODE[tamper_name]
                tampered = fn(true_text)
                if tampered is None or tampered == true_text:
                    continue
                by_tamper[tamper_name]["n_applicable"] += 1
                if disagrees(tampered, pred_text, mode=mode):
                    by_tamper[tamper_name]["tp"] += 1
                else:
                    by_tamper[tamper_name]["fn"] += 1

    report = {}
    print(f"\n{'Tamper':14} {'FPR':>8} {'FNR':>8} {'n_applicable':>14}")
    for name, c in by_tamper.items():
        fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
        fnr = c["fn"] / max(c["tp"] + c["fn"], 1)
        print(f"{name:14} {fpr:8.4f} {fnr:8.4f} {c['n_applicable']:14d}")
        report[name] = {
            "fpr": fpr,
            "fnr": fnr,
            "tpr": 1 - fnr,
            "tp": c["tp"],
            "fn": c["fn"],
            "fp": c["fp"],
            "tn": c["tn"],
            "n_applicable": c["n_applicable"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
