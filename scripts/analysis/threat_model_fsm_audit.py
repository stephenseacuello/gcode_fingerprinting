#!/usr/bin/env python3
"""Threat-model audit on FSM-equipped AR predictions.

Adapter over threat_model_tamper_injection.py that reads from the
'*_fsm/results/beam_1_all_predictions.json' paths produced by the FSM
inference batch.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from threat_model_tamper_injection import (
    TAMPERS, DETECT_MODE, disagrees, _detok
)


def audit_fsm_predictions(sweep_root: Path):
    by_tamper = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "n_applicable": 0})
    n_folds = 0
    for F in range(1, 6):
        cands = list(sweep_root.glob(f"fold_{F}/*_fsm/results/beam_1_all_predictions.json"))
        if not cands:
            print(f"fold_{F}: no fsm beam_1 predictions")
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        samples = json.loads(cands[0].read_text())
        n_folds += 1
        for s in samples:
            true_text = _detok(s["true"])
            pred_text = _detok(s["pred"])
            for tamper_name in TAMPERS:
                mode = DETECT_MODE[tamper_name]
                if disagrees(true_text, pred_text, mode=mode):
                    by_tamper[tamper_name]["fp"] += 1
                else:
                    by_tamper[tamper_name]["tn"] += 1
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
    return by_tamper, n_folds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    by_tamper, n_folds = audit_fsm_predictions(args.sweep_root)
    report = {}
    print(f"\n{args.sweep_root.name} (FSM, {n_folds} folds)")
    print(f"{'Tamper':14} {'FPR':>8} {'FNR':>8} {'TPR':>8} {'n_applicable':>14}")
    for name, c in by_tamper.items():
        fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
        fnr = c["fn"] / max(c["tp"] + c["fn"], 1)
        tpr = 1 - fnr
        print(f"{name:14} {fpr:8.4f} {fnr:8.4f} {tpr:8.4f} {c['n_applicable']:14d}")
        report[name] = {
            "fpr": fpr, "fnr": fnr, "tpr": tpr,
            "tp": c["tp"], "fn": c["fn"], "fp": c["fp"], "tn": c["tn"],
            "n_applicable": c["n_applicable"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
