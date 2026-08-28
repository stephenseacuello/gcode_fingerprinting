#!/usr/bin/env python3
"""Assemble the released non-neural baseline artifact `audit/nonneural_baselines.json`.

The decoder papers' Non-Neural Baseline section cites `audit/nonneural_baselines.json`
as the combined HGB + MLP comparison. The two underlying experiments exist as
separate artifacts:

  - audit/xgboost_baseline_v8.json  -- HistGradientBoosting per-head results
    (script: scripts/analysis/xgboost_baseline.py; NOTE its config.baseline
    label says "XGBoost" but the model actually trained is sklearn's
    HistGradientBoostingClassifier -- the installed xgboost build is GPU-only
    and crashes on CPU nodes; see that script's docstring).
  - audit/encoder_probe_v8.json     -- two-layer MLP probe accuracies
    (script: scripts/analysis/encoder_linear_probe.py). The probe run recorded
    accuracy only, so the command-head macro-F1 is recomputed here by retraining
    the identical probe (same architecture, data, epochs) with a fixed seed.

This script merges both into the promised combined artifact, with provenance.

Usage: python scripts/analysis/nonneural_baselines_export.py [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from encoder_linear_probe import MLPProbe, parse_split  # noqa: E402

HGB_JSON = REPO / "outputs/decoder20260511/audit/xgboost_baseline_v8.json"
PROBE_JSON = REPO / "outputs/decoder20260511/audit/encoder_probe_v8.json"
OUT_JSON = REPO / "outputs/decoder20260511/audit/nonneural_baselines.json"

SEED = 42
CMD_CLASSES = ["none", "G0", "G1", "G2", "G3"]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:  # class absent from both
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def rerun_mlp_command(memory_root: Path, data_root: Path, epochs: int, device: str) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    feats, labels = {}, {}
    for split in ("train", "val", "test"):
        d = np.load(data_root / f"{split}_sequences.npz", allow_pickle=True)
        parsed = parse_split(d["gcode_texts"])
        idx = {c: i for i, c in enumerate(CMD_CLASSES)}
        labels[split] = np.array([idx.get(c, 0) for c in parsed["cmd"]])
        mem_name = "train_memory.pt" if split == "train" else f"{split}_memory.pt"
        mem = torch.load(memory_root / mem_name, map_location="cpu")
        feats[split] = mem.mean(dim=1).numpy().astype(np.float32)
        del mem

    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    X = {s: torch.tensor(feats[s]).to(dev) for s in feats}
    y = {s: torch.tensor(labels[s], dtype=torch.long).to(dev) for s in labels}

    model = MLPProbe(feats["train"].shape[1], len(CMD_CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val, best_state, patience = -1.0, None, 10
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(X["train"]), y["train"]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(X["val"]).argmax(1) == y["val"]).float().mean().item()
        if val_acc > best_val:
            best_val, patience = val_acc, 10
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X["test"]).argmax(1).cpu().numpy()
    true = labels["test"]
    return {
        "seed": SEED,
        "test_accuracy": float((pred == true).mean()),
        "test_macro_f1": macro_f1(true, pred, len(CMD_CLASSES)),
        "val_accuracy_best": best_val,
        "n_test": int(len(true)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    hgb = json.loads(HGB_JSON.read_text())
    probe = json.loads(PROBE_JSON.read_text())

    memory_root = Path(probe["config"]["memory_root"])
    data_root = Path(probe["config"]["data_root"])
    print("Retraining MLP command probe (seeded) for macro-F1...")
    mlp_cmd = rerun_mlp_command(memory_root, data_root, args.epochs, args.device)
    print(f"  rerun: acc={mlp_cmd['test_accuracy']:.4f} macroF1={mlp_cmd['test_macro_f1']:.4f}"
          f" (original recorded acc={probe['probe_results']['probe_command_acc']:.4f})")

    out = {
        "description": (
            "Combined non-neural baselines on the mean-pooled encoder embedding "
            "the decoder consumes (fold 1, per-row V8 targets): a scikit-learn "
            "HistGradientBoosting (HGB) tree-booster per head, and the two-layer "
            "MLP probe. This is the combined artifact the decoder papers' "
            "Non-Neural Baseline section cites."
        ),
        "provenance": {
            "hgb_source": "audit/xgboost_baseline_v8.json",
            "hgb_note": (
                "That artifact's config.baseline string says 'XGBoost' but the "
                "trained model is sklearn HistGradientBoostingClassifier (the "
                "installed xgboost build is GPU-only); see "
                "scripts/analysis/xgboost_baseline.py docstring."
            ),
            "mlp_source": "audit/encoder_probe_v8.json",
            "mlp_note": (
                "The original probe run recorded accuracy only; the command-head "
                "macro-F1 here is a seeded retrain of the identical probe "
                "(scripts/analysis/encoder_linear_probe.py architecture, same "
                "memory/data/epochs). Its test accuracy is reported alongside as "
                "a validity anchor against the original recorded accuracy."
            ),
            "generator": "scripts/analysis/nonneural_baselines_export.py",
        },
        "config": {
            "memory_root": probe["config"]["memory_root"],
            "data_root": probe["config"]["data_root"],
            "hgb_config": hgb["config"],
        },
        "hgb_results_per_head": hgb["results_per_head"],
        "mlp_probe_results": probe["probe_results"],
        "mlp_command_rerun": mlp_cmd,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
