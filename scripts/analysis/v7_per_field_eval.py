#!/usr/bin/env python3
"""V7 actual decoder, evaluated per structured G-code field.

Phase-1 verification artifact. Read-only (loads existing V7 checkpoints,
does not modify them).

Produces the *ceiling* number for the manuscript reframe: given the V7
decoder (with sensors + window-position shortcut), what test accuracy /
MAE does it achieve per command / has_x / has_y / x_val / y_val / etc.?

Compare against `audit/recoverability_baseline.json` (metadata-only floor).
The gap is the empirical contribution of the sensor pathway.

Loads checkpoints from `outputs/decoder20260304/v7_best_5fold_multiseed/`
using the BEST_SEEDS map already used by `eval_full_per_class_v7.py`.

Output: `outputs/decoder20260511/audit/v7_per_field.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from run_decoder_quick_test import (  # type: ignore  # noqa: E402
    DecoderQuickTestDataset,
    CachedDecoderDataset,
    SensorMultiHeadDecoder,
    GCodeTokenizer,
    decoder_collate_fn,
)
from miracle.utilities.gcode_tokenizer import TokenizerConfig  # noqa: E402
from score_recoverability import parse_fields  # type: ignore  # noqa: E402

# Default G-code precision used at training time (mirrors DEFAULT_PRECISION in
# src/miracle/utilities/gcode_tokenizer.py). NUM_<ADDR>_<BIN> decodes to
# BIN * precision[ADDR].
DEFAULT_PRECISION = {
    "X": 1e-3, "Y": 1e-3, "Z": 1e-3,
    "A": 1e-3, "B": 1e-3, "C": 1e-3,
    "I": 1e-4, "J": 1e-4, "K": 1e-4,
    "F": 1.0,  "S": 10.0, "R": 1e-4,
    "P": 1e-3, "Q": 1e-3, "E": 1e-4,
}


def tokens_to_canonical_gcode(token_strs: list[str]) -> str:
    """Convert a stream of tokenizer token strings back to canonical G-code text.

    Examples:
      ["G0", "X", "NUM_X_1492", "Y", "NUM_Y_1485"] -> "G0 X1.492 Y1.485"
      ["G1", "NUM_X_-200"] -> "G1 X-0.2"      (orphan NUM still emits the address)
    """
    parts: list[str] = []
    i = 0
    while i < len(token_strs):
        t = token_strs[i]
        if not t or t in ("BOS", "EOS", "PAD", "UNK", "MASK", "?"):
            i += 1; continue
        if t.startswith("NUM_"):
            # Orphan numeric: try to recover address from token name itself.
            chunks = t.split("_")
            if len(chunks) >= 3:
                addr = chunks[1]
                try:
                    bin_val = int(chunks[2])
                    val = bin_val * DEFAULT_PRECISION.get(addr, 1.0)
                    parts.append(f"{addr}{val:g}")
                except ValueError:
                    pass
            i += 1; continue
        # Address token (single letter like X/Y/Z/I/J/R/F or G0/M3 etc.)
        if len(t) == 1 and t in "XYZIJKRFSABCPQE":
            # Look ahead for a NUM token of the matching address.
            if i + 1 < len(token_strs) and token_strs[i + 1].startswith(f"NUM_{t}_"):
                chunks = token_strs[i + 1].split("_")
                try:
                    bin_val = int(chunks[2])
                    val = bin_val * DEFAULT_PRECISION.get(t, 1.0)
                    parts.append(f"{t}{val:g}")
                    i += 2; continue
                except (ValueError, IndexError):
                    pass
            # Bare address token (no numeric follow-up)
            parts.append(t)
            i += 1; continue
        # G/M/T command tokens or anything else — keep as-is.
        parts.append(t)
        i += 1
    return " ".join(parts)

PROJECT = REPO
VOCAB_PATH = PROJECT / "data/gcode_vocab_712.json"
PREPROC_BASE = PROJECT / "outputs/decoder20260304/preprocessed_v7"
MULTISEED_BASE = PROJECT / "outputs/decoder20260304/v7_best_5fold_multiseed"
OUTPUT_PATH = PROJECT / "outputs/decoder20260511/audit/v7_per_field.json"

BEST_SEEDS = {1: 2024, 2: 123, 3: 456, 4: 2024, 5: 789}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer() -> tuple[GCodeTokenizer, dict[str, int], dict[int, str]]:
    with open(VOCAB_PATH) as f:
        vdata = json.load(f)
    vocab = vdata["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}
    cfg = TokenizerConfig(**vdata["config"])
    tokenizer = GCodeTokenizer(cfg, vocab=vocab)
    return tokenizer, vocab, id_to_token


def fold_inference(fold: int, seed: int, tokenizer, vocab, id_to_token) -> dict[str, Any]:
    BOS = vocab["BOS"]; EOS = vocab["EOS"]; PAD = vocab["PAD"]
    run_dir = MULTISEED_BASE / f"fold_{fold}_seed_{seed}"
    ckpt_path = run_dir / "decoder_checkpoint/best_decoder.pt"
    print(f"  loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["decoder_state_dict"]
    max_win = sd["window_pos_embed.weight"].shape[0] if "window_pos_embed.weight" in sd else 10

    decoder = SensorMultiHeadDecoder(
        vocab_size=712, d_model=384, n_heads=12, n_layers=8,
        sensor_dim=256, n_operations=9,
        memory_pos_encoding=True, use_sensor_prior=True,
        use_window_position=True, max_seq_len=16,
        max_windows_per_file=max_win,
    ).to(device)
    # set_vocab() registers grammar_mask + type_token_mask buffers so they appear
    # in state_dict and are overwritten by the checkpoint's values on load.
    decoder.set_vocab(vocab)
    model_state = decoder.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_state and v.shape == model_state[k].shape}
    decoder.load_state_dict(filtered, strict=False)
    decoder.eval()

    test_npz = PREPROC_BASE / f"fold_{fold}/test_sequences.npz"
    test_ds = DecoderQuickTestDataset(str(test_npz), tokenizer, max_token_len=16)

    memory = torch.load(run_dir / "encoder_memory/test_memory.pt", map_location=device, weights_only=False)
    op_pred = torch.load(run_dir / "encoder_memory/test_op_pred.pt", map_location=device, weights_only=False)

    npz_data = np.load(test_npz, allow_pickle=True)
    gcode_texts = [str(s) for s in npz_data["gcode_texts"]]
    op_names = [str(s) for s in npz_data["operation_type_names"]]

    aug_ds = CachedDecoderDataset(test_ds, memory, op_pred, multi_window_context=2, window_dropout=0.0, training=False)
    loader = DataLoader(aug_ds, batch_size=32, shuffle=False, collate_fn=decoder_collate_fn, num_workers=0)

    preds_str: list[str] = []
    targs_str: list[str] = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            inp = batch["input_tokens"].to(device)
            tgt = batch["target_tokens"].to(device)
            mem = batch["memory"].to(device)
            op = batch["op_pred"].to(device)
            pad_mask = (inp == PAD)
            extra = {}
            if "window_index" in batch:
                extra["window_index"] = batch["window_index"].to(device)
            if "total_windows" in batch:
                extra["total_windows"] = batch["total_windows"].to(device)
            outputs = decoder(
                tokens=inp, sensor_embeddings=mem, operation_type=op,
                tgt_key_padding_mask=pad_mask, **extra,
            )
            logits = outputs.get("legacy_logits", outputs.get("raw_legacy_logits"))
            pred_ids = logits.argmax(-1)

            B = inp.size(0)
            for b in range(B):
                if sample_idx >= len(gcode_texts):
                    break
                tgt_ids_full = [t.item() for t in tgt[b] if t.item() != PAD]
                tgt_no_eos = [t for t in tgt_ids_full if t != EOS]
                pred_no_special = [pred_ids[b, j].item() for j in range(len(tgt_no_eos))]
                pred_token_strs = [id_to_token.get(t, "?") for t in pred_no_special]
                # Reassemble tokenized prediction into canonical G-code text so the
                # regex field parser can extract command / addresses / values.
                pred_str = tokens_to_canonical_gcode(pred_token_strs)
                preds_str.append(pred_str)
                targs_str.append(gcode_texts[sample_idx])
                sample_idx += 1

    # ---- Parse both predicted and true text into structured fields ----
    pred_fields = [parse_fields(s) for s in preds_str]
    true_fields = [parse_fields(s) for s in targs_str]
    n = len(true_fields)

    categorical_fields = ["command", "has_x", "has_y", "has_z", "has_r", "has_f", "x_sign", "y_sign", "z_sign"]
    regression_fields = ["x_val", "y_val", "z_val", "f_val", "r_val", "i_val", "j_val"]

    out: dict[str, Any] = {"fold": fold, "seed": seed, "n": n, "categorical": {}, "regression": {}}

    for f in categorical_fields:
        y_true = np.array([d[f] for d in true_fields])
        y_pred = np.array([d[f] for d in pred_fields])
        # Mask-out NaN-equivalents — for the regression-coupled signs, treat 0 sign as "absent"
        n_total = int(len(y_true))
        match = int((y_true == y_pred).sum())
        out["categorical"][f] = {
            "n": n_total,
            "n_correct": match,
            "accuracy": float(match / max(n_total, 1)),
            "true_distribution": {str(k): int((y_true == k).sum()) for k in sorted(set(y_true.tolist()))},
        }

    for f in regression_fields:
        y_true = np.array([d[f] for d in true_fields], dtype=np.float64)
        y_pred = np.array([d[f] for d in pred_fields], dtype=np.float64)
        # Restrict to entries where the TRUE value is non-nan (i.e., the field is present in ground truth)
        mask_true = ~np.isnan(y_true)
        mask_pred = ~np.isnan(y_pred)
        both = mask_true & mask_pred
        n_true_present = int(mask_true.sum())
        n_pred_present_when_true = int(both.sum())
        # MAE where both present
        mae = float(np.abs(y_true[both] - y_pred[both]).mean()) if both.any() else float("nan")
        # Presence recall (did we predict any number when there should be one?)
        presence_recall = float(n_pred_present_when_true / max(n_true_present, 1))
        # False-positive rate (predicted a value when truth had none)
        false_pos = int((mask_pred & ~mask_true).sum())
        out["regression"][f] = {
            "n_true_present": n_true_present,
            "n_pred_present_when_true": n_pred_present_when_true,
            "presence_recall": presence_recall,
            "false_positive_count": false_pos,
            "mae_when_both_present": mae,
        }

    return out, preds_str, targs_str


def main() -> int:
    tokenizer, vocab, id_to_token = load_tokenizer()
    all_fold_reports = []
    all_examples = []

    for fold in [1, 2, 3, 4, 5]:
        seed = BEST_SEEDS[fold]
        print(f"\n=== Fold {fold} (seed={seed}) ===")
        try:
            r, preds, targs = fold_inference(fold, seed, tokenizer, vocab, id_to_token)
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            all_fold_reports.append({"fold": fold, "seed": seed, "error": repr(exc)})
            continue
        all_fold_reports.append(r)
        # Save up to 5 example mismatches per fold
        for p, t in zip(preds[:50], targs[:50]):
            all_examples.append({"fold": fold, "true": t, "pred": p})
        # Brief print
        c = r["categorical"]
        print(
            f"  cmd={c['command']['accuracy']:.3f}  "
            f"has_x={c['has_x']['accuracy']:.3f}  has_y={c['has_y']['accuracy']:.3f}  "
            f"x_sign={c['x_sign']['accuracy']:.3f}  y_sign={c['y_sign']['accuracy']:.3f}"
        )
        rg = r["regression"]
        print(
            f"  x_val: present {rg['x_val']['n_pred_present_when_true']}/{rg['x_val']['n_true_present']} "
            f"MAE={rg['x_val']['mae_when_both_present']:.3f}; "
            f"y_val MAE={rg['y_val']['mae_when_both_present']:.3f}"
        )

    # ---- Aggregate across folds ----
    aggregate = {"categorical": {}, "regression": {}}
    for f in ["command", "has_x", "has_y", "has_z", "has_r", "has_f", "x_sign", "y_sign", "z_sign"]:
        accs = []
        ns = 0
        for r in all_fold_reports:
            if "error" in r:
                continue
            s = r["categorical"].get(f)
            if s:
                accs.append(s["accuracy"])
                ns += s["n"]
        if accs:
            aggregate["categorical"][f] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "folds": len(accs), "n_total": ns}

    for f in ["x_val", "y_val", "z_val", "f_val", "r_val", "i_val", "j_val"]:
        maes = [r["regression"][f]["mae_when_both_present"] for r in all_fold_reports if "error" not in r and not np.isnan(r["regression"][f]["mae_when_both_present"])]
        rec = [r["regression"][f]["presence_recall"] for r in all_fold_reports if "error" not in r]
        if maes:
            aggregate["regression"][f] = {
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "presence_recall_mean": float(np.mean(rec)),
                "folds": len(maes),
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "folds": all_fold_reports,
        "aggregate": aggregate,
        "examples": all_examples[:100],
        "best_seeds": BEST_SEEDS,
    }, indent=2))
    print(f"\nwrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
