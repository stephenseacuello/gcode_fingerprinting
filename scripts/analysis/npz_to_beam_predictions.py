#!/usr/bin/env python3
"""Convert predictions.npz (from full_window 5-fold eval) to
beam_0_all_predictions.json format, so v8_per_field_eval.py can score it.

The NPZ stores PAD-padded token arrays; we strip PAD and detokenize back to
the space-joined token-string form that v8_per_field_eval._detok expects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAD_ID = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True,
                   help="V8 vocab JSON; id2token must be invertible.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    vocab = json.loads(args.vocab.read_text())
    if "vocab" in vocab and isinstance(vocab["vocab"], dict):
        id2tok = {int(v): k for k, v in vocab["vocab"].items()}
    elif "id_to_token" in vocab:
        id2tok = {int(k): v for k, v in vocab["id_to_token"].items()}
    elif "token_to_id" in vocab:
        id2tok = {int(v): k for k, v in vocab["token_to_id"].items()}
    else:
        raise SystemExit(f"unknown vocab shape: keys={list(vocab.keys())}")

    d = np.load(args.npz, allow_pickle=True)
    pred_tokens = d["pred_tokens"]
    target_tokens = d["target_tokens"]
    n = pred_tokens.shape[0]

    samples = []
    for i in range(n):
        t = [int(x) for x in target_tokens[i].tolist() if x != PAD_ID]
        # PAD positions in pred align with target; use target length to crop
        p = [int(x) for x in pred_tokens[i, :len(t)].tolist()]
        true_str = " ".join(id2tok.get(x, f"<UNK_{x}>") for x in t)
        pred_str = " ".join(id2tok.get(x, f"<UNK_{x}>") for x in p)
        samples.append({"true": true_str, "pred": pred_str})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(samples))
    print(f"wrote {args.out} ({n} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
