#!/usr/bin/env python3
"""Build the Design B "no-numbers" vocabulary + re-tokenized 5-fold targets.

Design B (action item 2, 2026-05-21 follow-up meeting): retrain the decoder
with every numeric coordinate VALUE collapsed to a single ``<NUM>`` placeholder
token, so the model cannot emit a wrong number and cannot desync the
autoregressive decode. Its autoregressive structural accuracy is the clean
answer Design A (eval-masking) could not give.

This script does the no-compute data prep:

  1. Collapse ``data/gcode_vocab_v8.json`` -> ``data/gcode_vocab_v8_nonum.json``:
     every ``NUM_*`` value token and every fused dotted literal (``Z0.``,
     ``F22.``, ...) is dropped; one ``<NUM>`` token is added. Structural tokens
     (axis letters, G/M command codes, specials) are kept. ~2418 -> ~24 entries.

  2. Re-tokenize the 5-fold full_window NPZ targets by a pure id-remap
     (numeric ids -> ``<NUM>`` id; structural ids unchanged), keeping every
     other array (sensor data, splits, gcode_texts) byte-identical. The ONLY
     change vs. the current data is numeric->placeholder, so Design B is a
     clean A/B against the current model.

The numeric/structural partition (``NUM_*`` or contains ``.``) is exactly the
one ``scripts/analysis/no_numeric_ablation.py`` used for Design A, so the two
experiments stay directly comparable.

Outputs:
  data/gcode_vocab_v8_nonum.json
  outputs/decoder20260511/preprocessed_f98_nonum/full_window/fold_{1..5}/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/seacuello/Documents/gcode_fingerprinting")
SRC_VOCAB = REPO / "data/gcode_vocab_v8.json"
DST_VOCAB = REPO / "data/gcode_vocab_v8_nonum.json"
SRC_ROOT = REPO / "outputs/decoder20260511/preprocessed_f98/full_window"
DST_ROOT = REPO / "outputs/decoder20260511/preprocessed_f98_nonum/full_window"
PLACEHOLDER = "<NUM>"
SPLITS = ["train", "val", "test"]


def is_numeric(tok: str) -> bool:
    """A coordinate-VALUE token: a NUM_* entry or a fused dotted literal."""
    return tok.startswith("NUM_") or ("." in tok)


def build_vocab():
    """Collapse the v8 vocab; return (old_inv, new_vocab, remap array)."""
    src = json.loads(SRC_VOCAB.read_text())
    old_vocab = src["vocab"]                       # {token: id}
    old_inv = {i: t for t, i in old_vocab.items()}

    # collapsed vocab: keep non-numeric tokens in original id order, append <NUM>
    new_vocab: dict[str, int] = {}
    for tok, _ in sorted(old_vocab.items(), key=lambda kv: kv[1]):
        if not is_numeric(tok):
            new_vocab[tok] = len(new_vocab)
    new_vocab[PLACEHOLDER] = len(new_vocab)

    # remap: old_id -> new_id  (numeric -> <NUM>, structural -> its new id)
    remap = np.full(len(old_vocab), -1, dtype=np.int64)
    for tok, oid in old_vocab.items():
        remap[oid] = new_vocab[PLACEHOLDER] if is_numeric(tok) else new_vocab[tok]
    assert (remap >= 0).all(), "unmapped old id"

    # collapse_numeric flag tells GCodeTokenizer to emit <NUM> when it
    # re-tokenizes gcode_texts at train/eval time (Design B).
    cfg = dict(src["config"])
    cfg["collapse_numeric"] = True
    DST_VOCAB.write_text(json.dumps({"config": cfg, "vocab": new_vocab}, indent=2))
    n_num = sum(1 for t in old_vocab if is_numeric(t))
    print(f"[vocab] {SRC_VOCAB.name}: {len(old_vocab)} tokens "
          f"({n_num} numeric collapsed) -> {DST_VOCAB.name}: {len(new_vocab)} tokens")
    print(f"[vocab] collapsed vocab: {list(new_vocab)}")
    return old_inv, new_vocab, remap


def retokenize(old_inv, new_vocab, remap):
    """Re-tokenize all 5 folds; verify each split."""
    placeholder_id = new_vocab[PLACEHOLDER]
    old_is_num = np.array([is_numeric(old_inv[i]) for i in range(len(remap))])
    new_inv = {i: t for t, i in new_vocab.items()}
    all_ok = True

    for fold in range(1, 6):
        sdir = SRC_ROOT / f"fold_{fold}"
        ddir = DST_ROOT / f"fold_{fold}"
        ddir.mkdir(parents=True, exist_ok=True)

        # copy json sidecars, patching vocab_size where present
        for jf in sorted(sdir.glob("*.json")):
            obj = json.loads(jf.read_text())
            if isinstance(obj, dict) and "vocab_size" in obj:
                obj["vocab_size"] = len(new_vocab)
            (ddir / jf.name).write_text(json.dumps(obj, indent=2))

        for split in SPLITS:
            d = np.load(sdir / f"{split}_sequences.npz", allow_pickle=True)
            arrays = {k: d[k] for k in d.files}
            old_tok = arrays["tokens"]
            assert int(old_tok.max()) < len(remap), "token id out of vocab range"

            new_tok = remap[old_tok]
            arrays["tokens"] = new_tok
            np.savez_compressed(ddir / f"{split}_sequences.npz", **arrays)

            # ---- verification --------------------------------------------
            num_mask = old_is_num[old_tok]                       # [N, L] bool
            ok = True
            ok &= new_tok.shape == old_tok.shape
            ok &= bool((new_tok[num_mask] == placeholder_id).all())   # numeric -> <NUM>
            ok &= bool((new_tok[~num_mask] != placeholder_id).all())  # structural never <NUM>
            ok &= bool((new_tok[old_tok == 0] == 0).all())            # PAD preserved
            ok &= bool((arrays["token_length"] == d["token_length"]).all())  # length-preserving
            nonpad = old_tok != 0
            frac_num = float((new_tok[nonpad] == placeholder_id).mean())
            all_ok &= ok
            print(f"[fold {fold}/{split:5s}] tokens {old_tok.shape} "
                  f"<NUM>-share={frac_num:.3f}  verify={'OK' if ok else 'FAIL'}")

    # decode a sample line (fold 1 test, row 0) old vs new
    d = np.load(SRC_ROOT / "fold_1/test_sequences.npz", allow_pickle=True)
    row = d["tokens"][0][:24]
    old_s = " ".join(old_inv[int(i)] for i in row)
    new_s = " ".join(new_inv[int(remap[int(i)])] for i in row)
    print(f"[sample] old: {old_s}")
    print(f"[sample] new: {new_s}")
    return all_ok


def check_decomposer():
    """Confirm the collapsed vocab decomposes cleanly (no crash, <NUM> -> SPECIAL)."""
    sys.path.insert(0, str(REPO / "src"))
    from miracle.dataset.target_utils import TokenDecomposer
    dec = TokenDecomposer(str(DST_VOCAB))
    checks = {PLACEHOLDER: TokenDecomposer.TYPE_SPECIAL,
              "X": TokenDecomposer.TYPE_PARAMETER,
              "G1": TokenDecomposer.TYPE_COMMAND}
    for tok, want_type in checks.items():
        tid = dec.vocab[tok]
        got = dec.decompose_token(tid)[0]
        flag = "OK" if got == want_type else "FAIL"
        print(f"[decomposer] {tok!r:>9} -> type={got} (want {want_type})  {flag}")
        assert got == want_type, f"{tok} decomposed to type {got}, expected {want_type}"


def main():
    old_inv, new_vocab, remap = build_vocab()
    ok = retokenize(old_inv, new_vocab, remap)
    check_decomposer()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'VERIFICATION FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
