"""
preprocess_alt.py
=================

Re-tokenize the v7 fold-1 NPZ files using an alternative tokenizer family.
Produces a new directory containing train/val/test NPZ files with everything
preserved EXCEPT the `tokens` array, which is replaced with the alternative
tokenizer's encoding of `gcode_texts`.

The output NPZ matches the schema that DecoderDataset expects:
    continuous, categorical, tokens, lengths, gcode_texts, operation_type,
    operation_type_names, window_index (optional), total_windows (optional),
    source_file (optional)

Usage:
    python3 scripts/experiments/grammar_design_space/preprocess_alt.py \
        --tokenizer domain_bpe \
        --in-fold outputs/decoder20260304/preprocessed_v7/fold_1 \
        --out-base outputs/decoder20260303/grammarpaper/alt_tokenizers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from alt_tokenizers import (  # noqa: E402
    BaseAltTokenizer, TOKENIZER_REGISTRY, make,
    PAD_ID, BOS_ID, EOS_ID,
)


def fit_tokenizer(name: str, train_texts: list[str]) -> BaseAltTokenizer:
    print(f"  Training tokenizer: {name} on {len(train_texts)} lines...")
    tok = make(name)
    tok.train(train_texts)
    print(f"  vocab_size = {tok.vocab_size()}")
    return tok


def encode_corpus(tok: BaseAltTokenizer, texts, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode each text into a fixed-width int64 array padded to max_len.

    Returns:
        tokens : [N, max_len] int64, padded with PAD_ID
        lengths: [N] int64, true token sequence lengths (incl. BOS/EOS)
    """
    n = len(texts)
    tokens = np.full((n, max_len), PAD_ID, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    n_truncated = 0
    for i, text in enumerate(texts):
        ids = tok.encode(str(text), add_special=True)
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [EOS_ID]
            n_truncated += 1
        tokens[i, :len(ids)] = ids
        lengths[i] = len(ids)
    return tokens, lengths, n_truncated


def process_split(in_npz: Path, out_npz: Path, tok: BaseAltTokenizer, max_len: int) -> dict:
    print(f"  -> {in_npz.name}")
    src = np.load(in_npz, allow_pickle=True)
    if "gcode_texts" not in src.files:
        raise KeyError(f"{in_npz} has no gcode_texts; cannot re-tokenize")
    texts = src["gcode_texts"]

    new_tokens, new_lengths, n_trunc = encode_corpus(tok, texts, max_len)

    # Preserve everything else
    payload = {k: src[k] for k in src.files if k != "tokens"}
    # Replace tokens (and overwrite the legacy `lengths` field with the *token* length,
    # since the v7 NPZ used `lengths` for sensor windows and that's confusing here).
    payload["tokens"] = new_tokens
    payload["token_lengths"] = new_lengths
    # Keep the original sensor-window lengths under a clear name
    if "lengths" in src.files:
        payload["sensor_lengths"] = src["lengths"]
    # Use token_lengths as the canonical lengths for downstream loaders
    payload["lengths"] = new_lengths

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)
    return {
        "n_samples": len(texts),
        "max_len": int(new_lengths.max()),
        "mean_len": float(new_lengths.mean()),
        "n_truncated": int(n_trunc),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, choices=list(TOKENIZER_REGISTRY))
    p.add_argument("--fold", type=int, default=1, help="Fold number (1..5)")
    p.add_argument("--in-base", default="outputs/decoder20260304/preprocessed_v7",
                   help="Base directory containing fold_<N>/ subdirs with v7 NPZ files")
    p.add_argument("--out-base", default="outputs/decoder20260303/grammarpaper/alt_tokenizers")
    p.add_argument("--max-len", type=int, default=64,
                   help="Pad/truncate token sequences to this length. Char-level needs ~64; BPE needs ~32; flat ~16.")
    args = p.parse_args()

    in_fold = Path(args.in_base) / f"fold_{args.fold}"
    out_dir = Path(args.out_base) / args.tokenizer / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Preprocessing fold_{args.fold} with tokenizer={args.tokenizer} ===")
    print(f"  in:  {in_fold}")
    print(f"  out: {out_dir}")
    print(f"  max_len: {args.max_len}")

    train_npz = in_fold / "train_sequences.npz"
    val_npz = in_fold / "val_sequences.npz"
    test_npz = in_fold / "test_sequences.npz"

    # 1. Train the tokenizer on the *train* split's gcode_texts
    print("\n[1/3] Loading train texts to fit tokenizer...")
    train_data = np.load(train_npz, allow_pickle=True)
    train_texts = [str(t) for t in train_data["gcode_texts"]]
    tok = fit_tokenizer(args.tokenizer, train_texts)

    # 2. Save the tokenizer
    tok_path = out_dir / "tokenizer.bin"
    print(f"\n[2/3] Saving tokenizer -> {tok_path}")
    tok.save(tok_path)

    # 3. Encode all three splits
    print("\n[3/3] Re-tokenizing splits...")
    stats = {}
    for split_npz in [train_npz, val_npz, test_npz]:
        if not split_npz.exists():
            print(f"  skipping {split_npz.name} (not found)")
            continue
        out_path = out_dir / split_npz.name
        s = process_split(split_npz, out_path, tok, args.max_len)
        stats[split_npz.name] = s
        print(f"     n={s['n_samples']}, mean_len={s['mean_len']:.1f}, "
              f"max_len_used={s['max_len']}, truncated={s['n_truncated']}")

    # Persist a small summary
    import json
    summary = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size(),
        "max_len": args.max_len,
        "in_fold": str(in_fold),
        "splits": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
