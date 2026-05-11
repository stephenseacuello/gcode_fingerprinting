"""Phase-3 (decoder20260511) preprocessing invariants.

These tests pin down the V8 NPZ schema so the V7 lengths-is-sensor-length
bug cannot return silently. They operate on the existing
`outputs/decoder20260511/preprocessed/` artifacts produced by
`scripts/preprocessing/run_preprocessing_v8_cv_fold.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PREPROC_ROOT = REPO / "outputs" / "decoder20260511" / "preprocessed"

PAD, BOS, EOS = 0, 1, 2

FOLDS = [1, 2, 3, 4, 5]
SPLITS = ["train", "val", "test"]
MODES = ["full_window", "per_row"]


def _all_v8_npzs():
    paths = []
    for mode in MODES:
        for fold in FOLDS:
            for split in SPLITS:
                p = PREPROC_ROOT / mode / f"fold_{fold}" / f"{split}_sequences.npz"
                if p.exists():
                    paths.append(p)
    return paths


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet — run preprocess_v8_dual_modes.sh first")
def test_schema_present_in_every_v8_npz():
    """Every V8 NPZ must include the new schema fields."""
    required = {
        "tokens", "lengths", "token_length", "window_length",
        "gcode_texts", "operation_type", "window_index",
        "line_in_window_index", "n_lines_in_window", "label_mode",
        "continuous",
    }
    for path in _all_v8_npzs():
        d = np.load(path, allow_pickle=True)
        missing = required - set(d.files)
        assert not missing, f"{path} missing required fields: {missing}"


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_lengths_matches_token_length():
    """`lengths` is now an alias of `token_length` — verify it on every NPZ."""
    for path in _all_v8_npzs():
        d = np.load(path, allow_pickle=True)
        lengths = d["lengths"]
        token_length = d["token_length"]
        assert np.array_equal(lengths, token_length), (
            f"{path}: `lengths` and `token_length` disagree. V7 bug returning?"
        )


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_token_length_matches_derived_content_length():
    """`token_length` must equal the actual non-PAD/BOS/EOS count in tokens."""
    for path in _all_v8_npzs():
        d = np.load(path, allow_pickle=True)
        tokens = d["tokens"]
        token_length = d["token_length"]
        content_mask = (tokens != PAD) & (tokens != BOS) & (tokens != EOS)
        derived = content_mask.sum(axis=1)
        assert np.array_equal(token_length, derived), (
            f"{path}: token_length disagrees with derived content length. "
            f"first_mismatch_idx={int(np.where(token_length != derived)[0][0]) if (token_length != derived).any() else -1}"
        )


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_window_length_is_sensor_length():
    """`window_length` must equal `continuous.shape[1]` for every row."""
    for path in _all_v8_npzs():
        d = np.load(path, allow_pickle=True)
        window_length = d["window_length"]
        T_s = d["continuous"].shape[1]
        assert np.all(window_length == T_s), (
            f"{path}: window_length expected {T_s}, got unique {np.unique(window_length)}"
        )


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_full_window_targets_are_multiline():
    """Full-window NPZs must show evidence of multi-line targets (newlines in
    gcode_texts AND content-token max > 6, the V7 ceiling)."""
    for path in _all_v8_npzs():
        if "full_window" not in str(path):
            continue
        d = np.load(path, allow_pickle=True)
        texts = [str(t) for t in d["gcode_texts"]]
        n_multi = sum(1 for t in texts if "\n" in t and t.strip())
        assert n_multi > 0, f"{path}: no multi-line gcode_texts found"
        max_content = int(d["token_length"].max())
        assert max_content > 6, (
            f"{path}: token_length max is {max_content}, expected > 6 "
            f"(V7 collapsed everything to 6)"
        )


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_per_row_targets_are_singleline():
    """Per_row NPZs must have single-line gcode_texts (no newlines)."""
    for path in _all_v8_npzs():
        if "per_row" not in str(path):
            continue
        d = np.load(path, allow_pickle=True)
        texts = [str(t) for t in d["gcode_texts"]]
        bad = [t for t in texts if "\n" in t]
        assert not bad, f"{path}: per_row produced multi-line gcode_text: {bad[:3]}"


@pytest.mark.skipif(not (PREPROC_ROOT.exists() and any(_all_v8_npzs())),
                    reason="V8 NPZs not generated yet")
def test_per_row_emits_one_sample_per_line_per_window():
    """For per_row mode, `n_lines_in_window` must equal the number of samples
    sharing the same `(source_file, window_index)` key (modulo empty-window
    placeholders that emit one sample with n_lines_in_window=0)."""
    for path in _all_v8_npzs():
        if "per_row" not in str(path):
            continue
        d = np.load(path, allow_pickle=True)
        wi = d["window_index"]
        sf = d["source_file"]
        n_lines = d["n_lines_in_window"]
        from collections import Counter
        counts = Counter(zip(sf.tolist(), wi.tolist()))
        for (file_, wi_v), c in counts.items():
            expected = max(int(n_lines[(sf == file_) & (wi == wi_v)][0]), 1)
            assert c == expected, (
                f"{path}: ({file_}, window_idx={wi_v}) emitted {c} samples, "
                f"expected {expected}"
            )
