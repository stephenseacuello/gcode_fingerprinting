"""Phase-3 (decoder20260511) DecoderDataset behavioral tests.

The hardened DecoderDataset (`src/miracle/dataset/decoder_dataset.py`) must:
  1. Refuse to load V7 NPZs that have `lengths == window_size` (sensor length).
  2. Refuse to silently truncate decoder targets when `max_token_len` is too small.
  3. Load V8 NPZs cleanly when `max_token_len` is large enough.
  4. Auto-detect the correct `token_length` field on V8 NPZs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(REPO / "src"))

from miracle.dataset.decoder_dataset import DecoderDataset  # noqa: E402

V7_NPZ = REPO / "outputs/decoder20260304/preprocessed_v7/fold_1/train_sequences.npz"
V8_FULL = REPO / "outputs/decoder20260511/preprocessed/full_window/fold_1/train_sequences.npz"
V8_PER = REPO / "outputs/decoder20260511/preprocessed/per_row/fold_1/train_sequences.npz"


@pytest.mark.skipif(not V7_NPZ.exists(), reason="V7 NPZ not on disk")
def test_v7_npz_rejected_with_clear_message():
    """Loading a V7 NPZ must raise ValueError pointing at the lengths bug."""
    with pytest.raises(ValueError) as exc:
        DecoderDataset(V7_NPZ, max_token_len=16)
    msg = str(exc.value)
    assert "SENSOR window length" in msg
    assert "AUDIT_REPORT.md" in msg


@pytest.mark.skipif(not V8_FULL.exists(), reason="V8 full_window NPZ not on disk")
def test_v8_full_window_loads_with_large_cap():
    """V8 full_window has content up to ~1339 tokens — must load with max_token_len=1500."""
    ds = DecoderDataset(V8_FULL, max_token_len=1500)
    assert len(ds) > 0
    assert ds.length_field == "token_length"
    # Every loaded sample preserves its full content (no truncation)
    assert ds._truncation_count == 0


@pytest.mark.skipif(not V8_FULL.exists(), reason="V8 full_window NPZ not on disk")
def test_v8_full_window_hard_asserts_on_small_cap():
    """V8 full_window with cap=64 must raise AssertionError (not silently truncate)."""
    with pytest.raises(AssertionError) as exc:
        DecoderDataset(V8_FULL, max_token_len=64)
    assert "Silent truncation prohibited" in str(exc.value)
    assert "max_token_len" in str(exc.value)


@pytest.mark.skipif(not V8_FULL.exists(), reason="V8 full_window NPZ not on disk")
def test_allow_truncation_opts_in_to_legacy_behavior():
    """If allow_truncation=True, the dataset truncates and counts the events."""
    ds = DecoderDataset(V8_FULL, max_token_len=64, allow_truncation=True)
    assert ds._truncation_count == len(ds)  # All 303 samples exceeded 62


@pytest.mark.skipif(not V8_PER.exists(), reason="V8 per_row NPZ not on disk")
def test_v8_per_row_loads_cleanly_at_default_cap():
    """V8 per_row has ≤ 8 content tokens; max_token_len=32 should succeed."""
    ds = DecoderDataset(V8_PER, max_token_len=32)
    assert len(ds) > 0
    assert ds._truncation_count == 0


@pytest.mark.skipif(not V8_FULL.exists(), reason="V8 full_window NPZ not on disk")
def test_token_length_field_takes_priority_in_auto_mode():
    """`length_field='auto'` picks `token_length` when present, not `lengths`."""
    ds = DecoderDataset(V8_FULL, max_token_len=1500, length_field="auto")
    assert ds.length_field == "token_length"


@pytest.mark.skipif(not V8_PER.exists(), reason="V8 per_row NPZ not on disk")
def test_input_target_shifted_by_one():
    """Verify BOS prefix / EOS suffix relationship on a sample."""
    ds = DecoderDataset(V8_PER, max_token_len=32)
    for idx in range(min(10, len(ds))):
        sample = ds[idx]
        inp = sample["input_tokens"]
        tgt = sample["target_tokens"]
        L = int(sample["length"].item())
        # input starts with BOS
        assert inp[0].item() == 1  # BOS
        # target ends with EOS at position L-1 (length includes EOS)
        if L >= 1:
            assert tgt[L - 1].item() == 2  # EOS
        # Shifted by one: tgt[:L-1] == inp[1:L]
        for j in range(L - 1):
            assert tgt[j].item() == inp[j + 1].item(), f"shift broken at idx {idx} pos {j}"
