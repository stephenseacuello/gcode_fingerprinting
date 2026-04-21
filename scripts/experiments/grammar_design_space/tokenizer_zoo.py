"""
tokenizer_zoo.py
================

Common interface + implementations for the tokenizer-zoo experiments
backing the design-space table in the grammar paper (Section 5).

Each wrapper exposes:
    name : str
    train(corpus_lines) -> None
    encode(line) -> List[int]
    decode(ids) -> str          # best-effort string reconstruction
    vocab_size() -> int          # nominal vocabulary capacity
    active_vocab() -> Set[int]   # token ids actually emitted on encode() calls
    reset_active_vocab() -> None

The active_vocab tracking is what powers Experiment A (cumulative distinct
token types observed across the corpus). reset_active_vocab() is called
between runs.

Tokenizers implemented:
    HierarchicalTokenizer  -- our hybrid hierarchical tokenizer (digit decomposition)
    GPT4BPETokenizer       -- tiktoken cl100k_base (frozen pretrained)
    GPT2BPETokenizer       -- tiktoken r50k_base (frozen pretrained)
    DomainBPETokenizer     -- HF tokenizers BPE trained on the G-code corpus
    WordPieceTokenizer     -- HF tokenizers WordPiece trained on the G-code corpus
    CharLevelTokenizer     -- one-byte alphabet
    FlatPerValueTokenizer  -- each canonicalized line element becomes a token
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Make src/ importable so we can use the existing GCodeTokenizer
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class ITokenizer:
    name: str = "abstract"
    # Design-space property declarations (used to populate the table).
    # These are the *theoretical* claims; experiments verify them.
    bounded_vocab: bool = False
    lossless_coords: bool = False
    grammar_compatible: bool = False

    def __init__(self) -> None:
        self._active: Set[int] = set()

    def train(self, corpus_lines: List[str]) -> None:
        raise NotImplementedError

    def encode(self, line: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def active_vocab(self) -> Set[int]:
        return self._active

    def reset_active_vocab(self) -> None:
        self._active = set()

    def _record(self, ids: Iterable[int]) -> None:
        self._active.update(ids)


# ---------------------------------------------------------------------------
# 1. Hierarchical (ours)
# ---------------------------------------------------------------------------
class HierarchicalTokenizer(ITokenizer):
    """Our hybrid tokenizer with digit-level decomposition.

    For experiment purposes we treat the structural vocabulary (G0..G3, X..R,
    BOS/EOS, etc.) plus a fixed digit alphabet (0..9 + sign) as the operative
    vocabulary. The "active vocab" we track is the set of *structural*
    tokens emitted, not bucketed NUM_X_* tokens, because the digit heads
    represent values, not the vocabulary.

    To compare apples to apples with other tokenizers in vocabulary-growth
    plots, we count both structural tokens *and* the implicit digit alphabet
    tokens (10 digits + sign). This is the operative alphabet the model
    actually emits.
    """

    name = "Hierarchical (ours)"
    bounded_vocab = True
    lossless_coords = True
    grammar_compatible = True

    # 19 structural tokens (Table 2 in paper) + 11 digit alphabet (10 digits + sign)
    _STRUCTURAL = [
        "PAD", "BOS", "EOS", "UNK", "MASK",
        "G0", "G1", "G2", "G3",
        "X", "Y", "Z", "F", "R",
        "G53", "M3", "M5", "M6", "M30",
    ]
    _DIGIT_ALPHABET = [f"D{i}" for i in range(10)] + ["SIGN+", "SIGN-"]

    def __init__(self) -> None:
        super().__init__()
        # Build a fixed vocabulary independent of any corpus
        self._vocab = {tok: i for i, tok in enumerate(self._STRUCTURAL + self._DIGIT_ALPHABET)}
        # Underlying GCodeTokenizer handles canonicalization and tokenization
        cfg = TokenizerConfig(mode="hybrid", min_freq=1, vocab_size=15000)
        cfg.precision = {
            "X": 1e-3, "Y": 1e-3, "Z": 1e-3,
            "I": 1e-4, "J": 1e-4, "K": 1e-4,
            "F": 1.0, "S": 10.0, "R": 1e-4,
            "P": 1e-3, "Q": 1e-3, "E": 1e-4,
            "A": 1e-3, "B": 1e-3, "C": 1e-3,
        }
        self._gtok = GCodeTokenizer(cfg)

    def train(self, corpus_lines: List[str]) -> None:
        # Vocabulary is fixed and grammar-derived; no training required.
        return

    def encode(self, line: str) -> List[int]:
        canon = self._gtok.canonicalize_line(line)
        if canon is None:
            return []
        ids: List[int] = []
        for word in canon.split(" "):
            toks = self._gtok._tokenize_word(word)
            for t in toks:
                if t in self._vocab:
                    ids.append(self._vocab[t])
                elif t.startswith("NUM_"):
                    # Decompose NUM_<addr>_<bucket> into digit-alphabet ids
                    parts = t.split("_", 2)
                    if len(parts) == 3:
                        bucket = parts[2]
                        sign = "SIGN-" if bucket.startswith("-") else "SIGN+"
                        ids.append(self._vocab[sign])
                        for ch in bucket.lstrip("-"):
                            if ch.isdigit():
                                ids.append(self._vocab[f"D{ch}"])
                else:
                    # Unknown structural token: add to UNK
                    ids.append(self._vocab["UNK"])
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Reconstruct a G-code-like line from id sequence.

        Walks the id stream and recomposes structural tokens, then for any
        SIGN/digit run, collects digits and emits a numeric word using the
        last-seen address letter and that address's quantization precision.
        """
        inv = {i: t for t, i in self._vocab.items()}
        toks = [inv.get(i, "[UNK]") for i in ids]
        out: List[str] = []
        i = 0
        last_addr: Optional[str] = None
        while i < len(toks):
            t = toks[i]
            if t in ("SIGN+", "SIGN-"):
                sign = -1 if t == "SIGN-" else 1
                j = i + 1
                digits = []
                while j < len(toks) and toks[j].startswith("D") and len(toks[j]) == 2 and toks[j][1].isdigit():
                    digits.append(toks[j][1])
                    j += 1
                if digits and last_addr is not None:
                    bucket = sign * int("".join(digits))
                    precision = self._gtok.cfg.precision.get(last_addr, 1e-3)
                    value = bucket * precision
                    if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
                        out.append(f"{last_addr}{int(round(value))}")
                    else:
                        out.append(f"{last_addr}{value}")
                i = j
                continue
            if t in ("X", "Y", "Z", "F", "R", "I", "J", "K", "S", "P", "Q", "E", "A", "B", "C"):
                last_addr = t
                # Don't emit the bare address; it'll come paired with the value
                i += 1
                continue
            if t in ("BOS", "EOS", "PAD", "UNK", "MASK"):
                i += 1
                continue
            out.append(t)
            i += 1
        return " ".join(out)

    def vocab_size(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# 2. GPT-4 BPE (tiktoken cl100k_base)
# ---------------------------------------------------------------------------
class GPT4BPETokenizer(ITokenizer):
    name = "GPT-4 BPE (cl100k)"
    bounded_vocab = True
    lossless_coords = False  # context-dependent fragmentation; see Sec. 5
    grammar_compatible = False

    def __init__(self) -> None:
        super().__init__()
        import tiktoken
        self._enc = tiktoken.get_encoding("cl100k_base")

    def train(self, corpus_lines: List[str]) -> None:
        return  # frozen pretrained

    def encode(self, line: str) -> List[int]:
        ids = self._enc.encode(line)
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)

    def vocab_size(self) -> int:
        return self._enc.n_vocab


# ---------------------------------------------------------------------------
# 3. GPT-2 BPE (tiktoken r50k_base)
# ---------------------------------------------------------------------------
class GPT2BPETokenizer(ITokenizer):
    name = "GPT-2 BPE (r50k)"
    bounded_vocab = True
    lossless_coords = False
    grammar_compatible = False

    def __init__(self) -> None:
        super().__init__()
        import tiktoken
        self._enc = tiktoken.get_encoding("r50k_base")

    def train(self, corpus_lines: List[str]) -> None:
        return

    def encode(self, line: str) -> List[int]:
        ids = self._enc.encode(line)
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)

    def vocab_size(self) -> int:
        return self._enc.n_vocab


# ---------------------------------------------------------------------------
# 4. Domain-trained BPE (HF tokenizers)
# ---------------------------------------------------------------------------
class DomainBPETokenizer(ITokenizer):
    name = "Domain BPE"
    bounded_vocab = True
    lossless_coords = False
    grammar_compatible = False

    def __init__(self, vocab_size: int = 1000) -> None:
        super().__init__()
        self._vocab_cap = vocab_size
        self._tok = None

    def train(self, corpus_lines: List[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self._vocab_cap,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        )
        tok.train_from_iterator(corpus_lines, trainer=trainer)
        self._tok = tok

    def encode(self, line: str) -> List[int]:
        ids = self._tok.encode(line).ids
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()


# ---------------------------------------------------------------------------
# 5. WordPiece (HF tokenizers)
# ---------------------------------------------------------------------------
class WordPieceTokenizer(ITokenizer):
    name = "WordPiece"
    bounded_vocab = True
    lossless_coords = False
    grammar_compatible = False

    def __init__(self, vocab_size: int = 1000) -> None:
        super().__init__()
        self._vocab_cap = vocab_size
        self._tok = None

    def train(self, corpus_lines: List[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tok = Tokenizer(WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=self._vocab_cap,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        )
        tok.train_from_iterator(corpus_lines, trainer=trainer)
        self._tok = tok

    def encode(self, line: str) -> List[int]:
        ids = self._tok.encode(line).ids
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()


# ---------------------------------------------------------------------------
# 6. Character-level
# ---------------------------------------------------------------------------
class CharLevelTokenizer(ITokenizer):
    name = "Character-level"
    bounded_vocab = True
    lossless_coords = True
    grammar_compatible = False  # FSA over digit strings explodes; see Sec. 5

    def __init__(self) -> None:
        super().__init__()
        # Fixed alphabet: printable ASCII covering G-code text
        chars = list("0123456789.- +XYZIJKFRSPQEABCGM")
        self._vocab = {c: i for i, c in enumerate(chars)}
        self._inv = {i: c for c, i in self._vocab.items()}
        self._unk = len(self._vocab)
        self._vocab["<UNK>"] = self._unk

    def train(self, corpus_lines: List[str]) -> None:
        return

    def encode(self, line: str) -> List[int]:
        ids = [self._vocab.get(ch, self._unk) for ch in line.upper() if ch != "\n"]
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self._inv.get(i, "?") for i in ids)

    def vocab_size(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# 7. Flat per-value
# ---------------------------------------------------------------------------
class FlatPerValueTokenizer(ITokenizer):
    """Each unique canonicalized whitespace-separated word becomes a token.

    This is the worst-case tokenizer for unbounded vocabulary growth: every
    distinct coordinate appears as a fresh token. Vocabulary scales with
    geometry.
    """
    name = "Flat per-value"
    bounded_vocab = False
    lossless_coords = True
    grammar_compatible = True

    def __init__(self) -> None:
        super().__init__()
        self._vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
        # Underlying GCodeTokenizer used only for canonicalization
        cfg = TokenizerConfig(mode="hybrid", min_freq=1, vocab_size=200000)
        self._gtok = GCodeTokenizer(cfg)

    def train(self, corpus_lines: List[str]) -> None:
        # Build vocabulary by encountering tokens; this happens lazily on encode
        return

    def encode(self, line: str) -> List[int]:
        canon = self._gtok.canonicalize_line(line)
        if canon is None:
            return []
        ids: List[int] = []
        for word in canon.split(" "):
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
            ids.append(self._vocab[word])
        self._record(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {i: t for t, i in self._vocab.items()}
        return " ".join(inv.get(i, "[UNK]") for i in ids)

    def vocab_size(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def all_tokenizers(
    train_corpus: Optional[List[str]] = None,
    domain_bpe_vocab: int = 1000,
    wordpiece_vocab: int = 1000,
) -> List[ITokenizer]:
    """Instantiate every tokenizer used in the design-space experiments.

    train_corpus is required for the trainable BPE/WordPiece variants;
    pass None to skip them (useful for fast smoke tests).
    """
    toks: List[ITokenizer] = [
        HierarchicalTokenizer(),
        GPT4BPETokenizer(),
        GPT2BPETokenizer(),
        CharLevelTokenizer(),
        FlatPerValueTokenizer(),
    ]
    if train_corpus is not None:
        dbpe = DomainBPETokenizer(vocab_size=domain_bpe_vocab)
        dbpe.train(train_corpus)
        toks.append(dbpe)

        wp = WordPieceTokenizer(vocab_size=wordpiece_vocab)
        wp.train(train_corpus)
        toks.append(wp)
    return toks
