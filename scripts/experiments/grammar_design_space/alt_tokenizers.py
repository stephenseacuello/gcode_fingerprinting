"""
alt_tokenizers.py
=================

Production-quality alternative tokenizers for the C/E experiments. Each
tokenizer guarantees the convention required by FlatVocabDecoder and
DecoderDataset:

    PAD = 0   BOS = 1   EOS = 2   UNK = 3   content tokens >= 4

Implemented:
    DomainBPETokenizer  -- HF tokenizers BPE trained on the corpus
    WordPieceTokenizer  -- HF tokenizers WordPiece trained on the corpus
    CharLevelTokenizer  -- one-byte alphabet (closed alphabet, fast train)
    FlatPerValueTokenizer -- one token per unique whitespace word (unbounded)
    GPT4CompactTokenizer -- tiktoken cl100k encoding remapped to corpus-active subset

Each exposes:
    name : str
    train(corpus_lines) -> None
    encode(line, add_special: bool) -> List[int]
    vocab_size() -> int
    save(path) / load(path)

Encoding never returns ids 0/1/2/3 from the *content* range; specials are
prepended/appended only when add_special=True.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

# Make src/ importable for the existing GCodeTokenizer (used for canonicalization)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig  # noqa: E402

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


# ---------------------------------------------------------------------------
# Helper: shared canonicalizer
# ---------------------------------------------------------------------------
_CANON = GCodeTokenizer(TokenizerConfig(mode="hybrid", min_freq=1, vocab_size=200_000))


def canonicalize(line: str) -> Optional[str]:
    return _CANON.canonicalize_line(line)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseAltTokenizer:
    name: str = "abstract"

    def train(self, corpus_lines: List[str]) -> None:
        raise NotImplementedError

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Reconstruct a G-code-like string from token ids.

        Specials (PAD/BOS/EOS) are stripped before reassembly.
        """
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BaseAltTokenizer":
        raise NotImplementedError

    @staticmethod
    def _strip_specials(ids: List[int]) -> List[int]:
        return [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]


# ---------------------------------------------------------------------------
# 1. Domain-trained BPE
# ---------------------------------------------------------------------------
class DomainBPETokenizer(BaseAltTokenizer):
    name = "domain_bpe"

    def __init__(self, vocab_cap: int = 1000) -> None:
        self.vocab_cap = vocab_cap
        self._tok = None

    def train(self, corpus_lines: List[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Canonicalize first so train and encode see the same input
        canonical = [canonicalize(ln) for ln in corpus_lines]
        canonical = [ln for ln in canonical if ln]

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        # IMPORTANT: order determines IDs. PAD=0, BOS=1, EOS=2, UNK=3.
        trainer = BpeTrainer(
            vocab_size=self.vocab_cap,
            min_frequency=2,
            special_tokens=SPECIALS,
            show_progress=False,
        )
        tok.train_from_iterator(canonical, trainer=trainer)
        self._tok = tok
        self._verify_specials()

    def _verify_specials(self) -> None:
        for tok_str, expected in zip(SPECIALS, range(4)):
            actual = self._tok.token_to_id(tok_str)
            if actual != expected:
                raise RuntimeError(
                    f"{self.name}: special {tok_str} got id {actual}, expected {expected}"
                )

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = list(self._tok.encode(canon).ids)
        # Strip any specials accidentally introduced (HF Tokenizer with no template
        # post-processor should not add specials, but just in case).
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "DomainBPETokenizer":
        from tokenizers import Tokenizer
        obj = cls()
        obj._tok = Tokenizer.from_file(str(path))
        return obj


# ---------------------------------------------------------------------------
# 2. WordPiece
# ---------------------------------------------------------------------------
class WordPieceTokenizer(BaseAltTokenizer):
    name = "wordpiece"

    def __init__(self, vocab_cap: int = 1000) -> None:
        self.vocab_cap = vocab_cap
        self._tok = None

    def train(self, corpus_lines: List[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace

        canonical = [canonicalize(ln) for ln in corpus_lines]
        canonical = [ln for ln in canonical if ln]

        tok = Tokenizer(WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=self.vocab_cap,
            min_frequency=2,
            special_tokens=SPECIALS,
            show_progress=False,
        )
        tok.train_from_iterator(canonical, trainer=trainer)
        self._tok = tok
        self._verify_specials()

    def _verify_specials(self) -> None:
        for tok_str, expected in zip(SPECIALS, range(4)):
            actual = self._tok.token_to_id(tok_str)
            if actual != expected:
                raise RuntimeError(
                    f"{self.name}: special {tok_str} got id {actual}, expected {expected}"
                )

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = list(self._tok.encode(canon).ids)
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "WordPieceTokenizer":
        from tokenizers import Tokenizer
        obj = cls()
        obj._tok = Tokenizer.from_file(str(path))
        return obj


# ---------------------------------------------------------------------------
# 3. Character-level
# ---------------------------------------------------------------------------
class CharLevelTokenizer(BaseAltTokenizer):
    name = "char_level"

    # Closed alphabet covering G-code text after canonicalization
    _CHARS = list(" 0123456789.- +XYZIJKFRSPQEABCGMT")

    def __init__(self) -> None:
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        # Build static vocab: 0-3 reserved, 4+ for chars
        self._char2id = {SPECIALS[0]: PAD_ID, SPECIALS[1]: BOS_ID,
                         SPECIALS[2]: EOS_ID, SPECIALS[3]: UNK_ID}
        for ch in self._CHARS:
            if ch not in self._char2id:
                self._char2id[ch] = len(self._char2id)
        self._id2char = {i: c for c, i in self._char2id.items()}

    def train(self, corpus_lines: List[str]) -> None:
        return  # static alphabet

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = [self._char2id.get(ch.upper(), UNK_ID) for ch in canon]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        return "".join(self._id2char.get(i, "?") for i in ids)

    def vocab_size(self) -> int:
        return len(self._char2id)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._char2id, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CharLevelTokenizer":
        obj = cls()
        obj._char2id = json.loads(Path(path).read_text())
        obj._id2char = {i: c for c, i in obj._char2id.items()}
        return obj


# ---------------------------------------------------------------------------
# 4. Flat per-value
# ---------------------------------------------------------------------------
class FlatPerValueTokenizer(BaseAltTokenizer):
    """Each canonicalized whitespace word becomes a token."""

    name = "flat_per_value"

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {SPECIALS[0]: PAD_ID, SPECIALS[1]: BOS_ID,
                                        SPECIALS[2]: EOS_ID, SPECIALS[3]: UNK_ID}

    def train(self, corpus_lines: List[str]) -> None:
        for ln in corpus_lines:
            canon = canonicalize(ln)
            if canon is None:
                continue
            for word in canon.split(" "):
                if word and word not in self._vocab:
                    self._vocab[word] = len(self._vocab)

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = [self._vocab.get(w, UNK_ID) for w in canon.split(" ") if w]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        inv = {i: t for t, i in self._vocab.items()}
        return " ".join(inv.get(i, "[UNK]") for i in ids)

    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._vocab, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FlatPerValueTokenizer":
        obj = cls()
        obj._vocab = json.loads(Path(path).read_text())
        return obj


# ---------------------------------------------------------------------------
# 5. Byte-level BPE (no Whitespace pre-tokenizer)
# ---------------------------------------------------------------------------
class ByteLevelBPETokenizer(BaseAltTokenizer):
    """Byte-level BPE trained on the corpus.

    Rationale: the DomainBPETokenizer uses a Whitespace pre-tokenizer, which
    causes the HuggingFace decoder to reinsert whitespace at every subword
    boundary on decode (e.g. "0.1755" -> "0 . 1755"). A reviewer correctly
    identified this as a configuration artifact rather than a fundamental
    BPE limitation. This variant trains BPE with a ByteLevel pre-tokenizer
    and a ByteLevel decoder, so the decode roundtrip is bit-exact on the
    canonicalized input and no whitespace is inserted between subword pieces.

    If grammar validity on this tokenizer is still low, the failure is in
    the BPE merge policy itself (ambiguous numeric fragmentation) rather
    than in decoder whitespace handling, and the impossibility argument
    strengthens. If it is high, the impossibility argument needs to be
    narrowed to "standard BPE pipelines with whitespace pre-tokenization".
    """

    name = "bytelevel_bpe"

    def __init__(self, vocab_cap: int = 1000) -> None:
        self.vocab_cap = vocab_cap
        self._tok = None

    def train(self, corpus_lines: List[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
        from tokenizers.decoders import ByteLevel as ByteLevelDec

        canonical = [canonicalize(ln) for ln in corpus_lines]
        canonical = [ln for ln in canonical if ln]

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tok.decoder = ByteLevelDec()
        trainer = BpeTrainer(
            vocab_size=self.vocab_cap,
            min_frequency=2,
            special_tokens=SPECIALS,
            initial_alphabet=ByteLevelPre.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(canonical, trainer=trainer)
        self._tok = tok
        self._verify_specials()

    def _verify_specials(self) -> None:
        for tok_str, expected in zip(SPECIALS, range(4)):
            actual = self._tok.token_to_id(tok_str)
            if actual != expected:
                raise RuntimeError(
                    f"{self.name}: special {tok_str} got id {actual}, expected {expected}"
                )

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = list(self._tok.encode(canon).ids)
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPETokenizer":
        from tokenizers import Tokenizer
        obj = cls()
        obj._tok = Tokenizer.from_file(str(path))
        return obj


# ---------------------------------------------------------------------------
# 6. GPT-4 cl100k compact
# ---------------------------------------------------------------------------
class GPT4CompactTokenizer(BaseAltTokenizer):
    """tiktoken cl100k encoding remapped to a compact corpus-specific vocab.

    Rationale: cl100k has 100k tokens; only ~1k are active on this corpus.
    Using the full 100k inflates the embedding layer with no signal. We keep
    the cl100k *tokenization* (so fragmentation behavior is identical to GPT-4)
    but compact the active subset onto a small id space, with PAD/BOS/EOS/UNK
    at 0-3.
    """

    name = "gpt4_compact"

    def __init__(self) -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._cl_to_compact: Dict[int, int] = {}

    def train(self, corpus_lines: List[str]) -> None:
        # Canonicalize, then encode through cl100k, collect active set
        active: Set[int] = set()
        for ln in corpus_lines:
            canon = canonicalize(ln)
            if canon is None:
                continue
            for tid in self._enc.encode(canon):
                active.add(tid)
        # Build compact map: 0-3 reserved
        compact_id = 4
        for tid in sorted(active):
            self._cl_to_compact[tid] = compact_id
            compact_id += 1

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = []
            for tid in self._enc.encode(canon):
                ids.append(self._cl_to_compact.get(tid, UNK_ID))
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        inv = {compact: cl for cl, compact in self._cl_to_compact.items()}
        cl_ids = [inv[i] for i in ids if i in inv]
        if not cl_ids:
            return ""
        return self._enc.decode(cl_ids)

    def vocab_size(self) -> int:
        return 4 + len(self._cl_to_compact)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store the cl→compact map keyed as strings for JSON
        path.write_text(json.dumps({str(k): v for k, v in self._cl_to_compact.items()}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GPT4CompactTokenizer":
        obj = cls()
        raw = json.loads(Path(path).read_text())
        obj._cl_to_compact = {int(k): v for k, v in raw.items()}
        return obj


# ---------------------------------------------------------------------------
# 6. Hierarchical (ours) — wraps the existing 712-token vocab
# ---------------------------------------------------------------------------
class HierarchicalAltTokenizer(BaseAltTokenizer):
    """Wraps the existing GCodeTokenizer (712-token hybrid vocab) so it
    fits the alt-tokenizer interface and can be trained / evaluated by the
    same slim pipeline as the alt baselines. The vocab is *fixed* (loaded
    from data/gcode_vocab_712.json) and not retrained per fold.

    Decoding reassembles a G-code-like line by mapping ``NUM_<addr>_<bucket>``
    tokens back to numeric values via the tokenizer's quantization precision.
    """

    name = "hierarchical"

    def __init__(self, vocab_path: str = "data/gcode_vocab_712.json") -> None:
        self._vocab_path = vocab_path
        self._gtok = GCodeTokenizer.load(Path(vocab_path))

    def train(self, corpus_lines: List[str]) -> None:
        # Fixed vocab; nothing to train
        return

    def encode(self, line: str, add_special: bool = True) -> List[int]:
        canon = canonicalize(line)
        if canon is None:
            ids: List[int] = []
        else:
            ids = []
            for word in canon.split(" "):
                for tok in self._gtok._tokenize_word(word):
                    if tok in self._gtok.vocab:
                        ids.append(self._gtok.vocab[tok])
                    else:
                        ids.append(UNK_ID)
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = self._strip_specials(ids)
        inv = {i: t for t, i in self._gtok.vocab.items()}
        toks = [inv.get(i, "?") for i in ids]
        out: List[str] = []
        i = 0
        last_addr: Optional[str] = None
        while i < len(toks):
            t = toks[i]
            if t in {"X", "Y", "Z", "F", "R", "I", "J", "K", "S", "P", "Q", "E", "A", "B", "C"}:
                last_addr = t
                if i + 1 < len(toks) and toks[i + 1].startswith("NUM_"):
                    parts = toks[i + 1].split("_", 2)
                    if len(parts) == 3:
                        try:
                            bucket = int(parts[2])
                        except ValueError:
                            bucket = None
                        if bucket is not None:
                            precision = self._gtok.cfg.precision.get(t, 1e-3)
                            value = bucket * precision
                            if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
                                out.append(f"{t}{int(round(value))}")
                            else:
                                out.append(f"{t}{value}")
                            i += 2
                            continue
                i += 1
                continue
            if t.startswith("NUM_"):
                # Orphan numeric token; skip
                i += 1
                continue
            if t in {"PAD", "BOS", "EOS", "UNK", "MASK", "?"}:
                i += 1
                continue
            out.append(t)
            i += 1
        return " ".join(out)

    def vocab_size(self) -> int:
        return len(self._gtok.vocab)

    def save(self, path: Path) -> None:
        # Just record the source vocab path for reproducibility
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"source_vocab": self._vocab_path}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "HierarchicalAltTokenizer":
        meta = json.loads(Path(path).read_text())
        return cls(vocab_path=meta.get("source_vocab", "data/gcode_vocab_712.json"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TOKENIZER_REGISTRY: Dict[str, type] = {
    "domain_bpe": DomainBPETokenizer,
    "bytelevel_bpe": ByteLevelBPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "char_level": CharLevelTokenizer,
    "flat_per_value": FlatPerValueTokenizer,
    "gpt4_compact": GPT4CompactTokenizer,
    "hierarchical": HierarchicalAltTokenizer,
}


def make(name: str) -> BaseAltTokenizer:
    if name not in TOKENIZER_REGISTRY:
        raise KeyError(f"Unknown tokenizer: {name}. Choices: {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name]()


if __name__ == "__main__":
    # Smoke test
    sample_lines = [
        "G0 X1.4919 Y1.4848",
        "X1.5748 Y1.4562 R0.1188",
        "G2 X3.5 Y0.7 R0.08",
        "G1 X10 Y20 Z-5 F500",
    ]
    for name in TOKENIZER_REGISTRY:
        print(f"\n=== {name} ===")
        tok = make(name)
        tok.train(sample_lines)
        for ln in sample_lines:
            ids = tok.encode(ln, add_special=True)
            print(f"  {ln!r:42s} -> {ids}")
        print(f"  vocab_size = {tok.vocab_size()}")
