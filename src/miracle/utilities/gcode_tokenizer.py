"""
gcode_tokenizer.py
Canonicalizer + tokenizer + vocab builder for G-code.
See module docstring for usage.
"""
from __future__ import annotations
import re, json, hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

DEFAULT_SPECIAL = {"PAD":0, "BOS":1, "EOS":2, "UNK":3, "MASK":4}
DEFAULT_PRECISION = {
    "X":1e-3, "Y":1e-3, "Z":1e-3,
    "A":1e-3, "B":1e-3, "C":1e-3,
    "I":1e-4, "J":1e-4, "K":1e-4,
    "F":1.0,  "S":10.0, "R":1e-4,
    "P":1e-3, "Q":1e-3, "E":1e-4,
}
KEEP_LITERAL = {"G","M","T"}
ADDR_CHARS  = set(list("XYZABCIJKFSRPQE"))

LINE_COMMENT_SEMI = re.compile(r";.*?$")
LINE_COMMENT_PARENS = re.compile(r"\(.*?\)")
LINE_NUMBER = re.compile(r"(^|\s)N\d+\b", re.IGNORECASE)
CHECKSUM = re.compile(r"\*[-+]?\d+$")
MULTISPACE = re.compile(r"\s+")
WORD_RE = re.compile(r"([A-Z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?$")

@dataclass
class TokenizerConfig:
    mode: str = "hybrid"       # 'literal' | 'split' | 'hybrid'
    precision: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRECISION))
    clip_bins: Dict[str, int] = field(default_factory=dict)
    canonical_decimal_places: int = 4
    ignore_for_pid: set = field(default_factory=lambda: {"F", "S"})
    special: Dict[str,int] = field(default_factory=lambda: dict(DEFAULT_SPECIAL))
    vocab_size: int = 50000
    min_freq: int = 1
    dynamic_numbers: bool = False
    bucket_digits: Optional[int] = None  # NEW: Use first N digits for bucketing (e.g., 2 → NUM_X_15 instead of NUM_X_1575)

class GCodeTokenizer:
    def __init__(self, config: TokenizerConfig, vocab: Optional[Dict[str,int]] = None):
        self.cfg = config
        self.vocab = vocab or dict(self.cfg.special)
        self.inv_vocab = {i:t for t,i in self.vocab.items()}

    # ---------- Canonicalization ----------
    def _strip_comments(self, line: str) -> str:
        line = LINE_COMMENT_PARENS.sub("", line)
        line = LINE_COMMENT_SEMI.sub("", line)
        return line

    def _normalize_words(self, parts: List[str]) -> List[str]:
        out = []
        for p in parts:
            p = p.strip().upper()
            if not p:
                continue
            p = CHECKSUM.sub("", p)
            if p in {"/", "%"}:
                continue
            m = re.match(r"^([GMT])0+(\d+)$", p)
            if m:
                p = f"{m.group(1)}{int(m.group(2))}"
            out.append(p)
        return out

    def canonicalize_line(self, line: str) -> Optional[str]:
        line = self._strip_comments(line)
        line = LINE_NUMBER.sub(" ", line)
        line = line.strip()
        if not line:
            return None
        line = line.upper()
        line = MULTISPACE.sub(" ", line)
        parts = self._normalize_words(line.split(" "))
        if not parts:
            return None
        return " ".join(parts)

    def canonicalize(self, lines: Iterable[str]) -> List[str]:
        out = []
        for ln in lines:
            c = self.canonicalize_line(ln)
            if c:
                out.append(c)
        return out

    # ---------- Tokenization ----------
    def _quantize(self, addr: str, value: float) -> int:
        step = self.cfg.precision.get(addr, 1e-3)
        return int(round(value / step))

    def _clip_bin(self, addr: str, b: int) -> int:
        if addr in self.cfg.clip_bins:
            lim = int(self.cfg.clip_bins[addr])
            if b >  lim: b =  lim
            if b < -lim: b = -lim
        return b

    def _bucket_value(self, b: int) -> str:
        """
        Bucket quantized value to first N digits.

        Examples (bucket_digits=2):
            1575 → "15"
            -125 → "-1"  (negative: sign + first digit)
            5 → "05" (pad with zero if needed)

        This reduces vocabulary size by grouping similar numeric values.
        """
        if self.cfg.bucket_digits is None:
            return str(b)

        # Handle negative numbers
        if b < 0:
            # For negatives, keep sign and first N-1 digits
            b_str = str(abs(b))
            if len(b_str) >= self.cfg.bucket_digits:
                return "-" + b_str[:self.cfg.bucket_digits-1]
            else:
                return "-" + b_str.zfill(self.cfg.bucket_digits-1)
        else:
            # For positives, take first N digits and zero-pad if needed
            b_str = str(b)
            if len(b_str) >= self.cfg.bucket_digits:
                return b_str[:self.cfg.bucket_digits]
            else:
                return b_str.zfill(self.cfg.bucket_digits)

    def _tokenize_word(self, token: str) -> List[str]:
        m = WORD_RE.match(token)
        if not m:
            return [token]
        head, num = m.group(1), m.group(2)
        if self.cfg.mode == "literal":
            return [token]
        if self.cfg.mode == "split":
            if num is None:
                return [head]
            try:
                val = float(num)
            except Exception:
                return [head, "NUM_UNPARSEABLE"]
            if head in ADDR_CHARS:
                b = self._clip_bin(head, self._quantize(head, val))
                b_str = self._bucket_value(b)  # Apply bucketing
                return [head, f"NUM_{head}_{b_str}"]
            else:
                b = self._quantize(head, val)
                b_str = self._bucket_value(b)
                return [f"{head}{int(val)}"] if head in {"G","M","T"} and float(val).is_integer() else [head, f"NUM_{head}_{b_str}"]
        if head in {"G","M","T"}:
            if num is not None:
                try:
                    val = float(num)
                    if float(val).is_integer():
                        return [f"{head}{int(val)}"]
                except Exception:
                    pass
            return [head] if num is None else [head, f"NUM_{head}_{num}"]
        if num is None or head not in ADDR_CHARS:
            return [token]
        try:
            val = float(num)
        except Exception:
            return [head, "NUM_UNPARSEABLE"]
        b = self._clip_bin(head, self._quantize(head, val))
        b_str = self._bucket_value(b)  # Apply bucketing
        return [head, f"NUM_{head}_{b_str}"]

    def tokenize_canonical(self, canon_lines: List[str]) -> List[str]:
        toks: List[str] = []
        for ln in canon_lines:
            for w in ln.split(" "):
                toks.extend(self._tokenize_word(w))
        return toks

    # ---------- Vocab ----------
    def train_vocab(self, corpus_files: List[Path], vocab_out: Optional[Path] = None) -> Dict[str,int]:
        from collections import Counter
        counts = Counter()
        for fp in corpus_files:
            text = fp.read_text(errors="ignore").splitlines()
            canon = self.canonicalize(text)
            toks = self.tokenize_canonical(canon)
            counts.update(toks)
        vocab = dict(self.cfg.special)
        items = sorted([(tok,cnt) for tok,cnt in counts.items() if cnt >= self.cfg.min_freq and tok not in vocab],
                       key=lambda x:(-x[1], x[0]))
        for tok,_ in items[: max(0, self.cfg.vocab_size - len(vocab)) ]:
            vocab[tok] = len(vocab)
        self.vocab = vocab
        self.inv_vocab = {i:t for t,i in vocab.items()}
        if vocab_out:
            cfg_dict = asdict(self.cfg)
            # Convert set to list for JSON serialization
            if 'ignore_for_pid' in cfg_dict and isinstance(cfg_dict['ignore_for_pid'], set):
                cfg_dict['ignore_for_pid'] = list(cfg_dict['ignore_for_pid'])
            vocab_out.write_text(json.dumps({"config": cfg_dict, "vocab": self.vocab}, indent=2))
        return self.vocab

    def save_vocab(self, path: Path) -> None:
        cfg_dict = asdict(self.cfg)
        # Convert set to list for JSON serialization
        if 'ignore_for_pid' in cfg_dict and isinstance(cfg_dict['ignore_for_pid'], set):
            cfg_dict['ignore_for_pid'] = list(cfg_dict['ignore_for_pid'])
        path.write_text(json.dumps({"config": cfg_dict, "vocab": self.vocab}, indent=2))

    @staticmethod
    def load(path: Path) -> "GCodeTokenizer":
        obj = json.loads(path.read_text())
        cfg_data = obj["config"]
        # Convert list back to set for ignore_for_pid
        if 'ignore_for_pid' in cfg_data and isinstance(cfg_data['ignore_for_pid'], list):
            cfg_data['ignore_for_pid'] = set(cfg_data['ignore_for_pid'])
        cfg = TokenizerConfig(**cfg_data)
        return GCodeTokenizer(cfg, obj["vocab"])

    # ---------- Encode / Decode ----------
    def _tok2id(self, tok: str) -> int:
        if tok in self.vocab:
            return self.vocab[tok]
        if self.cfg.dynamic_numbers and tok.startswith("NUM_"):
            self.vocab[tok] = len(self.vocab)
            self.inv_vocab[self.vocab[tok]] = tok
            return self.vocab[tok]
        return self.cfg.special["UNK"]

    def encode(self, lines: Iterable[str], add_bos_eos: bool = True) -> List[int]:
        canon = self.canonicalize(lines)
        toks = self.tokenize_canonical(canon)
        ids = [self.cfg.special["BOS"]] if add_bos_eos else []
        ids.extend(self._tok2id(t) for t in toks)
        if add_bos_eos:
            ids.append(self.cfg.special["EOS"])
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        toks = [ self.inv_vocab.get(i, "[UNK]") for i in ids ]
        if toks and toks[0] == "BOS":
            toks = toks[1:]
        if toks and toks[-1] == "EOS":
            toks = toks[:-1]
        return toks

    # ---------- Program ID ----------
    def program_id(self, lines: Iterable[str]) -> str:
        canon = self.canonicalize(lines)
        toks = self.tokenize_canonical(canon)
        filtered = []
        for t in toks:
            if t in self.cfg.ignore_for_pid:
                continue
            if t.startswith("NUM_"):
                parts = t.split("_")
                if len(parts) >= 3 and parts[1] in self.cfg.ignore_for_pid:
                    continue
            filtered.append(t)
        joined = " ".join(filtered).encode("utf-8", errors="ignore")
        return hashlib.sha1(joined).hexdigest()

# ---------------------------
# CLI
# ---------------------------
def _iter_files(root: Path) -> List[Path]:
    files = []
    root = Path(root)
    if root.is_file():
        return [root]
    for ext in ("*.gcode","*.nc","*.tap","*.txt"):
        files += list(root.rglob(ext))
    return files

def main():
    import argparse
    p = argparse.ArgumentParser(description="G-code tokenizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-vocab")
    p_build.add_argument("--corpus", type=str, required=True)
    p_build.add_argument("--vocab-out", type=str, required=True)
    p_build.add_argument("--mode", type=str, default="hybrid", choices=["literal","split","hybrid"])
    p_build.add_argument("--vocab-size", type=int, default=50000)
    p_build.add_argument("--min-freq", type=int, default=1)
    p_build.add_argument("--bucket-digits", type=int, default=None, help="Use first N digits for numeric bucketing (e.g., 2 for NUM_X_15)")

    p_encode = sub.add_parser("encode")
    p_encode.add_argument("--vocab", type=str, required=True)
    p_encode.add_argument("--in-file", type=str, required=True)
    p_encode.add_argument("--out", type=str, required=True)
    p_encode.add_argument("--no-bos-eos", action="store_true")

    p_hash = sub.add_parser("hash")
    p_hash.add_argument("--in-file", type=str, required=True)

    args = p.parse_args()
    if args.cmd == "build-vocab":
        cfg = TokenizerConfig(
            mode=args.mode,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            bucket_digits=args.bucket_digits  # Add bucketing parameter
        )
        tok = GCodeTokenizer(cfg)
        files = _iter_files(Path(args.corpus))
        if not files:
            raise SystemExit("No files found in corpus path.")
        tok.train_vocab(files, Path(args.vocab_out))
        print(f"Built vocab size={len(tok.vocab)} → {args.vocab_out}")
        if args.bucket_digits:
            print(f"  Using {args.bucket_digits}-digit bucketing for numeric values")
    elif args.cmd == "encode":
        tok = GCodeTokenizer.load(Path(args.vocab))
        ids = tok.encode(Path(args.in_file).read_text(errors="ignore").splitlines(), add_bos_eos=not args.no_bos_eos)
        Path(args.out).write_text(json.dumps({"ids": ids, "length": len(ids)}))
        print(f"Wrote tokens → {args.out} (len={len(ids)})")
    elif args.cmd == "hash":
        tok = GCodeTokenizer(TokenizerConfig())
        pid = tok.program_id(Path(args.in_file).read_text(errors='ignore').splitlines())
        print(pid)

if __name__ == "__main__":
    main()
