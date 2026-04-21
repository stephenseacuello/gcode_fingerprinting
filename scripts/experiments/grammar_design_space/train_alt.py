"""
train_alt.py
============

Slim trainer for FlatVocabDecoder on alt-tokenized v7 fold_1 data, using
the pre-cached encoder memory from
outputs/decoder20260304/v7_best_5fold_multiseed/fold_1_seed_42/encoder_memory/.

Vanilla cross-entropy + Adam, no curriculum, no scheduled sampling, no focal
loss, no data augmentation. The point is a clean apples-to-apples comparison
across alt-tokenizer arms; complexity in the trainer would muddy the signal.

Usage:
    python3 scripts/experiments/grammar_design_space/train_alt.py \
        --tokenizer domain_bpe \
        --epochs 80 \
        --out-base outputs/decoder20260303/grammarpaper/alt_decoders
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from miracle.model.flat_vocab_decoder import FlatVocabDecoder  # noqa: E402
from alt_tokenizers import PAD_ID, BOS_ID, EOS_ID, UNK_ID  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AltDecoderDataset(Dataset):
    """Pairs alt-tokenized G-code with cached encoder memory.

    The cached memory tensors (train_memory.pt) are aligned 1:1 with the
    rows of the corresponding NPZ split (train_sequences.npz, etc.), so we
    just index by row.
    """

    def __init__(self, npz_path: Path, memory_path: Path, op_pred_path: Path,
                 max_token_len: int = 32) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.tokens = torch.from_numpy(data["tokens"]).long()       # [N, max_len]
        self.lengths = torch.from_numpy(data["lengths"]).long()      # [N]
        self.gcode_texts = data.get("gcode_texts", None)
        self.memory = torch.load(memory_path, map_location="cpu", weights_only=False).float()
        self.op_pred = torch.load(op_pred_path, map_location="cpu", weights_only=False).long()
        self.max_token_len = max_token_len

        n_data = len(self.tokens)
        n_mem = len(self.memory)
        if n_data != n_mem:
            n = min(n_data, n_mem)
            print(f"  [warn] tokens={n_data}, memory={n_mem}; truncating to {n}")
            self.tokens = self.tokens[:n]
            self.lengths = self.lengths[:n]
            self.memory = self.memory[:n]
            self.op_pred = self.op_pred[:n]

        # Build input/target pairs (BOS/EOS already in self.tokens; we need
        # to construct teacher-forced inputs and shifted targets).
        # Saved tokens have format [BOS, t1, t2, ..., tn, EOS, PAD, PAD, ...]
        # input  = [BOS, t1, t2, ..., tn, EOS, PAD, ...]   (shift left? no)
        # target = [t1, t2, ..., tn, EOS, PAD, ...]
        # Standard convention: input = tokens[:, :-1], target = tokens[:, 1:]
        self.input_tokens = self.tokens[:, :-1].clone()    # [N, max_len-1]
        self.target_tokens = self.tokens[:, 1:].clone()    # [N, max_len-1]
        # Where target is PAD, mark as ignore (-100) so CE skips it
        self.padding_mask = (self.input_tokens == PAD_ID)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_tokens": self.input_tokens[i],
            "target_tokens": self.target_tokens[i],
            "memory": self.memory[i],
            "operation_type": self.op_pred[i],
            "padding_mask": self.padding_mask[i],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_tokens": torch.stack([b["input_tokens"] for b in batch]),
        "target_tokens": torch.stack([b["target_tokens"] for b in batch]),
        "memory": torch.stack([b["memory"] for b in batch]),
        "operation_type": torch.stack([b["operation_type"] for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optim, device) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        memory = batch["memory"].to(device)
        op = batch["operation_type"].to(device)
        pad_mask = batch["padding_mask"].to(device)

        outputs = model(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op,
            tgt_key_padding_mask=pad_mask,
        )
        logits = outputs["logits"]  # [B, L, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = target_tokens != PAD_ID
            total_correct += ((preds == target_tokens) & mask).sum().item()
            total_count += mask.sum().item()
        total_loss += loss.item() * input_tokens.size(0)

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "token_acc": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    seq_correct = 0
    seq_total = 0
    for batch in loader:
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        memory = batch["memory"].to(device)
        op = batch["operation_type"].to(device)
        pad_mask = batch["padding_mask"].to(device)

        outputs = model(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op,
            tgt_key_padding_mask=pad_mask,
        )
        logits = outputs["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        total_loss += loss.item() * input_tokens.size(0)

        preds = logits.argmax(-1)
        mask = target_tokens != PAD_ID
        total_correct += ((preds == target_tokens) & mask).sum().item()
        total_count += mask.sum().item()

        seq_match = ((preds == target_tokens) | ~mask).all(dim=1)
        seq_correct += seq_match.sum().item()
        seq_total += target_tokens.size(0)

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "token_acc": total_correct / max(total_count, 1),
        "seq_acc": seq_correct / max(seq_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--fold", type=int, default=1, help="Fold number (1..5)")
    p.add_argument("--seed-tag", default="42", help="Encoder-memory seed tag (default 42)")
    p.add_argument("--alt-base", default="outputs/decoder20260303/grammarpaper/alt_tokenizers")
    p.add_argument("--memory-base-template",
                   default="outputs/decoder20260304/v7_best_5fold_multiseed/fold_{fold}_seed_{seed}/encoder_memory")
    p.add_argument("--out-base", default="outputs/decoder20260303/grammarpaper/alt_decoders")
    p.add_argument("--max-token-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Tokenizer: {args.tokenizer}  Fold: {args.fold}")

    fold_dir = Path(args.alt_base) / args.tokenizer / f"fold_{args.fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(
            f"{fold_dir} not found. Run preprocess_alt.py --tokenizer {args.tokenizer} --fold {args.fold} first."
        )
    summary = json.loads((fold_dir / "summary.json").read_text())
    vocab_size = summary["vocab_size"]
    print(f"Vocab size: {vocab_size}")

    mem_base = Path(args.memory_base_template.format(fold=args.fold, seed=args.seed_tag))
    if not mem_base.exists():
        raise FileNotFoundError(f"Cached encoder memory not found at {mem_base}")
    train_ds = AltDecoderDataset(
        npz_path=fold_dir / "train_sequences.npz",
        memory_path=mem_base / "train_memory.pt",
        op_pred_path=mem_base / "train_op_pred.pt",
        max_token_len=args.max_token_len,
    )
    val_ds = AltDecoderDataset(
        npz_path=fold_dir / "val_sequences.npz",
        memory_path=mem_base / "val_memory.pt",
        op_pred_path=mem_base / "val_op_pred.pt",
        max_token_len=args.max_token_len,
    )
    test_ds = AltDecoderDataset(
        npz_path=fold_dir / "test_sequences.npz",
        memory_path=mem_base / "test_memory.pt",
        op_pred_path=mem_base / "test_op_pred.pt",
        max_token_len=args.max_token_len,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    sensor_dim = train_ds.memory.size(-1)
    print(f"Sensor dim: {sensor_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = FlatVocabDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sensor_dim=sensor_dim,
        n_operations=9,
        dropout=args.dropout,
        max_seq_len=args.max_token_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_base) / args.tokenizer / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.log"
    log_lines: list[str] = []

    best_val_loss = float("inf")
    best_metrics = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optim, device)
        val_m = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        line = (f"epoch {epoch:3d}  "
                f"train_loss={train_m['loss']:.4f} acc={train_m['token_acc']:.4f}  "
                f"val_loss={val_m['loss']:.4f} acc={val_m['token_acc']:.4f} seq={val_m['seq_acc']:.4f}")
        print(line)
        log_lines.append(line)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_metrics = val_m
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "sensor_dim": sensor_dim,
                "args": vars(args),
                "val_metrics": val_m,
            }, out_dir / "best_decoder.pt")

    elapsed = time.time() - start
    log_lines.append(f"\nTraining done in {elapsed:.1f}s")
    print(f"\nTraining done in {elapsed:.1f}s")

    # Final test eval using best checkpoint
    ckpt = torch.load(out_dir / "best_decoder.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["decoder_state_dict"])
    test_m = evaluate(model, test_loader, device)
    print(f"Test: loss={test_m['loss']:.4f} token_acc={test_m['token_acc']:.4f} seq_acc={test_m['seq_acc']:.4f}")
    log_lines.append(f"Test: loss={test_m['loss']:.4f} token_acc={test_m['token_acc']:.4f} seq_acc={test_m['seq_acc']:.4f}")

    log_path.write_text("\n".join(log_lines))
    (out_dir / "metrics.json").write_text(json.dumps({
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "n_params": n_params,
        "best_val": best_metrics,
        "test": test_m,
        "history": history,
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
