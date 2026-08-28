#!/usr/bin/env python3
"""Train + evaluate the from-scratch seq2seq baseline (see
src/miracle/model/seq2seq_baseline.py).

This script DELIBERATELY does not reuse the main paper's:
  - frozen MM-DTAE encoder
  - structured six-head decoder
  - grammar-constrained decoding mask
  - scheduled sampling
  - per-class auxiliary losses

Only the data, vocab, fold splits and metric definitions are shared
with the main paper so the comparison is apples-to-apples on the
single architectural-novelty axis.

Outputs metrics.json with the same headline keys as
scripts/evaluation/run_decoder_quick_test.py so the aggregator
can drop the baseline row directly into the headline table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.miracle.model.seq2seq_baseline import Seq2SeqBaseline  # noqa: E402


# =====================================================================
# Dataset (reuses the NPZ layout produced by
# scripts/preprocessing/run_preprocessing_v8_cv_fold.py)
# =====================================================================
class Seq2SeqDataset(Dataset):
    def __init__(self, npz_path: Path, max_token_len: int) -> None:
        z = np.load(npz_path, allow_pickle=True)
        self.continuous = torch.from_numpy(z["continuous"].astype(np.float32))
        self.tokens = torch.from_numpy(z["tokens"].astype(np.int64))
        self.token_length = torch.from_numpy(z["token_length"].astype(np.int64))
        self.gcode_texts = list(z["gcode_texts"])
        self.operation_type = z["operation_type"]
        # Truncate any sequences longer than max_token_len (keep BOS, drop tail).
        if self.tokens.shape[1] > max_token_len:
            self.tokens = self.tokens[:, :max_token_len]
            self.token_length = torch.clamp(self.token_length, max=max_token_len)
        self.max_token_len = max_token_len

    def __len__(self) -> int:
        return self.continuous.shape[0]

    def __getitem__(self, i: int) -> dict:
        tlen = int(self.token_length[i].item())
        tok = self.tokens[i, :tlen]  # already includes BOS + content + EOS as written
        return {
            "sensor": self.continuous[i],
            "tokens": tok,
            "tlen": tlen,
            "idx": i,
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    sensors = torch.stack([b["sensor"] for b in batch])
    T = max(b["tlen"] for b in batch)
    tokens = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for j, b in enumerate(batch):
        tokens[j, : b["tlen"]] = b["tokens"]
    lengths = torch.tensor([b["tlen"] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    return {"sensor": sensors, "tokens": tokens, "lengths": lengths, "idxs": idxs}


# =====================================================================
# Per-G-code-token classification of vocab IDs for headline metrics
# =====================================================================
def build_token_categories(vocab: dict[str, int]) -> dict[str, set[int]]:
    """Partition vocab IDs into command/type/param_type/numeric/special families
    so the same accuracy-key names as the main paper can be reported."""
    rev = {v: k for k, v in vocab.items()}
    commands = {"G0", "G1", "G2", "G3"}
    axis_letters = {"X", "Y", "Z", "F", "S", "I", "J", "K", "R"}
    specials = {"PAD", "BOS", "EOS", "UNK", "MASK"}

    cat = {
        "command": set(),
        "type": set(),
        "param_type": set(),
        "numeric": set(),
        "special": set(),
    }
    for tok_str, tok_id in vocab.items():
        if tok_str in specials:
            cat["special"].add(tok_id)
        elif tok_str in commands:
            cat["command"].add(tok_id)
        elif tok_str in axis_letters:
            cat["param_type"].add(tok_id)
        elif tok_str.startswith(("M", "T")):
            cat["type"].add(tok_id)
        else:
            cat["numeric"].add(tok_id)
    return cat


def accuracy_by_category(
    pred: torch.Tensor, target: torch.Tensor, cat_ids: set[int], mask: torch.Tensor
) -> float:
    """pred, target: (B, T) ; mask True for valid positions ; cat_ids: vocab ids
    in this category. Reports accuracy on positions where TARGET is in this
    category (matches the main paper's per-head accuracy convention)."""
    if not cat_ids:
        return 0.0
    cat_tensor = torch.tensor(sorted(cat_ids), device=target.device)
    in_cat = torch.isin(target, cat_tensor) & mask
    if in_cat.sum().item() == 0:
        return 0.0
    correct = (pred.eq(target) & in_cat).sum().item()
    return correct / in_cat.sum().item()


def sequence_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of sequences where every non-pad position matches."""
    matches = (pred.eq(target) | ~mask).all(dim=-1)
    return matches.float().mean().item()


def token_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    return ((pred.eq(target) & mask).sum().item() / mask.sum().item())


# =====================================================================
# Evaluation
# =====================================================================
@torch.no_grad()
def evaluate(
    model: Seq2SeqBaseline,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    categories: dict[str, set[int]],
    max_token_len: int,
    mode: str,  # "tf" or "ar"
) -> dict[str, float]:
    model.eval()
    metric_keys = ["command", "type", "param_type", "numeric"]
    sums = {f"{k}_accuracy": 0.0 for k in metric_keys}
    sums["token_accuracy"] = 0.0
    sums["sequence_accuracy"] = 0.0
    n_batches = 0

    for batch in loader:
        sensor = batch["sensor"].to(device)
        tokens = batch["tokens"].to(device)  # (B, T) with BOS at [:,0]
        if tokens.size(1) < 2:
            continue
        target_input = tokens[:, :-1]
        target_output = tokens[:, 1:]
        mask = target_output.ne(pad_id)

        if mode == "tf":
            logits = model(sensor, target_input)
            pred = logits.argmax(-1)
        else:  # ar
            gen = model.generate(sensor, max_len=max_token_len)
            # Align predicted positions with target_output (drop BOS at gen[:,0])
            T = target_output.size(1)
            pred = gen[:, 1 : T + 1]
            if pred.size(1) < T:
                pad = torch.full(
                    (pred.size(0), T - pred.size(1)),
                    pad_id,
                    dtype=torch.long,
                    device=device,
                )
                pred = torch.cat([pred, pad], dim=1)
            elif pred.size(1) > T:
                pred = pred[:, :T]

        sums["token_accuracy"] += token_accuracy(pred, target_output, mask)
        sums["sequence_accuracy"] += sequence_accuracy(pred, target_output, mask)
        for k in metric_keys:
            sums[f"{k}_accuracy"] += accuracy_by_category(
                pred, target_output, categories[k], mask
            )
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in sums}
    return {k: v / n_batches for k, v in sums.items()}


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers_enc", type=int, default=4)
    ap.add_argument("--n_layers_dec", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ff_dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_token_len", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[seq2seq-baseline] device={device}  fold={args.fold}")

    # Vocab + token categories
    vocab_obj = json.loads(args.vocab.read_text())
    vocab = vocab_obj["vocab"] if "vocab" in vocab_obj else vocab_obj
    if isinstance(vocab, list):
        vocab = {tok: i for i, tok in enumerate(vocab)}
    vocab_size = len(vocab)
    categories = build_token_categories(vocab)
    pad_id = vocab.get("PAD", 0)
    bos_id = vocab.get("BOS", 1)
    eos_id = vocab.get("EOS", 2)
    print(f"[seq2seq-baseline] vocab_size={vocab_size}  PAD={pad_id} BOS={bos_id} EOS={eos_id}")

    # Datasets
    train_ds = Seq2SeqDataset(args.data_dir / "train_sequences.npz", args.max_token_len)
    val_ds = Seq2SeqDataset(args.data_dir / "val_sequences.npz", args.max_token_len)
    test_ds = Seq2SeqDataset(args.data_dir / "test_sequences.npz", args.max_token_len)
    print(f"[seq2seq-baseline] sizes train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate(b, pad_id),
            pin_memory=("cuda" in str(device)),
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    sensor_channels = train_ds.continuous.shape[-1]
    window_size = train_ds.continuous.shape[1]

    model = Seq2SeqBaseline(
        sensor_channels=sensor_channels,
        window_size=window_size,
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers_enc=args.n_layers_enc,
        n_layers_dec=args.n_layers_dec,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_target_len=args.max_token_len,
    ).to(device)

    n_params = model.count_parameters()
    print(f"[seq2seq-baseline] params={n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history = []
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            sensor = batch["sensor"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            if tokens.size(1) < 2:
                continue
            target_input = tokens[:, :-1]
            target_output = tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(sensor, target_input)
            loss = criterion(logits.reshape(-1, vocab_size), target_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                sensor = batch["sensor"].to(device, non_blocking=True)
                tokens = batch["tokens"].to(device, non_blocking=True)
                if tokens.size(1) < 2:
                    continue
                target_input = tokens[:, :-1]
                target_output = tokens[:, 1:]
                logits = model(sensor, target_input)
                loss = criterion(logits.reshape(-1, vocab_size), target_output.reshape(-1))
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "dt_s": dt})
        msg = f"epoch {epoch+1:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  dt={dt:.1f}s"
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            msg += "  ** BEST **"
        else:
            patience_left -= 1
        print(msg, flush=True)
        if patience_left <= 0:
            print(f"[seq2seq-baseline] early stop at epoch {epoch+1}; best_epoch={best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_dir / "best_seq2seq.pt")

    # Final evaluation: TF + AR on test set
    print("[seq2seq-baseline] evaluating test set (teacher-forced)...")
    test_tf = evaluate(model, test_loader, device, pad_id, bos_id, eos_id, categories, args.max_token_len, "tf")
    print(f"  TF: {test_tf}")
    print("[seq2seq-baseline] evaluating test set (autoregressive)...")
    test_ar = evaluate(model, test_loader, device, pad_id, bos_id, eos_id, categories, args.max_token_len, "ar")
    print(f"  AR: {test_ar}")

    # Also val for completeness
    val_tf = evaluate(model, val_loader, device, pad_id, bos_id, eos_id, categories, args.max_token_len, "tf")

    metrics = {
        "test_metrics": {**test_tf, **{f"{k}_ar": v for k, v in test_ar.items()}},
        "val_metrics": val_tf,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val,
        "params_millions": n_params / 1e6,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[seq2seq-baseline] wrote {results_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
