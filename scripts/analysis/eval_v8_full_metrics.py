#!/usr/bin/env python3
"""Re-evaluate V8 decoder checkpoints with full per-class metrics.

Round-2 Phase A. The training pipeline argmaxes predictions and discards raw
logits, so the existing `metrics.json` files have only aggregate accuracies.
This script loads a V8 checkpoint, re-runs inference on the V8 test set,
captures predictions for every head, then computes per-class precision /
recall / F1 / confusion matrix via `per_class_metrics.compute_full_classification_metrics`.

For each checkpoint we emit:
  - `full_metrics.json` next to the existing `metrics.json`
  - `predictions.npz` with all per-head predictions + targets for downstream analysis

Heads scored:
  - sequence-level exact match
  - token-level (legacy_logits argmax)
  - type   (4 classes: SPECIAL / COMMAND / PARAM / NUMERIC)
  - command (G0 / G1 / G2 / G3 / G53 / OTHER)
  - param_type (X / Y / Z / I / J / K / R / F / S / P)
  - numeric digit head (per digit-position accuracy + overall)

CLI:
    python scripts/analysis/eval_v8_full_metrics.py \\
        --checkpoint outputs/decoder20260511/checkpoints/per_row_5fold/fold_1 \\
        --data-dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

from run_decoder_quick_test import (  # type: ignore  # noqa: E402
    DecoderQuickTestDataset,
    CachedDecoderDataset,
    SensorMultiHeadDecoder,
    GCodeTokenizer,
    decoder_collate_fn,
    build_modality_indices,
    cache_encoder_memory,
    load_frozen_encoder,
    ENCODER_BASE,
    ENCODER_CONFIGS,
    TYPE_SPECIAL,
    TYPE_COMMAND,
    TYPE_PARAM,
    TYPE_NUMERIC,
    CMD2ID,
    PARAM2ID,
    DIGIT_PAD,
    SIGN_PAD,
)
from miracle.utilities.gcode_tokenizer import TokenizerConfig  # noqa: E402
from miracle.training.per_class_metrics import (  # noqa: E402
    compute_full_classification_metrics,
    sequence_level_accuracy,
    regression_metrics,
)

PAD, BOS, EOS = 0, 1, 2

TYPE_NAMES = {
    TYPE_SPECIAL: "SPECIAL",
    TYPE_COMMAND: "COMMAND",
    TYPE_PARAM: "PARAM",
    TYPE_NUMERIC: "NUMERIC",
}
COMMAND_NAMES = {v: k for k, v in CMD2ID.items()}
PARAM_NAMES = {v: k for k, v in PARAM2ID.items()}


def _load_tokenizer(vocab_path: Path) -> tuple[GCodeTokenizer, dict[str, int], dict[int, str]]:
    vdata = json.loads(vocab_path.read_text())
    vocab = vdata["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}
    cfg = TokenizerConfig(**vdata["config"])
    tok = GCodeTokenizer(cfg, vocab=vocab)
    return tok, vocab, id_to_token


def _resolve_encoder_dir(encoder_config: str, fold: int) -> Path:
    if encoder_config not in ENCODER_CONFIGS:
        raise ValueError(f"unknown encoder_config '{encoder_config}'")
    sub = ENCODER_CONFIGS[encoder_config]
    return ENCODER_BASE / sub / f"fold_{fold}" / "encoder" / "checkpoint" / "best_model.pt"


def _ckpt_meta(ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def _build_model(ckpt_state: dict, vocab: dict, vocab_size: int, max_seq_len: int,
                 use_window_position: bool, max_windows_per_file: int, device: torch.device) -> SensorMultiHeadDecoder:
    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=384,
        n_heads=12,
        n_layers=8,
        sensor_dim=256,
        n_operations=9,
        memory_pos_encoding=True,
        use_sensor_prior=True,
        use_window_position=use_window_position,
        max_seq_len=max_seq_len,
        max_windows_per_file=max_windows_per_file,
    ).to(device)
    decoder.set_vocab(vocab)
    model_state = decoder.state_dict()
    filtered = {k: v for k, v in ckpt_state.items() if k in model_state and v.shape == model_state[k].shape}
    decoder.load_state_dict(filtered, strict=False)
    decoder.eval()
    return decoder


def _infer_settings_from_ckpt(ckpt_state: dict) -> dict:
    return {
        "vocab_size": ckpt_state["legacy_token_head.weight"].shape[0] if "legacy_token_head.weight" in ckpt_state else None,
        "uses_window_position": "window_pos_embed.weight" in ckpt_state,
        "max_windows": int(ckpt_state["window_pos_embed.weight"].shape[0]) if "window_pos_embed.weight" in ckpt_state else 32,
        "pos_encoding_max_len": int(ckpt_state["pos_encoding.pe"].shape[1]) if "pos_encoding.pe" in ckpt_state else 32,
    }


def evaluate_checkpoint(checkpoint_dir: Path, data_dir: Path, vocab_path: Path,
                        encoder_config: str, fold: int, device: torch.device,
                        batch_size: int = 32,
                        out_dir: Path | None = None) -> dict[str, Any]:
    """Run inference + full metric capture on one checkpoint."""
    tokenizer, vocab, id_to_token = _load_tokenizer(vocab_path)

    ckpt_path = checkpoint_dir / "decoder_checkpoint" / "best_decoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = _ckpt_meta(ckpt_path)
    sd = ckpt["decoder_state_dict"]
    settings = _infer_settings_from_ckpt(sd)

    encoder_ckpt = _resolve_encoder_dir(encoder_config, fold)
    encoder, _enc_config, _enc_ckpt = load_frozen_encoder(str(encoder_ckpt), device)
    encoder.eval()

    test_npz = data_dir / "test_sequences.npz"
    # IMPORTANT: max_token_len must match what training used so input shapes
    # align with the model's pos_encoding capacity. The training scripts
    # (train_v8_*.sh) pass --max_token_len 32 for per_row, 1400 for full_window.
    # We infer from the checkpoint's pos_encoding.pe.
    inferred_max = settings["pos_encoding_max_len"]
    test_ds = DecoderQuickTestDataset(str(test_npz), tokenizer, max_token_len=inferred_max)

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    group_names, group_indices, sensor_dims = build_modality_indices(metadata["continuous_columns"])

    memory, op_pred, encoder_cls_acc = cache_encoder_memory(
        encoder, test_ds, group_indices, device, batch_size=batch_size
    )

    decoder = _build_model(
        ckpt_state=sd,
        vocab=vocab,
        vocab_size=settings["vocab_size"] or len(vocab),
        max_seq_len=settings["pos_encoding_max_len"],
        use_window_position=settings["uses_window_position"],
        max_windows_per_file=settings["max_windows"],
        device=device,
    )

    aug_ds = CachedDecoderDataset(test_ds, memory, op_pred, multi_window_context=0,
                                  window_dropout=0.0, training=False)
    loader = DataLoader(aug_ds, batch_size=batch_size, shuffle=False, collate_fn=decoder_collate_fn, num_workers=0)

    # Buffers for token + per-head predictions / targets.
    pred_tokens_list: list[np.ndarray] = []
    target_tokens_list: list[np.ndarray] = []
    type_p_list: list[np.ndarray] = []; type_t_list: list[np.ndarray] = []
    cmd_p_list: list[np.ndarray] = [];  cmd_t_list: list[np.ndarray] = []
    pt_p_list: list[np.ndarray] = [];   pt_t_list: list[np.ndarray] = []
    sign_p_list: list[np.ndarray] = []; sign_t_list: list[np.ndarray] = []
    digit_p_list: list[np.ndarray] = []; digit_t_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inp = batch["input_tokens"].to(device)
            tgt = batch["target_tokens"].to(device)
            mem = batch["memory"].to(device)
            op = batch["op_pred"].to(device)
            pad_mask = (inp == PAD)
            extra = {}
            if "window_index" in batch:
                extra["window_index"] = batch["window_index"].to(device)
            if "total_windows" in batch:
                extra["total_windows"] = batch["total_windows"].to(device)
            outputs = decoder(tokens=inp, sensor_embeddings=mem, operation_type=op,
                              tgt_key_padding_mask=pad_mask, **extra)

            legacy_logits = outputs.get("legacy_logits", outputs.get("raw_legacy_logits"))
            tok_pred = legacy_logits.argmax(-1)
            pred_tokens_list.append(tok_pred.cpu().numpy())
            target_tokens_list.append(tgt.cpu().numpy())

            if "type_logits" in outputs:
                tp = outputs["type_logits"].argmax(-1).cpu().numpy()
                tt = batch["type_targets"].cpu().numpy()
                type_p_list.append(tp); type_t_list.append(tt)
            if "command_logits" in outputs:
                cp = outputs["command_logits"].argmax(-1).cpu().numpy()
                ct = batch["command_targets"].cpu().numpy()
                cmd_p_list.append(cp); cmd_t_list.append(ct)
            if "param_type_logits" in outputs:
                pp = outputs["param_type_logits"].argmax(-1).cpu().numpy()
                pt = batch["param_type_targets"].cpu().numpy()
                pt_p_list.append(pp); pt_t_list.append(pt)
            if "sign_logits" in outputs:
                sp = outputs["sign_logits"].argmax(-1).cpu().numpy()
                st = batch["sign_targets"].cpu().numpy()
                sign_p_list.append(sp); sign_t_list.append(st)
            if "digit_logits" in outputs:
                # digit_logits: [B, L, 6, 11]
                dp = outputs["digit_logits"].argmax(-1).cpu().numpy()
                dt = batch["digit_targets"].cpu().numpy()
                digit_p_list.append(dp); digit_t_list.append(dt)

    pred_tokens = np.concatenate(pred_tokens_list, axis=0)
    target_tokens = np.concatenate(target_tokens_list, axis=0)

    def _maybe_concat(lst):
        return np.concatenate(lst, axis=0) if lst else None

    type_p = _maybe_concat(type_p_list); type_t = _maybe_concat(type_t_list)
    cmd_p = _maybe_concat(cmd_p_list);   cmd_t = _maybe_concat(cmd_t_list)
    pt_p = _maybe_concat(pt_p_list);     pt_t = _maybe_concat(pt_t_list)
    sign_p = _maybe_concat(sign_p_list); sign_t = _maybe_concat(sign_t_list)
    digit_p = _maybe_concat(digit_p_list); digit_t = _maybe_concat(digit_t_list)

    # ---- METRIC COMPUTATION ----
    metrics: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "encoder_cls_acc": float(encoder_cls_acc),
        "n_test_samples": int(pred_tokens.shape[0]),
    }

    # Token level (legacy head)
    metrics["token"] = compute_full_classification_metrics(
        target_tokens.flatten(), pred_tokens.flatten(),
        ignore_labels={PAD, BOS},  # keep EOS in scoring
    )

    # Sequence level (exact match, ignoring PAD)
    true_seqs = [list(row) for row in target_tokens]
    pred_seqs = [list(row[:len(t)]) for row, t in zip(pred_tokens, [[x for x in row if x != PAD] for row in target_tokens])]
    # simpler: trim each pred to the length of the corresponding non-PAD target
    pred_seqs = []
    true_seqs = []
    for t, p in zip(target_tokens, pred_tokens):
        t_clean = [x for x in t.tolist() if x != PAD]
        L = len(t_clean)
        p_clean = p[:L].tolist()
        true_seqs.append(t_clean)
        pred_seqs.append(p_clean)
    metrics["sequence"] = sequence_level_accuracy(true_seqs, pred_seqs, ignore_labels={PAD})

    # Type
    if type_p is not None:
        metrics["type"] = compute_full_classification_metrics(
            type_t.flatten(), type_p.flatten(),
            label_names=TYPE_NAMES,
            ignore_labels={-1},
        )
    # Command
    if cmd_p is not None:
        metrics["command"] = compute_full_classification_metrics(
            cmd_t.flatten(), cmd_p.flatten(),
            label_names=COMMAND_NAMES,
            ignore_labels={-1},
        )
    # Param type
    if pt_p is not None:
        metrics["param_type"] = compute_full_classification_metrics(
            pt_t.flatten(), pt_p.flatten(),
            label_names=PARAM_NAMES,
            ignore_labels={-1},
        )
    # Sign
    if sign_p is not None:
        metrics["sign"] = compute_full_classification_metrics(
            sign_t.flatten(), sign_p.flatten(),
            label_names={0: "POSITIVE", 1: "NEGATIVE", SIGN_PAD: "PAD"},
            ignore_labels={SIGN_PAD},
        )
    # Numeric digit (multi-position) — collapse to per-position then overall
    if digit_p is not None:
        n_pos = digit_p.shape[-1]
        per_position = []
        for pos in range(n_pos):
            dt = digit_t[..., pos].flatten()
            dp = digit_p[..., pos].flatten()
            per_position.append(compute_full_classification_metrics(
                dt, dp,
                label_names={i: str(i) for i in range(11)} | {DIGIT_PAD: "PAD"},
                ignore_labels={DIGIT_PAD},
            ))
        digit_overall = compute_full_classification_metrics(
            digit_t.flatten(), digit_p.flatten(),
            label_names={i: str(i) for i in range(11)} | {DIGIT_PAD: "PAD"},
            ignore_labels={DIGIT_PAD},
        )
        metrics["numeric"] = {
            "overall": digit_overall,
            "per_position": per_position,
        }

    # ---- WRITE ----
    out_dir = out_dir or (checkpoint_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Predictions NPZ for downstream analysis
    save_kwargs = dict(
        pred_tokens=pred_tokens, target_tokens=target_tokens,
    )
    for name, arr in (
        ("type_p", type_p), ("type_t", type_t),
        ("cmd_p", cmd_p), ("cmd_t", cmd_t),
        ("pt_p", pt_p), ("pt_t", pt_t),
        ("sign_p", sign_p), ("sign_t", sign_t),
        ("digit_p", digit_p), ("digit_t", digit_t),
    ):
        if arr is not None:
            save_kwargs[name] = arr
    np.savez(out_dir / "predictions.npz", **save_kwargs)

    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="Directory containing decoder_checkpoint/best_decoder.pt")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--vocab", type=Path, default=REPO / "data" / "gcode_vocab_v8.json")
    p.add_argument("--encoder-config", default="f98_w256_s64")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint,
        data_dir=args.data_dir,
        vocab_path=args.vocab,
        encoder_config=args.encoder_config,
        fold=args.fold,
        device=device,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )
    print()
    print(f"checkpoint: {args.checkpoint}")
    print(f"data:       {args.data_dir}")
    print(f"n_test:     {metrics['n_test_samples']}")
    print(f"token:      acc={metrics['token']['accuracy']:.4f}  macroF1={metrics['token']['macro_f1']:.4f}")
    print(f"sequence:   acc={metrics['sequence']['accuracy']:.4f}")
    if "command" in metrics:
        print(f"command:    acc={metrics['command']['accuracy']:.4f}  macroF1={metrics['command']['macro_f1']:.4f}")
    if "type" in metrics:
        print(f"type:       acc={metrics['type']['accuracy']:.4f}  macroF1={metrics['type']['macro_f1']:.4f}")
    if "param_type" in metrics:
        print(f"param_type: acc={metrics['param_type']['accuracy']:.4f}  macroF1={metrics['param_type']['macro_f1']:.4f}")
    if "numeric" in metrics:
        print(f"numeric:    acc={metrics['numeric']['overall']['accuracy']:.4f}  macroF1={metrics['numeric']['overall']['macro_f1']:.4f}")
    print(f"out: {args.out_dir or (args.checkpoint / 'results')}/full_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
