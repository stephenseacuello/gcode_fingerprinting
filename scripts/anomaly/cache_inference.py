#!/usr/bin/env python3
"""Phase 0: Cache all decoder inference outputs for 5 folds.

Runs teacher-forced inference once per fold and saves:
- logits.pt: All head logits {legacy, type, command, param_type, sign, digit, raw_legacy}
- nll_scores.pt: Per-token NLL for each head
- predictions.pt: Argmax predictions from each head
- targets.pt: Ground truth targets, lengths, operation types, padding masks
- embeddings.pt: Encoder fingerprints, memory, reconstruction, anomaly logit
- metadata.json: Window counts, class distribution, checkpoint paths

All downstream experiments read from this cache instead of re-running inference.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    CACHE_DIR,
    DECODER_CKPT_TEMPLATE,
    ENCODER_MEMORY_TEMPLATE,
    PREPROC_TEMPLATE,
    VOCAB_PATH,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    FOLDS,
    load_vocab,
    load_decoder,
    load_encoder_memory,
    load_preprocessed_npz,
    build_teacher_forcing_batch,
    compute_per_token_nll,
    setup_logging,
)

import logging

logger = logging.getLogger(__name__)


def cache_fold(fold: int, output_dir: Path, device: str = "cuda"):
    """Run inference for one fold and cache all outputs."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Fold {fold} ===")
    t0 = time.time()

    # Load decoder
    logger.info("Loading decoder...")
    decoder = load_decoder(fold, device=device)
    vocab = load_vocab()
    vocab_size = len(vocab["vocab"])

    # Load cached encoder outputs (skip encoder forward pass)
    logger.info("Loading encoder memory...")
    test_memory, test_op_pred = load_encoder_memory(fold, split="test")
    val_memory, val_op_pred = load_encoder_memory(fold, split="val")
    train_memory, train_op_pred = load_encoder_memory(fold, split="train")

    # Load preprocessed data
    logger.info("Loading preprocessed NPZ...")
    test_data = load_preprocessed_npz(fold, split="test")
    val_data = load_preprocessed_npz(fold, split="val")

    # Build teacher-forcing batches
    test_input, test_target, test_mask = build_teacher_forcing_batch(
        test_data["tokens"], test_data["lengths"]
    )
    val_input, val_target, val_mask = build_teacher_forcing_batch(
        val_data["tokens"], val_data["lengths"]
    )

    # Truncate to match encoder memory size (V7 NPZ may have slightly more windows)
    N_test = min(test_input.shape[0], test_memory.shape[0])
    test_input = test_input[:N_test]
    test_target = test_target[:N_test]
    test_mask = test_mask[:N_test]
    test_memory = test_memory[:N_test]
    test_op_pred = test_op_pred[:N_test]
    # Also truncate metadata
    for key in ["gcode_texts", "operation_type", "operation_type_names"]:
        if key in test_data and len(test_data[key]) > N_test:
            test_data[key] = test_data[key][:N_test]

    N_val = min(val_input.shape[0], val_memory.shape[0])
    val_input = val_input[:N_val]
    val_target = val_target[:N_val]
    val_mask = val_mask[:N_val]
    val_memory = val_memory[:N_val]
    val_op_pred = val_op_pred[:N_val]
    for key in ["operation_type"]:
        if key in val_data and len(val_data[key]) > N_val:
            val_data[key] = val_data[key][:N_val]

    L = test_input.shape[1]
    logger.info(f"Test: {N_test} windows, seq_len={L}, memory={test_memory.shape}")

    # Run decoder inference (teacher-forced)
    logger.info("Running decoder inference...")
    batch_size = 64

    all_logits = {
        "legacy_logits": [],
        "raw_legacy_logits": [],
        "type_logits": [],
        "command_logits": [],
        "param_type_logits": [],
        "sign_logits": [],
        "digit_logits": [],
    }

    with torch.no_grad():
        for start in range(0, N_test, batch_size):
            end = min(start + batch_size, N_test)

            b_input = test_input[start:end].to(device)
            b_memory = test_memory[start:end].to(device)
            b_op = test_op_pred[start:end].to(device)
            b_mask = test_mask[start:end].to(device)

            outputs = decoder(
                tokens=b_input,
                sensor_embeddings=b_memory,
                operation_type=b_op,
                tgt_key_padding_mask=b_mask,
            )

            for key in all_logits:
                if key in outputs:
                    all_logits[key].append(outputs[key].cpu())

            if (start // batch_size) % 5 == 0:
                logger.info(f"  Batch {start//batch_size + 1}/{(N_test + batch_size - 1)//batch_size}")

    # Concatenate
    for key in all_logits:
        if all_logits[key]:
            all_logits[key] = torch.cat(all_logits[key], dim=0)
        else:
            all_logits[key] = None

    # Also run on validation set (for calibration / threshold setting)
    logger.info("Running validation inference...")
    val_logits = {"legacy_logits": []}
    N_val = val_input.shape[0]
    with torch.no_grad():
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            b_input = val_input[start:end].to(device)
            b_memory = val_memory[start:end].to(device)
            b_op = val_op_pred[start:end].to(device)
            b_mask = val_mask[start:end].to(device)

            outputs = decoder(
                tokens=b_input,
                sensor_embeddings=b_memory,
                operation_type=b_op,
                tgt_key_padding_mask=b_mask,
            )
            val_logits["legacy_logits"].append(outputs["legacy_logits"].cpu())

    val_logits["legacy_logits"] = torch.cat(val_logits["legacy_logits"], dim=0)

    # Compute NLL scores
    logger.info("Computing NLL scores...")
    nll_legacy = compute_per_token_nll(
        all_logits["legacy_logits"], test_target, test_mask
    )
    nll_type = compute_per_token_nll(
        all_logits["type_logits"],
        # Need type targets — derive from token type map
        _get_type_targets(test_target, vocab),
        test_mask,
    ) if all_logits["type_logits"] is not None else None

    # Compute predictions
    logger.info("Computing predictions...")
    predictions = {}
    if all_logits["legacy_logits"] is not None:
        predictions["argmax_tokens"] = all_logits["legacy_logits"].argmax(dim=-1)
    if all_logits["type_logits"] is not None:
        predictions["type_preds"] = all_logits["type_logits"].argmax(dim=-1)
    if all_logits["command_logits"] is not None:
        predictions["cmd_preds"] = all_logits["command_logits"].argmax(dim=-1)
    if all_logits["param_type_logits"] is not None:
        predictions["param_preds"] = all_logits["param_type_logits"].argmax(dim=-1)
    if all_logits["digit_logits"] is not None:
        predictions["digit_preds"] = all_logits["digit_logits"].argmax(dim=-1)
    if all_logits["sign_logits"] is not None:
        predictions["sign_preds"] = all_logits["sign_logits"].argmax(dim=-1)

    # Save everything
    logger.info("Saving cached data...")

    # Logits (biggest file — save only what we need)
    torch.save(
        {k: v for k, v in all_logits.items() if v is not None},
        fold_dir / "logits.pt",
    )

    # NLL scores
    nll_dict = {"nll_legacy": nll_legacy}
    if nll_type is not None:
        nll_dict["nll_type"] = nll_type
    torch.save(nll_dict, fold_dir / "nll_scores.pt")

    # Predictions
    torch.save(predictions, fold_dir / "predictions.pt")

    # Targets and metadata
    torch.save(
        {
            "target_tokens": test_target,
            "input_tokens": test_input,
            "lengths": (~test_mask).sum(dim=1),  # actual token seq lengths (not sensor window lengths)
            "sensor_lengths": torch.from_numpy(test_data["lengths"].copy()),  # sensor window lengths (256)
            "operation_type": torch.from_numpy(test_data["operation_type"].copy()),
            "padding_mask": test_mask,
            "gcode_texts": test_data["gcode_texts"].tolist(),
            "operation_type_names": test_data["operation_type_names"].tolist(),
        },
        fold_dir / "targets.pt",
    )

    # Validation data (for threshold setting, calibration)
    torch.save(
        {
            "val_legacy_logits": val_logits["legacy_logits"],
            "val_target_tokens": val_target,
            "val_padding_mask": val_mask,
            "val_lengths": (~val_mask).sum(dim=1),  # actual token seq lengths
            "val_operation_type": torch.from_numpy(val_data["operation_type"].copy()),
        },
        fold_dir / "val_data.pt",
    )

    # Embeddings (from encoder memory cache)
    torch.save(
        {
            "test_memory": test_memory,
            "test_op_pred": test_op_pred,
            "train_memory": train_memory,
            "train_op_pred": train_op_pred,
        },
        fold_dir / "embeddings.pt",
    )

    # Metadata
    op_types = test_data["operation_type"]
    unique, counts = np.unique(op_types, return_counts=True)
    class_counts = {
        test_data["operation_type_names"][np.where(op_types == u)[0][0]]: int(c)
        for u, c in zip(unique, counts)
    }

    metadata = {
        "fold": fold,
        "n_test_windows": int(N_test),
        "n_val_windows": int(N_val),
        "seq_len": int(L),
        "vocab_size": vocab_size,
        "class_counts": class_counts,
        "decoder_checkpoint": DECODER_CKPT_TEMPLATE.format(fold=fold),
        "encoder_memory_dir": ENCODER_MEMORY_TEMPLATE.format(fold=fold),
        "data_dir": PREPROC_TEMPLATE.format(fold=fold),
        "cache_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - t0, 1),
    }
    with open(fold_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Fold {fold} cached in {elapsed:.1f}s ({N_test} test + {N_val} val windows)")

    # Free GPU memory
    del decoder
    torch.cuda.empty_cache()


def _get_type_targets(target_tokens: torch.Tensor, vocab: dict) -> torch.Tensor:
    """Convert token targets to type targets (SPECIAL/COMMAND/PARAM/NUMERIC)."""
    from scripts.anomaly.anomaly_scoring_utils import build_token_type_map

    type_map = build_token_type_map(vocab)
    N, L = target_tokens.shape
    type_targets = torch.zeros_like(target_tokens)
    for i in range(N):
        for j in range(L):
            tid = target_tokens[i, j].item()
            type_targets[i, j] = type_map.get(tid, 0)
    return type_targets


def main():
    parser = argparse.ArgumentParser(description="Cache decoder inference for all folds")
    parser.add_argument("--output-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    setup_logging("cache_inference")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")
    logger.info(f"Device: {args.device}")

    t_start = time.time()
    for fold in args.folds:
        cache_fold(fold, args.output_dir, device=args.device)

    # Write completion flag
    flag = args.output_dir / "cache_complete.flag"
    flag.write_text(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total = time.time() - t_start
    logger.info(f"All folds cached in {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
