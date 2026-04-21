#!/usr/bin/env python3
"""
Full per-class evaluation across all 9 operation classes for the decoder paper.

Loads each fold's checkpoint (2B5_no_label_smoothing), runs teacher-forced
evaluation on ALL test windows, and computes per-operation-class token accuracy.

Saves results to: outputs/decoder20260304/paper/per_class_accuracy.json

Author: Claude Code
Date: March 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.utilities.gcode_tokenizer import GCodeTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────
PAD = 0
BOS = 1
EOS = 2
UNK = 3

BASE_DIR = Path("/home/seacuello/Documents/gcode_fingerprinting")
ABLATION_DIR = BASE_DIR / "outputs/decoder20260304/ablations/2B5_no_label_smoothing"
DATA_DIR = BASE_DIR / "outputs/decoder20260304/preprocessed_v7"
VOCAB_PATH = BASE_DIR / "data/gcode_vocab_712.json"
OUTPUT_PATH = BASE_DIR / "outputs/decoder20260304/paper/per_class_accuracy.json"

FOLDS = [1, 2, 3, 4, 5]
MAX_TOKEN_LEN = 16


def get_mwc_memory(idx, memory_tensor, file_groups, idx_to_file, mwc=2):
    """Get multi-window context memory by concatenating neighboring windows."""
    sf = idx_to_file[idx]
    file_idxs = file_groups[sf]
    try:
        my_pos = file_idxs.index(idx)
    except ValueError:
        my_pos = 0
    start = max(0, my_pos - mwc)
    end = min(len(file_idxs), my_pos + mwc + 1)
    neighbor_idxs = [file_idxs[i] for i in range(start, end)]
    memories = [memory_tensor[ni] for ni in neighbor_idxs]
    return torch.cat(memories, dim=0)  # [T_s * n_neighbors, 256]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer from vocab JSON
    tokenizer = GCodeTokenizer.load(VOCAB_PATH)
    vocab = tokenizer.vocab
    vocab_size = len(vocab)
    id2token = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {vocab_size}")

    # Results accumulator
    all_results = {}
    global_per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    global_correct = 0
    global_total = 0

    for fold in FOLDS:
        print(f"\n{'='*70}")
        print(f"Fold {fold}")
        print(f"{'='*70}")

        # Paths
        ckpt_path = ABLATION_DIR / f"fold_{fold}" / "decoder_checkpoint" / "best_decoder.pt"
        memory_path = ABLATION_DIR / f"fold_{fold}" / "encoder_memory" / "test_memory.pt"
        data_path = DATA_DIR / f"fold_{fold}" / "test_sequences.npz"

        assert ckpt_path.exists(), f"Missing: {ckpt_path}"
        assert memory_path.exists(), f"Missing: {memory_path}"
        assert data_path.exists(), f"Missing: {data_path}"

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        state_dict = ckpt["decoder_state_dict"]

        # Load test data
        npz = np.load(data_path, allow_pickle=True)
        gcode_texts = list(npz["gcode_texts"])
        operation_type = torch.from_numpy(npz["operation_type"].astype(np.int64))
        op_names = list(npz["operation_type_names"])
        window_index = torch.from_numpy(npz["window_index"].astype(np.int64))
        total_windows = torch.from_numpy(npz["total_windows"].astype(np.int64))
        source_files = list(npz["source_file"])
        N = len(gcode_texts)
        print(f"  Test samples: {N}")

        # Re-tokenize gcode_texts using the tokenizer (same as training)
        precision = tokenizer.cfg.precision
        all_input_tokens = []
        all_target_tokens = []
        for text in gcode_texts:
            text_str = str(text)
            canon = tokenizer.canonicalize([text_str])
            tok_strings = tokenizer.tokenize_canonical(canon)
            tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
            # Truncate to max_token_len - 1 (leave room for BOS/EOS)
            tok_ids = tok_ids[:MAX_TOKEN_LEN - 1]
            # Input: [BOS, t1, t2, ..., tn]
            inp = [BOS] + tok_ids
            # Target: [t1, t2, ..., tn, EOS]
            tgt = tok_ids + [EOS]
            all_input_tokens.append(inp)
            all_target_tokens.append(tgt)

        # Load encoder memory
        memory_tensor = torch.load(memory_path, map_location=device, weights_only=False)
        print(f"  Memory shape: {memory_tensor.shape}")

        # Build file groups for MWC
        file_groups = defaultdict(list)
        for i, sf in enumerate(source_files):
            file_groups[sf].append(i)
        for sf in file_groups:
            file_groups[sf].sort()
        idx_to_file = {i: sf for i, sf in enumerate(source_files)}

        # Determine max_windows_per_file from checkpoint weight
        if "window_pos_embed.weight" in state_dict:
            max_win = state_dict["window_pos_embed.weight"].shape[0]
        else:
            max_win = 32

        # Construct model with same args as training
        decoder = SensorMultiHeadDecoder(
            vocab_size=vocab_size,
            d_model=args["d_model"],
            sensor_dim=256,  # encoder d_model
            n_operations=9,
            n_heads=args["n_heads"],
            n_layers=args["n_layers"],
            max_int_digits=2,
            n_decimal_digits=4,
            dropout=args["dropout"],
            max_seq_len=args["max_token_len"],
            hierarchical=args.get("hierarchical", False),
            memory_pos_encoding=args.get("memory_pos_encoding", False),
            use_regression_head=args.get("use_regression_head", False),
            use_window_position=args.get("use_window_position", False),
            max_windows_per_file=max_win,
            use_sequence_classifier=args.get("use_sequence_classifier", False),
            use_pointer_network=False,
            use_sensor_prior=args.get("use_sensor_prior", False),
            drop_path_rate=args.get("drop_path_rate", 0.0),
        )

        # Load state dict with filtering for size mismatches
        model_state = decoder.state_dict()
        filtered_state = {}
        size_mismatched = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                size_mismatched.append(f"{k}: ckpt={v.shape} vs model={model_state[k].shape}")
            elif k in model_state:
                filtered_state[k] = v
        missing, unexpected = decoder.load_state_dict(filtered_state, strict=False)
        if size_mismatched:
            print(f"  Size-mismatched (skipped): {size_mismatched}")
        if missing:
            sm_keys = {s.split(":")[0].strip() for s in size_mismatched}
            truly_missing = [k for k in missing if k not in sm_keys]
            if truly_missing:
                print(f"  Missing keys ({len(truly_missing)}): {truly_missing[:10]}")

        # Set vocab for grammar/type constraints
        decoder.set_vocab(vocab)
        decoder.use_grammar_constraint = args.get("grammar_constraint", False)
        decoder = decoder.to(device)
        decoder.eval()

        param_count = sum(p.numel() for p in decoder.parameters())
        print(f"  Model params: {param_count:,}")

        # Sample check: print a few input/target pairs
        for si in range(min(3, N)):
            inp_str = [id2token.get(t, f"?{t}") for t in all_input_tokens[si]]
            tgt_str = [id2token.get(t, f"?{t}") for t in all_target_tokens[si]]
            print(f"  Sample {si}: inp={inp_str}, tgt={tgt_str}")

        # Run teacher-forced evaluation on ALL test samples
        mwc = args.get("multi_window_context", 0)
        fold_per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        fold_correct = 0
        fold_total = 0

        with torch.no_grad():
            for i in range(N):
                # Get MWC memory
                if mwc > 0:
                    mem = get_mwc_memory(i, memory_tensor, file_groups, idx_to_file, mwc)
                else:
                    mem = memory_tensor[i]
                mem = mem.unsqueeze(0).to(device)  # [1, T_s_mwc, 256]

                inp = torch.tensor(all_input_tokens[i], dtype=torch.long, device=device).unsqueeze(0)
                tgt = torch.tensor(all_target_tokens[i], dtype=torch.long, device=device)
                op = operation_type[i].unsqueeze(0).to(device)
                win_idx = window_index[i].unsqueeze(0).to(device)
                tot_win = total_windows[i].unsqueeze(0).to(device)
                cls_name = op_names[i]
                seq_len = len(all_target_tokens[i])

                # Forward pass (teacher-forced)
                outputs = decoder(
                    tokens=inp,
                    sensor_embeddings=mem,
                    operation_type=op,
                    window_index=win_idx,
                    total_windows=tot_win,
                    teacher_forcing_ratio=1.0,
                )

                # Get legacy predictions
                legacy_logits = outputs["legacy_logits"]  # [1, seq_len, vocab_size]
                preds = legacy_logits.argmax(dim=-1).squeeze(0)  # [seq_len]

                # Compare to target
                correct = (preds == tgt).sum().item()
                total = seq_len

                fold_correct += correct
                fold_total += total
                fold_per_class[cls_name]["correct"] += correct
                fold_per_class[cls_name]["total"] += total
                global_per_class[cls_name]["correct"] += correct
                global_per_class[cls_name]["total"] += total

        global_correct += fold_correct
        global_total += fold_total

        fold_acc = fold_correct / fold_total if fold_total > 0 else 0
        print(f"\n  Fold {fold} overall: {fold_correct}/{fold_total} = {fold_acc:.4f}")
        fold_class_results = {}
        for cls in sorted(fold_per_class.keys()):
            c = fold_per_class[cls]["correct"]
            t = fold_per_class[cls]["total"]
            acc = c / t if t > 0 else 0
            n_windows = sum(1 for name in op_names if name == cls)
            fold_class_results[cls] = {
                "correct": c, "total": t, "accuracy": round(acc, 6),
                "n_windows": n_windows,
            }
            print(f"    {cls:20s}: {c:5d}/{t:5d} = {acc:.4f}  ({n_windows} windows)")

        all_results[f"fold_{fold}"] = {
            "overall_accuracy": round(fold_acc, 6),
            "overall_correct": fold_correct,
            "overall_total": fold_total,
            "n_test_samples": N,
            "per_class": fold_class_results,
        }

    # Aggregate across all folds
    print(f"\n{'='*70}")
    print(f"AGGREGATE (all folds)")
    print(f"{'='*70}")
    agg_acc = global_correct / global_total if global_total > 0 else 0
    print(f"Overall: {global_correct}/{global_total} = {agg_acc:.4f}")

    agg_per_class = {}
    for cls in sorted(global_per_class.keys()):
        c = global_per_class[cls]["correct"]
        t = global_per_class[cls]["total"]
        acc = c / t if t > 0 else 0
        agg_per_class[cls] = {
            "correct": c, "total": t, "accuracy": round(acc, 6),
        }
        print(f"  {cls:20s}: {c:6d}/{t:6d} = {acc:.4f}")

    all_results["aggregate"] = {
        "overall_accuracy": round(agg_acc, 6),
        "overall_correct": global_correct,
        "overall_total": global_total,
        "per_class": agg_per_class,
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
