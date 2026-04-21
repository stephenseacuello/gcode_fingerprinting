#!/usr/bin/env python3
"""
Generate publication-quality PCA and t-SNE figures of decoder hidden states.

Figure 1: PCA of all decoder hidden states colored by token type.
Figure 2: t-SNE of numeric tokens only, colored by coordinate axis.

Uses the V7 best checkpoint (fold 5, seed 789) with teacher-forced forward pass.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig
from run_decoder_quick_test import (
    DecoderQuickTestDataset,
    CachedDecoderDataset,
    decoder_collate_fn,
    PAD,
)

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_PATH = ROOT / "outputs/decoder20260304/v7_best_5fold_multiseed/fold_5_seed_789/decoder_checkpoint/best_decoder.pt"
MEMORY_DIR = ROOT / "outputs/decoder20260304/v7_best_5fold_multiseed/fold_5_seed_789/encoder_memory"
DATA_DIR = ROOT / "outputs/decoder20260304/preprocessed_v7/fold_5"
VOCAB_PATH = ROOT / "data/gcode_vocab_712.json"
OUT_DIR = ROOT / "outputs/decoder20260304/paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Publication matplotlib settings ──────────────────────────────────────────
RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_vocab(path):
    with open(path) as f:
        data = json.load(f)
    return data["vocab"], data.get("config", {})


def classify_token_type(tok_name):
    """Classify token name into type category."""
    if tok_name in ("PAD", "BOS", "EOS", "UNK", "MASK"):
        return "Special"
    if tok_name in ("G0", "G1", "G2", "G3", "G53", "M30"):
        return "Command"
    if tok_name in ("X", "Y", "Z", "F", "R", "S", "I", "J", "K"):
        return "Parameter"
    if tok_name.startswith("NUM_"):
        return "Numeric"
    # Compound tokens like "F22.", "Z0.", "Y0."
    return "Special"


def get_numeric_axis(tok_name):
    """Extract axis from numeric token name (e.g., NUM_X_1650 -> 'X')."""
    if tok_name.startswith("NUM_"):
        parts = tok_name.split("_")
        if len(parts) >= 3:
            return parts[1]
    return None


def get_numeric_value(tok_name, precision_map):
    """Extract the physical value from a numeric token name."""
    if not tok_name.startswith("NUM_"):
        return None
    parts = tok_name.split("_")
    if len(parts) < 3:
        return None
    axis = parts[1]
    try:
        bucket = int(parts[2])
    except ValueError:
        return None
    step = precision_map.get(axis, 0.001)
    return bucket * step


def main():
    print("=" * 70)
    print("Decoder PCA/t-SNE Figure Generation")
    print("=" * 70)

    # ── 1. Load vocabulary ──────────────────────────────────────────────────
    vocab, vocab_config = load_vocab(VOCAB_PATH)
    id2tok = {v: k for k, v in vocab.items()}
    precision_map = {k: float(v) for k, v in vocab_config.get("precision", {}).items()}
    print(f"Vocabulary: {len(vocab)} tokens")

    # ── 2. Build tokenizer ──────────────────────────────────────────────────
    cfg = TokenizerConfig(
        mode=vocab_config.get("mode", "hybrid"),
        precision=precision_map,
    )
    tokenizer = GCodeTokenizer(cfg, vocab=vocab)

    # ── 3. Load dataset ─────────────────────────────────────────────────────
    test_npz = DATA_DIR / "test_sequences.npz"
    print(f"Loading test data: {test_npz}")
    base_dataset = DecoderQuickTestDataset(str(test_npz), tokenizer, max_token_len=16)
    print(f"  Samples: {len(base_dataset)}")
    print(f"  Stats: {base_dataset.stats}")

    # ── 4. Load cached encoder memory ────────────────────────────────────────
    test_memory = torch.load(MEMORY_DIR / "test_memory.pt", map_location="cpu", weights_only=True)
    test_op_pred = torch.load(MEMORY_DIR / "test_op_pred.pt", map_location="cpu", weights_only=True)
    print(f"  Memory shape: {test_memory.shape}")
    print(f"  Op pred shape: {test_op_pred.shape}")

    # ── 5. Build CachedDecoderDataset with MWC ──────────────────────────────
    # MWC=2 matches training config for V7 best
    cached_dataset = CachedDecoderDataset(
        base_dataset, test_memory, test_op_pred,
        multi_window_context=2, training=False,
    )
    loader = DataLoader(
        cached_dataset, batch_size=32, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=0,
    )

    # ── 6. Load decoder ─────────────────────────────────────────────────────
    print(f"\nLoading decoder checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    decoder = SensorMultiHeadDecoder(
        vocab_size=712,
        d_model=384,
        n_heads=12,
        n_layers=8,
        sensor_dim=256,
        n_operations=9,
        memory_pos_encoding=True,
        use_sensor_prior=True,
        use_window_position=True,
        max_seq_len=16,
        max_windows_per_file=13,
    )
    # Load with strict=False to handle buffer keys (type_token_mask, grammar_mask)
    # that were saved in the checkpoint but are registered dynamically via set_vocab()
    missing, unexpected = decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    if unexpected:
        print(f"  Unexpected keys (buffers from set_vocab): {unexpected}")
    if missing:
        print(f"  Missing keys: {missing}")
    decoder = decoder.to(DEVICE)
    decoder.eval()
    print("  Decoder loaded and set to eval mode")

    # ── 7. Extract hidden states ─────────────────────────────────────────────
    print("\nExtracting decoder hidden states...")
    all_hidden = []
    all_target_ids = []

    with torch.no_grad():
        for batch in loader:
            input_tokens = batch["input_tokens"].to(DEVICE)
            memory = batch["memory"].to(DEVICE)
            op_pred = batch["op_pred"].to(DEVICE)
            target_tokens = batch["target_tokens"]
            padding_mask = (input_tokens == PAD)

            extra_kwargs = {}
            if "window_index" in batch:
                extra_kwargs["window_index"] = batch["window_index"].to(DEVICE)
            if "total_windows" in batch:
                extra_kwargs["total_windows"] = batch["total_windows"].to(DEVICE)

            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=memory,
                operation_type=op_pred,
                tgt_key_padding_mask=padding_mask,
                return_hidden=True,
                **extra_kwargs,
            )

            hidden = outputs["hidden"].cpu()  # [B, L, 384]
            B, L, D = hidden.shape

            for b in range(B):
                for pos in range(L):
                    tid = target_tokens[b, pos].item()
                    if tid == PAD:
                        continue  # skip padding positions
                    all_hidden.append(hidden[b, pos].numpy())
                    all_target_ids.append(tid)

    all_hidden = np.array(all_hidden)
    all_target_ids = np.array(all_target_ids)
    print(f"  Collected {len(all_hidden)} hidden states from {len(cached_dataset)} windows")

    # ── 8. Classify tokens ───────────────────────────────────────────────────
    token_types = []
    token_names = []
    for tid in all_target_ids:
        name = id2tok.get(tid, f"UNK_{tid}")
        token_names.append(name)
        token_types.append(classify_token_type(name))

    token_types = np.array(token_types)
    token_names = np.array(token_names)

    type_counts = {}
    for t in np.unique(token_types):
        type_counts[t] = int(np.sum(token_types == t))
    print(f"  Token type distribution: {type_counts}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1: PCA by Token Type
    # ──────────────────────────────────────────────────────────────────────────
    print("\nGenerating Figure 1: PCA by token type...")

    plt.rcParams.update(RCPARAMS)

    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(all_hidden)

    type_colors = {
        "Command":   "#c0392b",   # strong red
        "Parameter": "#2471a3",   # strong blue
        "Numeric":   "#27ae60",   # strong green
        "Special":   "#7f8c8d",   # gray
    }
    type_markers = {
        "Command":   "^",
        "Parameter": "s",
        "Numeric":   "o",
        "Special":   "X",
    }
    type_sizes = {
        "Command":   55,
        "Parameter": 42,
        "Numeric":   30,
        "Special":   42,
    }
    type_zorder = {
        "Special":   1,
        "Numeric":   2,
        "Parameter": 3,
        "Command":   4,
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Plot each type separately for legend; draw in z-order
    plot_order = ["Special", "Numeric", "Parameter", "Command"]
    for ttype in plot_order:
        mask = token_types == ttype
        if mask.sum() == 0:
            continue
        ax.scatter(
            hidden_2d[mask, 0],
            hidden_2d[mask, 1],
            c=type_colors[ttype],
            marker=type_markers[ttype],
            s=type_sizes[ttype],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"{ttype} ($n$={int(mask.sum())})",
            zorder=type_zorder[ttype],
        )

    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100

    ax.set_xlabel(f"PC 1 ({ev1:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({ev2:.1f}% variance)")
    ax.set_title("Decoder Hidden States by Token Type (PCA)", pad=10)

    # Legend with a clean frame
    leg = ax.legend(
        loc="upper left",
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=False,
        borderpad=0.6,
        handletextpad=0.5,
    )

    # Annotate the two main clusters with text callouts
    # Cluster 1 (upper-left): Commands + Parameters + some Special
    cmd_center = hidden_2d[token_types == "Command"].mean(axis=0)
    # Cluster 2 (right side): Numerics + EOS tokens that follow them
    num_center = hidden_2d[token_types == "Numeric"].mean(axis=0)

    text_style = dict(fontsize=8.5, fontstyle="italic", color="#555555")
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]

    ax.annotate(
        "Commands &\nparameters",
        xy=(cmd_center[0], cmd_center[1] + 2),
        fontsize=8.5, fontstyle="italic", color="#555555",
        ha="center", va="bottom",
        path_effects=outline,
    )
    ax.annotate(
        "Numeric\nvalues",
        xy=(num_center[0], num_center[1] + 3),
        fontsize=8.5, fontstyle="italic", color="#555555",
        ha="center", va="bottom",
        path_effects=outline,
    )

    ax.grid(True, alpha=0.15, linestyle="--", color="#888888")

    for fmt in ("pdf", "png"):
        path = OUT_DIR / f"decoder_token_pca.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2: t-SNE of Numeric Tokens by Coordinate Axis
    # ──────────────────────────────────────────────────────────────────────────
    print("\nGenerating Figure 2: t-SNE of numeric tokens by axis...")

    numeric_mask = token_types == "Numeric"
    numeric_hidden = all_hidden[numeric_mask]
    numeric_names_arr = token_names[numeric_mask]

    # Classify by axis and extract physical values
    axes_labels = []
    physical_values = []
    for name in numeric_names_arr:
        axis = get_numeric_axis(name)
        if axis in ("X", "Y", "Z", "R", "F"):
            axes_labels.append(axis)
        else:
            axes_labels.append("Other")
        val = get_numeric_value(name, precision_map)
        physical_values.append(val if val is not None else 0.0)

    axes_labels = np.array(axes_labels)
    physical_values = np.array(physical_values)

    # Filter out "Other" axis tokens
    valid_mask = axes_labels != "Other"
    numeric_hidden_valid = numeric_hidden[valid_mask]
    axes_valid = axes_labels[valid_mask]
    values_valid = physical_values[valid_mask]
    names_valid = numeric_names_arr[valid_mask]

    axis_counts = {}
    for a in np.unique(axes_valid):
        axis_counts[a] = int(np.sum(axes_valid == a))
    print(f"  Numeric axis distribution: {axis_counts}")
    print(f"  Total numeric points for t-SNE: {len(numeric_hidden_valid)}")

    # t-SNE
    perplexity = min(30, max(5, len(numeric_hidden_valid) // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=2000,
        learning_rate="auto",
        init="pca",
    )
    tsne_2d = tsne.fit_transform(numeric_hidden_valid)

    axis_colors = {
        "X": "#2471a3",   # blue
        "Y": "#e67e22",   # orange
        "Z": "#27ae60",   # green
        "R": "#8e44ad",   # purple
        "F": "#c0392b",   # red
    }
    axis_markers = {
        "X": "o",         # circle
        "Y": "^",         # triangle up
        "Z": "s",         # square
        "R": "D",         # diamond
        "F": "*",         # star
    }
    # Larger markers for rare axes so they remain visible
    axis_sizes = {
        "X": 40,
        "Y": 50,
        "Z": 65,
        "R": 55,
        "F": 80,
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))

    for axis_name in ["X", "Y", "Z", "R", "F"]:
        mask = axes_valid == axis_name
        if mask.sum() == 0:
            continue
        ax.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=axis_colors[axis_name],
            marker=axis_markers[axis_name],
            s=axis_sizes[axis_name],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=f"{axis_name}-axis ($n$={int(mask.sum())})",
            zorder=3 if mask.sum() < 10 else 2,  # bring rare axes to front
        )

    # Annotate a few representative points within the X and Y clusters
    # to show within-cluster value structure
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    annotated = set()
    for axis_name, max_annot in [("X", 4), ("Y", 3), ("Z", 1), ("R", 1)]:
        mask = axes_valid == axis_name
        if mask.sum() == 0:
            continue
        axis_indices = np.where(mask)[0]
        # Select a diverse subset: min, max, and middle values
        axis_vals = values_valid[mask]
        sort_order = np.argsort(axis_vals)
        if len(sort_order) <= max_annot:
            pick = sort_order
        else:
            # Pick evenly spaced indices through sorted values
            pick_idx = np.linspace(0, len(sort_order) - 1, max_annot, dtype=int)
            pick = sort_order[pick_idx]

        for local_idx in pick:
            global_idx = axis_indices[local_idx]
            val = values_valid[global_idx]
            name = names_valid[global_idx]
            # Format the value label
            if abs(val) < 0.01:
                label = f"{val:.4f}"
            elif abs(val) < 10:
                label = f"{val:.3f}"
            else:
                label = f"{val:.1f}"

            ax.annotate(
                label,
                xy=(tsne_2d[global_idx, 0], tsne_2d[global_idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=axis_colors[axis_name],
                fontweight="bold",
                path_effects=outline,
            )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Decoder Numeric Representations by Coordinate Axis (t-SNE)", pad=10)

    leg = ax.legend(
        loc="best",
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=False,
        borderpad=0.6,
        handletextpad=0.5,
    )

    ax.grid(True, alpha=0.15, linestyle="--", color="#888888")

    for fmt in ("pdf", "png"):
        path = OUT_DIR / f"decoder_numeric_tsne.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)

    print("\nDone! All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
