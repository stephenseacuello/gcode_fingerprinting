#!/usr/bin/env python3
"""CPU inference cost benchmark.

Times one decoder forward pass per_row + full_window on CPU at batch=1
to estimate edge-hardware deployment latency.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch


def main():
    device = torch.device("cpu")
    base_dir = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / "fold_1" / "6o90io5p"
    memory = torch.load(base_dir / "encoder_memory" / "test_memory.pt", map_location=device, weights_only=False)[:1]
    op_pred = torch.load(base_dir / "encoder_memory" / "test_op_pred.pt", map_location=device, weights_only=False)[:1]
    print(f"sample memory shape {memory.shape}, op_pred {op_pred.shape}")

    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    blob = torch.load(base_dir / "decoder_checkpoint" / "best_decoder.pt", map_location=device, weights_only=False)
    a = blob["args"]
    vocab = json.load(open(REPO / "data" / "gcode_vocab_v8.json"))["vocab"]
    decoder = SensorMultiHeadDecoder(
        vocab_size=len(vocab),
        sensor_dim=memory.shape[-1],
        d_model=a.get("d_model", 384),
        n_layers=a.get("n_layers", 8),
        n_heads=a.get("n_heads", 12),
        dropout=0.0,
        max_seq_len=a.get("max_token_len", 1400),
        n_operations=9,
    ).to(device)
    decoder.set_vocab(vocab)
    sd = blob["decoder_state_dict"]
    ms = decoder.state_dict()
    decoder.load_state_dict({k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}, strict=False)
    decoder.eval()

    results = {}
    for seq_len, label in [(32, "per-row (32 tokens)"), (256, "medium (256 tokens)"), (1024, "full-window (1024 tokens)")]:
        in_tok = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        in_tok[0, 0] = 1  # BOS
        padding = torch.zeros_like(in_tok, dtype=torch.bool)
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = decoder(tokens=in_tok, sensor_embeddings=memory, operation_type=op_pred, tgt_key_padding_mask=padding)
        # Time
        n_runs = 10
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = decoder(tokens=in_tok, sensor_embeddings=memory, operation_type=op_pred, tgt_key_padding_mask=padding)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        med = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        results[label] = {"seq_len": seq_len, "median_ms": med, "p95_ms": p95, "n_runs": n_runs}
        print(f"  {label:35s}: median {med:.1f} ms, p95 {p95:.1f} ms  ({n_runs} runs)")

    # Same for autoregressive generation (greedy, no beam)
    from copy import deepcopy
    print("\n=== autoregressive (greedy) ===")
    for target_len, label in [(32, "per-row AR (32 tokens)"), (256, "medium AR (256 tokens)")]:
        n_runs = 3
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                seq = [1]  # BOS
                for _ in range(target_len - 1):
                    tok_t = torch.tensor([seq], device=device)
                    pad = torch.zeros_like(tok_t, dtype=torch.bool)
                    out = decoder(tokens=tok_t, sensor_embeddings=memory, operation_type=op_pred, tgt_key_padding_mask=pad)
                    nxt = int(out["legacy_logits"][0, -1].argmax())
                    if nxt == 2:  # EOS
                        break
                    seq.append(nxt)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        med = float(np.median(times))
        results[label] = {"target_len": target_len, "median_ms": med, "n_runs": n_runs}
        print(f"  {label:35s}: median {med:.0f} ms  ({n_runs} runs)")

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "inference_cost_cpu.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
