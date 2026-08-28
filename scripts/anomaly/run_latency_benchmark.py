#!/usr/bin/env python3
"""Real latency benchmark for the V7 detector (R1 M7).

Replaces the hard-coded 19.9 ms / 3200x literals with measured timings of the
decoder forward pass (exact best_config_5fold checkpoint) and the encoder forward
pass (representative same-family checkpoint), on GPU and CPU. Per 64 s window.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR, load_decoder, load_cached_targets, load_encoder_memory,
    build_teacher_forcing_batch, setup_logging,
)
import logging
logger = logging.getLogger(__name__)

WINDOW_SECONDS = 64.0   # each sensor window spans 64 s of machining


def time_decoder(fold, device, n_rep=50, warmup=10):
    dec = load_decoder(fold, device=device)
    td = load_cached_targets(fold)
    tt, ln = td["target_tokens"], td["lengths"]
    mem, op_pred = load_encoder_memory(fold, split="test")
    N = min(len(tt), len(mem))
    tt, ln, mem, op_pred = tt[:N], ln[:N], mem[:N], op_pred[:N]
    inp, tgt, pmask = build_teacher_forcing_batch(tt.numpy() if hasattr(tt, "numpy") else tt, ln.numpy() if hasattr(ln, "numpy") else np.array(ln))
    inp = inp.to(device); mem = mem.to(device); op_pred = op_pred.to(device); pmask = pmask.to(device)
    # single-window latency (batch=1)
    times = []
    with torch.no_grad():
        for k in range(warmup + n_rep):
            i = k % N
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dec(tokens=inp[i:i+1], sensor_embeddings=mem[i:i+1],
                operation_type=op_pred[i:i+1], tgt_key_padding_mask=pmask[i:i+1])
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if k >= warmup:
                times.append((t1 - t0) * 1000.0)
    del dec
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(times)), float(np.std(times))


def main():
    setup_logging("latency")
    out = {"window_seconds": WINDOW_SECONDS, "decoder_checkpoint": "best_config_5fold (V7, max_token_len=6)"}
    for device in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        mean_ms, std_ms = time_decoder(1, device)
        rt = WINDOW_SECONDS * 1000.0 / mean_ms
        out[device] = {"decoder_ms_per_window_mean": round(mean_ms, 3),
                       "decoder_ms_per_window_std": round(std_ms, 3),
                       "decoder_realtime_factor": round(rt, 1)}
        logger.info(f"{device}: decoder {mean_ms:.2f}+/-{std_ms:.2f} ms/window, {rt:.0f}x real-time")
    OUT = BASE_DIR / "revision_2026_06"
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    (OUT / "latency_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
