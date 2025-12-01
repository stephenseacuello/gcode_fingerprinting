#!/usr/bin/env python3
"""
mcc_helper.py - MCC DAQ Interface for 4-Channel Current Monitoring
==================================================================
Run under x86_64 conda env (mcc-intel) with: libuldaq, uldaq, pyzmq.
Publishes JSON blocks over ZeroMQ to localhost.

Hardware: USB-1608G with 4x BAYITE Hall Effect Current Sensors
Channels:
  0 - Spindle motor current
  1 - X-axis stepper motor current
  2 - Y-axis stepper motor current
  3 - Z-axis stepper motor current

Usage:
  python mcc_helper.py --pub tcp://127.0.0.1:5557 --rate 1000 --block 200 --ch-low 0 --ch-high 3
"""

import argparse
import socket
import time
from typing import List, Tuple

import zmq
from uldaq import (
    get_daq_device_inventory, DaqDevice, InterfaceType,
    AiInputMode, Range, AInFlag, ULException
)

# Channel labels for identification
CHANNEL_LABELS = {
    0: "Spindle",
    1: "X_motor",
    2: "Y_motor",
    3: "Z_motor"
}

def pick_mode_and_range(ai) -> Tuple[AiInputMode, Range, int]:
    """
    Return (mode, range, n_chans_for_mode).
    ULDAQ: discover support by probing modes with get_num_chans_by_mode(mode),
    and ranges with get_ranges(input_mode). No get_input_modes() exists.
    """
    info = ai.get_info()
    # Try SE first (USB-1608G supports 16 SE, 8 DIFF), then DIFF.
    for mode in (AiInputMode.SINGLE_ENDED, AiInputMode.DIFFERENTIAL):
        try:
            n = info.get_num_chans_by_mode(mode)
            if n and n > 0:
                ranges = list(info.get_ranges(mode))  # requires input_mode arg
                # Prefer ±10V if present, else first available
                rng = Range.BIP10VOLTS if Range.BIP10VOLTS in ranges else ranges[0]
                return mode, rng, int(n)
        except ULException:
            continue
        except Exception:
            continue
    # Fallback (very defensive): DIFF with its first range
    ranges = list(info.get_ranges(AiInputMode.DIFFERENTIAL))
    rng = ranges[0]
    return AiInputMode.DIFFERENTIAL, rng, int(info.get_num_chans_by_mode(AiInputMode.DIFFERENTIAL))

def publish_blocks(pub_url: str, rate: float, block: int, low_ch: int, high_ch: int):
    """
    Main publishing loop - acquires data from MCC DAQ and publishes via ZMQ.
    
    Args:
        pub_url: ZeroMQ publish URL (e.g., tcp://127.0.0.1:5557)
        rate: Per-channel sample rate in Hz
        block: Number of samples per channel before publishing
        low_ch: Starting channel number
        high_ch: Ending channel number (inclusive)
    """
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.set_hwm(10)
    pub.bind(pub_url)
    
    print(f"[MCC] ZeroMQ publisher bound to {pub_url}")

    # Wait for an MCC USB device on macOS host
    dev_desc = None
    print("[MCC] Searching for MCC USB device...")
    while True:
        devs = get_daq_device_inventory(InterfaceType.USB)
        if devs:
            dev_desc = devs[0]
            print(f"[MCC] Found device: {dev_desc.product_name} (ID: {dev_desc.unique_id})")
            break
        print("[MCC] No device detected. Connect USB to Mac host (not VM). Retrying in 1s...")
        time.sleep(1)

    dev = DaqDevice(dev_desc)
    ai = dev.get_ai_device()
    dev.connect()
    print(f"[MCC] Connected to {dev_desc.product_name}")

    mode, rng, n_mode = pick_mode_and_range(ai)

    # Clamp channels to mode's valid range
    if high_ch >= n_mode:
        print(f"[MCC] WARNING: Requested channel {high_ch} exceeds {n_mode-1} for {mode.name}; clamping.")
        high_ch = n_mode - 1
    if low_ch < 0:
        low_ch = 0
    if low_ch > high_ch:
        low_ch = high_ch

    ch_list = list(range(low_ch, high_ch + 1))
    nch = len(ch_list)
    
    # Get channel labels
    ch_labels = [CHANNEL_LABELS.get(ch, f"Channel_{ch}") for ch in ch_list]
    
    print(f"[MCC] Configuration:")
    print(f"  Mode: {mode.name}")
    print(f"  Range: {rng.name}")
    print(f"  Channels: {ch_list}")
    print(f"  Labels: {ch_labels}")
    print(f"  Sample rate: {rate} Hz per channel")
    print(f"  Block size: {block} samples per channel")
    print(f"  Publish rate: {rate/block:.2f} Hz ({block/rate*1000:.1f}ms per block)")
    print(f"[MCC] Starting acquisition...")

    # Single-sample loop with correct signature: a_in(chan, input_mode, range, flags)
    period = 1.0 / (rate * nch) if rate > 0 else 0.0  # Period per sample across all channels
    buf: List[float] = []
    host = socket.gethostname()
    t_next = time.perf_counter()
    block_start_time = None
    sample_count = 0

    try:
        while True:
            # Mark start of block
            if block_start_time is None:
                block_start_time = time.time()
            
            # Acquire one sample from each channel
            for ch in ch_list:
                val = ai.a_in(ch, mode, rng, AInFlag.DEFAULT)
                buf.append(float(val))
                sample_count += 1

            # When we have a complete block, publish it
            if len(buf) >= block * nch:
                block_end_time = time.time()
                block_mid_time = (block_start_time + block_end_time) / 2.0
                
                msg = {
                    "ts_start": block_start_time,
                    "ts_mid": block_mid_time,
                    "ts_end": block_end_time,
                    "host": host,
                    "device": dev_desc.product_name,
                    "rate": rate,          # per-channel target rate
                    "channels": ch_list,
                    "channel_labels": ch_labels,
                    "block": block,
                    "n": len(buf),
                    "samples": buf,        # interleaved by channel
                }
                pub.send_json(msg)
                
                # Debug output (rate-limited)
                if sample_count % (rate * nch) == 0:  # Once per second
                    duration = block_end_time - block_start_time
                    actual_rate = block / duration if duration > 0 else 0
                    print(f"[MCC] Published block: {len(buf)} samples, "
                          f"duration: {duration*1000:.1f}ms, "
                          f"actual rate: {actual_rate:.1f} Hz")
                
                buf = []
                block_start_time = None

            # Timing control
            if period > 0:
                t_next += period
                sleep_s = t_next - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # We're behind - resync
                    t_next = time.perf_counter()
                    
    except KeyboardInterrupt:
        print("\n[MCC] Shutting down gracefully...")
    finally:
        try:
            dev.disconnect()
            print("[MCC] Device disconnected")
        except Exception as e:
            print(f"[MCC] Error during disconnect: {e}")
        dev.release()
        print("[MCC] Device released")

def main():
    p = argparse.ArgumentParser(
        description="MCC DAQ Helper - Acquires data and publishes via ZeroMQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 4-channel acquisition at 1000 Hz
  python mcc_helper.py --pub tcp://127.0.0.1:5557 --rate 1000 --block 200 --ch-low 0 --ch-high 3
  
  # Single channel at 2000 Hz
  python mcc_helper.py --pub tcp://127.0.0.1:5557 --rate 2000 --block 100 --ch-low 0 --ch-high 0
        """
    )
    p.add_argument("--pub", default="tcp://127.0.0.1:5557",
                   help="ZeroMQ publish URL (default: tcp://127.0.0.1:5557)")
    p.add_argument("--rate", type=float, default=1000.0,
                   help="Per-channel sample rate in Hz (default: 1000)")
    p.add_argument("--block", type=int, default=200,
                   help="Samples per channel per published block (default: 200)")
    p.add_argument("--ch-low", type=int, default=0,
                   help="Starting channel number (default: 0)")
    p.add_argument("--ch-high", type=int, default=0,
                   help="Ending channel number, inclusive (default: 0)")
    
    args = p.parse_args()
    
    print("=" * 70)
    print("MCC DAQ Helper - 4-Channel Current Monitor")
    print("=" * 70)
    
    publish_blocks(args.pub, args.rate, args.block, args.ch_low, args.ch_high)

if __name__ == "__main__":
    main()
