#!/usr/bin/env python3
"""
combined_logger.py - Multithreaded MCC DAQ + Serial Sensor Logger

This script simultaneously records:
  1. MCC DAQ data (voltage/current from USB-1608G via ZeroMQ)
  2. Serial sensor data (IMU, magnetometer, pressure, etc. from multiple ports)

Both data streams are recorded to separate CSV files with synchronized timestamps.

Usage:
    ./combined_logger.py \
        --mcc-out logs/experiment_mcc.csv \
        --sensor-out logs/experiment_sensors.csv \
        --duration 60 \
        --sensor-ports /dev/ttyACM0,/dev/ttyACM1
"""

import argparse
import os
import signal
import sys
import time
import csv
import struct
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# MCC imports
try:
    from mcc_recorder import MCCRecorder
    MCC_AVAILABLE = True
except ImportError:
    MCC_AVAILABLE = False
    print("WARNING: mcc_recorder not available. MCC logging will be disabled.", file=sys.stderr)

# Serial imports
try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not available. Serial logging will be disabled.", file=sys.stderr)


# =============================================================================
# MCC DAQ Configuration
# =============================================================================

CURRENT_CHANNELS = {
    "spindle": {
        "column": "spindle",
        "scale_a_per_v": 1.0,
        "offset_v": 0.0,
    },
    "x_motor": {
        "column": "x_motor",
        "scale_a_per_v": 1.0,
        "offset_v": 0.0,
    },
    "y_motor": {
        "column": "y_motor",
        "scale_a_per_v": 1.0,
        "offset_v": 0.0,
    },
    "z_motor": {
        "column": "z_motor",
        "scale_a_per_v": 1.0,
        "offset_v": 0.0,
    },
}


def _convert_row_voltage_to_current(row, mapping=CURRENT_CHANNELS):
    """Convert voltage columns to current using linear mapping."""
    out = dict(row)
    for name, cfg in mapping.items():
        col = cfg.get("column")
        if not col or col not in row:
            continue
        v_str = row[col]
        try:
            v = float(v_str)
        except (TypeError, ValueError):
            continue
        scale = float(cfg.get("scale_a_per_v", 1.0))
        offset = float(cfg.get("offset_v", 0.0))
        current_a = (v - offset) * scale
        out[f"{col}_A"] = current_a
    return out


def apply_voltage_to_current_conversion(csv_path, mapping=CURRENT_CHANNELS):
    """Post-process MCC CSV to add current columns."""
    tmp_path = csv_path + ".with_current_tmp"
    with open(csv_path, "r", newline="") as fin, open(tmp_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []

        seen = set(fieldnames)
        for name, cfg in mapping.items():
            col = cfg.get("column")
            if not col:
                continue
            new_name = f"{col}_A"
            if new_name not in seen:
                fieldnames.append(new_name)
                seen.add(new_name)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_with_current = _convert_row_voltage_to_current(row, mapping)
            writer.writerow(row_with_current)

    os.replace(tmp_path, csv_path)


# =============================================================================
# Serial Sensor Configuration
# =============================================================================

HDR0, HDR1 = 0xA5, 0x5A
FRAME_SIZE = 47
CRC_POLY = 0x1021

NUMERIC_FIELDS = ["Ax","Ay","Az","Gx","Gy","Gz","Mx","My","Mz","Pressure","Temperature","Proximity","ColorR","ColorG","ColorB","ColorA","RMS"]
CATEGORICAL_FIELDS = ["Gesture"]
ALL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

GESTURE = {0:"NONE", 1:"UP", 2:"DOWN", 3:"LEFT", 4:"RIGHT"}


def crc16_ccitt(data: bytes, poly=CRC_POLY, init=0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def unpack_rgba10(b: bytes):
    v = b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24) | (b[4]<<32)
    R =  v        & 0x3FF
    G = (v >> 10) & 0x3FF
    B = (v >> 20) & 0x3FF
    A = (v >> 30) & 0x3FF
    return R, G, B, A


def pick_ports_auto(max_ports: int) -> List[str]:
    """Auto-detect serial ports, preferring USB devices."""
    ports = list(list_ports.comports())
    devs = []
    for p in ports:
        if "usbmodem" in p.device or "USB" in p.description.upper():
            devs.append(p.device)
    if not devs:
        devs = [p.device for p in ports]
    return devs[:max_ports]


# =============================================================================
# MCC DAQ Thread
# =============================================================================

class MCCThread(threading.Thread):
    """Thread for MCC DAQ recording."""
    
    def __init__(self, out_path: str, experiment_name: str, session_ts: str, 
                 stop_event: threading.Event, status_queue: queue.Queue):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.experiment_name = experiment_name
        self.session_ts = session_ts
        self.stop_event = stop_event
        self.status_queue = status_queue
        self.recorder = None
        self.temp_filename = None
        
    def run(self):
        try:
            print("[MCC] Initializing recorder...")
            self.recorder = MCCRecorder()
            
            print("[MCC] Testing connection to MCC helper...")
            if not self.recorder.test_connection(timeout_ms=2000):
                print("[MCC] ERROR: Could not connect to MCC helper.", file=sys.stderr)
                self.status_queue.put(("mcc_error", "Connection failed"))
                return
            
            print("[MCC] Connection OK. Starting recording...")
            self.temp_filename = self.recorder.start_recording(
                experiment_name=self.experiment_name,
                session_ts=self.session_ts,
            )
            
            if not self.temp_filename:
                print("[MCC] ERROR: Recorder did not return a filename.", file=sys.stderr)
                self.status_queue.put(("mcc_error", "No filename"))
                return
            
            print(f"[MCC] Recording to: {self.temp_filename}")
            self.status_queue.put(("mcc_started", self.temp_filename))
            
            # Monitor loop
            while not self.stop_event.is_set():
                try:
                    stats = self.recorder.get_current_stats()
                    self.status_queue.put(("mcc_stats", stats))
                except Exception as e:
                    print(f"[MCC] Error getting stats: {e}", file=sys.stderr)
                time.sleep(1.0)
                
        except Exception as e:
            print(f"[MCC] Thread error: {e}", file=sys.stderr)
            self.status_queue.put(("mcc_error", str(e)))
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Stop recording and finalize file."""
        if not self.recorder:
            return
            
        print("\n[MCC] Stopping recorder...")
        recorded_file = self.recorder.stop_recording()
        
        if recorded_file:
            try:
                # Rename to final path
                if os.path.abspath(recorded_file) != self.out_path:
                    os.replace(recorded_file, self.out_path)
                    final_file = self.out_path
                else:
                    final_file = recorded_file
                
                # Apply voltage->current conversion
                apply_voltage_to_current_conversion(final_file, CURRENT_CHANNELS)
                print(f"[MCC] Log saved to: {final_file}")
                self.status_queue.put(("mcc_complete", final_file))
                
            except Exception as e:
                print(f"[MCC] Error finalizing: {e}", file=sys.stderr)
                self.status_queue.put(("mcc_error", str(e)))
        else:
            print("[MCC] No file recorded.", file=sys.stderr)


# =============================================================================
# Serial Sensor Threads
# =============================================================================

class SerialReader(threading.Thread):
    """Thread for reading from a single serial port."""
    
    def __init__(self, port: str, baud: int, out_queue: queue.Queue, 
                 stop_event: threading.Event, alpha: float = 0.9):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_q = out_queue
        self.stop_event = stop_event
        self.alpha = alpha
        self.dev_accum_ms = 0
        self.host_origin_ms: Optional[float] = None

    def run(self):
        try:
            ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.2)
            ser.reset_input_buffer()
            print(f"[SERIAL] Opened {self.port}")
        except Exception as e:
            print(f"[SERIAL] Failed to open {self.port}: {e}", file=sys.stderr)
            return

        while not self.stop_event.is_set():
            try:
                # Sync to header
                b = ser.read(1)
                if not b or b[0] != HDR0:
                    continue
                b2 = ser.read(1)
                if not b2 or b2[0] != HDR1:
                    continue

                rest = ser.read(FRAME_SIZE - 2)
                if len(rest) != (FRAME_SIZE - 2):
                    continue

                # Parse frame
                gest_byte = rest[0]
                id_bytes = rest[1:9]
                dev_id = id_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                seq = int.from_bytes(rest[9:11], "little", signed=False)
                dt_ms = int.from_bytes(rest[11:13], "little", signed=False)

                off = 13
                Ax, Ay, Az = struct.unpack_from("<hhh", rest, off); off += 6
                Gx, Gy, Gz = struct.unpack_from("<hhh", rest, off); off += 6
                Mx, My, Mz = struct.unpack_from("<hhh", rest, off); off += 6
                pressure_dPa = struct.unpack_from("<H", rest, off)[0]; off += 2
                temp_cC = struct.unpack_from("<h", rest, off)[0]; off += 2
                proximity = rest[off]; off += 1
                rgba10 = rest[off:off+5]; off += 5
                rms = struct.unpack_from("<H", rest, off)[0]; off += 2
                crc_recv = struct.unpack_from("<H", rest, off)[0]; off += 2

                # Verify CRC
                crc_data = rest[0:43]
                crc_calc = crc16_ccitt(crc_data)
                if crc_calc != crc_recv:
                    continue

                # Timing synchronization
                host_ms = time.monotonic() * 1000.0
                if self.host_origin_ms is None:
                    self.host_origin_ms = host_ms
                    self.dev_accum_ms = 0
                else:
                    self.dev_accum_ms = (self.dev_accum_ms + dt_ms) & 0xFFFFFFFF
                abs_ms = self.alpha * (self.host_origin_ms + self.dev_accum_ms) + (1.0 - self.alpha) * host_ms

                R, G, B, A = unpack_rgba10(rgba10)
                gest = GESTURE.get(gest_byte & 0x07, "NONE")

                rec = {
                    "id": dev_id,
                    "abs_ms": abs_ms,
                    "seq": seq,
                    "Ax": Ax/1000.0, "Ay": Ay/1000.0, "Az": Az/1000.0,
                    "Gx": Gx/1000.0, "Gy": Gy/1000.0, "Gz": Gz/1000.0,
                    "Mx": float(Mx), "My": float(My), "Mz": float(Mz),
                    "Pressure": pressure_dPa/10.0,
                    "Temperature": temp_cC/100.0,
                    "Proximity": float(proximity),
                    "Gesture": gest,
                    "ColorR": float(R), "ColorG": float(G), "ColorB": float(B), "ColorA": float(A),
                    "RMS": float(rms),
                }
                
                try:
                    self.out_q.put_nowait(rec)
                except queue.Full:
                    pass

            except Exception:
                continue

        try:
            ser.close()
            print(f"[SERIAL] Closed {self.port}")
        except Exception:
            pass


class InterpolatorWriter(threading.Thread):
    """Thread for interpolating and writing sensor data to CSV."""
    
    def __init__(self, out_path: str, out_queue: queue.Queue, stop_event: threading.Event, 
                 period_ms: int, discover_seconds: float):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.q = out_queue
        self.stop_event = stop_event
        self.period_ms = period_ms
        self.discover_until = time.monotonic() + discover_seconds
        self.buffers: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        self.header_written = False
        self.device_ids: List[str] = []
        self.csv_file = open(self.out_path, "w", newline="")
        self.writer = None
        self.next_tick_ms: Optional[float] = None

    def _write_header(self):
        cols = ["ts_ms"]
        for dev in self.device_ids:
            for f in ALL_FIELDS:
                cols.append(f"{dev}.{f}")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=cols)
        self.writer.writeheader()
        self.csv_file.flush()
        self.header_written = True

    def _append_sample(self, rec: dict):
        dev = rec["id"]
        t = rec["abs_ms"]
        buf = self.buffers.setdefault(dev, [])
        buf.append((t, rec))
        cutoff = t - 10000.0
        while len(buf) >= 2 and buf[0][0] < cutoff:
            buf.pop(0)
        if dev not in self.device_ids:
            self.device_ids.append(dev)

    def _interp_value(self, dev: str, t_ms: float, key: str) -> Optional[float]:
        buf = self.buffers.get(dev)
        if not buf or len(buf) == 0:
            return None
        prev_t, prev_v = None, None
        for (ti, rec) in buf:
            if ti <= t_ms:
                prev_t, prev_v = ti, rec
            if ti >= t_ms:
                if prev_t is None:
                    return rec[key] if key in rec and isinstance(rec[key], (int,float)) else None
                if ti == prev_t:
                    return rec[key] if key in rec and isinstance(rec[key], (int,float)) else None
                if key in rec and key in prev_v and isinstance(rec[key], (int,float)) and isinstance(prev_v[key], (int,float)):
                    frac = (t_ms - prev_t) / (ti - prev_t)
                    return (1.0-frac) * prev_v[key] + frac * rec[key]
                else:
                    return rec.get(key, None)
        return prev_v.get(key, None) if prev_v else None

    def _ffill_value(self, dev: str, t_ms: float, key: str) -> Optional[Any]:
        buf = self.buffers.get(dev)
        if not buf:
            return None
        last = None
        for (ti, rec) in buf:
            if ti <= t_ms:
                last = rec.get(key, last)
            else:
                break
        return last

    def run(self):
        print(f"[SENSOR] Discovering devices for {self.discover_until - time.monotonic():.1f}s...")
        
        # Discovery phase
        while time.monotonic() < self.discover_until and not self.stop_event.is_set():
            try:
                rec = self.q.get(timeout=0.1)
                self._append_sample(rec)
            except queue.Empty:
                continue

        if self.device_ids:
            print(f"[SENSOR] Discovered devices: {self.device_ids}")
        else:
            print("[SENSOR] WARNING: No devices discovered during discovery window", file=sys.stderr)
            
        if not self.header_written:
            self._write_header()

        all_first_times = [buf[0][0] for buf in self.buffers.values() if buf]
        self.next_tick_ms = min(all_first_times) if all_first_times else (time.monotonic()*1000.0)

        print(f"[SENSOR] Writing to: {self.out_path}")

        # Main recording loop
        while not self.stop_event.is_set():
            # Drain queue
            while True:
                try:
                    rec = self.q.get_nowait()
                    self._append_sample(rec)
                except queue.Empty:
                    break

            # Write interpolated rows
            now_ms = time.monotonic() * 1000.0
            while self.next_tick_ms is not None and self.next_tick_ms + self.period_ms <= now_ms:
                row = {"ts_ms": f"{self.next_tick_ms:.3f}"}
                for dev in self.device_ids:
                    for f in NUMERIC_FIELDS:
                        val = self._interp_value(dev, self.next_tick_ms, f)
                        row[f"{dev}.{f}"] = (f"{val:.6f}" if isinstance(val, float) else (str(val) if val is not None else ""))
                    for f in CATEGORICAL_FIELDS:
                        val = self._ffill_value(dev, self.next_tick_ms, f)
                        row[f"{dev}.{f}"] = (str(val) if val is not None else "")
                self.writer.writerow(row)
                self.csv_file.flush()
                self.next_tick_ms += self.period_ms

            time.sleep(0.005)

        print("[SENSOR] Closing file...")
        self.csv_file.close()


# =============================================================================
# Main Application
# =============================================================================

_stop_requested = False


def _handle_sigint(signum, frame):
    """Signal handler for Ctrl+C."""
    global _stop_requested
    print("\n[MAIN] Ctrl+C received, stopping all threads...")
    _stop_requested = True


def main():
    parser = argparse.ArgumentParser(
        description="Combined MCC DAQ + Serial Sensor Logger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # MCC options
    mcc_group = parser.add_argument_group("MCC DAQ options")
    mcc_group.add_argument(
        "--mcc-out",
        help="MCC output CSV filename (if omitted, MCC logging is disabled)",
    )
    mcc_group.add_argument(
        "--session-ts",
        dest="session_ts",
        default=None,
        help="Session timestamp for MCC recording (auto-generated if omitted)",
    )
    
    # Serial sensor options
    sensor_group = parser.add_argument_group("Serial sensor options")
    sensor_group.add_argument(
        "--sensor-out",
        help="Sensor output CSV filename (if omitted, sensor logging is disabled)",
    )
    sensor_group_ex = sensor_group.add_mutually_exclusive_group()
    sensor_group_ex.add_argument(
        "--sensor-ports",
        help="Comma-separated list of serial ports",
    )
    sensor_group_ex.add_argument(
        "--sensor-auto",
        type=int,
        metavar="N",
        help="Auto-detect up to N serial ports",
    )
    sensor_group.add_argument(
        "--sensor-baud",
        type=int,
        default=115200,
        help="Serial baud rate",
    )
    sensor_group.add_argument(
        "--sensor-period-ms",
        type=int,
        default=50,
        help="Sensor resample period in ms (e.g., 50=20Hz)",
    )
    sensor_group.add_argument(
        "--sensor-discover-seconds",
        type=float,
        default=5.0,
        help="Discovery window to learn device IDs",
    )
    sensor_group.add_argument(
        "--sensor-queue-size",
        type=int,
        default=20000,
        help="Queue capacity for sensor records",
    )
    
    # Common options
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--logfile",
        help="Optional logfile path for stdout/stderr",
    )

    args = parser.parse_args()

    # Validate at least one output is specified
    if not args.mcc_out and not args.sensor_out:
        parser.error("At least one of --mcc-out or --sensor-out must be specified")

    # Check availability
    if args.mcc_out and not MCC_AVAILABLE:
        parser.error("MCC logging requested but mcc_recorder module not available")
    
    if args.sensor_out and not SERIAL_AVAILABLE:
        parser.error("Sensor logging requested but pyserial not available")

    # Setup logging to file if requested
    if args.logfile:
        logfile_path = args.logfile if os.path.isabs(args.logfile) else os.path.join(os.getcwd(), args.logfile)
        logfile_dir = os.path.dirname(logfile_path)
        if logfile_dir and not os.path.exists(logfile_dir):
            os.makedirs(logfile_dir, exist_ok=True)
        logfile_fh = open(logfile_path, "a", buffering=1)
        print(f"[MAIN] Logging to {logfile_path}")

        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                return len(data)
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(sys.stdout, logfile_fh)
        sys.stderr = Tee(sys.stderr, logfile_fh)

    # Setup signal handler
    signal.signal(signal.SIGINT, _handle_sigint)

    # Initialize shared objects
    stop_event = threading.Event()
    status_queue = queue.Queue()
    threads = []

    # Setup MCC thread
    mcc_thread = None
    if args.mcc_out:
        out_dir, out_base = os.path.split(args.mcc_out)
        if not out_base:
            parser.error("--mcc-out must be a filename")
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "logs")
        mcc_out_path = os.path.join(out_dir, out_base)
        
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        experiment_name = os.path.splitext(out_base)[0]
        session_ts = args.session_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        mcc_thread = MCCThread(
            out_path=mcc_out_path,
            experiment_name=experiment_name,
            session_ts=session_ts,
            stop_event=stop_event,
            status_queue=status_queue,
        )
        threads.append(mcc_thread)

    # Setup sensor threads
    sensor_readers = []
    sensor_writer = None
    if args.sensor_out:
        # Determine ports
        if args.sensor_ports:
            ports = [p.strip() for p in args.sensor_ports.split(",") if p.strip()]
        elif args.sensor_auto:
            ports = pick_ports_auto(max_ports=args.sensor_auto)
        else:
            parser.error("Sensor logging enabled but no ports specified (use --sensor-ports or --sensor-auto)")
        
        if not ports:
            parser.error("No serial ports available")
        
        # Setup output path
        out_dir, out_base = os.path.split(args.sensor_out)
        if not out_base:
            parser.error("--sensor-out must be a filename")
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "logs")
        sensor_out_path = os.path.join(out_dir, out_base)
        
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # Create threads
        sensor_queue = queue.Queue(maxsize=args.sensor_queue_size)
        
        for port in ports:
            reader = SerialReader(
                port=port,
                baud=args.sensor_baud,
                out_queue=sensor_queue,
                stop_event=stop_event,
            )
            sensor_readers.append(reader)
            threads.append(reader)
        
        sensor_writer = InterpolatorWriter(
            out_path=sensor_out_path,
            out_queue=sensor_queue,
            stop_event=stop_event,
            period_ms=args.sensor_period_ms,
            discover_seconds=args.sensor_discover_seconds,
        )
        threads.append(sensor_writer)

    # Start all threads
    print(f"[MAIN] Starting {len(threads)} thread(s)...")
    for t in threads:
        t.start()

    # Main monitoring loop
    start_time = time.time()
    duration = float(args.duration) if args.duration else 0.0
    
    if duration > 0:
        print(f"[MAIN] Recording for {duration:.1f} seconds. Press Ctrl+C to stop early.")
    else:
        print("[MAIN] Recording until Ctrl+C...")

    try:
        last_mcc_stats = None
        while True:
            if _stop_requested:
                break

            if duration > 0 and (time.time() - start_time) >= duration:
                print("\n[MAIN] Reached requested duration.")
                break

            # Check status queue for updates
            try:
                while True:
                    msg_type, data = status_queue.get_nowait()
                    if msg_type == "mcc_stats":
                        last_mcc_stats = data
                    elif msg_type == "mcc_error":
                        print(f"[MAIN] MCC error: {data}", file=sys.stderr)
                    elif msg_type == "mcc_started":
                        print(f"[MAIN] MCC recording started")
                    elif msg_type == "mcc_complete":
                        print(f"[MAIN] MCC recording complete: {data}")
            except queue.Empty:
                pass

            # Display status
            status_parts = []
            if last_mcc_stats:
                status_parts.append(
                    f"MCC: X={last_mcc_stats['x_motor']['voltage']:.2f}V "
                    f"Y={last_mcc_stats['y_motor']['voltage']:.2f}V "
                    f"Z={last_mcc_stats['z_motor']['voltage']:.2f}V "
                    f"S={last_mcc_stats['spindle']['voltage']:.2f}V"
                )
            
            if sensor_readers:
                status_parts.append(f"Sensors: {len(sensor_readers)} ports")
            
            if status_parts:
                elapsed = time.time() - start_time
                print(f"[{elapsed:>6.1f}s] {' | '.join(status_parts)}", end="\r", flush=True)

            time.sleep(1.0)

    finally:
        print("\n[MAIN] Stopping all threads...")
        stop_event.set()

        # Wait for threads to complete
        for t in threads:
            t.join(timeout=2.0)
            if t.is_alive():
                print(f"[MAIN] Warning: Thread {t.name} did not exit cleanly", file=sys.stderr)

        print("[MAIN] All threads stopped.")
        print("[MAIN] Done.")


if __name__ == "__main__":
    main()
