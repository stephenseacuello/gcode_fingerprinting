#!/usr/bin/env python3
"""
mcc_recorder.py - MCC DAQ Data Recorder for Flask Integration
==============================================================

This module provides the MCCRecorder class for integrating MCC DAQ data
acquisition into the Flask CNC Controller application.

Features:
- ZeroMQ subscriber for receiving MCC helper data
- Real-time statistics tracking
- CSV logging with raw sample data
- Connection status monitoring
- Thread-safe operation

Usage:
    from mcc_recorder import MCCRecorder
    
    recorder = MCCRecorder(zmq_url="tcp://127.0.0.1:5557")
    recorder.test_connection()  # Check if helper is running
    recorder.start_recording()   # Begin logging
    stats = recorder.get_current_stats()  # Get real-time values
    recorder.stop_recording()    # Stop and close file
"""

import os
import csv
import time
import threading
from typing import Dict, List, Optional
from datetime import datetime

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("WARNING: pyzmq not installed. MCC recording will not be available.")
    print("Install with: pip install pyzmq --break-system-packages")


class MCCRecorder:
    """
    Records data from MCC DAQ helper via ZeroMQ and logs to CSV.
    
    Attributes:
        zmq_url: ZeroMQ subscription URL
        log_dir: Directory for CSV log files
        is_recording: Whether currently recording
        connected: Whether connected to MCC helper
        current_stats: Real-time statistics for UI display
    """
    
    def __init__(self, zmq_url: str = "tcp://127.0.0.1:5557", 
                 log_dir: str = "~/gcode_align/sensor_logging/logs"):
        """
        Initialize MCC recorder.
        
        Args:
            zmq_url: ZeroMQ URL to subscribe to (default: tcp://127.0.0.1:5557)
            log_dir: Directory for log files (default: ~/gcode_align/sensor_logging/logs)
        """
        self.zmq_url = zmq_url
        self.log_dir = os.path.expanduser(log_dir)
        
        # State
        self.is_recording = False
        self.connected = False
        self.recording_thread: Optional[threading.Thread] = None
        
        # File handles
        self.csv_file = None
        self.csv_writer = None
        self.csv_filename = None
        
        # Real-time statistics (for UI display)
        self.current_stats = {
            "x_motor": {"voltage": 0.0, "max": 0.0, "avg": 0.0},
            "y_motor": {"voltage": 0.0, "max": 0.0, "avg": 0.0},
            "z_motor": {"voltage": 0.0, "max": 0.0, "avg": 0.0},
            "spindle": {"voltage": 0.0, "max": 0.0, "avg": 0.0}
        }
        
        # Connection monitoring
        self.last_message_time: Optional[float] = None
        self.device_name: Optional[str] = None
        
        # Statistics
        self.total_samples = 0
        self.total_blocks = 0
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def test_connection(self, timeout_ms: int = 1000) -> bool:
        """
        Test if MCC helper is publishing data.
        
        Args:
            timeout_ms: Timeout in milliseconds (default: 1000)
            
        Returns:
            True if connection successful, False otherwise
        """
        if not ZMQ_AVAILABLE:
            print("ERROR: pyzmq not available")
            return False
        
        try:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(self.zmq_url)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            
            # Try to receive one message
            msg = socket.recv_json()
            
            # Extract device info
            self.device_name = msg.get("device", "Unknown")
            self.connected = True
            
            socket.close()
            context.term()
            
            print(f"[MCC Recorder] Connection successful: {self.device_name}")
            return True
            
        except zmq.error.Again:
            print(f"[MCC Recorder] Timeout: No data received on {self.zmq_url}")
            self.connected = False
            return False
        except Exception as e:
            print(f"[MCC Recorder] Connection error: {e}")
            self.connected = False
            return False
    
    def start_recording(self, experiment_name: str = "", session_ts: str = None) -> str:
        """
        Start recording MCC data to CSV file.
        
        Args:
            experiment_name: Optional experiment name for filename
            session_ts: Optional session timestamp (YYYYMMDD_HHMMSS format)
            
        Returns:
            Filename of created CSV file
        """
        if self.is_recording:
            print("[MCC Recorder] Already recording")
            return self.csv_filename
        
        if not ZMQ_AVAILABLE:
            print("[MCC Recorder] ERROR: Cannot record, pyzmq not available")
            return None
        
        # Use provided session timestamp or generate new one
        if session_ts is None:
            session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename.  If an experiment name is provided, place it at
        # the very beginning of the filename so that all logs share a common prefix.
        if experiment_name:
            self.csv_filename = os.path.join(self.log_dir, f"{experiment_name}_mcc_data_{session_ts}.csv")
        else:
            self.csv_filename = os.path.join(self.log_dir, f"mcc_data_{session_ts}.csv")
        
        # Reset statistics
        self.total_samples = 0
        self.total_blocks = 0
        
        # Start recording thread
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        print(f"[MCC Recorder] Started recording to: {self.csv_filename}")
        return self.csv_filename
    
    def stop_recording(self) -> str:
        """
        Stop recording and close CSV file.
        
        Returns:
            Filename of completed CSV file
        """
        if not self.is_recording:
            print("[MCC Recorder] Not currently recording")
            return None
        
        print("[MCC Recorder] Stopping recording...")
        self.is_recording = False
        
        # Wait for thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        filename = self.csv_filename
        print(f"[MCC Recorder] Recording stopped. Total blocks: {self.total_blocks}, "
              f"Total samples: {self.total_samples}")
        print(f"[MCC Recorder] File saved: {filename}")
        
        return filename
    
    def _recording_loop(self):
        """
        Main recording loop - runs in separate thread.
        Subscribes to ZMQ, receives blocks, deinterleaves, and writes CSV.
        """
        context = None
        socket = None
        
        try:
            # Setup ZMQ subscriber
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(self.zmq_url)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            
            print(f"[MCC Recorder] Subscribed to {self.zmq_url}")
            
            # Open CSV file
            self.csv_file = open(self.csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow([
                'timestamp',
                'spindle',
                'x_motor',
                'y_motor',
                'z_motor'
            ])
            self.csv_file.flush()
            
            # Channel name mapping
            channel_map = {
                "X_motor": "x_motor",
                "Y_motor": "y_motor",
                "Z_motor": "z_motor",
                "Spindle": "spindle"
            }
            
            print("[MCC Recorder] Waiting for data...")
            
            while self.is_recording:
                try:
                    # Receive message with timeout
                    msg = socket.recv_json()
                    
                    # Update connection status
                    self.last_message_time = time.time()
                    self.connected = True
                    
                    # Extract message data
                    ts_start = msg.get("ts_start", time.time())
                    ts_end = msg.get("ts_end", time.time())
                    rate = msg.get("rate", 1000)
                    channels = msg.get("channels", [])
                    channel_labels = msg.get("channel_labels", [])
                    block_size = msg.get("block", 200)
                    samples = msg.get("samples", [])
                    
                    n_channels = len(channels)
                    n_samples = len(samples)
                    samples_per_channel = n_samples // n_channels
                    
                    # Deinterleave samples
                    # samples = [ch0_s0, ch1_s0, ch2_s0, ch3_s0, ch0_s1, ch1_s1, ...]
                    deinterleaved = {ch: [] for ch in channels}
                    for i in range(samples_per_channel):
                        for j, ch in enumerate(channels):
                            idx = i * n_channels + j
                            if idx < n_samples:
                                deinterleaved[ch].append(samples[idx])
                    
                    # Calculate individual sample timestamps
                    # Distribute evenly across the block time span
                    time_span = ts_end - ts_start
                    time_per_sample = time_span / samples_per_channel if samples_per_channel > 0 else 0
                    
                    # Write samples to CSV
                    for i in range(samples_per_channel):
                        sample_time = ts_start + (i * time_per_sample)
                        
                        # Get voltages for each channel
                        # Channels [0, 1, 2, 3] = [Spindle, X_motor, Y_motor, Z_motor]
                        voltages = [
                            deinterleaved[0][i] if 0 in deinterleaved and i < len(deinterleaved[0]) else 0.0,
                            deinterleaved[1][i] if 1 in deinterleaved and i < len(deinterleaved[1]) else 0.0,
                            deinterleaved[2][i] if 2 in deinterleaved and i < len(deinterleaved[2]) else 0.0,
                            deinterleaved[3][i] if 3 in deinterleaved and i < len(deinterleaved[3]) else 0.0
                        ]
                        
                        self.csv_writer.writerow([sample_time] + voltages)
                    
                    # Flush periodically
                    self.csv_file.flush()
                    
                    # Update statistics for UI
                    with self.stats_lock:
                        # Calculate stats from last block
                        for i, ch in enumerate(channels):
                            label = channel_labels[i] if i < len(channel_labels) else f"ch{ch}"
                            mapped_name = channel_map.get(label, f"ch{ch}")
                            
                            if mapped_name in self.current_stats:
                                values = deinterleaved[ch]
                                if values:
                                    self.current_stats[mapped_name]["voltage"] = values[-1]  # Latest
                                    self.current_stats[mapped_name]["max"] = max(values)
                                    self.current_stats[mapped_name]["avg"] = sum(values) / len(values)
                    
                    # Update counters
                    self.total_samples += n_samples
                    self.total_blocks += 1
                    
                    # Periodic status
                    if self.total_blocks % 10 == 0:
                        print(f"[MCC Recorder] Blocks: {self.total_blocks}, "
                              f"Samples: {self.total_samples}, "
                              f"File size: {os.path.getsize(self.csv_filename) / 1024:.1f} KB")
                
                except zmq.error.Again:
                    # Timeout - check if we should continue
                    if time.time() - (self.last_message_time or 0) > 5.0:
                        print("[MCC Recorder] WARNING: No data received for 5 seconds")
                        self.connected = False
                    continue
                
                except Exception as e:
                    print(f"[MCC Recorder] Error processing message: {e}")
                    continue
        
        except Exception as e:
            print(f"[MCC Recorder] Fatal error in recording loop: {e}")
        
        finally:
            # Cleanup
            if self.csv_file:
                self.csv_file.close()
                print(f"[MCC Recorder] CSV file closed")
            
            if socket:
                socket.close()
            
            if context:
                context.term()
            
            self.is_recording = False
            print("[MCC Recorder] Recording loop ended")
    
    def get_current_stats(self) -> Dict:
        """
        Get current real-time statistics for UI display.
        
        Returns:
            Dictionary with current voltage readings and statistics
        """
        with self.stats_lock:
            return self.current_stats.copy()
    
    def get_status(self) -> Dict:
        """
        Get recorder status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "connected": self.connected,
            "recording": self.is_recording,
            "zmq_url": self.zmq_url,
            "device": self.device_name,
            "last_message": self.last_message_time,
            "total_blocks": self.total_blocks,
            "total_samples": self.total_samples,
            "csv_filename": self.csv_filename if self.is_recording else None
        }


# Test/demo code
if __name__ == "__main__":
    print("MCC Recorder Test")
    print("=" * 60)
    
    recorder = MCCRecorder()
    
    print("\n1. Testing connection...")
    if recorder.test_connection(timeout_ms=2000):
        print("   ✓ Connection successful")
        
        print("\n2. Starting recording...")
        filename = recorder.start_recording(experiment_name="test")
        print(f"   Recording to: {filename}")
        
        print("\n3. Recording for 5 seconds...")
        for i in range(5):
            time.sleep(1)
            stats = recorder.get_current_stats()
            print(f"   [{i+1}s] X: {stats['x_motor']['voltage']:.3f}V, "
                  f"Y: {stats['y_motor']['voltage']:.3f}V, "
                  f"Z: {stats['z_motor']['voltage']:.3f}V, "
                  f"Spindle: {stats['spindle']['voltage']:.3f}V")
        
        print("\n4. Stopping recording...")
        recorder.stop_recording()
        print("   ✓ Recording stopped")
        
        print(f"\n✓ Test complete. Check file: {filename}")
    else:
        print("   ✗ Connection failed")
        print("   Make sure mcc_helper.py is running!")
