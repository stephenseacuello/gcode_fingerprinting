"""
Device selection utilities for PyTorch.

Automatically detects and selects the best available device (GPU/CPU).
"""
import torch
from typing import Union

__all__ = ["get_device", "get_device_name", "print_device_info"]


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Get PyTorch device with automatic GPU detection and CPU fallback.

    Priority order:
    1. If device is explicitly specified, use it
    2. If CUDA GPU available, use 'cuda'
    3. If Apple MPS available, use 'mps'
    4. Otherwise, use 'cpu'

    Args:
        device: Optional device specification ('cuda', 'mps', 'cpu', or torch.device)
                If None, auto-detects best available device.

    Returns:
        torch.device object

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('cpu')  # Force CPU
    """
    if device is not None:
        # User explicitly specified device
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str):
            # Validate device string
            if device in ['cuda', 'mps', 'cpu']:
                # Check availability
                if device == 'cuda' and not torch.cuda.is_available():
                    print(f"⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
                    return torch.device('cpu')
                elif device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    print(f"⚠️  Warning: MPS requested but not available. Falling back to CPU.")
                    return torch.device('cpu')
                return torch.device(device)
            else:
                # Assume it's a cuda device specification like 'cuda:0'
                return torch.device(device)
        else:
            raise TypeError(f"device must be str, torch.device, or None, got {type(device)}")

    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_name(device: Union[str, torch.device, None] = None) -> str:
    """
    Get human-readable device name.

    Args:
        device: Device to get name for (auto-detects if None)

    Returns:
        Human-readable device name

    Examples:
        >>> get_device_name()
        'NVIDIA GeForce RTX 3090'
        >>> get_device_name('cpu')
        'CPU'
    """
    device_obj = get_device(device)

    if device_obj.type == 'cuda':
        if torch.cuda.is_available():
            device_idx = device_obj.index if device_obj.index is not None else 0
            gpu_name = torch.cuda.get_device_name(device_idx)
            return f"{gpu_name} (CUDA:{device_idx})"
        else:
            return "CUDA (Not Available)"
    elif device_obj.type == 'mps':
        return "Apple Metal Performance Shaders (MPS)"
    else:
        return "CPU"


def print_device_info(device: Union[str, torch.device, None] = None) -> None:
    """
    Print detailed device information.

    Args:
        device: Device to print info for (auto-detects if None)

    Examples:
        >>> print_device_info()
        🖥️  Device: NVIDIA GeForce RTX 3090 (CUDA:0)
        📊 CUDA Version: 11.8
        💾 GPU Memory: 24.0 GB
    """
    device_obj = get_device(device)

    print(f"\n{'='*60}")
    print(f"🖥️  Device: {get_device_name(device_obj)}")
    print(f"📍 Device Type: {device_obj.type}")

    if device_obj.type == 'cuda':
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"🔢 Number of GPUs: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            device_idx = device_obj.index if device_obj.index is not None else 0
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            print(f"💾 GPU Memory: {total_memory:.1f} GB")
            print(f"🏗️  Compute Capability: {torch.cuda.get_device_capability(device_idx)}")

    elif device_obj.type == 'mps':
        print(f"🍎 Apple Silicon GPU acceleration enabled")

    else:  # CPU
        print(f"⚡ PyTorch CPU threads: {torch.get_num_threads()}")

    print(f"{'='*60}\n")


def set_device_optimizations(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Set device and apply performance optimizations.

    Args:
        device: Device to use (auto-detects if None)

    Returns:
        torch.device object with optimizations applied
    """
    device_obj = get_device(device)

    if device_obj.type == 'cuda':
        # Enable TF32 for Ampere GPUs (faster mixed precision)
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Enabled TF32 for faster training (Ampere+ GPU)")

        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled cuDNN benchmarking")

    return device_obj
