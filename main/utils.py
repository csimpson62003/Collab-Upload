"""
UTILITY FUNCTIONS FOR THE DIFFUSION MODEL
========================================
Various helper functions for setup, training, and inference.
"""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List, Optional

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "1"


# TPU support (PJRT path). Import guarded so this file works on any runtime.
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except Exception:
    TPU_AVAILABLE = False


def _is_tpu_ready() -> bool:
    """True if torch_xla is present AND PJRT reports TPU device."""
    if not TPU_AVAILABLE:
        return False
    try:
        return xr.device_type() == "TPU"
    except Exception:
        return False


def set_seed(seed: int = 42):
    """
    REPRODUCIBILITY SETUP
    ====================
    Sets random seeds for all libraries to ensure reproducible results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Only touch CUDA knobs if CUDA exists (keeps TPU happy)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_cuda_device(preferred_gpu: int = 0):
    """
    DEVICE SETUP - SUPPORTS TPU (PJRT), CUDA GPU, AND CPU
    =====================================================
    Automatically selects the best available device:
      1) TPU via torch_xla (PJRT)
      2) CUDA GPU
      3) CPU

    Returns:
        A device handle suitable for `.to(device)`:
          - XLA device (TPU) when on TPU runtime
          - torch.device("cuda:X") when on GPU
          - torch.device("cpu") otherwise
    """
    print("=" * 50)
    print("DEVICE SETUP AND DETECTION (TPU -> CUDA -> CPU)")
    print("=" * 50)

    # PRIORITY 1: TPU (PJRT)
    if _is_tpu_ready():
        try:
            dev = xm.xla_device()
            # world size via PJRT
            try:
                world = xr.world_size()
            except Exception:
                world = 1
            print("✅ TPU detected via PJRT")
            print(f"   XLA device: {dev}")
            print(f"   TPU world size: {world}")
            print("💡 Tips: use batch sizes multiple of 8; avoid frequent CPU<->TPU transfers.")
            return dev
        except Exception as e:
            print(f"⚠️  TPU initialization failed: {e}")
            print("   Falling back to CUDA/CPU...")

    # PRIORITY 2: CUDA GPU
    if torch.cuda.is_available():
        print("✅ CUDA GPU available")
        try:
            print(f"   CUDA version: {torch.version.cuda}")
        except Exception:
            pass
        n_gpu = torch.cuda.device_count()
        print(f"   Number of GPUs: {n_gpu}")

        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")

        if preferred_gpu < n_gpu:
            selected = preferred_gpu
        else:
            selected = 0
            print(f"⚠️  Preferred GPU {preferred_gpu} not available; using GPU 0")

        torch.cuda.set_device(selected)
        dev = torch.device(f"cuda:{selected}")
        print(f"✅ Using GPU {selected}: {torch.cuda.get_device_name(selected)}")
        return dev

    # PRIORITY 3: CPU
    print("⚠️  No TPU/GPU detected — using CPU (slow).")
    print("💡 To accelerate:")
    print("   - Colab TPU: Runtime > Change runtime type > TPU")
    print("   - Colab GPU: Runtime > Change runtime type > GPU")
    print("   - Local GPU: Install NVIDIA drivers + PyTorch w/ CUDA")
    return torch.device("cpu")


def display_reverse(images: List[torch.Tensor], max_steps: Optional[int] = 10):
    """
    VISUALIZATION FUNCTION FOR DIFFUSION PROCESS
    ===========================================
    Shows the step-by-step reverse diffusion process from noise to image.

    Args:
        images: list of CHW tensors (range roughly [-1, 1])
        max_steps: cap the number of frames plotted (default 10)
    """
    if not images:
        print("No images to display.")
        return

    steps = min(len(images), max_steps if max_steps is not None else len(images))
    fig, axes = plt.subplots(1, steps, figsize=(1.5 * steps, 3))
    if steps == 1:
        axes = [axes]

    for i in range(steps):
        ax = axes[i]
        x = images[i].squeeze(0)  # (C,H,W)
        x = rearrange(x, 'c h w -> h w c').detach().cpu().numpy()
        x = (x + 1) / 2.0
        x = np.clip(x, 0, 1)
        ax.imshow(x)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
