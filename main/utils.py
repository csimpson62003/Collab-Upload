"""
UTILITY FUNCTIONS FOR THE DIFFUSION MODEL
========================================
Various helper functions for setup, training, and inference.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List

# TPU support for Google Colab
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


def set_seed(seed: int = 42):
    """
    REPRODUCIBILITY SETUP
    ====================
    Sets random seeds for all libraries to ensure reproducible results.
    Critical for scientific experiments and debugging.
    
    For face-swapping: Ensures consistent results when testing different
    face-swapping configurations and comparing model performance.
    """
    # PyTorch CPU random number generator
    torch.manual_seed(seed)
    
    # PyTorch GPU random number generators (all devices)
    torch.cuda.manual_seed_all(seed)
    
    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # NumPy random number generator
    np.random.seed(seed)
    
    # Python's built-in random module
    random.seed(seed)


def setup_cuda_device(preferred_gpu: int = 0):
    """
    DEVICE SETUP - SUPPORTS TPU, CUDA GPU, AND CPU
    ==============================================
    Automatically detects and sets up the best available device:
    1. TPU (Google Colab TPU runtime)
    2. CUDA GPU (NVIDIA GPUs)
    3. CPU (fallback)
    
    Args:
        preferred_gpu: Which GPU to prefer if multiple CUDA GPUs available
    Returns:
        torch.device: The selected device (xla for TPU, cuda:X for GPU, or cpu)
    """
    print("=" * 50)
    print("DEVICE SETUP AND DETECTION")
    print("=" * 50)
    
    # PRIORITY 1: Check for TPU (Google Colab)
    if TPU_AVAILABLE:
        try:
            device = xm.xla_device()
            print("✅ TPU DETECTED AND ACTIVATED!")
            print(f"   Device: {device}")
            print(f"   TPU cores: {xm.xrt_world_size()}")
            print("💡 For optimal TPU performance:")
            print("   - Use batch sizes that are multiples of 8")
            print("   - Avoid frequent CPU<->TPU transfers")
            print("   - Use XLA-compatible operations")
            return device
        except Exception as e:
            print(f"⚠️  TPU initialization failed: {e}")
            print("   Falling back to CUDA/CPU...")
    
    # PRIORITY 2: Check for CUDA GPU
    if torch.cuda.is_available():
        print(f"✅ CUDA GPU AVAILABLE")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List all available GPUs
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Select GPU
        if preferred_gpu < torch.cuda.device_count():
            selected_gpu = preferred_gpu
        else:
            selected_gpu = 0
            print(f"⚠️  Preferred GPU {preferred_gpu} not available, using GPU 0")
        
        torch.cuda.set_device(selected_gpu)
        device = torch.device(f"cuda:{selected_gpu}")
        print(f"✅ Using GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
        return device
    
    # PRIORITY 3: Fallback to CPU
    print("⚠️  No GPU/TPU detected! Using CPU (this will be VERY slow)")
    print("💡 To enable hardware acceleration:")
    print("   - For TPU in Colab: Runtime > Change runtime type > TPU")
    print("   - For GPU in Colab: Runtime > Change runtime type > GPU")
    print("   - For local GPU: Install CUDA drivers and PyTorch with CUDA support")
    return torch.device("cpu")




def display_reverse(images: List):
    """
    VISUALIZATION FUNCTION FOR DIFFUSION PROCESS
    ===========================================
    Shows the step-by-step reverse diffusion process from noise to image.
    Useful for understanding how the model generates images.
    
    For face-swapping: This will help visualize how faces emerge from noise,
    showing the progression from random noise -> rough face shape -> detailed features.
    """
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))  # Larger figure for face images
    for i, ax in enumerate(axes.flat):
        # Convert tensor to displayable format
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')  # Change from channels-first to channels-last
        x = x.numpy()
        
        # Denormalize from [-1, 1] to [0, 1] for RGB display
        x = (x + 1) / 2
        x = np.clip(x, 0, 1)  # Ensure values are in valid range
        
        # Display RGB face image
        ax.imshow(x)
        ax.axis('off')  # Remove axis labels for cleaner visualization
    plt.show()