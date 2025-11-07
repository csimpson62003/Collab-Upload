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
    CUDA SETUP AND DEVICE SELECTION
    ===============================
    Sets up CUDA and selects the best available GPU device.
    
    Args:
        preferred_gpu: Which GPU to prefer (0 for first GPU, 1 for second, etc.)
    Returns:
        torch.device: The selected device (cuda:X or cpu)
    """
    print("=" * 50)
    print("CUDA SETUP AND DETECTION")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Using CPU instead.")
        print("💡 To enable CUDA:")
        print("   1. Make sure you have an NVIDIA GPU")
        print("   2. Install CUDA drivers from NVIDIA")
        print("   3. Install PyTorch with CUDA support:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return torch.device("cpu")
    
    # Display CUDA information
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # List all available GPUs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Select the best GPU
    if preferred_gpu < torch.cuda.device_count():
        selected_gpu = preferred_gpu
    else:
        # Default to GPU 0 if preferred GPU doesn't exist
        selected_gpu = 0
        print(f"⚠️  Preferred GPU {preferred_gpu} not available, using GPU {selected_gpu}")
    
    # Set the default GPU
    torch.cuda.set_device(selected_gpu)
    device = torch.device(f"cuda:{selected_gpu}")
    
    print(f"✅ Selected GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
    print(f"   Device: {device}")
    
    # Test GPU with a simple operation
    try:
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor @ test_tensor.T
        print(f"✅ GPU test successful! Tensor computation working on {device}")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("   Falling back to CPU")
        return torch.device("cpu")
    
    print("=" * 50)
    return device


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