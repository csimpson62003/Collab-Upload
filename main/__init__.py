"""
FACE-SWAPPING DIFFUSION MODEL PACKAGE
====================================
This package contains all the components for a face-swapping diffusion model.

Core Components:
- FaceSwapDataset: Loads and processes face-swap training data
- UNET: The main neural network that learns to remove noise
- DDPM_Scheduler: Manages the noise schedule for training and generation
- SinusoidalEmbeddings: Time encoding for the diffusion process
- ResBlock: Residual blocks for the U-Net architecture
- Attention: Self-attention mechanism for long-range dependencies
- UnetLayer: Individual layers of the U-Net

Training & Inference:
- train: Main training function
- inference: Image generation function
- utils: Helper functions for setup and visualization
"""

from .face_swap_dataset import FaceSwapDataset
from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .sinusoidal_embeddings import SinusoidalEmbeddings
from .res_block import ResBlock
from .attention import Attention
from .unet_layer import UnetLayer
from .train import train
from .inference import inference
from .utils import set_seed, setup_cuda_device, display_reverse
from .face_swapper import swap_faces, batch_face_swap, example_usage

__all__ = [
    'FaceSwapDataset',
    'UNET', 
    'DDPM_Scheduler',
    'SinusoidalEmbeddings',
    'ResBlock',
    'Attention', 
    'UnetLayer',
    'train',
    'inference',
    'set_seed',
    'setup_cuda_device',
    'display_reverse',
    'swap_faces',
    'batch_face_swap',
    'example_usage'
]