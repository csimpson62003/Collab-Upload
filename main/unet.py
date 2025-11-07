"""
U-NET ARCHITECTURE - THE CORE OF THE DIFFUSION MODEL
===================================================
The main neural network that learns to predict and remove noise from images.
Uses an encoder-decoder architecture with skip connections for preserving details.

ARCHITECTURE FOR FACE-SWAPPING:
- Encoder: Progressively downsamples input to capture hierarchical features
- Bottleneck: Processes the most compressed representation with attention
- Decoder: Progressively upsamples while using skip connections to preserve details
- Skip connections: Ensure fine facial details are maintained during generation

FUTURE FACE-SWAPPING ENHANCEMENTS:
- Add identity encoding branch to preserve source identity
- Include pose/expression conditioning inputs
- Add face landmark guidance for better alignment
- Implement style transfer mechanisms for lighting/texture adaptation
"""

import torch
import torch.nn as nn
from typing import List
from .sinusoidal_embeddings import SinusoidalEmbeddings
from .unet_layer import UnetLayer


class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],        # Channel progression through network
            Attentions: List = [False, True, False, False, False, True],  # Which layers use attention
            Upscales: List = [False, False, False, True, True, True],     # Which layers upsample vs downsample
            num_groups: int = 32,           # Groups for GroupNorm (affects normalization granularity)
            dropout_prob: float = 0.1,     # Dropout probability for regularization
            num_heads: int = 8,             # Number of attention heads
            input_channels: int = 3,        # Input image channels (3 for RGB faces)
            output_channels: int = 3,       # Output channels (3 for RGB face generation)
            time_steps: int = 1000):        # Number of diffusion timesteps
        super().__init__()
        
        self.num_layers = len(Channels)
        
        # Initial convolution to map input channels to first feature dimension
        # For face-swapping: This will process the initial noisy face image
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        
        # Output processing layers
        out_channels = (Channels[-1]//2) + Channels[0]  # Combine decoder output with skip connection
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Time embedding module for encoding diffusion timesteps
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        
        # Dynamically create all U-Net layers based on configuration
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],        # First half: False (downsampling), Second half: True (upsampling)
                attention=Attentions[i],    # Add attention at specified layers for long-range dependencies
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            # Register each layer as a module attribute
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Noisy input image tensor [batch, channels, height, width]
            t: Timestep tensor indicating noise level [batch]
        Returns:
            Predicted noise tensor with same shape as input
        """
        # Initial feature extraction
        x = self.shallow_conv(x)
        
        # Store skip connections for the decoder
        residuals = []
        
        # ENCODER PATH: Downsample and extract hierarchical features
        # Each layer reduces spatial resolution while increasing feature depth
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            # Generate time embeddings for this timestep
            embeddings = self.embeddings(x, t)
            # Process through layer and store residual for skip connection
            x, r = layer(x, embeddings)
            residuals.append(r)
        
        # DECODER PATH: Upsample and reconstruct image details
        # Each layer increases spatial resolution while using skip connections
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            # Concatenate decoder features with corresponding encoder features (skip connection)
            # This preserves fine details lost during downsampling
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        
        # Final output processing to generate the predicted noise
        return self.output_conv(self.relu(self.late_conv(x)))