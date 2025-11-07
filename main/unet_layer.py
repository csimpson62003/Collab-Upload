"""
INDIVIDUAL U-NET LAYER (ENCODER OR DECODER BLOCK)
================================================
A single layer in the U-Net architecture that can either downsample (encode) or
upsample (decode) the feature maps while processing them through residual blocks.

For face-swapping: Each layer will learn different levels of facial features:
- Early layers: Low-level features (edges, textures, skin details)
- Middle layers: Mid-level features (facial parts, local structures)  
- Later layers: High-level features (overall face shape, global appearance)
"""

import torch
import torch.nn as nn
from .res_block import ResBlock
from .attention import Attention


class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool,           # Whether this layer upsamples (decoder) or downsamples (encoder)
            attention: bool,         # Whether to include self-attention in this layer
            num_groups: int,         # Number of groups for GroupNorm
            dropout_prob: float,     # Dropout probability for regularization
            num_heads: int,          # Number of attention heads (if attention=True)
            C: int):                 # Number of channels
        super().__init__()
        
        # Two residual blocks for feature processing
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        
        # Convolutional layer for spatial dimension changes
        if upscale:
            # Decoder: Transpose convolution for upsampling (increasing spatial resolution)
            # Reduces channels by half while doubling spatial dimensions
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            # Encoder: Regular convolution for downsampling (decreasing spatial resolution)
            # Doubles channels while halving spatial dimensions
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        
        # Optional self-attention layer for capturing long-range dependencies
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        """
        Forward pass through the U-Net layer.
        
        Args:
            x: Input feature tensor
            embeddings: Time embeddings for the current diffusion timestep
        Returns:
            Tuple of (processed_tensor, residual_for_skip_connection)
        """
        # First residual block with time embedding injection
        x = self.ResBlock1(x, embeddings)
        
        # Apply attention if this layer includes it
        # Attention helps maintain facial feature relationships
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        
        # Second residual block with time embedding injection
        x = self.ResBlock2(x, embeddings)
        
        # Return both the processed tensor and the residual for skip connections
        # Skip connections help preserve fine details during upsampling
        return self.conv(x), x