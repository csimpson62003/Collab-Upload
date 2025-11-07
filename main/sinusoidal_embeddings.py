"""
TIME EMBEDDING MODULE FOR DIFFUSION PROCESS
==========================================
Generates sinusoidal positional embeddings to encode the current timestep in the diffusion process.
This allows the model to understand how much noise has been added at each step.

For face-swapping: This component will help the model understand the denoising progression,
crucial for generating high-quality facial features at the right denoising stage.
"""

import torch
import torch.nn as nn
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        # Create positional encodings for each timestep (0 to time_steps-1)
        position = torch.arange(time_steps).unsqueeze(1).float()
        
        # Calculate the divisor for the sinusoidal functions
        # This creates different frequencies for different embedding dimensions
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        # Initialize the embedding matrix
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        
        # Apply sine to even indices (0, 2, 4, ...)
        embeddings[:, 0::2] = torch.sin(position * div)
        # Apply cosine to odd indices (1, 3, 5, ...)
        embeddings[:, 1::2] = torch.cos(position * div)
        
        self.embeddings = embeddings

    def forward(self, x, t):
        """
        Returns time embeddings for the given timesteps, reshaped for broadcasting with image tensors.
        
        Args:
            x: Input tensor (used for device placement)
            t: Timestep tensor
        Returns:
            Time embeddings shaped for broadcasting [batch, embed_dim, 1, 1]
        """
        # Get embeddings for the specified timesteps and ensure they're on the correct device
        embeds = self.embeddings.to(x.device)[t]
        # Reshape for broadcasting across spatial dimensions (height, width)
        return embeds[:, :, None, None]