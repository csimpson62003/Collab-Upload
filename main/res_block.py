"""
RESIDUAL BLOCK - CORE BUILDING BLOCK FOR U-NET
==============================================
A residual connection block that helps train deep networks by allowing gradients to flow
more easily. Essential for the U-Net architecture used in diffusion models.

For face-swapping: These blocks will learn to preserve and transform facial features
while maintaining important identity information through residual connections.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Group normalization layers (more stable than batch norm for small batches)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        
        # Convolutional layers that preserve spatial dimensions
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        """
        Forward pass with time embedding injection and residual connection.
        
        Args:
            x: Input feature tensor [batch, channels, height, width]
            embeddings: Time embeddings to inject [batch, embed_dim, 1, 1]
        Returns:
            Output tensor with residual connection applied
        """
        # Inject time embeddings into the input features
        # This tells the model what stage of denoising we're at
        x = x + embeddings[:, :x.shape[1], :, :]
        
        # First convolution block: GroupNorm -> ReLU -> Conv2d
        r = self.conv1(self.relu(self.gnorm1(x)))
        
        # Apply dropout for regularization
        r = self.dropout(r)
        
        # Second convolution block: GroupNorm -> ReLU -> Conv2d
        r = self.conv2(self.relu(self.gnorm2(r)))
        
        # Residual connection: add input to output
        # This helps preserve important features while allowing transformations
        return r + x