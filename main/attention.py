"""
SELF-ATTENTION MECHANISM FOR LONG-RANGE DEPENDENCIES
===================================================
Implements multi-head self-attention to capture long-range spatial relationships
in the image. Critical for maintaining coherent facial features across the image.

For face-swapping: This mechanism will be crucial for:
- Maintaining facial symmetry and proportions
- Ensuring consistent lighting and shadows across the face
- Preserving relationships between facial landmarks (eyes, nose, mouth)
- Generating coherent hair and background transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        # Linear projection to generate Query, Key, Value tensors (C*3 for all three)
        self.proj1 = nn.Linear(C, C*3)
        
        # Output projection layer
        self.proj2 = nn.Linear(C, C)
        
        # Store attention parameters
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        Apply multi-head self-attention to the input tensor.
        
        Args:
            x: Input feature tensor [batch, channels, height, width]
        Returns:
            Attention-processed tensor with same shape as input
        """
        # Store spatial dimensions for later reconstruction
        h, w = x.shape[2:]
        
        # Reshape from spatial format to sequence format
        # [batch, channels, height, width] -> [batch, height*width, channels]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Generate Query, Key, Value tensors through linear projection
        x = self.proj1(x)
        
        # Reshape for multi-head attention
        # Split into 3 tensors (Q, K, V) and organize by attention heads
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]  # Query, Key, Value tensors
        
        # Apply scaled dot-product attention
        # This computes attention weights and applies them to values
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        
        # Reshape back to spatial format
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        
        # Final linear projection
        x = self.proj2(x)
        
        # Convert back to original tensor format
        return rearrange(x, 'b h w C -> b C h w')