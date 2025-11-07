"""
DIFFUSION PROCESS SCHEDULER
==========================
Manages the noise schedule for the forward and reverse diffusion processes.
Controls how noise is added during training and how it's removed during generation.

DIFFUSION PROCESS EXPLANATION:
- Forward Process: Gradually adds Gaussian noise to images over T timesteps
- Reverse Process: Learns to remove noise step by step to generate images
- Beta Schedule: Controls noise addition rate (starts small, increases over time)
- Alpha Schedule: Cumulative noise retention (decreases over time)

For face-swapping: This scheduler will control the quality of face generation
by managing how facial features emerge during the denoising process.
"""

import torch
import torch.nn as nn


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        
        # Beta schedule: Defines how much noise to add at each timestep
        # Linear schedule from 1e-4 to 0.02 (commonly used for images)
        # For faces: This gradual schedule helps preserve facial structure
        beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        
        # Alpha values: How much of the original signal to retain
        alpha = 1 - beta
        
        # Cumulative alpha: How much original signal remains after t timesteps
        # This is used for efficient noise addition in training
        alpha_cumprod = torch.cumprod(alpha, dim=0).requires_grad_(False)
        
        # Register as buffers so they move with the model to GPU/CPU
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha_cumprod)

    def forward(self, t):
        """
        Get noise schedule parameters for given timesteps.
        
        Args:
            t: Timestep indices
        Returns:
            Tuple of (beta_t, alpha_t) for the given timesteps
        """
        return self.beta[t], self.alpha[t]