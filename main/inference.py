"""
IMAGE GENERATION THROUGH REVERSE DIFFUSION
==========================================
Generates new images by starting with pure noise and gradually denoising it
using the trained model. This is where the magic happens!

REVERSE DIFFUSION PROCESS:
1. Start with pure random noise
2. For each timestep (from T to 0):
   - Use model to predict noise at current step
   - Remove predicted noise to get cleaner image
   - Add small amount of random noise (except at final step)
3. Result: A generated image from learned data distribution

FACE-SWAPPING ADAPTATIONS NEEDED:
- Accept source face and target identity as inputs
- Add face alignment and preprocessing steps
- Include identity conditioning throughout denoising
- Add post-processing for face blending and color matching
- Implement face landmark guidance for better results
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from timm.utils import ModelEmaV3

from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import setup_cuda_device, display_reverse


def inference(checkpoint_path: str=None,
            num_time_steps: int=1000,
            ema_decay: float=0.9999):
    
    # STEP 1: SET UP GPU
    device = setup_cuda_device(preferred_gpu=0)  # Use RTX 2080 Super (GPU 0)
    
    # STEP 2: LOAD TRAINED MODEL FROM CHECKPOINT
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    
    # Use EMA weights for better generation quality
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    
    # Load diffusion scheduler
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)
    
    # Timesteps to save images for visualization (shows denoising progression)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    # STEP 3: GENERATION LOOP
    with torch.no_grad():  # No gradient computation needed for inference
        model = ema.module.eval()  # Set model to evaluation mode
        
        # Generate 10 sample face images
        for i in range(10):
            # STEP 3a: START WITH PURE RANDOM NOISE
            # For face-swapping: This would be conditioned noise based on source face
            z = torch.randn(1, 3, 64, 64).to(device)  # 3 channels (RGB), 64x64 resolution
            
            # STEP 3b: REVERSE DIFFUSION PROCESS
            # Gradually denoise from timestep T-1 down to 1
            for t in reversed(range(1, num_time_steps)):
                # Create timestep tensor
                t_tensor = torch.tensor([t], device=device)
                
                # Get noise schedule parameters for current timestep
                beta_t = scheduler.beta[t_tensor]
                alpha_t = scheduler.alpha[t_tensor]
                
                # Calculate denoising term
                # This determines how much of the model's noise prediction to subtract
                temp = (beta_t / ((torch.sqrt(1-alpha_t)) * (torch.sqrt(1-beta_t))))
                
                # DENOISING STEP: Remove predicted noise from current image
                # z = (z - noise_prediction) / scaling_factor
                z = (1/(torch.sqrt(1-beta_t)))*z - (temp*model(z, t_tensor))
                
                # Save intermediate images for visualization
                if t in times:
                    images.append(z.cpu())
                
                # Add small amount of random noise (except at final step)
                # This prevents the denoising from being too aggressive
                e = torch.randn(1, 3, 64, 64, device=device)  # Match the image dimensions
                z = z + (e*torch.sqrt(beta_t))
            
            # STEP 3c: FINAL DENOISING STEP (t=0)
            # No noise is added after this step
            t0_tensor = torch.tensor([0], device=device)
            beta_0 = scheduler.beta[t0_tensor]
            alpha_0 = scheduler.alpha[t0_tensor]
            temp = beta_0 / ((torch.sqrt(1-alpha_0)) * (torch.sqrt(1-beta_0)))
            x = (1/(torch.sqrt(1-beta_0)))*z - (temp*model(z, t0_tensor))

            # STEP 3d: SAVE AND DISPLAY RESULTS
            images.append(x.cpu())
            
            # Convert tensor to displayable format
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().cpu()
            x = x.numpy()
            
            # Denormalize from [-1, 1] to [0, 1] for RGB display
            x = (x + 1) / 2
            x = np.clip(x, 0, 1)
            
            # Display final generated face image
            plt.figure(figsize=(4, 4))
            plt.imshow(x)
            plt.axis('off')
            plt.title(f"Generated Face {i+1}")
            plt.show()
            
            # Show the complete denoising process
            display_reverse(images)
            images = []  # Reset for next generation