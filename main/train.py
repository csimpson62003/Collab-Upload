"""
MAIN TRAINING FUNCTION FOR DIFFUSION MODEL
==========================================
Trains the U-Net to predict noise added to images at various timesteps.

TRAINING PROCESS:
1. Load and preprocess training images
2. For each image: randomly sample a timestep t
3. Add noise to the image according to timestep t
4. Train model to predict the added noise
5. Use predicted noise to calculate loss and update weights

PARAMETERS:
- max_dataset_size: Limit how many image pairs to train on (None = all ~7000 pairs)
  Examples: 100 (quick test), 1000 (medium training), None (full dataset)

FACE-SWAPPING TRAINING ADAPTATIONS NEEDED:
- Replace MNIST with face datasets (CelebA-HQ, FFHQ)
- Add identity conditioning inputs  
- Include face alignment preprocessing
- Add perceptual losses for better face quality
- Implement progressive training for high-resolution faces
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import kagglehub
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV3

from .face_swap_dataset import FaceSwapDataset
from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import set_seed, setup_cuda_device


def train(batch_size: int=64,
      num_time_steps: int=1000,
      num_epochs: int=15,
      seed: int=-1,
      ema_decay: float=0.9999,  
      lr=2e-5,
      checkpoint_path: str=None,
      max_dataset_size: int=None):
    
    # STEP 1: SET UP REPRODUCIBILITY
    # Set random seed for reproducibility
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    # STEP 2: DOWNLOAD AND LOAD FACE-SWAP DATASET
    # Download the face-swap dataset from Kaggle
    dataset_path = kagglehub.dataset_download("rdjarbeng/face-swap-images")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Create face-swap dataset and dataloader
    # max_dataset_size controls how many of the 7000 pairs to use for training
    if max_dataset_size is not None:
        print(f"🎯 Training on {max_dataset_size} image pairs (out of ~7000 available)")
    else:
        print("🎯 Training on ALL available image pairs (~7000)")
    
    train_dataset = FaceSwapDataset(
        dataset_path, 
        split="train", 
        image_size=64,  # Start with 64x64 for testing
        max_pairs=max_dataset_size  # Limit dataset size if specified
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # STEP 3: SET UP GPU/CPU DEVICE
    device = setup_cuda_device(preferred_gpu=0)  # Use RTX 2080 Super (GPU 0)
    
    # STEP 4: INITIALIZE ALL MODEL COMPONENTS
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)  # Noise schedule manager
    model = UNET().to(device)                                  # Main denoising network
    optimizer = optim.Adam(model.parameters(), lr=lr)          # Adam optimizer
    ema = ModelEmaV3(model, decay=ema_decay)                  # Exponential moving average for stable training
    
    # STEP 5: LOAD EXISTING CHECKPOINT (if it exists)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Starting training from scratch")
    
    # Ensure scheduler is on correct device
    scheduler = scheduler.to(device)
    
    # Loss function for comparing predicted vs actual noise
    criterion = nn.MSELoss(reduction='mean')

    # STEP 6: MAIN TRAINING LOOP
    for i in range(num_epochs):
        total_loss = 0
        
        # Process each batch of face-swap data
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            # STEP 6a: EXTRACT FACE IMAGES FROM BATCH
            # Extract face images from batch dictionary
            # For face-swapping, we train the model to generate the 'altered' (face-swapped) image
            # from the 'original' image by learning to remove noise
            original_faces = batch_data['original'].to(device)  # Source faces
            altered_faces = batch_data['altered'].to(device)    # Target face-swapped results
            
            # For diffusion training, we use the altered (face-swapped) images as our target
            # The model learns to generate face-swapped results by denoising
            x = altered_faces
            
            # STEP 6b: DIFFUSION TRAINING STEP
            # 1. Sample random timesteps for each image in the batch
            t = torch.randint(0, num_time_steps, (batch_size,), device=device)
            
            # 2. Sample random noise to add to images
            e = torch.randn_like(x, requires_grad=False)
            
            # 3. Get noise schedule parameter for sampled timesteps
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1)
            
            # 4. Add noise to images according to diffusion formula
            # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
            x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            # 5. Train model to predict the noise that was added
            output = model(x, t)
            
            # STEP 6c: BACKPROPAGATION
            optimizer.zero_grad()
            
            # Calculate loss: how well did model predict the noise?
            loss = criterion(output, e)
            total_loss += loss.item()
            
            # Update model weights
            loss.backward()
            optimizer.step()
            
            # Update exponential moving average (stabilizes training)
            ema.update(model)
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {i+1} | Loss {avg_loss:.5f} | Processed {len(train_dataset)} face-swap pairs')

    # STEP 7: SAVE TRAINED MODEL
    checkpoint = {
        'weights': model.state_dict(),      # Model parameters
        'optimizer': optimizer.state_dict(), # Optimizer state  
        'ema': ema.state_dict()            # EMA parameters (used for generation)
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")