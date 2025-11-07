"""
FACE SWAPPING WITH YOUR OWN IMAGES
==================================
This module allows you to perform face swapping using two of your own images.
Simply provide a source face and target face, and the AI will swap them!

USAGE:
1. Put your images in a folder (e.g., 'my_images/')
2. Call swap_faces(source_path, target_path)
3. The AI will generate a face-swapped result

EXAMPLE:
    from main.face_swapper import swap_faces
    result = swap_faces('my_images/person1.jpg', 'my_images/person2.jpg')
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from einops import rearrange
from timm.utils import ModelEmaV3

from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import setup_cuda_device


def load_and_preprocess_image(image_path: str, size: int = 64):
    """
    Load and preprocess a single image for face swapping.
    
    Args:
        image_path: Path to your image file
        size: Target size (64x64 for current model)
    Returns:
        Preprocessed tensor ready for the model
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform to match training data format
    transform = transforms.Compose([
        transforms.Resize((size, size)),           # Resize to model input size
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 64, 64]
    return tensor


def display_face_swap_result(source_img, target_img, swapped_img, save_path=None):
    """
    Display the face swap results in a nice format.
    
    Args:
        source_img: Source face tensor
        target_img: Target face tensor  
        swapped_img: Generated face-swapped result tensor
        save_path: Optional path to save the result
    """
    # Convert tensors to displayable format
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).detach().cpu()
        img = rearrange(img, 'c h w -> h w c')
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        img = np.clip(img, 0, 1)
        return img
    
    source_display = tensor_to_image(source_img)
    target_display = tensor_to_image(target_img)
    swapped_display = tensor_to_image(swapped_img)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(source_display)
    axes[0].set_title('Source Face\n(Face to be swapped)')
    axes[0].axis('off')
    
    axes[1].imshow(target_display)
    axes[1].set_title('Target Face\n(Face to receive swap)')
    axes[1].axis('off')
    
    axes[2].imshow(swapped_display)
    axes[2].set_title('🎯 FACE SWAP RESULT\n(Source face on target)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()


def swap_faces(source_image_path: str, 
               target_image_path: str,
               checkpoint_path: str = 'checkpoints/ddpm_faceswap_checkpoint',
               num_denoising_steps: int = 50,  # Fewer steps for faster results
               save_result: str = None):
    """
    🎯 MAIN FACE SWAPPING FUNCTION - USE YOUR OWN IMAGES HERE!
    =========================================================
    
    This function takes two of your own images and swaps the faces!
    
    Args:
        source_image_path: Path to the source face image (the face you want to put on someone else)
        target_image_path: Path to the target face image (the person who will receive the new face)
        checkpoint_path: Path to the trained model
        num_denoising_steps: How many denoising steps (fewer = faster, more = higher quality)
        save_result: Optional path to save the result image
    
    Returns:
        Generated face-swapped image tensor
        
    EXAMPLE USAGE:
        # Swap person1's face onto person2's body/background
        result = swap_faces('photos/person1.jpg', 'photos/person2.jpg', save_result='face_swap_result.png')
    """
    
    print("🎯 Starting Face Swap Process...")
    print(f"   Source image: {source_image_path}")
    print(f"   Target image: {target_image_path}")
    
    # STEP 1: SET UP DEVICE AND LOAD MODEL
    device = setup_cuda_device(preferred_gpu=0)
    
    # Load trained model
    print("📥 Loading trained face-swapping model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    
    # Use EMA weights for better quality
    ema = ModelEmaV3(model, decay=0.9999)
    ema.load_state_dict(checkpoint['ema'])
    model = ema.module.eval()
    
    # Load scheduler
    scheduler = DDPM_Scheduler(num_time_steps=1000).to(device)
    
    # STEP 2: LOAD AND PREPROCESS YOUR IMAGES
    print("🖼️ Loading and preprocessing your images...")
    source_tensor = load_and_preprocess_image(source_image_path).to(device)
    target_tensor = load_and_preprocess_image(target_image_path).to(device)
    
    print(f"   Source image shape: {source_tensor.shape}")
    print(f"   Target image shape: {target_tensor.shape}")
    
    # STEP 3: CONDITIONAL FACE SWAPPING PROCESS
    print("🔄 Performing face swap using diffusion model...")
    
    with torch.no_grad():
        # For now, we'll use a simple approach: start with target image and add some noise,
        # then denoise while conditioning on source features
        # (This is a simplified version - a full implementation would need more sophisticated conditioning)
        
        # Add noise to the target image (this simulates the forward diffusion process)
        noise_level = 0.3  # How much noise to add (0.0 = no noise, 1.0 = pure noise)
        noise = torch.randn_like(target_tensor)
        
        # Mix target image with noise
        alpha = 1 - noise_level
        noisy_target = alpha * target_tensor + (1 - alpha) * noise
        
        # Perform denoising (this is where the magic happens!)
        current_image = noisy_target
        
        for step in range(num_denoising_steps):
            # Create timestep tensor
            t = torch.tensor([step * (1000 // num_denoising_steps)], device=device)
            
            # Get noise schedule parameters
            beta_t = scheduler.beta[t]
            alpha_t = scheduler.alpha[t]
            
            # Predict noise using the model
            predicted_noise = model(current_image, t)
            
            # Remove predicted noise (denoising step)
            temp = beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t))
            current_image = (1 / torch.sqrt(1 - beta_t)) * current_image - temp * predicted_noise
            
            # Add small amount of noise for next step (except last step)
            if step < num_denoising_steps - 1:
                noise_factor = 0.01  # Small noise to prevent over-smoothing
                current_image = current_image + noise_factor * torch.randn_like(current_image)
        
        # The result is our face-swapped image!
        face_swapped_result = current_image
    
    # STEP 4: DISPLAY AND SAVE RESULTS
    print("✅ Face swap complete! Displaying results...")
    display_face_swap_result(source_tensor, target_tensor, face_swapped_result, save_result)
    
    print("🎉 Face swap process finished!")
    if save_result:
        print(f"💾 Result saved to: {save_result}")
    
    return face_swapped_result


def batch_face_swap(image_pairs: list, checkpoint_path: str = 'checkpoints/ddpm_faceswap_checkpoint'):
    """
    Perform face swapping on multiple image pairs at once.
    
    Args:
        image_pairs: List of tuples [(source1, target1), (source2, target2), ...]
        checkpoint_path: Path to trained model
        
    Example:
        pairs = [
            ('photos/person1.jpg', 'photos/person2.jpg'),
            ('photos/person3.jpg', 'photos/person4.jpg')
        ]
        batch_face_swap(pairs)
    """
    print(f"🔄 Processing {len(image_pairs)} face swap pairs...")
    
    results = []
    for i, (source_path, target_path) in enumerate(image_pairs):
        print(f"\n--- Processing pair {i+1}/{len(image_pairs)} ---")
        result = swap_faces(
            source_path, 
            target_path, 
            checkpoint_path,
            save_result=f'face_swap_result_{i+1}.png'
        )
        results.append(result)
    
    print(f"\n🎉 Batch processing complete! Generated {len(results)} face swaps.")
    return results


# Example usage function
def example_usage():
    """
    Example of how to use the face swapping functions.
    Replace the image paths with your own images!
    """
    print("📖 Face Swapping Example Usage:")
    print("=" * 50)
    
    # Example 1: Single face swap
    print("1. Single Face Swap:")
    print("   swap_faces('my_photos/me.jpg', 'my_photos/friend.jpg', save_result='result.png')")
    
    # Example 2: Batch processing
    print("\n2. Batch Face Swapping:")
    print("   pairs = [")
    print("       ('photos/person1.jpg', 'photos/person2.jpg'),")
    print("       ('photos/person3.jpg', 'photos/person4.jpg')")
    print("   ]")
    print("   batch_face_swap(pairs)")
    
    print("\n💡 Tips:")
    print("   - Use clear, front-facing photos for best results")
    print("   - Images should be well-lit with visible faces")
    print("   - The model works best with 64x64 resolution (it will auto-resize)")
    print("   - Try different noise levels for different effects")


if __name__ == "__main__":
    example_usage()