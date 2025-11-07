"""
GOOGLE COLAB TPU SETUP - COPY THIS INTO YOUR COLAB NOTEBOOK
===========================================================
This script sets up everything needed for TPU training in Google Colab.
Just copy-paste this into a Colab cell and run it!
"""

# ============================================
# CELL 1: Install Dependencies
# ============================================
def install_dependencies():
    """Install all required packages for TPU training"""
    print("📦 Installing TPU-optimized PyTorch and dependencies...")
    
    # Install torch-xla for TPU support
    import subprocess
    import sys
    
    commands = [
        # Install torch-xla with TPU support
        "pip install torch torchvision",
        "pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html",
        # Install other requirements
        "pip install einops timm tqdm pillow matplotlib kagglehub"
    ]
    
    for cmd in commands:
        print(f"\n🔧 Running: {cmd}")
        subprocess.check_call(cmd.split())
    
    print("\n✅ All dependencies installed!")


# ============================================
# CELL 2: Verify TPU Setup
# ============================================
def verify_tpu():
    """Verify that TPU is properly configured"""
    print("🔍 Verifying TPU setup...")
    
    import torch
    try:
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        num_cores = xm.xrt_world_size()
        
        print(f"✅ TPU SUCCESSFULLY DETECTED!")
        print(f"   Device: {device}")
        print(f"   Number of TPU cores: {num_cores}")
        
        # Test computation
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y
        xm.mark_step()  # Sync TPU
        
        print(f"✅ TPU computation test passed!")
        print(f"   Test result shape: {z.shape}")
        
        return True
        
    except ImportError:
        print("❌ torch_xla not found!")
        print("   Make sure you've installed dependencies (run CELL 1)")
        return False
    except Exception as e:
        print(f"❌ TPU verification failed: {e}")
        print("\n💡 Solutions:")
        print("   1. Make sure Runtime Type is set to TPU")
        print("   2. Restart runtime and try again")
        print("   3. Check that you're using a TPU-enabled Colab account")
        return False


# ============================================
# CELL 3: Clone Repository and Setup
# ============================================
def setup_repository():
    """Clone the face-swap repository"""
    import os
    import subprocess
    
    # Check if already cloned
    if os.path.exists('Collab-Upload'):
        print("📁 Repository already exists, pulling latest changes...")
        subprocess.run(["git", "-C", "Collab-Upload", "pull"])
    else:
        print("📥 Cloning repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/csimpson62003/Collab-Upload.git"
        ])
    
    # Change to repository directory
    os.chdir('Collab-Upload')
    print(f"✅ Working directory: {os.getcwd()}")
    
    # Add to Python path
    import sys
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    print("✅ Repository ready!")


# ============================================
# CELL 4: Run Training with TPU
# ============================================
def train_on_tpu(
    num_epochs=10,
    batch_size=8,
    max_dataset_size=1000,
    learning_rate=1e-4
):
    """
    Start training the face-swap model on TPU
    
    Args:
        num_epochs: Number of training epochs (default: 10)
        batch_size: Batch size - MUST be multiple of 8 for TPU (default: 8)
        max_dataset_size: Number of image pairs to train on (default: 1000)
        learning_rate: Learning rate for optimizer (default: 1e-4)
    """
    print("🚀 Starting TPU-accelerated training...")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {max_dataset_size} pairs")
    print(f"  Learning rate: {learning_rate}")
    print("=" * 60)
    
    # Validate batch size
    if batch_size % 8 != 0:
        print(f"⚠️  WARNING: Batch size {batch_size} is not a multiple of 8")
        print("   TPU has 8 cores, so batch sizes like 8, 16, 24, 32 work best")
        suggested_batch = ((batch_size // 8) + 1) * 8
        print(f"   Suggested batch size: {suggested_batch}")
        batch_size = suggested_batch
    
    from main import train
    
    # Run training
    train(
        checkpoint_path='checkpoints/ddpm_faceswap_tpu_checkpoint',
        lr=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_dataset_size=max_dataset_size
    )
    
    print("\n🎉 Training complete!")
    print("💾 Checkpoint saved to: checkpoints/ddpm_faceswap_tpu_checkpoint")


# ============================================
# CELL 5: Generate Face Swaps
# ============================================
def generate_face_swap(source_image, target_image, output_path='face_swap_result.png'):
    """
    Generate a face swap using trained model
    
    Args:
        source_image: Path to source face image
        target_image: Path to target face image
        output_path: Where to save the result
    """
    from main import swap_faces
    
    print(f"🎨 Generating face swap...")
    print(f"   Source: {source_image}")
    print(f"   Target: {target_image}")
    
    result = swap_faces(
        source_image_path=source_image,
        target_image_path=target_image,
        checkpoint_path='checkpoints/ddpm_faceswap_tpu_checkpoint',
        num_denoising_steps=50,
        save_result=output_path
    )
    
    print(f"✅ Face swap saved to: {output_path}")
    return result


# ============================================
# ALL-IN-ONE SETUP
# ============================================
def complete_setup_and_train():
    """Run complete setup and start training in one go"""
    print("🚀 COMPLETE TPU SETUP AND TRAINING")
    print("=" * 60)
    
    # Step 1: Install dependencies
    print("\n📦 STEP 1: Installing dependencies...")
    install_dependencies()
    
    # Step 2: Verify TPU
    print("\n🔍 STEP 2: Verifying TPU...")
    if not verify_tpu():
        print("\n❌ TPU verification failed. Please fix the issues above.")
        return
    
    # Step 3: Setup repository
    print("\n📁 STEP 3: Setting up repository...")
    setup_repository()
    
    # Step 4: Start training
    print("\n🎓 STEP 4: Starting training...")
    train_on_tpu(
        num_epochs=10,
        batch_size=8,
        max_dataset_size=1000
    )
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETE! Your model is trained and ready to use!")
    print("=" * 60)


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    print("""
    📚 USAGE EXAMPLES FOR GOOGLE COLAB
    ===================================
    
    QUICK START (Recommended):
    --------------------------
    Run everything in one command:
    >>> complete_setup_and_train()
    
    
    STEP-BY-STEP (More control):
    -----------------------------
    1. Install dependencies:
    >>> install_dependencies()
    
    2. Verify TPU is working:
    >>> verify_tpu()
    
    3. Clone and setup repository:
    >>> setup_repository()
    
    4. Train the model:
    >>> train_on_tpu(num_epochs=10, batch_size=8, max_dataset_size=1000)
    
    5. Generate face swaps:
    >>> generate_face_swap('my_photos/person1.jpg', 'my_photos/person2.jpg')
    
    
    CUSTOM TRAINING:
    ----------------
    Train with custom parameters:
    >>> train_on_tpu(
    ...     num_epochs=50,           # More epochs = better quality
    ...     batch_size=16,           # Larger batch = faster but more memory
    ...     max_dataset_size=5000,   # More data = better results
    ...     learning_rate=2e-4       # Higher LR = faster learning
    ... )
    
    
    💡 TIPS:
    --------
    - First time? Use complete_setup_and_train()
    - For quick testing: max_dataset_size=100, num_epochs=2
    - For serious training: max_dataset_size=None (all 7000 pairs)
    - Batch size MUST be multiple of 8 for optimal TPU performance
    - Training 1000 pairs for 10 epochs takes ~5-10 minutes on TPU
    """)
