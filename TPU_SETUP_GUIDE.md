# 🚀 Google Colab TPU Setup Guide

## Quick Setup for TPU Training

### Step 1: Enable TPU Runtime in Google Colab
1. Click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **TPU**
3. Click **Save**
4. Your notebook will restart with TPU enabled

### Step 2: Install Required Packages in Colab
Run this in a Colab cell:
```python
# Install TPU-optimized PyTorch and dependencies
!pip install torch torchvision
!pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install -r requirements.txt
```

### Step 3: Verify TPU is Working
Run this to confirm TPU detection:
```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TPU Device: {device}")
print(f"Number of TPU cores: {xm.xrt_world_size()}")
```

You should see output like:
```
TPU Device: xla:0
Number of TPU cores: 8
```

### Step 4: Run Training
```python
from main import train

# Train with TPU acceleration
train(
    checkpoint_path='checkpoints/ddpm_faceswap_checkpoint',
    lr=1e-4,
    num_epochs=200,
    batch_size=8,  # TPU works best with batch sizes that are multiples of 8
    max_dataset_size=1000  # Start with 1000 pairs for testing
)
```

## TPU Performance Tips

### ✅ DO:
- **Use batch sizes that are multiples of 8** (8, 16, 24, 32, etc.)
  - TPU has 8 cores, so divisible batch sizes utilize all cores efficiently
- **Minimize CPU ↔ TPU data transfers**
  - Keep data on TPU during training loops
- **Use `xm.mark_step()`** after each epoch
  - Already implemented in the updated `train.py`
- **Increase batch size** compared to GPU
  - TPU v2 has 8GB per core = 64GB total
  - Can handle larger batches than typical consumer GPUs

### ❌ DON'T:
- Use batch size 1 or odd numbers (wastes TPU cores)
- Print inside tight training loops (causes TPU-CPU sync overhead)
- Use operations that aren't XLA-compatible (most PyTorch ops are fine)
- Move tensors between CPU/TPU frequently

## Expected Performance

### Training Speed Comparison (approximate):
- **CPU**: ~10-20 seconds per batch (Very slow ⚠️)
- **GPU (T4 in Colab)**: ~0.5-1 second per batch
- **TPU (v2-8)**: ~0.1-0.3 seconds per batch (Fastest! 🚀)

### For 1000 image pairs:
- Batch size 8 = 125 batches per epoch
- On TPU: ~30-40 seconds per epoch
- On GPU: ~60-120 seconds per epoch
- On CPU: ~1200-2400 seconds per epoch (20-40 minutes!)

## Troubleshooting

### Problem: "TPU not detected" or "Using CPU"
**Solution:**
1. Check runtime type is set to TPU
2. Restart runtime after changing to TPU
3. Reinstall torch-xla: `!pip install torch-xla[tpu]`

### Problem: "Slow training even on TPU"
**Check these issues:**
1. **Wrong batch size**: Use multiples of 8
2. **Too many print statements**: Remove prints from training loop
3. **Data loading bottleneck**: Set `num_workers=0` in DataLoader (TPU works better with single-threaded loading)
4. **Small dataset**: TPU overhead only pays off with enough data

### Problem: "Out of memory on TPU"
**Solutions:**
1. Reduce batch size (try 8 instead of 16)
2. Reduce image size (currently 64x64, don't go much higher)
3. Enable gradient accumulation
4. Use mixed precision training (FP16)

## Monitoring TPU Usage

Check TPU utilization in Colab:
```python
# Check if TPU is actually being used
!nvidia-smi  # Should show "No devices found" (we're using TPU, not GPU)

# Monitor TPU metrics
import torch_xla.debug.metrics as met
print(met.metrics_report())
```

## Code Changes Made for TPU Support

The following files have been updated to support TPU:

### 1. `main/utils.py`
- Added `torch_xla` import with fallback
- Updated `setup_cuda_device()` to auto-detect TPU
- Priority: TPU > CUDA > CPU

### 2. `main/train.py`
- Added TPU-optimized data loading with `ParallelLoader`
- Used `xm.optimizer_step()` for proper gradient sync
- Added `xm.mark_step()` for performance
- Fixed checkpoint saving to work with TPU tensors

### 3. This guide
- Complete setup instructions for Google Colab

## Quick Test Script for Colab

Copy this into a Colab cell to test everything:

```python
# Test TPU setup
import torch
import torch_xla.core.xla_model as xm

print("Testing TPU setup...")
device = xm.xla_device()
print(f"✅ TPU Device: {device}")

# Test computation
x = torch.randn(10, 10).to(device)
y = torch.randn(10, 10).to(device)
z = x @ y
print(f"✅ TPU computation works! Result shape: {z.shape}")

# Now run actual training
from main import train
train(
    checkpoint_path='checkpoints/test_checkpoint',
    num_epochs=2,
    batch_size=8,
    max_dataset_size=100  # Small test
)
print("✅ Training complete!")
```

## Additional Resources

- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [Google Cloud TPU Guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm)
- [Colab TPU Tutorial](https://colab.research.google.com/notebooks/tpu.ipynb)

---

**Summary:** Your code now automatically detects and uses TPU when available in Google Colab. Just enable TPU runtime and run your training script!
