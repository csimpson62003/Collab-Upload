"""
FACE-SWAP DATASET LOADER
=======================
Custom dataset class for loading face-swap image pairs.
Handles original faces and their corresponding face-swapped versions.

Dataset Structure:
- original/: Original face images (ID_frame.png)
- altered/: Face-swapped images (sourceID_targetID_frame.png)

For each training sample, loads:
- Original face image (source identity)
- Face-swapped target image (what we want to generate)
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FaceSwapDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = "train", image_size: int = 256, max_pairs: int = None):
        """
        Args:
            dataset_path: Path to the face-swap dataset
            split: "train", "val", or "test"
            image_size: Target image size (will resize to image_size x image_size)
            max_pairs: Maximum number of image pairs to load (None = load all 7000 pairs)
        """
        self.image_size = image_size
        self.max_pairs = max_pairs
        # Correct path construction based on actual dataset structure
        self.split_path = os.path.join(dataset_path, "kaggle", "working", "dataset", "Faceswap_images", split)
        
        # Get all original and altered image paths
        self.original_dir = os.path.join(self.split_path, "original")
        self.altered_dir = os.path.join(self.split_path, "altered")
        
        print(f"Looking for data in: {self.split_path}")
        print(f"Original dir: {self.original_dir}")
        print(f"Altered dir: {self.altered_dir}")
        
        # Find matching pairs
        self.image_pairs = self._find_matching_pairs()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        print(f"Loaded {len(self.image_pairs)} face-swap pairs from {split} set")
    
    def _find_matching_pairs(self):
        """Find matching original and altered image pairs"""
        pairs = []
        
        # Get all original images
        original_files = glob.glob(os.path.join(self.original_dir, "*.png"))
        
        # Apply max_pairs limit if specified
        if self.max_pairs is not None:
            print(f"Limiting dataset to {self.max_pairs} pairs (out of ~7000 available)")
            # We'll process enough original files to likely get max_pairs
            # Since each original might have multiple altered matches, we'll be conservative
            estimated_files_needed = min(len(original_files), self.max_pairs * 2)
            original_files = original_files[:estimated_files_needed]
        
        print(f"Processing {len(original_files)} original files...")
        
        for orig_path in original_files:
            # Stop if we've reached the desired number of pairs
            if self.max_pairs is not None and len(pairs) >= self.max_pairs:
                break
                
            orig_filename = os.path.basename(orig_path)
            # Extract ID and frame number from original filename (e.g., "004_0.png")
            parts = orig_filename.replace('.png', '').split('_')
            if len(parts) >= 2:
                orig_id = parts[0]
                frame_num = parts[1]
                
                # Look for corresponding altered images with this target ID and frame
                # Pattern: sourceID_targetID_frame.png where targetID matches orig_id
                altered_pattern = os.path.join(self.altered_dir, f"*_{orig_id}_{frame_num}.png")
                altered_matches = glob.glob(altered_pattern)
                
                # Add pairs for each matching altered image
                for altered_path in altered_matches:
                    # Stop if we've reached the desired number of pairs
                    if self.max_pairs is not None and len(pairs) >= self.max_pairs:
                        break
                        
                    altered_filename = os.path.basename(altered_path)
                    altered_parts = altered_filename.replace('.png', '').split('_')
                    if len(altered_parts) >= 3:
                        source_id = altered_parts[0]  # The person being swapped onto target
                        target_id = altered_parts[1]  # The target identity (should match orig_id)
                        frame = altered_parts[2]      # Frame number
                        
                        pairs.append({
                            'original': orig_path,     # Target person's original face
                            'altered': altered_path,   # Face-swapped result (source onto target)
                            'source_id': source_id,    # Source identity
                            'target_id': target_id,    # Target identity
                            'frame': frame             # Frame number
                        })
        
        final_count = len(pairs)
        if self.max_pairs is not None:
            print(f"Successfully loaded {final_count} face-swap pairs (requested: {self.max_pairs})")
        else:
            print(f"Found {final_count} matching face-swap pairs")
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load images
        original_img = Image.open(pair['original']).convert('RGB')
        altered_img = Image.open(pair['altered']).convert('RGB')
        
        # Apply transformations
        original_tensor = self.transform(original_img)
        altered_tensor = self.transform(altered_img)
        
        return {
            'original': original_tensor,      # Target person's original face
            'altered': altered_tensor,        # Face-swapped result (source onto target)
            'source_id': pair['source_id'],   # Source identity
            'target_id': pair['target_id'],   # Target identity  
            'frame': pair['frame']            # Frame number
        }