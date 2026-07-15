import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from utils.data_synthesis import (
    blend_bleed_through,
    generate_stains,
    apply_illumination_gradient,
    degrade_character_strokes
)

class DocumentRestorationDataset(Dataset):
    """
    PyTorch Dataset that takes clean document images and applies on-the-fly, 
    physics-based degradations to generate paired (degraded, target, mask) samples.
    """
    def __init__(self, clean_dir, patch_size=512, is_training=True):
        """
        Args:
            clean_dir (str): Path to directory with clean target images.
            patch_size (int): Image size to crop for training.
            is_training (bool): If True, applies random cropping and degradations.
        """
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.is_training = is_training
        
        # Supported extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        self.file_list = []
        for ext in extensions:
            self.file_list.extend(glob.glob(os.path.join(clean_dir, ext)))
            self.file_list.extend(glob.glob(os.path.join(clean_dir, ext.upper())))
            
        self.file_list = list(set(self.file_list))
        if len(self.file_list) == 0:
            # Fallback/Dummy warning for instantiation if clean folder is empty
            print(f"Warning: No images found in {clean_dir}. Dataset will be empty.")

    def __len__(self):
        return len(self.file_list)

    def _get_binary_mask(self, img_gray):
        """Generates ground truth binary mask of text using Sauvola/Adaptive thresholding."""
        # Simple thresholding on clean typewritten text is highly effective
        _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def __getitem__(self, idx):
        # Load clean target image
        target_path = self.file_list[idx]
        target_img = cv2.imread(target_path)
        if target_img is None:
            # Fallback if image fails to load: return black patches
            return (torch.zeros(3, self.patch_size, self.patch_size), 
                    torch.zeros(3, self.patch_size, self.patch_size),
                    torch.zeros(1, self.patch_size, self.patch_size))
            
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        h, w, c = target_img.shape

        # Crop to patch_size during training
        if self.is_training:
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                target_img = target_img[y:y+self.patch_size, x:x+self.patch_size]
            else:
                target_img = cv2.resize(target_img, (self.patch_size, self.patch_size))
        else:
            # Resize during validation if not matching patch size
            if target_img.shape[0] != self.patch_size or target_img.shape[1] != self.patch_size:
                target_img = cv2.resize(target_img, (self.patch_size, self.patch_size))

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        
        # 1. Generate ground truth binary mask (1 for text, 0 for background)
        mask = self._get_binary_mask(target_gray)
        # Convert mask to [0.0, 1.0] scale
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Normalized target image to [0.0, 1.0]
        target_normalized = target_img.astype(np.float32) / 255.0
        
        # 2. Synthesize degraded image starting from target
        # Load a separate image as the bleed background
        bg_idx = random.randint(0, len(self.file_list) - 1)
        bg_path = self.file_list[bg_idx]
        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        if bg_img is None:
            bg_img = np.ones_like(target_gray) * 255
        
        # Apply Bleed-Through
        degraded = blend_bleed_through(target_normalized, bg_img, alpha_range=(0.1, 0.35))
        
        # Apply Stains
        degraded = generate_stains(degraded, num_stains_range=(1, 3))
        
        # Apply Illumination Gradients
        degraded = apply_illumination_gradient(degraded, strength_range=(0.3, 0.7))
        
        # Apply character stroke degradation (breaks/fading) to simulate damaged letters
        # Run on the target mask to create degraded characters for the model to reconstruct
        # For training, we want the network to reconstruct clean, non-degraded target
        # so we inject character breaks only into the input image text.
        # To do this, we can mask the degraded image and make text strokes in it more faint/broken:
        if random.random() < 0.7:
            # Create a degraded mask (with broken strokes and thinning)
            degraded_mask = degrade_character_strokes(mask_normalized, probability=1.0)
            degraded_mask = np.expand_dims(degraded_mask, axis=-1)
            # Make the text locally lighter (closer to background value 1.0) where the mask was broken/eroded
            faded_factor = random.uniform(0.3, 0.8)
            degraded = degraded * (1.0 - (mask_normalized - degraded_mask) * faded_factor)

        # Ensure bounds
        degraded = np.clip(degraded, 0.0, 1.0)

        # Convert all to PyTorch tensors [C, H, W]
        degraded_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(target_normalized).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()

        return degraded_tensor, target_tensor, mask_tensor
