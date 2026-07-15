import numpy as np
import torch
import math

def create_blend_window(patch_size: int, margin: int = 64) -> np.ndarray:
    """
    Generates a 2D blending weight window where the weights taper smoothly (using a cosine function)
    to zero in the margins, ensuring seamless boundary stitching.
    """
    w_1d = np.ones(patch_size, dtype=np.float32)
    for i in range(margin):
        # Sine-based taper from 0 to 1
        taper_val = np.sin((i / margin) * (np.pi / 2))
        w_1d[i] = taper_val
        w_1d[patch_size - 1 - i] = taper_val
    w_2d = np.outer(w_1d, w_1d)
    return np.expand_dims(w_2d, axis=-1)


def tile_inference(model, image: np.ndarray, patch_size: int = 512, overlap: int = 64, device='cpu') -> np.ndarray:
    """
    Runs model inference on high-resolution images by breaking them into overlapping patches
    and stitching the outputs back together using a blending window.
    
    Args:
        model: PyTorch model or callable taking a tensor of shape [1, C, H_p, W_p] and returning a dict or tensor
        image (np.ndarray): HWC numpy array, normalized to [0, 1]
        patch_size (int): Dimensions of the square patch
        overlap (int): Number of pixels to overlap between adjacent patches
        device: Torch device to run inference on
    """
    h, w, c = image.shape
    stride = patch_size - overlap
    
    # Calculate padding size to make image fit exactly into patches
    pad_h = (math.ceil((h - patch_size) / stride) * stride + patch_size) - h
    pad_w = (math.ceil((w - patch_size) / stride) * stride + patch_size) - w
    
    # Pad image (replicate edge pixels)
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    hp, wp, _ = padded_image.shape
    
    # Canvas for accumulation and weights
    accumulated_img = np.zeros_like(padded_image, dtype=np.float32)
    accumulated_weight = np.zeros((hp, wp, 1), dtype=np.float32)
    
    blend_window = create_blend_window(patch_size, overlap)
    
    # Disable gradient computation for inference
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad():
        for y in range(0, hp - patch_size + 1, stride):
            for x in range(0, wp - patch_size + 1, stride):
                # Extract patch
                patch = padded_image[y:y+patch_size, x:x+patch_size]
                
                # Convert to Torch tensor [1, C, H, W]
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                # Model inference
                pred_out = model(patch_tensor)
                
                # Handle pipeline dictionary outputs vs standard tensor output
                if isinstance(pred_out, dict):
                    # Use final output stage
                    pred_tensor = pred_out["final"]
                else:
                    pred_tensor = pred_out
                    
                # Convert back to HWC numpy array
                pred_patch = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Apply blending window
                accumulated_img[y:y+patch_size, x:x+patch_size] += pred_patch * blend_window
                accumulated_weight[y:y+patch_size, x:x+patch_size] += blend_window
                
    # Normalize by accumulated weights
    reconstructed = accumulated_img / (accumulated_weight + 1e-8)
    
    # Crop back to original dimensions
    restored_image = reconstructed[:h, :w]
    return np.clip(restored_image, 0.0, 1.0)
