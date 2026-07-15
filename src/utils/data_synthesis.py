import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import random

def blend_bleed_through(fg: np.ndarray, bg: np.ndarray, alpha_range=(0.1, 0.35), blur_range=(5, 15)) -> np.ndarray:
    """
    Blends a background image (reversed page) onto a foreground page to simulate bleed-through.
    Args:
        fg (np.ndarray): Foreground RGB or Grayscale image, normalized to [0, 1] or uint8.
        bg (np.ndarray): Background image to bleed through, same size as fg.
        alpha_range (tuple): Min/max opacity of the bleed-through.
        blur_range (tuple): Min/max kernel sizes for Gaussian blur simulating ink diffusion.
    """
    # Standardize to float32 [0, 1]
    if fg.dtype == np.uint8:
        fg = fg.astype(np.float32) / 255.0
    if bg.dtype == np.uint8:
        bg = bg.astype(np.float32) / 255.0
        
    # Ensure they are the same size
    if fg.shape[:2] != bg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    # Flip the background image horizontally to simulate the reverse side of the page
    bg_flipped = cv2.flip(bg, 1)

    # Blur background to simulate diffusion through the paper fibers
    blur_kernel = random.choice([k for k in range(blur_range[0], blur_range[1] + 1) if k % 2 == 1])
    bg_blurred = cv2.GaussianBlur(bg_flipped, (blur_kernel, blur_kernel), 0)

    # Draw bleed-through strength coefficient
    alpha = random.uniform(alpha_range[0], alpha_range[1])

    # Blend using physics-based multiplicative model: I_bleed = I_fg * (1 - alpha * (1 - I_bg_blurred))
    # This reflects light transmission absorption.
    bleed_img = fg * (1.0 - alpha * (1.0 - bg_blurred))
    
    return np.clip(bleed_img, 0.0, 1.0)


def generate_stains(img: np.ndarray, num_stains_range=(1, 4), stain_scale_range=(0.05, 0.25)) -> np.ndarray:
    """
    Adds synthetic stains (e.g. coffee/water/aging spots) using blended colored blobs.
    Args:
        img (np.ndarray): RGB image normalized to [0, 1] or uint8.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    h, w = img.shape[:2]
    num_stains = random.randint(num_stains_range[0], num_stains_range[1])
    stain_layer = np.ones_like(img)

    for _ in range(num_stains):
        # Generate random center and radius
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        r = int(random.uniform(stain_scale_range[0], stain_scale_range[1]) * min(h, w))
        if r < 10:
            continue

        # Create localized brown/yellow stain color (in RGB)
        stain_color = np.array([
            random.uniform(0.55, 0.75),  # R
            random.uniform(0.45, 0.65),  # G
            random.uniform(0.30, 0.50)   # B
        ], dtype=np.float32)

        # Draw smooth blob on mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (r | 1, r | 1), 0)
        mask = np.expand_dims(mask, axis=-1)

        # Multiply/blend stain layer
        stain_layer = stain_layer * (1.0 - mask * (1.0 - stain_color))

    # Blend with original image
    output = img * stain_layer
    return np.clip(output, 0.0, 1.0)


def apply_illumination_gradient(img: np.ndarray, strength_range=(0.4, 0.85)) -> np.ndarray:
    """
    Simulates uneven lighting gradients across the page.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    h, w = img.shape[:2]
    
    # Generate linear or radial gradient map
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    gradient_type = random.choice(['linear_x', 'linear_y', 'radial'])
    
    if gradient_type == 'linear_x':
        grad = (xx + 1.0) / 2.0
    elif gradient_type == 'linear_y':
        grad = (yy + 1.0) / 2.0
    else:
        # Radial gradient representing center-based lighting
        grad = np.sqrt(xx**2 + yy**2)
        grad = 1.0 - (grad / grad.max())

    strength = random.uniform(strength_range[0], strength_range[1])
    # Interpolate between flat and gradient illumination
    grad = grad * strength + (1.0 - strength)
    grad = np.expand_dims(grad, axis=-1)
    
    output = img * grad
    return np.clip(output, 0.0, 1.0)


def degrade_character_strokes(img: np.ndarray, probability=0.8) -> np.ndarray:
    """
    Degrades character strokes via random erosion, dilation, and micro-breaks.
    Typically applied to binary masks/inputs.
    """
    if random.random() > probability:
        return img

    is_uint8 = (img.dtype == np.uint8)
    if not is_uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.copy()

    # Determine kernel size for morphological operations
    k_size = random.choice([2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    degrade_type = random.choice(['erosion', 'dilation', 'scratch'])
    
    if degrade_type == 'erosion':
        # Text is dark in RGB, so erosion of foreground (thinning/breaking strokes) 
        # is actually equivalent to morphological dilation on high-intensity (white background) values.
        img_uint8 = cv2.dilate(img_uint8, kernel, iterations=1)
    elif degrade_type == 'dilation':
        # Dilation of foreground (thickening/smudging strokes) is equivalent to erosion of high-intensity.
        img_uint8 = cv2.erode(img_uint8, kernel, iterations=1)
    else:
        # Add random thin micro-breaks/scratches
        h, w = img_uint8.shape[:2]
        num_scratches = random.randint(3, 8)
        for _ in range(num_scratches):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(15, 60)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            thickness = random.choice([1, 2])
            # Draw scratch line (light color matching paper background)
            cv2.line(img_uint8, (x1, y1), (x2, y2), 255, thickness)

    # Re-apply slight blur to smooth out aliased edges
    blur_kernel = random.choice([3, 5])
    img_uint8 = cv2.GaussianBlur(img_uint8, (blur_kernel, blur_kernel), 0)

    if not is_uint8:
        return img_uint8.astype(np.float32) / 255.0
    return img_uint8
