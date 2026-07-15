import os
import sys
import glob
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy import signal, ndimage
from scipy.spatial import KDTree

# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Configure timestamped logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"restore_{timestamp}.log")

    logger = logging.getLogger("DocRestore")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console handler with UTF-8 encoding for Windows compatibility
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


# ============================================================
# Stage 1: Preprocessing
# ============================================================

def preprocess(image: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """Convert to grayscale and apply gentle bilateral denoising."""
    if not cfg.get("enabled", True):
        logger.info("  [Stage 1] Preprocessing -- SKIPPED")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    logger.info("  [Stage 1] Preprocessing -- bilateral denoise")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Bilateral filter -- edge-preserving smoothing
    bf = cfg.get("bilateral_filter", {})
    gray = cv2.bilateralFilter(
        gray,
        d=bf.get("diameter", 9),
        sigmaColor=bf.get("sigma_color", 75),
        sigmaSpace=bf.get("sigma_space", 75),
    )

    # Optional: Non-local means denoising for heavy damage
    nlm = cfg.get("non_local_means", {})
    if nlm.get("enabled", False):
        logger.debug("    Applying non-local means denoising")
        gray = cv2.fastNlMeansDenoising(
            gray,
            h=nlm.get("h", 10),
            templateWindowSize=nlm.get("template_window", 7),
            searchWindowSize=nlm.get("search_window", 21),
        )

    return gray


# ============================================================
# Stage 2: Illumination Correction
# ============================================================

def correct_illumination(gray: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """Flatten uneven illumination using morphological background estimation + CLAHE.

    Uses background-preserving blending: CLAHE enhancement is smoothly faded out
    as pixel intensity approaches white, preventing noise amplification in
    uniform background regions.
    """
    if not cfg.get("enabled", True):
        logger.info("  [Stage 2] Illumination correction -- SKIPPED")
        return gray

    logger.info("  [Stage 2] Illumination correction -- morph background + CLAHE")

    # Morphological closing to estimate the background illumination
    k = cfg.get("morph_kernel_size", 51)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide original by background to flatten illumination
    corrected = cv2.divide(gray, background, scale=255.0)

    # CLAHE for local contrast enhancement with background-preserving blending
    clahe_cfg = cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        tile = clahe_cfg.get("tile_grid_size", 8)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_cfg.get("clip_limit", 2.0),
            tileGridSize=(tile, tile),
        )
        clahe_img = clahe.apply(corrected)

        # Background-preserving blend: as pixels approach white (background),
        # use the original corrected value instead of CLAHE output.
        # This prevents CLAHE from amplifying faint texture in white regions.
        blend_low = float(clahe_cfg.get("blend_low", 180))
        blend_high = float(clahe_cfg.get("blend_high", 230))
        weight = np.clip(
            (corrected.astype(np.float32) - blend_low) / (blend_high - blend_low),
            0.0, 1.0,
        )
        # weight=1 in white regions → keep corrected; weight=0 in dark regions → keep CLAHE
        final = corrected.astype(np.float32) * weight + clahe_img.astype(np.float32) * (1.0 - weight)
        corrected = np.clip(final, 0, 255).astype(np.uint8)

    return corrected


# ============================================================
# Stage 3: Stain & Bleed-Through Removal
# ============================================================

def remove_stains(image_bgr: np.ndarray, gray: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """
    Detect coloured stains via HSV thresholding and inpaint them.
    Protect dark text pixels from being affected by stain removal.
    Suppress bleed-through conservatively.
    """
    if not cfg.get("enabled", True):
        logger.info("  [Stage 3] Stain & bleed-through removal -- SKIPPED")
        return gray

    logger.info("  [Stage 3] Stain & bleed-through removal")
    result = gray.copy()

    if image_bgr is not None and len(image_bgr.shape) == 3:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Detect blue stains
        blue = cfg.get("blue_stain", {})
        blue_lower = np.array([blue.get("h_low", 90), blue.get("s_low", 30), blue.get("v_low", 50)])
        blue_upper = np.array([blue.get("h_high", 130), blue.get("s_high", 255), blue.get("v_high", 255)])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Detect yellow/brown stains
        yellow = cfg.get("yellow_stain", {})
        y_lower = np.array([yellow.get("h_low", 15), yellow.get("s_low", 30), yellow.get("v_low", 100)])
        y_upper = np.array([yellow.get("h_high", 40), yellow.get("s_high", 255), yellow.get("v_high", 255)])
        yellow_mask = cv2.inRange(hsv, y_lower, y_upper)

        stain_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        # Dilate mask slightly to cover stain boundaries
        stain_mask = cv2.dilate(stain_mask, np.ones((3, 3), np.uint8), iterations=1)

        # CRITICAL: Protect dark text pixels from being inpainted.
        # If the grayscale value is dark (< text_protect_threshold), it's likely text, not stain.
        text_protect = cfg.get("text_protect_threshold", 120)
        text_mask = gray < text_protect
        stain_mask[text_mask] = 0

        stain_pixels = cv2.countNonZero(stain_mask)
        total_pixels = stain_mask.shape[0] * stain_mask.shape[1]
        stain_pct = (stain_pixels / total_pixels) * 100
        logger.debug(f"    Stain coverage: {stain_pct:.1f}% of image ({stain_pixels} pixels)")

        if stain_pixels > 0:
            # Inpaint stained regions
            radius = cfg.get("inpaint_radius", 3)
            result = cv2.inpaint(result, stain_mask, radius, cv2.INPAINT_TELEA)
            logger.debug(f"    Inpainted {stain_pixels} stain pixels (text-protected)")

    # Bleed-through suppression - CONSERVATIVE approach
    bt_cfg = cfg.get("bleed_through", {})
    if bt_cfg.get("enabled", True):
        threshold = bt_cfg.get("intensity_threshold", 220)
        min_contrast = bt_cfg.get("min_contrast", 15)

        # Only suppress pixels that are VERY light AND have very low local contrast
        # This targets faint ghost text without touching real text
        local_mean = cv2.blur(result, (51, 51))
        contrast = np.abs(result.astype(np.int16) - local_mean.astype(np.int16))

        # Very conservative: only push near-white low-contrast areas to white
        bleed_mask = (result > threshold) & (contrast < min_contrast)
        result[bleed_mask] = 255
        bleed_count = np.count_nonzero(bleed_mask)
        if bleed_count > 0:
            logger.debug(f"    Suppressed {bleed_count} bleed-through pixels")

    return result


# ============================================================
# Stage 4: Deblur
# ============================================================

def deblur(gray: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """Apply Wiener deconvolution and/or unsharp masking if blur is detected."""
    if not cfg.get("enabled", True):
        logger.info("  [Stage 4] Deblur -- SKIPPED")
        return gray

    # Estimate blur via Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = cfg.get("blur_threshold", 100.0)
    logger.info(f"  [Stage 4] Deblur -- Laplacian variance: {laplacian_var:.1f} (threshold: {threshold})")

    result = gray.copy()

    if laplacian_var < threshold:
        logger.debug("    Image detected as blurry -- applying Wiener deconvolution")
        wiener_cfg = cfg.get("wiener", {})
        k_size = wiener_cfg.get("kernel_size", 5)
        noise_var = wiener_cfg.get("noise_variance", 0.01)

        # Create a simple Gaussian PSF (point spread function)
        psf = np.ones((k_size, k_size), dtype=np.float64) / (k_size * k_size)

        # Wiener deconvolution in frequency domain
        img_float = gray.astype(np.float64) / 255.0
        psf_padded = np.zeros_like(img_float)
        psf_padded[:k_size, :k_size] = psf
        psf_padded = np.roll(psf_padded, -k_size // 2, axis=0)
        psf_padded = np.roll(psf_padded, -k_size // 2, axis=1)

        img_fft = np.fft.fft2(img_float)
        psf_fft = np.fft.fft2(psf_padded)

        # Wiener filter: H* / (|H|^2 + NSR)
        wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + noise_var)
        restored = np.fft.ifft2(img_fft * wiener_filter)
        restored = np.abs(restored)
        restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
        result = restored

    # Edge-guided sharpening: enhance only near character boundaries,
    # not uniform background regions. This prevents noise amplification.
    usm_cfg = cfg.get("unsharp_mask", {})
    if usm_cfg.get("enabled", True):
        sigma = usm_cfg.get("sigma", 1.0)
        strength = usm_cfg.get("strength", 0.3)
        blurred = cv2.GaussianBlur(result, (0, 0), sigma)
        high_pass = result.astype(np.float64) - blurred.astype(np.float64)

        # Compute gradient magnitude as an edge guidance mask
        grad_x = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_mask = grad_mag / (grad_mag + 15.0)  # soft sigmoid-like normalization
        edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)

        # Sharpen only where edges are strong
        sharpened = result.astype(np.float64) + strength * high_pass * edge_mask
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        logger.debug(f"    Applied edge-guided sharpening (sigma={sigma}, strength={strength})")

    return result


# ============================================================
# Stage 5: Super-Resolution (Real-ESRGAN)
# ============================================================

def super_resolve(gray: np.ndarray, cfg: dict, logger: logging.Logger, device: str = "cpu") -> np.ndarray:
    """Upscale low-resolution images using Real-ESRGAN."""
    if not cfg.get("enabled", True):
        logger.info("  [Stage 5] Super-resolution -- SKIPPED")
        return gray

    h, w = gray.shape[:2]
    min_res = cfg.get("min_resolution", 1500)
    scale = cfg.get("scale", 2)

    if max(h, w) >= min_res:
        logger.info(f"  [Stage 5] Super-resolution -- SKIPPED (image {max(h,w)}px >= {min_res}px threshold)")
        return gray

    logger.info(f"  [Stage 5] Super-resolution -- Real-ESRGAN {scale}x upscale")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch

        # Configure model based on scale
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            model_name = "RealESRGAN_x2plus"
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            model_name = "RealESRGAN_x4plus"

        # Find or download model weights
        model_path = os.path.join(os.path.dirname(__file__), "weights", f"{model_name}.pth")
        if not os.path.exists(model_path):
            model_path = None  # RealESRGANer will auto-download

        use_gpu = "cuda" in device
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=cfg.get("tile_size", 512),
            tile_pad=cfg.get("tile_pad", 10),
            half=use_gpu,
            device=device,
        )

        # Convert grayscale to 3-channel for the model
        if len(gray.shape) == 2:
            img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            img_3ch = gray

        output, _ = upsampler.enhance(img_3ch, outscale=scale)

        # Convert back to grayscale
        if len(gray.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        logger.info(f"    Upscaled: {w}x{h} -> {output.shape[1]}x{output.shape[0]}")
        return output

    except ImportError as e:
        logger.warning(f"    Real-ESRGAN not available ({e}). Using bicubic upscale fallback.")
        upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        return upscaled
    except Exception as e:
        logger.warning(f"    Super-resolution failed ({e}). Using bicubic upscale fallback.")
        upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        return upscaled


# ============================================================
# Stage 6: Binarization (moved BEFORE text restoration)
# ============================================================

def binarize(gray: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """Apply Sauvola, Otsu, or adaptive binarization with text-preserving parameters."""
    if not cfg.get("enabled", True):
        logger.info("  [Stage 6] Binarization -- SKIPPED")
        return gray

    method = cfg.get("method", "sauvola")
    logger.info(f"  [Stage 6] Binarization -- {method}")

    if method == "sauvola":
        scfg = cfg.get("sauvola", {})
        win = scfg.get("window_size", 25)
        k = scfg.get("k", 0.08)  # Lower k = more text preserved (was 0.2, too aggressive)
        r = scfg.get("r", 128)
        min_contrast = scfg.get("min_contrast", 20)
        edge_alpha = scfg.get("edge_alpha", 0.15)

        # Sauvola threshold: T(x,y) = mean(x,y) * [1 + k * (std(x,y)/r - 1)]
        img_f = gray.astype(np.float64)
        mean = cv2.blur(img_f, (win, win))
        mean_sq = cv2.blur(img_f ** 2, (win, win))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        threshold = mean * (1.0 + k * (std / r - 1.0))

        # --- Local contrast gating ---
        # If the local contrast within the window is below min_contrast,
        # the region is uniform background and should never be marked as text.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        local_max = cv2.dilate(gray, kern)
        local_min = cv2.erode(gray, kern)
        local_contrast = local_max.astype(np.float64) - local_min.astype(np.float64)
        threshold[local_contrast < min_contrast] = 0.0  # force these to background

        # --- Edge guidance ---
        # Lower the threshold slightly near strong edges (real character strokes)
        # and raise it in flat areas to suppress bleed-through/ghost text.
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_map = grad_mag / (grad_mag + 15.0)
        edge_influence = cv2.GaussianBlur(edge_map, (5, 5), 0)
        threshold = threshold * (1.0 - edge_alpha * (1.0 - edge_influence))

        binary = np.where(img_f > threshold, 255, 0).astype(np.uint8)

    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        acfg = cfg.get("adaptive", {})
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=acfg.get("block_size", 31),
            C=acfg.get("c", 10),
        )
    else:
        logger.warning(f"    Unknown binarization method '{method}', falling back to Otsu")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if cfg.get("invert_output", False):
        binary = cv2.bitwise_not(binary)

    return binary


# ============================================================
# Stage 7: Text Restoration (CC analysis AFTER binarization)
# ============================================================

def _filter_isolated_noise(binary_img: np.ndarray, min_large_area: int = 12,
                           max_dist: float = 70.0, logger: logging.Logger = None) -> np.ndarray:
    """
    Remove small connected components that are spatially isolated from text.

    Punctuation (dots, commas) is preserved because it sits near large text
    components.  Background speckles are far from any text and get removed.
    Uses a KDTree for O(N log N) nearest-neighbour search.
    """
    inv = cv2.bitwise_not(binary_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

    if num_labels <= 1:
        return binary_img

    large_indices = []
    small_indices = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_large_area:
            large_indices.append(i)
        else:
            small_indices.append(i)

    if not large_indices or not small_indices:
        return binary_img

    large_centroids = centroids[large_indices]
    small_centroids = centroids[small_indices]

    tree = KDTree(large_centroids)
    dists, _ = tree.query(small_centroids, distance_upper_bound=max_dist)

    cleaned_inv = np.zeros_like(inv)
    for idx in large_indices:
        cleaned_inv[labels == idx] = 255
    kept_small = 0
    removed_small = 0
    for idx, d in zip(small_indices, dists):
        if d < max_dist:
            cleaned_inv[labels == idx] = 255
            kept_small += 1
        else:
            removed_small += 1

    if logger:
        logger.debug(f"    Spatial filter: kept {kept_small} small components, "
                     f"removed {removed_small} isolated speckles")

    return cv2.bitwise_not(cleaned_inv)


def _clean_borders(binary: np.ndarray, margin: int = 15,
                   logger: logging.Logger = None) -> np.ndarray:
    """
    Remove dark border artifacts (scanner shadows, binding creases) by
    flood-filling connected dark regions that touch the image margins.
    """
    h, w = binary.shape[:2]
    inv = cv2.bitwise_not(binary)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    filled = 0
    # Top and bottom margins
    for x in range(0, w, 5):
        for offset in range(margin):
            if inv[offset, x] == 255:
                ret, _, _, _ = cv2.floodFill(inv, mask, (x, offset), 0)
                filled += ret
            if inv[h - 1 - offset, x] == 255:
                ret, _, _, _ = cv2.floodFill(inv, mask, (x, h - 1 - offset), 0)
                filled += ret
    # Left and right margins
    for y in range(0, h, 5):
        for offset in range(margin):
            if inv[y, offset] == 255:
                ret, _, _, _ = cv2.floodFill(inv, mask, (offset, y), 0)
                filled += ret
            if inv[y, w - 1 - offset] == 255:
                ret, _, _, _ = cv2.floodFill(inv, mask, (w - 1 - offset, y), 0)
                filled += ret

    if logger and filled > 0:
        logger.debug(f"    Border cleanup: removed {filled} border-touching pixels")

    return cv2.bitwise_not(inv)


def restore_text(binary: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """
    Reconnect broken strokes via morphological closing.
    Filter noise using connected component analysis on the binary image.
    Remove isolated background speckles via spatial density filtering.
    This stage runs AFTER binarization for proper text/noise separation.
    """
    if not cfg.get("enabled", True):
        logger.info("  [Stage 7] Text restoration -- SKIPPED")
        return binary

    logger.info("  [Stage 7] Text restoration -- morph close + CC filter")

    # Invert: text becomes white (foreground) for morphological ops
    inverted = cv2.bitwise_not(binary)

    # Morphological closing to reconnect broken character strokes
    k = cfg.get("morph_close_kernel", 2)
    iters = cfg.get("morph_iterations", 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=iters)

    # Connected component analysis to filter noise
    cc_cfg = cfg.get("connected_component", {})
    min_area = cc_cfg.get("min_area", 3)  # Very small threshold to keep punctuation
    max_ratio = cc_cfg.get("max_area_ratio", 0.3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    total_area = binary.shape[0] * binary.shape[1]
    max_area = int(total_area * max_ratio)

    cleaned = np.zeros_like(closed)
    kept = 0
    removed = 0

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cleaned[labels == i] = 255
            kept += 1
        else:
            removed += 1

    logger.debug(f"    Connected components: kept {kept}, removed {removed} (noise)")

    # Re-invert back to black text on white background
    result = cv2.bitwise_not(cleaned)

    # Spatial density filter: remove small components far from any text
    spatial_cfg = cfg.get("spatial_filter", {})
    if spatial_cfg.get("enabled", True):
        result = _filter_isolated_noise(
            result,
            min_large_area=spatial_cfg.get("min_large_area", 12),
            max_dist=spatial_cfg.get("max_distance", 70.0),
            logger=logger,
        )

    return result


# ============================================================
# Stage 8: Post-Processing
# ============================================================

def postprocess(binary: np.ndarray, cfg: dict, logger: logging.Logger) -> np.ndarray:
    """Median filter, deskew, morphological cleanup, and autocrop."""
    if not cfg.get("enabled", True):
        logger.info("  [Stage 8] Post-processing -- SKIPPED")
        return binary

    logger.info("  [Stage 8] Post-processing -- median, deskew, crop")
    result = binary.copy()

    # Morphological cleanup -- small opening to remove speckle noise
    morph_cfg = cfg.get("morphological_clean", {})
    if morph_cfg.get("enabled", True):
        mk = morph_cfg.get("kernel_size", 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Border cleanup -- remove scanner shadows and binding crease artifacts
    border_cfg = cfg.get("border_clean", {})
    if border_cfg.get("enabled", True):
        result = _clean_borders(
            result,
            margin=border_cfg.get("margin", 35),
            logger=logger,
        )

    # Median filter for salt-and-pepper noise
    med_cfg = cfg.get("median_filter", {})
    if med_cfg.get("enabled", True):
        ks = med_cfg.get("kernel_size", 3)
        result = cv2.medianBlur(result, ks)

    # Deskew
    deskew_cfg = cfg.get("deskew", {})
    if deskew_cfg.get("enabled", True):
        max_angle = deskew_cfg.get("max_angle", 5.0)
        angle = _estimate_skew(result, max_angle)
        if abs(angle) > 0.1:
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
            logger.debug(f"    Deskewed by {angle:.2f} degrees")

    # Auto-crop whitespace
    crop_cfg = cfg.get("autocrop", {})
    if crop_cfg.get("enabled", True):
        margin = crop_cfg.get("border_margin", 20)
        result = _autocrop(result, margin)

    return result


def _estimate_skew(binary: np.ndarray, max_angle: float = 5.0) -> float:
    """Estimate skew angle using projection profile analysis."""
    # Invert so text is white
    inv = cv2.bitwise_not(binary)

    best_angle = 0.0
    best_score = 0.0

    # Test angles from -max_angle to +max_angle in 0.5 degree steps (faster)
    for angle_10x in range(int(-max_angle * 2), int(max_angle * 2) + 1):
        angle = angle_10x / 2.0
        h, w = inv.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(inv, M, (w, h), borderValue=0)

        # Compute horizontal projection (sum of each row)
        projection = np.sum(rotated, axis=1, dtype=np.float64)
        # Score = variance of projection (higher = more aligned text lines)
        score = np.var(projection)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _autocrop(image: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crop whitespace borders, keeping a small margin."""
    if len(image.shape) == 2:
        mask = image < 250
    else:
        mask = np.any(image < 250, axis=2)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Add margin
    h, w = image.shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin)
    x1 = min(w, x1 + margin)

    return image[y0:y1, x0:x1]


# ============================================================
# Comparison Image Generator
# ============================================================

def create_comparison(original: np.ndarray, restored: np.ndarray, filename: str,
                      output_path: str, logger: logging.Logger):
    """Create a side-by-side before/after comparison image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resize both to same height for comparison
    target_h = 1200

    # Process original
    if len(original.shape) == 3:
        orig_display = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_display = original.copy()
    scale_o = target_h / orig_display.shape[0]
    orig_resized = cv2.resize(orig_display, None, fx=scale_o, fy=scale_o, interpolation=cv2.INTER_AREA)

    # Process restored
    scale_r = target_h / restored.shape[0]
    rest_resized = cv2.resize(restored, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_AREA)

    # Create separator bar
    separator = np.full((target_h, 4), 128, dtype=np.uint8)

    # Concatenate horizontally with labels
    combined = np.hstack([orig_resized, separator, rest_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "BEFORE", (10, 30), font, 0.8, (0,), 2)
    cv2.putText(combined, "AFTER", (orig_resized.shape[1] + 14, 30), font, 0.8, (0,), 2)

    cv2.imwrite(output_path, combined)
    logger.info(f"    Comparison saved: {output_path}")


# ============================================================
# Chandra OCR Engine (ONLY OCR Engine - Tesseract DISABLED)
# ============================================================

class ChandraOCREngine:
    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.enabled = cfg.get("enabled", False)
        self.method = cfg.get("method", "vllm")
        self.model_checkpoint = cfg.get("model_checkpoint", "datalab-to/chandra-ocr-2")
        self.vllm_api_base = cfg.get("vllm_api_base", "http://localhost:8000/v1")
        self.vllm_api_key = cfg.get("vllm_api_key", "EMPTY")
        self.vllm_model_name = cfg.get("vllm_model_name", "chandra")
        self.save_txt = cfg.get("save_txt", True)
        self.save_format = cfg.get("save_format", "markdown").lower()
        self.use_mock_fallback = cfg.get("use_mock_fallback", True)
        self.manager = None
        self.server_checked = False
        self.server_online = False

        if self.enabled:
            self._init_engine()
    
    def _init_engine(self):
        self.logger.info(f"Initializing Chandra OCR Engine (method: {self.method})...")
        
        if self.method == "vllm" and self.use_mock_fallback:
            self.logger.info(f"Checking connection to vLLM server: {self.vllm_api_base}...")
            self.server_online = self._check_server_online()
            self.server_checked = True
            if not self.server_online:
                self.logger.warning("vLLM server is offline. Bypassing API requests and using mock fallback directly.")
                return

        try:
            from chandra.model import InferenceManager
            from chandra.settings import settings
            settings.VLLM_API_KEY = self.vllm_api_key
            settings.VLLM_API_BASE = self.vllm_api_base
            settings.VLLM_MODEL_NAME = self.vllm_model_name
            settings.MODEL_CHECKPOINT = self.model_checkpoint
            settings.TORCH_DEVICE = "cpu"

            # Try to initialize Chandra in a CPU-safe way first.
            self.manager = InferenceManager(method=self.method)
            self.logger.info("Chandra OCR Engine initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chandra OCR Engine: {e}")
            self.logger.warning("Chandra OCR could not be loaded in this environment. The pipeline will continue to process images but the OCR output will be generated from the available Chandra fallback metadata.")
            self.manager = None
            self.enabled = True

    def _check_server_online(self) -> bool:
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(self.vllm_api_base + "/models")
            with urllib.request.urlopen(req, timeout=1.5) as response:
                return True
        except Exception as e:
            import urllib.error
            if isinstance(e, urllib.error.HTTPError):
                # Any HTTP status response (e.g. 401 Unauthorized, 404 Not Found) means server is running
                return True
            return False

    def run_ocr(self, image_np: np.ndarray, filename: str) -> str:
        """Runs Chandra OCR on a numpy image. Returns text content in the requested format."""
        if not self.enabled:
            return ""

        self.logger.info(f"  [Stage 9] Chandra OCR Verification on: {filename}")
        start_time = time.time()

        # If we know the server is offline, go straight to fallback
        if self.method == "vllm" and self.server_checked and not self.server_online:
            fallback_text = self._get_fallback_output(filename, self.save_format, image_np)
            elapsed = time.time() - start_time
            self.logger.info(f"    Chandra OCR (FALLBACK - OFFLINE SERVER) completed in {elapsed:.3f}s")
            return fallback_text

        # Convert numpy array (BGR or Grayscale) to PIL Image (RGB)
        try:
            if len(image_np.shape) == 2:
                pil_img = Image.fromarray(image_np).convert("RGB")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.logger.error(f"    Failed to convert image for OCR: {e}")
            return self._get_fallback_output(filename, self.save_format, image_np)

        # Try to run OCR using the manager
        if self.manager is not None:
            try:
                from chandra.model.schema import BatchInputItem
                batch = [BatchInputItem(image=pil_img, prompt_type="ocr_layout")]
                
                # Run layout OCR
                results = self.manager.generate(batch, vllm_api_base=self.vllm_api_base)
                
                if results and not results[0].error:
                    ocr_output = results[0]
                    elapsed = time.time() - start_time
                    self.logger.info(f"    Chandra OCR completed in {elapsed:.1f}s (tokens: {ocr_output.token_count})")
                    
                    if self.save_format == "html":
                        return ocr_output.html or self._get_fallback_output(filename, self.save_format, image_np)
                    elif self.save_format == "markdown":
                        return ocr_output.markdown or self._get_fallback_output(filename, self.save_format, image_np)
                    else:
                        return ocr_output.raw or self._get_fallback_output(filename, self.save_format, image_np)
                else:
                    err_msg = results[0].error if results else "Empty results"
                    raise RuntimeError(f"OCR generation failed: {err_msg}")

            except Exception as e:
                self.logger.error(f"    Chandra OCR execution failed: {e}")
                self.logger.warning("    Falling back to built-in OCR output generation.")

        # Fallback output that still produces a valid Markdown file
        fallback_text = self._get_fallback_output(filename, self.save_format, image_np)
        elapsed = time.time() - start_time
        self.logger.info(f"    Chandra OCR (FALLBACK) completed in {elapsed:.1f}s")
        return fallback_text

    def _get_fallback_output(self, filename: str, format_str: str, image_np: np.ndarray | None = None) -> str:
        height = width = 0
        if image_np is not None:
            if len(image_np.shape) >= 2:
                height, width = image_np.shape[:2]
        if width == 0 or height == 0:
            width = height = 0

        md = (
            f"# OCR Transcription for {filename}\n\n"
            f"## Status\n"
            f"- Chandra OCR initialization failed in this environment.\n"
            f"- The pipeline could not extract actual OCR text from the model at this time.\n"
            f"\n## Image details\n"
            f"- Source file: {filename}\n"
            f"- Dimensions: {width}x{height}\n"
        )
        html = f"<div><h1>OCR Transcription for {filename}</h1><pre>{md}</pre></div>"

        if format_str == "html":
            return html
        elif format_str == "markdown":
            return md
        else:
            return md


# ============================================================
# Main Pipeline Orchestrator

class DocumentRestorer:
    """
    Orchestrates the full 8-stage document restoration pipeline.

    Pipeline order (corrected):
      1. Preprocessing (grayscale + bilateral denoise)
      2. Illumination correction (morph background + CLAHE)
      3. Stain & bleed-through removal (HSV segmentation + inpainting, text-protected)
      4. Deblur (Wiener filter + gentle unsharp mask)
      5. Super-resolution (Real-ESRGAN, conditional)
      6. Binarization (Sauvola with conservative k)
      7. Text restoration (morph close + CC analysis on binary)
      8. Post-processing (median filter, deskew, autocrop)
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        paths = self.config.get("paths", {})
        general = self.config.get("general", {})

        self.input_dir = paths.get("input_dir", "./input")
        self.output_dir = paths.get("output_dir", "./output")
        self.comparison_dir = paths.get("comparison_dir", os.path.join(self.output_dir, "comparisons"))
        log_dir = paths.get("log_dir", "./logs")

        self.output_format = general.get("output_format", "png")
        self.output_dpi = general.get("output_dpi", 300)
        self.use_gpu = general.get("use_gpu", True)
        self.log_level = general.get("log_level", "INFO")

        self.logger = setup_logging(log_dir, self.log_level)

        # Determine device
        self.device = "cpu"
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    self.logger.info(f"GPU detected: {gpu_name}")
                else:
                    self.logger.info("CUDA not available -- using CPU")
            except ImportError:
                self.logger.info("PyTorch not available -- using CPU")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)

        ocr_cfg = self.config.get("ocr", {})
        
        # ONLY Chandra OCR is enabled - Tesseract is DISABLED
        self.ocr_engine = ChandraOCREngine(ocr_cfg, self.logger)

    def _prepare_output_paths(self, input_path: str, base_input_dir: str | None = None) -> tuple[str, str, str, str]:
        """
        Create and return output paths for the restored image, OCR markdown,
        and comparison image while preserving original directory structure.
        """
        if base_input_dir:
            rel_dir = os.path.relpath(os.path.dirname(input_path), base_input_dir)
            if rel_dir.startswith(".."):
                rel_dir = "."
        else:
            rel_dir = "."

        output_subdir = self.output_dir if rel_dir == "." else os.path.join(self.output_dir, rel_dir)
        comparison_subdir = self.comparison_dir if rel_dir == "." else os.path.join(self.comparison_dir, rel_dir)

        os.makedirs(output_subdir, exist_ok=True)
        os.makedirs(comparison_subdir, exist_ok=True)

        basename = Path(os.path.basename(input_path)).stem
        image_out_path = os.path.join(output_subdir, f"{basename}_restored.{self.output_format}")
        ext = "md" if self.ocr_engine.save_format == "markdown" else ("html" if self.ocr_engine.save_format == "html" else "txt")
        ocr_out_path = os.path.join(output_subdir, f"{basename}.{ext}")
        comparison_out_path = os.path.join(comparison_subdir, f"{basename}_comparison.png")

        return image_out_path, ocr_out_path, comparison_out_path, output_subdir


    def restore_image_data(self, image_bgr: np.ndarray, filename: str) -> tuple[np.ndarray, str | None]:
        """Runs the 8 restoration stages and Chandra OCR on a BGR image. Returns (restored_image, ocr_text)."""
        # Stage 1: Preprocessing
        gray = preprocess(image_bgr, self.config.get("preprocessing", {}), self.logger)

        # Stage 2: Illumination Correction
        gray = correct_illumination(gray, self.config.get("illumination", {}), self.logger)

        # Stage 3: Stain & Bleed-Through Removal (text-protected)
        gray = remove_stains(image_bgr, gray, self.config.get("stain_removal", {}), self.logger)

        # Stage 4: Deblur
        gray = deblur(gray, self.config.get("deblur", {}), self.logger)

        # Stage 5: Super-Resolution
        gray = super_resolve(gray, self.config.get("super_resolution", {}),
                              self.logger, self.device)

        # Stage 6: Binarization (BEFORE text restoration)
        binary = binarize(gray, self.config.get("binarization", {}), self.logger)

        # Stage 7: Text Restoration (CC analysis on binary image)
        binary = restore_text(binary, self.config.get("text_restoration", {}), self.logger)

        # Stage 8: Post-Processing
        result = postprocess(binary, self.config.get("postprocessing", {}), self.logger)

        # Stage 9: OCR Verification (Chandra OCR)
        ocr_text = None
        if self.ocr_engine.enabled:
            ocr_text = self.ocr_engine.run_ocr(result, filename)

        return result, ocr_text

    def restore_single(self, image_path: str, base_input_dir: str | None = None) -> np.ndarray | None:
        """Run the full pipeline on a single image. Returns the restored image or None on error."""
        filename = os.path.basename(image_path)
        self.logger.info(f"Processing: {filename}")
        start = time.time()

        try:
            # Load original image (BGR)
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                self.logger.error(f"  Failed to read image: {image_path}")
                return None

            h, w = image_bgr.shape[:2]
            self.logger.info(f"  Input: {w}x{h} ({os.path.getsize(image_path) / 1024:.0f} KB)")

            result, ocr_text = self.restore_image_data(image_bgr, filename)

            # Save outputs preserving input folder structure
            image_out_path, ocr_out_path, comparison_out_path, _ = self._prepare_output_paths(image_path, base_input_dir)

            self._save_with_dpi(result, image_out_path, self.output_dpi)
            self.logger.info(f"  Saved: {image_out_path}")

            if self.ocr_engine.save_txt:
                if ocr_text is None:
                    ocr_text = self.ocr_engine._get_fallback_output(filename, self.ocr_engine.save_format, result)
                with open(ocr_out_path, "w", encoding="utf-8") as f_ocr:
                    f_ocr.write(ocr_text)
                self.logger.info(f"  Saved OCR text output: {ocr_out_path}")

            create_comparison(image_bgr, result, filename, comparison_out_path, self.logger)

            elapsed = time.time() - start
            self.logger.info(f"  Completed in {elapsed:.1f}s -- output: {result.shape[1]}x{result.shape[0]}")

            return result

        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"  FAILED after {elapsed:.1f}s: {e}", exc_info=True)
            return None

    def restore_pdf(self, pdf_path: str) -> bool:
        """Processes a PDF file, restoring each page and saving both individual pages and compiled outputs."""
        filename = os.path.basename(pdf_path)
        self.logger.info(f"Processing PDF document: {filename}")
        start = time.time()

        try:
            import pypdfium2 as pdfium
        except ImportError:
            self.logger.error("pypdfium2 is not installed. Cannot process PDF files.")
            return False

        try:
            doc = pdfium.PdfDocument(pdf_path)
            num_pages = len(doc)
            self.logger.info(f"  PDF has {num_pages} pages.")

            restored_pil_pages = []
            compiled_ocr_texts = []
            
            basename = Path(filename).stem

            for page_idx in range(num_pages):
                page_num = page_idx + 1
                self.logger.info(f"  --- Processing Page {page_num}/{num_pages} ---")
                
                # Render page at 3x scale (~216 DPI) or 4x scale (~288 DPI) for high-accuracy OCR.
                page = doc[page_idx]
                bitmap = page.render(scale=3)
                pil_image = bitmap.to_pil()
                
                # Convert to BGR for OpenCV pipeline
                image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Run restoration and OCR
                page_filename = f"{basename}_page_{page_num}.png"
                restored_bgr, ocr_text = self.restore_image_data(image_bgr, page_filename)
                
                # Convert back to PIL Image (RGB) for compiled PDF generation
                restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
                restored_pil = Image.fromarray(restored_rgb)
                restored_pil_pages.append(restored_pil)
                
                # Save individual restored page image
                page_out_path = os.path.join(self.output_dir, f"{basename}_page_{page_num}_restored.{self.output_format}")
                self._save_with_dpi(restored_bgr, page_out_path, self.output_dpi)
                self.logger.info(f"    Saved page image: {page_out_path}")
                
                # Save individual page comparison image
                comp_path = os.path.join(self.comparison_dir, f"{basename}_page_{page_num}_comparison.png")
                create_comparison(image_bgr, restored_bgr, page_filename, comp_path, self.logger)

                # Save individual page OCR if enabled
                if self.ocr_engine.save_txt:
                    if ocr_text is None:
                        ocr_text = self.ocr_engine._get_fallback_output(page_filename, self.ocr_engine.save_format, restored_bgr)
                    ext = "md" if self.ocr_engine.save_format == "markdown" else ("html" if self.ocr_engine.save_format == "html" else "txt")
                    page_ocr_out_path = os.path.join(self.output_dir, f"{basename}_page_{page_num}_restored.{ext}")
                    with open(page_ocr_out_path, "w", encoding="utf-8") as f_ocr:
                        f_ocr.write(ocr_text)
                    self.logger.info(f"    Saved page OCR text: {page_ocr_out_path}")
                    
                    compiled_ocr_texts.append(f"## PAGE {page_num}\n\n{ocr_text}\n\n")
                else:
                    compiled_ocr_texts.append(f"## PAGE {page_num}\n\n[OCR is disabled or failed for this page]\n\n")

            # Save document-wide compiled PDF containing all restored pages
            if restored_pil_pages:
                compiled_pdf_path = os.path.join(self.output_dir, f"{basename}_restored.pdf")
                restored_pil_pages[0].save(
                    compiled_pdf_path,
                    save_all=True,
                    append_images=restored_pil_pages[1:],
                    resolution=self.output_dpi
                )
                self.logger.info(f"  Saved compiled multi-page restored PDF: {compiled_pdf_path}")

            # Save document-wide compiled OCR text file
            if compiled_ocr_texts and self.ocr_engine.enabled and self.ocr_engine.save_txt:
                ext = "md" if self.ocr_engine.save_format == "markdown" else ("html" if self.ocr_engine.save_format == "html" else "txt")
                compiled_ocr_path = os.path.join(self.output_dir, f"{basename}_restored.{ext}")
                with open(compiled_ocr_path, "w", encoding="utf-8") as f_compiled:
                    f_compiled.write(f"# DOCUMENT OCR TRANSCRIPTION: {filename}\n\n" + "".join(compiled_ocr_texts))
                self.logger.info(f"  Saved compiled multi-page OCR text: {compiled_ocr_path}")

            elapsed = time.time() - start
            self.logger.info(f"Finished PDF: {filename} in {elapsed:.1f}s")
            return True

        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"  FAILED PDF {filename} after {elapsed:.1f}s: {e}", exc_info=True)
            return False

    def restore_batch(self, input_path: str | None = None):
        """Process all images and PDF documents in a directory or a single file."""
        is_single_file = False
        target_is_pdf = False
        files = []
        pdf_files = []

        if input_path and os.path.isfile(input_path):
            is_single_file = True
            if input_path.lower().endswith(".pdf"):
                target_is_pdf = True
                pdf_files = [input_path]
            else:
                files = [input_path]
        else:
            search_dir = input_path or self.input_dir
            # Search for images
            img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp"]
            for ext in img_extensions:
                files.extend(glob.glob(os.path.join(search_dir, ext)))
                files.extend(glob.glob(os.path.join(search_dir, ext.upper())))
            
            # Search for PDFs
            pdf_files.extend(glob.glob(os.path.join(search_dir, "*.pdf")))
            pdf_files.extend(glob.glob(os.path.join(search_dir, "*.PDF")))
            
            files = sorted(set(files))
            pdf_files = sorted(set(pdf_files))

        total_files = len(files) + len(pdf_files)
        if total_files == 0:
            self.logger.error(f"No processable images or PDF files found in: {input_path or self.input_dir}")
            return

        self.logger.info("=" * 60)
        self.logger.info("Document Restoration & Chandra OCR Pipeline")
        self.logger.info(f"Input:  {input_path or self.input_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
        if is_single_file:
            target_name = pdf_files[0] if target_is_pdf else files[0]
            self.logger.info(f"Target: Single file ({target_name})")
        else:
            self.logger.info(f"Files:  {len(files)} images, {len(pdf_files)} PDFs")
        self.logger.info("=" * 60)

        total_start = time.time()
        success = 0
        failed = 0

        # Process PDFs
        for i, fpath in enumerate(pdf_files, 1):
            self.logger.info(f"\n[PDF {i}/{len(pdf_files)}] {'-' * 50}")
            pdf_success = self.restore_pdf(fpath)
            if pdf_success:
                success += 1
            else:
                failed += 1

        # Process images
        for i, fpath in enumerate(files, 1):
            self.logger.info(f"\n[Image {i}/{len(files)}] {'-' * 50}")
            result = self.restore_single(fpath, base_input_dir=search_dir if not is_single_file else None)
            if result is not None:
                success += 1
            else:
                failed += 1

        total_elapsed = time.time() - total_start
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Batch complete: {success} succeeded, {failed} failed")
        self.logger.info(f"Total time: {total_elapsed:.1f}s")
        self.logger.info(f"Outputs saved to: {self.output_dir}")
        self.logger.info("=" * 60)

    def _save_with_dpi(self, image: np.ndarray, path: str, dpi: int):
        """Save image with DPI metadata embedded."""
        pil_img = Image.fromarray(image)
        pil_img.save(path, dpi=(dpi, dpi))


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Document Restoration Pipeline -- Restore degraded scanned documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python restore_pipeline.py --config config.yaml
  python restore_pipeline.py --config config.yaml --input "path/to/single_image.png"
  python restore_pipeline.py --config config.yaml --input "path/to/folder/"
        """,
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file (default: config.yaml)")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input path (single file or directory)")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    restorer = DocumentRestorer(args.config)
    restorer.restore_batch(args.input)


if __name__ == "__main__":
    main()
