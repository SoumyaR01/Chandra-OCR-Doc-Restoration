import os
import argparse
import cv2
import numpy as np
import torch
import glob

from models import DocumentRestorationPipeline
from utils.tiling import tile_inference

def run_inference(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = DocumentRestorationPipeline().to(device)

    # Load weights if checkpoint is provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle dict format vs raw state_dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: No checkpoint weight path provided or file does not exist.")
    model.eval()

    # Find input files
    if os.path.isdir(args.input):
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        input_files = []
        for ext in extensions:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
            input_files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        input_files = list(set(input_files))
    else:
        input_files = [args.input] if os.path.exists(args.input) else []

    if len(input_files) == 0:
        print(f"Error: No valid input files found at {args.input}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(input_files)} files. Processing document restoration pipeline...")

    for i, file_path in enumerate(input_files):
        filename = os.path.basename(file_path)
        basename, ext = os.path.splitext(filename)
        print(f"[{i+1}/{len(input_files)}] Processing {filename}...")

        # Load image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Failed to read image {file_path}")
            continue

        # Convert BGR to RGB and normalize to [0.0, 1.0]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Execute overlapping tile-based inference to handle high resolution safely
        print("  Running final sharpening and reconnection module...")
        restored = tile_inference(
            model=model, 
            image=img_rgb, 
            patch_size=args.patch_size, 
            overlap=args.overlap, 
            device=device
        )

        # Save final output
        restored_bgr = cv2.cvtColor((restored * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(args.output_dir, f"{basename}_restored{ext}")
        cv2.imwrite(output_path, restored_bgr)

        # If requested, save intermediate stages (useful for validation and debugging)
        if args.save_intermediates:
            print("  Running intermediate steps visualization extraction...")
            # Extract bleed-suppressed image
            bleed_suppressed = tile_inference(
                model=lambda x: model(x)["suppressed"],
                image=img_rgb,
                patch_size=args.patch_size,
                overlap=args.overlap,
                device=device
            )
            bleed_bgr = cv2.cvtColor((bleed_suppressed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, f"{basename}_1_bleed_suppressed{ext}"), bleed_bgr)

            # Extract binarization soft mask
            bin_mask = tile_inference(
                model=lambda x: model(x)["binarization_mask"],
                image=img_rgb,
                patch_size=args.patch_size,
                overlap=args.overlap,
                device=device
            )
            # Replicate grayscale mask to 3 channels for visualization
            bin_uint8 = (bin_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, f"{basename}_2_binarization_mask{ext}"), bin_uint8)

            # Extract faint stroke enhanced image
            enhanced = tile_inference(
                model=lambda x: model(x)["enhanced"],
                image=img_rgb,
                patch_size=args.patch_size,
                overlap=args.overlap,
                device=device
            )
            enhanced_bgr = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, f"{basename}_3_enhanced{ext}"), enhanced_bgr)

    print(f"Restoration complete. Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Script for Document Text Restoration Pipeline")
    
    # Path configuration
    parser.add_argument('--input', type=str, required=True, help="Path to input degraded image file or directory containing scans")
    parser.add_argument('--output_dir', type=str, default='./restored_outputs', help="Directory to save restoration results")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to pipeline model weights checkpoint (.pth)")

    # Inference settings
    parser.add_argument('--patch_size', type=int, default=512, help="Patch size for tiled processing")
    parser.add_argument('--overlap', type=int, default=64, help="Border overlap pixels for seamless blending")
    parser.add_argument('--save_intermediates', action='store_true', help="Save bleed suppression, binarization, and enhancement stages")

    args = parser.parse_args()
    run_inference(args)
