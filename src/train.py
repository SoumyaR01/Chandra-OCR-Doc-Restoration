import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DocumentRestorationPipeline
from losses.edge_loss import EdgeAwareLoss
from losses.hybrid_loss import HybridBinarizationLoss
from dataset import DocumentRestorationDataset

def train_pipeline(args):
    # Create checkpoint directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from: {args.clean_dir}")
    dataset = DocumentRestorationDataset(
        clean_dir=args.clean_dir, 
        patch_size=args.patch_size, 
        is_training=True
    )
    if len(dataset) == 0:
        print("Error: Dataset is empty. Make sure you provide a folder containing clean document images.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = DocumentRestorationPipeline().to(device)
    
    # Initialize losses
    criterion_bleed = nn.L1Loss()
    criterion_binarization = HybridBinarizationLoss()
    criterion_edge = EdgeAwareLoss().to(device)
    criterion_sharpening = nn.L1Loss()

    # Configure Optimizer
    # We can separate optimizer weights for phase training or train everything end-to-end
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load checkpoint if exists
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    print("Starting training...")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_loss_bleed = 0.0
        epoch_loss_bin = 0.0
        epoch_loss_enhance = 0.0
        epoch_loss_sharp = 0.0

        for i, (degraded, target, mask) in enumerate(dataloader):
            degraded = degraded.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # Forward pass
            out = model(degraded)

            # 1. Bleed suppression loss (reconstruction loss with clean page)
            loss_bleed = criterion_bleed(out["suppressed"], target)

            # 2. Binarization loss (hybrid dice + focal loss)
            loss_bin = criterion_binarization(out["binarization_logits"], mask)

            # 3. Faint stroke enhancement loss (mask-weighted L1 loss)
            # Focuses optimization on the text strokes (guided by soft mask)
            # Mask weight contains an epsilon to keep background supervised but with lower weight
            mask_weight = out["binarization_mask"] + 0.1
            loss_enhance = torch.mean(mask_weight * torch.abs(out["enhanced"] - target))

            # 4. Sharpening & Stroke reconnection loss (combined L1 + Edge-Aware Gradient Loss)
            loss_edge = criterion_edge(out["final"], target)
            loss_reconstruct = criterion_sharpening(out["final"], target)
            loss_sharp = loss_reconstruct + 0.5 * loss_edge

            # Multi-task loss combining stages
            # Coefficients can be adjusted depending on modular priorities
            total_loss = (
                args.w_bleed * loss_bleed +
                args.w_bin * loss_bin +
                args.w_enhance * loss_enhance +
                args.w_sharp * loss_sharp
            )

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Record stats
            epoch_loss += total_loss.item()
            epoch_loss_bleed += loss_bleed.item()
            epoch_loss_bin += loss_bin.item()
            epoch_loss_enhance += loss_enhance.item()
            epoch_loss_sharp += loss_sharp.item()

            if (i + 1) % args.log_step == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], "
                    f"Loss: {total_loss.item():.4f} | Bleed: {loss_bleed.item():.4f} | "
                    f"Bin: {loss_bin.item():.4f} | Enhance: {loss_enhance.item():.4f} | "
                    f"Sharp: {loss_sharp.item():.4f}"
                )

        scheduler.step()
        
        # Log epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(
            f"=== Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} | "
            f"Bleed: {epoch_loss_bleed/len(dataloader):.4f} | Bin: {epoch_loss_bin/len(dataloader):.4f} | "
            f"Enhance: {epoch_loss_enhance/len(dataloader):.4f} | Sharp: {epoch_loss_sharp/len(dataloader):.4f} ==="
        )

        # Save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pipeline_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "pipeline_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model weights to: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Character-Aware Text Restoration Pipeline")
    
    # Path configuration
    parser.add_argument('--clean_dir', type=str, required=True, help="Directory containing clean document scans for synthetic degradation")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save weights/checkpoints")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint file to resume from")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument('--patch_size', type=int, default=512, help="Patch size for training crop")
    parser.add_argument('--num_workers', type=int, default=2, help="Dataloader multi-processing workers")

    # Loss weights configuration
    parser.add_argument('--w_bleed', type=float, default=1.0, help="Weight factor for Bleed-Through Suppression module loss")
    parser.add_argument('--w_bin', type=float, default=1.0, help="Weight factor for Binarization mask module loss")
    parser.add_argument('--w_enhance', type=float, default=1.5, help="Weight factor for Faint Stroke Enhancement module loss")
    parser.add_argument('--w_sharp', type=float, default=2.0, help="Weight factor for Character Sharpening module loss")

    # Logging and checkpoints frequency
    parser.add_argument('--log_step', type=int, default=10, help="Log information frequency step size")
    parser.add_argument('--save_epoch', type=int, default=5, help="Epoch frequency to save checkpoints")

    args = parser.parse_args()
    train_pipeline(args)
