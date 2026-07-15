import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes Dice Loss to optimize the overlap of predicted binary text segmentation masks
    with ground truth, particularly effective at preserving thin character strokes.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_probs, targets):
        """
        Args:
            pred_probs (torch.Tensor): Soft predictions after sigmoid of shape [B, 1, H, W]
            targets (torch.Tensor): Binary target mask of shape [B, 1, H, W]
        """
        pred_flat = pred_probs.view(-1)
        target_flat = targets.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss to focus training on hard-to-classify pixels (e.g., character boundaries, 
    low-contrast faint strokes) by down-weighting easy background pixels.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_probs, targets):
        """
        Args:
            pred_probs (torch.Tensor): Soft predictions after sigmoid of shape [B, 1, H, W]
            targets (torch.Tensor): Binary target mask of shape [B, 1, H, W]
        """
        pred_flat = pred_probs.view(-1)
        target_flat = targets.view(-1)
        
        # Clip predictions to prevent log(0) instabilities
        pred_flat = torch.clamp(pred_flat, 1e-7, 1.0 - 1e-7)
        
        # Compute focal loss terms
        loss_pos = -self.alpha * ((1.0 - pred_flat) ** self.gamma) * torch.log(pred_flat) * target_flat
        loss_neg = -(1.0 - self.alpha) * (pred_flat ** self.gamma) * torch.log(1.0 - pred_flat) * (1.0 - target_flat)
        
        loss = loss_pos + loss_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HybridBinarizationLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss to balance pixel-level classification accuracy
    and character stroke shape overlaps.
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_smooth=1e-5, weight_focal=1.0, weight_dice=1.0):
        super(HybridBinarizationLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw logits before sigmoid of shape [B, 1, H, W]
            targets (torch.Tensor): Target mask of shape [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        loss_f = self.focal(probs, targets)
        loss_d = self.dice(probs, targets)
        return self.weight_focal * loss_f + self.weight_dice * loss_d
