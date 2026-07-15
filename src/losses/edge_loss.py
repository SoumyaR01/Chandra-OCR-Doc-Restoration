import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareLoss(nn.Module):
    """
    Computes first-order (Sobel) and second-order (Laplacian) gradient losses
    between predicted and target images to enforce sharp edges and preserve
    fine character strokes/topology.
    """
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        # Sobel filters for x and y gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Laplacian filter for second-order edges (curvature/corners)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Model prediction tensor of shape [B, C, H, W]
            target (torch.Tensor): Target clean image tensor of shape [B, C, H, W]
        """
        # Convert to grayscale if inputs are RGB to calculate gradients on intensity
        if pred.shape[1] == 3:
            pred = 0.2989 * pred[:, 0:1, :, :] + 0.5870 * pred[:, 1:2, :, :] + 0.1140 * pred[:, 2:3, :, :]
            target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:3, :, :]

        # Compute Sobel gradients
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

        # Compute Laplacian (second-order gradients)
        lap_pred = F.conv2d(pred, self.laplacian, padding=1)
        lap_target = F.conv2d(target, self.laplacian, padding=1)

        # L1 losses on gradients
        loss_sobel = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)
        loss_laplacian = F.l1_loss(lap_pred, lap_target)

        # Return weighted sum of Sobel and Laplacian losses
        return loss_sobel + 0.5 * loss_laplacian
