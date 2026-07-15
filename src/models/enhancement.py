import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature modulation."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention block to weigh feature maps according to pixel locations."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mean and Max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class AttentiveResidualBlock(nn.Module):
    """Residual block equipped with channel and spatial attention gates."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention modulation
        out = self.ca(out)
        out = self.sa(out)
        
        return out + residual


class FaintStrokeEnhancementNet(nn.Module):
    """
    Spatial-Channel Attention Network (SCAN) for Faint Stroke Enhancement.
    Takes 4-channel input: Suppressed RGB/Grayscale image (3 channels) + Soft Binarization Mask (1 channel).
    Outputs 3-channel contrast-enhanced text regions while preserving clean background areas.
    """
    def __init__(self, num_blocks=4, init_features=32):
        super().__init__()
        
        # Initial projection: 4 channels -> init_features
        self.in_conv = nn.Sequential(
            nn.Conv2d(4, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # Sequence of attentive residual blocks
        self.res_blocks = nn.Sequential(
            *[AttentiveResidualBlock(init_features) for _ in range(num_blocks)]
        )
        
        # Output project: init_features -> 3 channels
        self.out_conv = nn.Conv2d(init_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Concat of image and soft binarization mask of shape [B, 4, H, W]
        """
        # Save original 3-channel input image for residual learning
        orig_img = x[:, :3, :, :]
        
        # Forward pass
        feat = self.in_conv(x)
        feat = self.res_blocks(feat)
        out = self.out_conv(feat)
        
        # Predict an enhancement residual to adjust brightness/contrast locally
        # Bounded between -1 and 1 to allow darkening/brightening adjustments
        enhancement_residual = torch.tanh(out)
        
        # Apply residual modification to original image
        return torch.clamp(orig_img + 0.5 * enhancement_residual, 0.0, 1.0)
