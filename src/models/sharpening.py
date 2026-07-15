import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Standard residual block to maintain high-frequency information."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))


class SharpeningNet(nn.Module):
    """
    Character Sharpening Network.
    Uses an encoder-decoder architecture with dense residual skip connections
    to reconstruct crisp, high-contrast character edges and reconnect broken strokes.
    """
    def __init__(self, in_channels=3, out_channels=3, init_features=32, num_res_blocks=4):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(init_features, init_features * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_features * 4),
            nn.ReLU(inplace=True)
        )
        
        # Deep Residual Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock(init_features * 4) for _ in range(num_res_blocks)]
        )
        
        # Decoder Upsampling (PixelShuffle for artifact-free upscaling)
        self.up2 = nn.Sequential(
            nn.Conv2d(init_features * 4, init_features * 8, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        self.conv_up2 = ResBlock(init_features * 2)
        
        self.up1 = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True)
        )
        self.conv_up1 = ResBlock(init_features)
        
        # Output prediction
        self.conv_out = nn.Conv2d(init_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Enhanced image tensor [B, 3, H, W]
        """
        # Encoder
        x_in = self.conv_in(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        
        # Bottleneck
        b = self.bottleneck(x_d2)
        
        # Decoder with pixel-shuffle upsampling and skip-connection summation
        u2 = self.up2(b)
        u2 = self.conv_up2(u2 + x_d1) # Add skip connection
        
        u1 = self.up1(u2)
        u1 = self.conv_up1(u1 + x_in) # Add skip connection
        
        # Final residual prediction added to input image
        out = self.conv_out(u1)
        return torch.sigmoid(x + out)
