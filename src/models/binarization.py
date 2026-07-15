import torch
import torch.nn as nn
import torch.nn.functional as F

class AddCoords(nn.Module):
    """
    Appends normalized X and Y coordinate grids as extra channels.
    Allows the model to learn location-dependent biases like illumination gradients.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, h, w = x.size()
        
        # Create coordinate channels
        xx_channel = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        yy_channel = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        
        # Normalize to [-1, 1]
        xx_channel = (xx_channel / (w - 1)) * 2.0 - 1.0
        yy_channel = (yy_channel / (h - 1)) * 2.0 - 1.0
        
        # Concatenate coordinate channels with input
        out = torch.cat([x, xx_channel, yy_channel], dim=1)
        return out


class CoordConv2d(nn.Module):
    """Convolutional layer with AddCoords coordinate injection."""
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.add_coords = AddCoords()
        # In channels increase by 2 (for X and Y coordinates)
        self.conv = nn.Conv2d(in_channels + 2, out_channels, *args, **kwargs)

    def forward(self, x):
        return self.conv(self.add_coords(x))


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, use_coordconv=False):
        super().__init__()
        conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) if use_coordconv else \
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.double_conv = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BinarizationNet(nn.Module):
    """
    CoordConv-enabled U-Net architecture for robust document text binarization.
    Outputs a single-channel logits map.
    """
    def __init__(self, in_channels=3, init_features=32):
        super().__init__()
        
        # Encoder: use CoordConv in the first block to absorb lighting gradient coordinate cues
        self.conv1 = DoubleConv(in_channels, init_features, use_coordconv=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 4, init_features * 8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 8, init_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 2, init_features)
        
        # Output Logits (Single channel for binary classification)
        self.out_conv = nn.Conv2d(init_features, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = self.conv_up3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.conv_up2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.conv_up1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)
