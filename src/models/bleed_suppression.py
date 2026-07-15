import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate to filter skip connection features using a gating signal from coarser scales.
    Helps isolate foreground text from low-contrast background bleed-through.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g is gating signal (coarse scale), x is skip connection (fine scale)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Align spatial dimensions of gating signal and skip connection if different
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class BleedSuppressionNet(nn.Module):
    """
    Attention U-Net architecture designed for bleed-through suppression.
    Learns to isolate and suppress back-page text ghosting using multi-scale context.
    """
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 4, init_features * 8)
        
        # Decoder skip-connection attention gates
        self.att3 = AttentionGate(F_g=init_features * 8, F_l=init_features * 4, F_int=init_features * 4)
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 8, init_features * 4)
        
        self.att2 = AttentionGate(F_g=init_features * 4, F_l=init_features * 2, F_int=init_features * 2)
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.att1 = AttentionGate(F_g=init_features * 2, F_l=init_features, F_int=init_features)
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 2, init_features)
        
        # Output prediction
        self.out_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder passes with skip-connection attention filtering
        g3 = self.upconv3(b)
        x3 = self.att3(g=b, x=e3)
        d3 = self.conv_up3(torch.cat([g3, x3], dim=1))
        
        g2 = self.upconv2(d3)
        x2 = self.att2(g=d3, x=e2)
        d2 = self.conv_up2(torch.cat([g2, x2], dim=1))
        
        g1 = self.upconv1(d2)
        x1 = self.att1(g=d2, x=e1)
        d1 = self.conv_up1(torch.cat([g1, x1], dim=1))
        
        # Output with residual learning (predict modification to input to make training easier)
        out = self.out_conv(d1)
        return torch.sigmoid(x + out) # Residual skip connection bounded to [0,1]
