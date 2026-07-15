import torch
import torch.nn as nn
from .bleed_suppression import BleedSuppressionNet
from .binarization import BinarizationNet
from .enhancement import FaintStrokeEnhancementNet
from .sharpening import SharpeningNet

class DocumentRestorationPipeline(nn.Module):
    """
    End-to-End Character-Aware Restoration Pipeline.
    Integrates all four sub-networks:
      1. BleedSuppressionNet: Removes reverse-page ghosting.
      2. BinarizationNet: Segments text from background, outputting raw logits.
      3. FaintStrokeEnhancementNet: Restores local contrast of text regions guided by the binarization mask.
      4. SharpeningNet: Reconnects broken loops and sharpens character boundaries.
    """
    def __init__(self, init_features=32):
        super(DocumentRestorationPipeline, self).__init__()
        self.bleed_suppression = BleedSuppressionNet(in_channels=3, out_channels=3, init_features=init_features)
        self.binarization = BinarizationNet(in_channels=3, init_features=init_features)
        self.faint_enhancement = FaintStrokeEnhancementNet(num_blocks=4, init_features=init_features)
        self.character_sharpening = SharpeningNet(in_channels=3, out_channels=3, init_features=init_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw degraded input image [B, 3, H, W]
        Returns:
            dict: Dictionary containing:
                "suppressed": Grayscale/RGB image with bleed-through removed
                "binarization_logits": Raw binarization outputs (for loss computing)
                "binarization_mask": Sigmoid probability map of the text strokes
                "enhanced": Faint stroke enhanced image
                "final": Final sharp restored output image
        """
        # 1. Bleed-Through Suppression
        suppressed = self.bleed_suppression(x)
        
        # 2. Binarization & Soft Mask extraction
        bin_logits = self.binarization(suppressed)
        soft_mask = torch.sigmoid(bin_logits)
        
        # 3. Faint Stroke Enhancement
        # Concat suppressed image and mask to guide enhancement spatially
        enhancement_input = torch.cat([suppressed, soft_mask], dim=1)
        enhanced = self.faint_enhancement(enhancement_input)
        
        # 4. Character Sharpening & Stroke Reconnection
        final_output = self.character_sharpening(enhanced)
        
        return {
            "suppressed": suppressed,
            "binarization_logits": bin_logits,
            "binarization_mask": soft_mask,
            "enhanced": enhanced,
            "final": final_output
        }
