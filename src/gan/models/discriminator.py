# src/gan/models/discriminator.py
"""
Patch Discriminator for Medical WGAN-GP
======================================

Improvements:
- PatchGAN (focus on local texture realism)
- Spectral Normalization for stability
- No BatchNorm / InstanceNorm (WGAN-GP friendly)
- Designed for single-class GAN (no conditional labels)

Best paired with:
- Upsample + Conv Generator
- Progressive training
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# ======================================================
# Patch Discriminator
# ======================================================
class Discriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        image_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        assert image_size % 16 == 0, "image_size must be divisible by 16"

        # --------------------------------------------------
        # Patch-based convolutional backbone
        # 128 → 64 → 32 → 16 → 8
        # --------------------------------------------------
        self.features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(image_channels, base_channels, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --------------------------------------------------
        # Patch output head
        # Output shape: (B, 1, H_patch, W_patch)
        # --------------------------------------------------
        self.patch_out = spectral_norm(
            nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)
        )

        self._initialize_weights()

    # --------------------------------------------------
    def forward(self, img: torch.Tensor):
        """
        Args:
            img: (B, C, H, W)
        Returns:
            Patch-wise critic scores (B, H_patch, W_patch)
        """
        x = self.features(img)
        x = self.patch_out(x)
        return x.squeeze(1)

    # --------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
