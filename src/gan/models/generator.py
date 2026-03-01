# src/gan/models/generator.py

"""
Improved Generator for Medical Image GAN (WGAN-GP)
=================================================

Designed for:
- Skin lesion images (HAM10000)
- Resolution: 128x128
- High-frequency texture preservation
- Stable training with WGAN-GP

Key features:
- Upsample + Conv2d (no checkerboard)
- Residual refinement for sharp textures
- InstanceNorm (no BatchNorm)
"""

import torch
import torch.nn as nn


# ======================================================
# Upsampling Block
# ======================================================
class GenBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.block(x)


# ======================================================
# Residual Refinement Block (IMPORTANT)
# ======================================================
class ResidualRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ======================================================
# Generator (128x128)
# ======================================================
class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_size: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()

        if image_size != 128:
            raise ValueError("Generator currently supports image_size=128 only")

        self.latent_dim = latent_dim
        self.init_spatial = image_size // 16  # 8

        # ----------------------------------
        # Latent projection
        # ----------------------------------
        self.fc = nn.Linear(
            latent_dim,
            base_channels * 8 * self.init_spatial * self.init_spatial,
        )

        # ----------------------------------
        # Upsampling path
        # ----------------------------------
        self.up1 = GenBlock(base_channels * 8, base_channels * 4)  # 8 → 16
        self.up2 = GenBlock(base_channels * 4, base_channels * 2)  # 16 → 32
        self.up3 = GenBlock(base_channels * 2, base_channels)      # 32 → 64

        # ----------------------------------
        # High-resolution refinement
        # ----------------------------------
        self.refine = nn.Sequential(
            ResidualRefine(base_channels),
            ResidualRefine(base_channels),
        )

        # ----------------------------------
        # Final upsample + RGB
        # ----------------------------------
        self.to_rgb = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64 → 128
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

        self._initialize_weights()

    # --------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError("z must have shape (B, latent_dim)")

        # Latent normalization (helps stability)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        x = self.fc(z)
        x = x.view(
            z.size(0),
            -1,
            self.init_spatial,
            self.init_spatial,
        )

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.refine(x)
        return self.to_rgb(x)

    # --------------------------------------------------
    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.forward(z)

    # --------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
