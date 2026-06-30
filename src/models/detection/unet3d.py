"""
unet3d.py
─────────
3D U-Net for volumetric cell segmentation.

Architecture
────────────
Encoder → Bottleneck → Decoder with skip connections.
Each block: Conv3d → Norm → Activation (×2).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv3d-Norm-Activation blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "instance",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _get_norm(norm, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _get_norm(norm, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, features: List[int], norm: str, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        for i in range(len(features) - 1):
            self.blocks.append(ConvBlock(features[i], features[i + 1], norm, dropout))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, features: List[int], norm: str, dropout: float) -> None:
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        rev = list(reversed(features))
        for i in range(len(rev) - 2):
            self.upconvs.append(
                nn.ConvTranspose3d(rev[i], rev[i + 1], kernel_size=2, stride=2)
            )
            self.blocks.append(ConvBlock(rev[i], rev[i + 1], norm, dropout))

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor]
    ) -> torch.Tensor:
        for up, block, skip in zip(self.upconvs, self.blocks, reversed(skips)):
            x = up(x)
            # Handle size mismatches from odd-dimension inputs
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """3D U-Net for semantic / instance segmentation.

    Parameters
    ----------
    in_channels:
        Number of input channels (typically 1 for single-channel microscopy).
    out_channels:
        Number of output classes (e.g., 2 for binary foreground/background).
    features:
        Channel progression at each encoder stage.
    norm:
        Normalization type: ``"batch"``, ``"instance"``, or ``"group"``.
    dropout:
        Dropout probability applied after each ConvBlock.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: List[int] = [32, 64, 128, 256],
        norm: str = "instance",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        enc_features = [in_channels] + features
        self.input_block = ConvBlock(in_channels, features[0], norm, dropout)
        self.encoder = Encoder(enc_features, norm, dropout)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, norm, dropout)
        dec_features = [features[-1] * 2] + list(reversed(features))
        self.decoder = Decoder(dec_features, norm, dropout)
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_block(x)
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_norm(norm: str, num_features: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm3d(num_features)
    elif norm == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    elif norm == "group":
        return nn.GroupNorm(8, num_features)
    else:
        raise ValueError(f"Unknown normalization: {norm!r}")
