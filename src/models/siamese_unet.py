import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net baseline for binary change detection."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.pool = nn.MaxPool2d(2)

        # Shared encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        # Decoder: concatenated channel counts are explicit.
        self.dec3 = ConvBlock(c3 + c2, c2)
        self.dec2 = ConvBlock(c2 + c1, c1)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> torch.Tensor:
        if image_a.shape != image_b.shape:
            raise ValueError(
                "Input images must have identical shapes."
            )

        a1, a2, a3, _ = self.encode(image_a)
        b1, b2, b3, _ = self.encode(image_b)

        d1 = torch.abs(a1 - b1)
        d2 = torch.abs(a2 - b2)
        d3 = torch.abs(a3 - b3)

        x = F.interpolate(
            d3,
            size=d2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.dec3(torch.cat([x, d2], dim=1))

        x = F.interpolate(
            x,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.dec2(torch.cat([x, d1], dim=1))

        return self.head(x)