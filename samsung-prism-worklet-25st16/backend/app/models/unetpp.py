"""U-Net++ (nested U-Net) for binary leaf/background segmentation.

Implements the dense nested skip pathway from Zhou et al. 2018. Deep
supervision is available for training; inference uses the final head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """Nested U-Net with optional deep supervision.

    Args:
        in_channels: input image channels (3 for RGB).
        num_classes: output channels (1 for a binary mask).
        widths: encoder channel widths per depth level.
        deep_supervision: return all four heads instead of only the last one.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        widths: tuple = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        w = widths
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Backbone column (j = 0)
        self.conv0_0 = ConvBlock(in_channels, w[0])
        self.conv1_0 = ConvBlock(w[0], w[1])
        self.conv2_0 = ConvBlock(w[1], w[2])
        self.conv3_0 = ConvBlock(w[2], w[3])
        self.conv4_0 = ConvBlock(w[3], w[4])

        # Nested decoder columns
        self.conv0_1 = ConvBlock(w[0] + w[1], w[0])
        self.conv1_1 = ConvBlock(w[1] + w[2], w[1])
        self.conv2_1 = ConvBlock(w[2] + w[3], w[2])
        self.conv3_1 = ConvBlock(w[3] + w[4], w[3])

        self.conv0_2 = ConvBlock(w[0] * 2 + w[1], w[0])
        self.conv1_2 = ConvBlock(w[1] * 2 + w[2], w[1])
        self.conv2_2 = ConvBlock(w[2] * 2 + w[3], w[2])

        self.conv0_3 = ConvBlock(w[0] * 3 + w[1], w[0])
        self.conv1_3 = ConvBlock(w[1] * 3 + w[2], w[1])

        self.conv0_4 = ConvBlock(w[0] * 4 + w[1], w[0])

        if deep_supervision:
            self.heads = nn.ModuleList(
                [nn.Conv2d(w[0], num_classes, 1) for _ in range(4)]
            )
        else:
            self.head = nn.Conv2d(w[0], num_classes, 1)

    def _up_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Upsample x to the spatial size of ref (handles odd sizes)."""
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_0)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_0)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_to(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_to(x3_1, x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_to(x2_2, x1_0)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_to(x1_3, x0_0)], 1)
        )

        if self.deep_supervision:
            return [
                self.heads[0](x0_1),
                self.heads[1](x0_2),
                self.heads[2](x0_3),
                self.heads[3](x0_4),
            ]
        return self.head(x0_4)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Section 6 of the spec: BCE + Dice."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter / union if union else 1.0


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    num = 2 * (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return num / den if den else 1.0
