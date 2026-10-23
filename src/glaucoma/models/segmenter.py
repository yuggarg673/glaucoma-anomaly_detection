"""Small U-Net for optic disc and optic cup segmentation (2 output channels: disc, cup)."""
from __future__ import annotations

import torch
from torch import nn

from .autoencoder import conv_block


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 16, out_ch: int = 2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.factor = 8
        self.d1 = nn.Sequential(conv_block(in_ch, c[0]), conv_block(c[0], c[0]))
        self.d2 = nn.Sequential(conv_block(c[0], c[1]), conv_block(c[1], c[1]))
        self.d3 = nn.Sequential(conv_block(c[1], c[2]), conv_block(c[2], c[2]))
        self.mid = nn.Sequential(conv_block(c[2], c[3]), conv_block(c[3], c[3]))
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.e3 = nn.Sequential(conv_block(c[2] * 2, c[2]), conv_block(c[2], c[2]))
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.e2 = nn.Sequential(conv_block(c[1] * 2, c[1]), conv_block(c[1], c[1]))
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.e1 = nn.Sequential(conv_block(c[0] * 2, c[0]), conv_block(c[0], c[0]))
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.factor or w % self.factor:
            raise ValueError(f"Input size {h}x{w} must be divisible by {self.factor}.")
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        m = self.mid(self.pool(x3))
        y = self.e3(torch.cat([self.u3(m), x3], 1))
        y = self.e2(torch.cat([self.u2(y), x2], 1))
        y = self.e1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)  # logits


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel Dice over the batch: tensor of shape (C,)."""
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum(dim=(0, 2, 3))
    union = p.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return (2 * inter + eps) / (union + eps)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(0, 2, 3))
    union = p.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * inter + eps) / (union + eps)).mean()
    return bce + dice
