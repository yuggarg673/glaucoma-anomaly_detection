"""Convolutional autoencoder trained on healthy fundus images only.

Reconstruction error is the anomaly signal. Skip connections are OFF by default:
they let the network copy glaucoma features straight to the output, which would
hide the anomaly. Turn them on only for an ablation.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch: int = 3, channels=(32, 64, 128, 256), latent_channels: int = 64, skip: bool = False):
        super().__init__()
        self.channels = list(channels)
        self.skip = skip
        self.factor = 2 ** len(self.channels)

        self.enc_a = nn.ModuleList()   # stride-1 block, its output is the skip feature
        self.enc_b = nn.ModuleList()   # stride-2 downsampling block
        prev = in_ch
        for ch in self.channels:
            self.enc_a.append(conv_block(prev, ch))
            self.enc_b.append(conv_block(ch, ch, stride=2))
            prev = ch
        self.to_latent = nn.Conv2d(prev, latent_channels, 1)
        self.from_latent = nn.Conv2d(latent_channels, prev, 1)

        self.dec = nn.ModuleList()
        dec_in = prev
        for ch in reversed(self.channels):
            cin = dec_in + (ch if skip else 0)
            self.dec.append(nn.Sequential(conv_block(cin, ch), conv_block(ch, ch)))
            dec_in = ch
        self.out = nn.Conv2d(dec_in, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.factor or w % self.factor:
            raise ValueError(f"Input size {h}x{w} must be divisible by {self.factor} for this depth.")
        feats = []
        for a, b in zip(self.enc_a, self.enc_b):
            x = a(x)
            feats.append(x)
            x = b(x)
        x = self.from_latent(self.to_latent(x))
        for block, f in zip(self.dec, reversed(feats)):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            if self.skip:
                x = torch.cat([x, f], dim=1)
            x = block(x)
        return torch.sigmoid(self.out(x))

    @torch.no_grad()
    def error_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel absolute reconstruction error averaged over colour channels: (B, H, W)."""
        was_training = self.training
        self.eval()
        err = (self(x) - x).abs().mean(dim=1)
        self.train(was_training)
        return err


def build_autoencoder(cfg: dict) -> ConvAutoencoder:
    a = cfg["autoencoder"]
    return ConvAutoencoder(channels=a["channels"], latent_channels=a["latent_channels"], skip=a["skip_connections"])
