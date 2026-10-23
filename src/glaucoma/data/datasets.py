"""PyTorch datasets. Images are read from the split CSVs; augmentation only when train=True."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..cdr import masks_from_refuge
from ..preprocessing import augment, load_image, resize_square, to_chw, to_float01
from .index import absolute

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageDataset(Dataset):
    """Returns (image tensor, label, row index). Used by the autoencoder (no normalisation) and baseline (ImageNet)."""

    def __init__(self, df: pd.DataFrame, size: int, train: bool = False, imagenet_norm: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.size, self.train, self.imagenet_norm = size, train, imagenet_norm
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = resize_square(load_image(absolute(row["path"])), self.size)
        if self.train:
            img = augment(img, self.rng)
        f = to_float01(img)
        if self.imagenet_norm:
            f = (f - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(to_chw(f))), int(row["label"]), int(i)


class SegDataset(Dataset):
    """Returns (image tensor, target (2,H,W) with channel 0 = disc, 1 = cup)."""

    def __init__(self, df: pd.DataFrame, size: int, train: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.size, self.train = size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        import cv2

        row = self.df.iloc[i]
        img = resize_square(load_image(absolute(row["image"])), self.size)
        mask = cv2.imread(str(absolute(row["mask"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {row['mask']}")
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        disc, cup = masks_from_refuge(mask)
        if self.train and self.rng.random() < 0.5:
            img, disc, cup = img[:, ::-1], disc[:, ::-1], cup[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(to_chw(to_float01(img))))
        y = torch.from_numpy(np.stack([disc, cup]).astype(np.float32).copy())
        return x, y
