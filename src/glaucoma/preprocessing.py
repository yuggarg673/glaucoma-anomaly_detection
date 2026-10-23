"""Image loading and preprocessing (numpy/OpenCV only, no PyTorch needed)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """Raised when bytes or a file cannot be decoded as an image."""


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(f"Could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes (jpg/png) as RGB uint8."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        raise ImageDecodeError("The uploaded file is empty.")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("The file is not a readable JPG or PNG image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_square(img: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize to size x size (stretches non-square images)."""
    return cv2.resize(img, (size, size), interpolation=interpolation)


def to_float01(img: np.ndarray) -> np.ndarray:
    """uint8 HxWxC -> float32 in [0, 1]."""
    return img.astype(np.float32) / 255.0


def to_chw(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (2, 0, 1))


def crop_to_mask(img: np.ndarray, mask: np.ndarray, margin: float = 0.5) -> np.ndarray:
    """Crop around the bounding box of a binary mask, with a relative margin. Used for optic-disc ROI."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Mask is empty; cannot crop the region of interest.")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * (1 + margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max(side // 2, 1)
    ya, yb = max(cy - half, 0), min(cy + half, img.shape[0])
    xa, xb = max(cx - half, 0), min(cx + half, img.shape[1])
    return img[ya:yb, xa:xb]


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training-time augmentation: flip, small rotation, brightness/contrast. Never use on val/test."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    angle = float(rng.uniform(-15, 15))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha = float(rng.uniform(0.85, 1.15))   # contrast
    beta = float(rng.uniform(-15, 15))        # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img
