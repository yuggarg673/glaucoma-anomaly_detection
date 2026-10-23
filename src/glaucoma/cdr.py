"""Cup-to-disc ratio (CDR) from optic disc and cup masks."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np


@dataclass
class CDRResult:
    value: Optional[float]            # None when it cannot be computed
    reason: Optional[str]             # why it failed (None when value is set)
    disc_height: int = 0
    cup_height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def vertical_extent(mask: np.ndarray) -> int:
    """Height in pixels of the bounding box of a non-empty binary mask."""
    ys = np.where(mask.any(axis=1))[0]
    return int(ys.max() - ys.min() + 1) if ys.size else 0


def compute_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray, min_disc_height: int = 4) -> CDRResult:
    """CDR = cup height / disc height, using the largest component of each mask.

    Returns value=None plus a reason instead of inventing a number when masks are
    empty, too small, or inconsistent.
    """
    disc = largest_component(disc_mask)
    cup = largest_component(cup_mask)
    dh, ch = vertical_extent(disc), vertical_extent(cup)
    if dh == 0:
        return CDRResult(None, "optic disc not found", dh, ch)
    if dh < min_disc_height:
        return CDRResult(None, "optic disc too small to measure", dh, ch)
    if ch == 0:
        return CDRResult(None, "optic cup not found", dh, ch)
    if ch > dh:
        return CDRResult(None, "cup is taller than disc (segmentation error)", dh, ch)
    return CDRResult(round(ch / dh, 4), None, dh, ch)


def masks_from_refuge(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a REFUGE annotation (cup=0, disc rim=128, background=255) into disc and cup masks.

    VERIFY this encoding with scripts/inspect_data.py after downloading the data.
    The disc mask includes the cup (cup pixels are inside the disc).
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    disc = (mask < 192).astype(np.uint8)
    cup = (mask < 64).astype(np.uint8)
    return disc, cup
