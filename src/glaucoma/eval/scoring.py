"""Anomaly score from a per-pixel reconstruction error map (numpy only)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def anomaly_score(error_map: np.ndarray, disc_mask: Optional[np.ndarray] = None, disc_only: bool = False) -> float:
    """Mean absolute reconstruction error. If disc_only and a disc mask is given, average inside the disc region."""
    if disc_only and disc_mask is not None and disc_mask.any():
        m = cv2.resize((disc_mask > 0).astype(np.uint8), (error_map.shape[1], error_map.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.any():
            return float(error_map[m].mean())
    return float(error_map.mean())
