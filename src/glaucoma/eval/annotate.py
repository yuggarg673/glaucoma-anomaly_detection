"""Build the labelled, annotated image shown to the user.

The image is made from the autoencoder's per-pixel reconstruction error and the
optic disc / cup outlines. It shows where the image differs most from healthy
eyes. It is a localisation aid, NOT a confirmed lesion detector.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# colours are RGB here and converted to BGR for OpenCV drawing
RED = (255, 90, 78)
AMBER = (255, 176, 32)
TEAL = (79, 209, 197)
WHITE = (255, 255, 255)
DARK = (16, 24, 24)


def _bgr(c):
    return (c[2], c[1], c[0])


def _dashed_rect(img, x, y, w, h, color, thick, dash):
    def seg(p0, p1):
        length = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        for i in range(0, length, dash * 2):
            a = i / max(length, 1)
            b = min(i + dash, length) / max(length, 1)
            q0 = (int(p0[0] + (p1[0] - p0[0]) * a), int(p0[1] + (p1[1] - p0[1]) * a))
            q1 = (int(p0[0] + (p1[0] - p0[0]) * b), int(p0[1] + (p1[1] - p0[1]) * b))
            cv2.line(img, q0, q1, _bgr(color), thick, cv2.LINE_AA)

    seg((x, y), (x + w, y)); seg((x + w, y), (x + w, y + h))
    seg((x + w, y + h), (x, y + h)); seg((x, y + h), (x, y))


def _chip(img, text, x, y, scale, text_color=WHITE):
    """Draw text on a dark chip so it stays readable on any background. (x, y) = top-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = max(2, int(scale * 6))
    H, W = img.shape[:2]
    x = int(np.clip(x, 0, max(W - tw - 2 * pad, 0)))
    y = int(np.clip(y, 0, max(H - th - base - 2 * pad, 0)))
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), _bgr(DARK), -1)
    cv2.putText(img, text, (x + pad, y + pad + th), font, scale, _bgr(text_color), thick, cv2.LINE_AA)


def smooth_error(error_map: np.ndarray, blur_sigma: float = 4.0) -> np.ndarray:
    """Gaussian-smooth an error map. Sigma scales with width so the smoothing is the same relative size at any
    resolution. Training uses this too, so the percentile thresholds match what annotate() sees."""
    e = error_map.astype(np.float32)
    return cv2.GaussianBlur(e, (0, 0), sigmaX=max(blur_sigma * e.shape[1] / 256.0, 0.5))


def find_regions(error_map: np.ndarray, size_hw: tuple[int, int], moderate_thr: float, high_thr: float,
                 blur_sigma: float = 4.0, max_regions: int = 3, min_area_frac: float = 0.002) -> list[dict]:
    """Connected regions whose smoothed error is above moderate_thr, strongest first."""
    h, w = size_hw
    emap = cv2.resize(error_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    emap = smooth_error(emap, blur_sigma)
    binary = (emap >= moderate_thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    found = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_frac * w * h:
            continue
        pix = emap[labels == i]
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        found.append({
            "box": [x, y, bw, bh],
            "mean_error": float(pix.mean()),
            "max_error": float(pix.max()),
            "severity": "high" if float(pix.max()) >= high_thr else "moderate",
        })
    found.sort(key=lambda r: (r["severity"] != "high", -r["mean_error"]))
    found = found[:max_regions]
    for k, r in enumerate(found, start=1):
        r["id"] = f"A{k}"
    return found


def annotate(image_rgb: np.ndarray, error_map: np.ndarray, moderate_thr: float, high_thr: float,
             disc_mask: Optional[np.ndarray] = None, cup_mask: Optional[np.ndarray] = None,
             cdr_value: Optional[float] = None, blur_sigma: float = 4.0, max_regions: int = 3,
             min_area_frac: float = 0.002) -> tuple[bytes, list[dict]]:
    """Return (PNG bytes, regions). Same pixel size as image_rgb."""
    h, w = image_rgb.shape[:2]
    regions = find_regions(error_map, (h, w), moderate_thr, high_thr, blur_sigma, max_regions, min_area_frac)
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    scale = max(0.4, w / 900.0)
    thick = max(1, int(round(scale * 2)))
    dash = max(4, int(w / 60))

    def draw_mask(mask, color, label):
        if mask is None:
            return
        m = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [big], -1, _bgr(color), thick, cv2.LINE_AA)
        x, y, bw, bh = cv2.boundingRect(big)
        _chip(out, label, x, y - int(24 * scale) - 4, scale * 0.8, color if color != WHITE else WHITE)

    draw_mask(disc_mask, TEAL, "Optic disc")
    cup_label = f"Cup | CDR {cdr_value:.2f}" if cdr_value is not None else "Cup | CDR unavailable"
    draw_mask(cup_mask, WHITE, cup_label)

    for r in regions:
        x, y, bw, bh = r["box"]
        color = RED if r["severity"] == "high" else AMBER
        _dashed_rect(out, x, y, bw, bh, color, thick, dash)
        _chip(out, f'{r["id"]} | {r["severity"]} error', x, y - int(24 * scale) - 4, scale * 0.8)

    _chip(out, "Annotated by model | screening aid", 6, h - int(30 * scale) - 8, scale * 0.7)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("Could not encode the annotated image as PNG.")
    return buf.tobytes(), regions
