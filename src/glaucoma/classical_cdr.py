"""EXPERIMENTAL, NOT WIRED INTO THE APP. Mean absolute error was 0.24 on simple synthetic drawings with vessels
(accurate only without vessels), so it is not reliable. Kept as a starting point; do not report its output as CDR.

Cup-to-disc ratio WITHOUT a trained segmenter (classical image processing, numpy + OpenCV only).

Why it exists: ACRIMA has no disc/cup masks, so a U-Net cannot be trained from it. This estimates the optic disc
and cup directly from the image. It is designed for images cropped around the optic disc, like ACRIMA's.

How it works
  disc: smooth the luminance, Otsu threshold, keep the large compact bright region near the centre, take its convex hull
  cup : inside the disc, Otsu threshold on the green channel (the cup is the palest part), keep the largest bright
        blob, take its convex hull
  CDR : cup height / disc height (same definition as the trained version)

Honest limits: it assumes the disc is the brightest compact region and the cup is paler than the rim. Glare,
peripapillary atrophy, big vessels or poor contrast can make it wrong. It returns None with a reason instead of
inventing a number when it cannot find a disc or a distinguishable cup. Always look at the drawn outlines.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .cdr import CDRResult, compute_cdr

WORK_SIZE = 256            # all processing happens at this size; masks are resized back at the end
MIN_DISC_FRAC = 0.03       # disc must cover at least 3 % of the image
MAX_DISC_FRAC = 0.75       # ... and at most 75 % (else the threshold just split the image in half)
CENTER_FRAC = 0.8          # disc centroid must lie inside the central 80 % of the image
MIN_CUP_CONTRAST = 15      # grey levels between cup and rim in the green channel
MIN_CUP_FRAC, MAX_CUP_FRAC = 0.02, 0.95   # cup area relative to disc area


def _hull_mask(mask: np.ndarray) -> np.ndarray:
    """Filled convex hull of the largest contour: removes notches cut by vessels."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask, dtype=np.uint8)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out


def _central_component(mask: np.ndarray, min_frac: float) -> Optional[np.ndarray]:
    h, w = mask.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best, best_area = None, 0
    lo, hi = (1 - CENTER_FRAC) / 2, 1 - (1 - CENTER_FRAC) / 2
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = cents[i]
        if area < min_frac * h * w or not (lo * w <= cx <= hi * w and lo * h <= cy <= hi * h):
            continue
        if area > best_area:
            best, best_area = i, area
    return None if best is None else (labels == best).astype(np.uint8)


def segment_disc_cup(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (disc_mask, cup_mask, reason) at the input size. reason is None on success, else why it failed.
    A failed cup returns an empty cup mask but may still return the disc."""
    h, w = rgb.shape[:2]
    img = cv2.resize(rgb, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    empty = np.zeros((h, w), np.uint8)

    def back(m: np.ndarray) -> np.ndarray:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    lum = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 3)
    if lum.max() - lum.min() < 10:
        return empty, empty, "optic disc not found (image has almost no contrast)"
    thr, _ = cv2.threshold(lum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (lum > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright = cv2.morphologyEx(cv2.morphologyEx(bright, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    comp = _central_component(bright, MIN_DISC_FRAC)
    if comp is None:
        return empty, empty, "optic disc not found"
    disc = _hull_mask(comp)
    frac = disc.mean()
    if frac > MAX_DISC_FRAC or frac < MIN_DISC_FRAC:
        return empty, empty, "optic disc could not be separated from the background"

    green = cv2.GaussianBlur(img[..., 1], (0, 0), 2)
    vals = green[disc > 0]
    # Two-level Otsu: the first split removes dark vessels from the disc, the second separates cup from rim.
    t1, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = vals[vals > t1]
    if tissue.size < 20:
        return back(disc), empty, "optic cup not distinguishable (low contrast)"
    t, _ = cv2.threshold(tissue.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi, lo = tissue[tissue > t], tissue[tissue <= t]
    if hi.size == 0 or lo.size == 0 or float(hi.mean() - lo.mean()) < MIN_CUP_CONTRAST:
        return back(disc), empty, "optic cup not distinguishable (low contrast)"
    cup = ((green > t) & (disc > 0)).astype(np.uint8)
    cup = cv2.morphologyEx(cup, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cup_c = _central_component(cup, 0.0)
    if cup_c is None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cup, connectivity=8)
        if n <= 1:
            return back(disc), empty, "optic cup not found"
        cup_c = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # Vessels can split the cup into pieces: keep pieces near the main one (within the disc) and take the hull.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cup, connectivity=8)
    main_area = int(cup_c.sum())
    keep = cup_c.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.1 * main_area:
            keep |= (labels == i).astype(np.uint8)
    cup = _hull_mask(keep) & disc
    ratio = cup.sum() / max(disc.sum(), 1)
    if ratio < MIN_CUP_FRAC or ratio > MAX_CUP_FRAC:
        return back(disc), empty, "optic cup size is implausible"
    return back(disc), back(cup), None


def estimate_cdr_classical(rgb: np.ndarray) -> tuple[CDRResult, np.ndarray, np.ndarray]:
    """(CDRResult, disc_mask, cup_mask), masks at the input size."""
    disc, cup, reason = segment_disc_cup(rgb)
    if reason is not None:
        return CDRResult(None, reason), disc, cup
    return compute_cdr(disc, cup), disc, cup
