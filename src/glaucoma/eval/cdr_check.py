"""Is the cup-to-disc ratio predicted on ACRIMA informative at all?

ACRIMA has no cup/disc outlines, so there is no ground truth for the ratio. The segmenter is trained on another dataset
(G1020) and its ratio on ACRIMA images is an estimate. The only honest check available is indirect: glaucoma images
should, on average, get a larger ratio than normal ones, and the ratio alone should separate the two classes better than
chance. This module computes that, plus warnings to show the user. No PyTorch needed.

It cannot prove the outlines are right (an image can get the correct class by luck of a wrong outline). Always also
look at `python scripts/preview_cdr.py`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import bootstrap_auc_ci, compute_metrics

MIN_N = 10                # fewer usable images than this: report "not enough to say"
WEAK_AUC = 0.55           # validation AUC below this: the ratio is close to uninformative
MAX_MISSING = 0.20        # more than this share of images without a ratio is a problem worth a warning


def _auc(c: np.ndarray, y: np.ndarray) -> Optional[float]:
    ok = ~np.isnan(c)
    if ok.sum() < MIN_N or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], c[ok]))


def _class_mean(c: np.ndarray, y: np.ndarray, label: int) -> Optional[float]:
    m = (~np.isnan(c)) & (y == label)
    return float(c[m].mean()) if m.sum() else None


def cdr_sanity(cdr_val, y_val, cdr_test, y_test, threshold: Optional[float], n_boot: int = 1000,
               seed: int = 42) -> dict:
    """Summary dict (JSON-safe) with a `warnings` list. `threshold` is the CDR cut-off chosen on validation."""
    cv, yv = np.asarray(cdr_val, dtype=float), np.asarray(y_val).astype(int)
    ct, yt = np.asarray(cdr_test, dtype=float), np.asarray(y_test).astype(int)
    info = {
        "unavailable_val": float(np.isnan(cv).mean()), "unavailable_test": float(np.isnan(ct).mean()),
        "val_auc": _auc(cv, yv), "test_auc": _auc(ct, yt),
        "val_mean_cdr_normal": _class_mean(cv, yv, 0), "val_mean_cdr_glaucoma": _class_mean(cv, yv, 1),
        "test_mean_cdr_normal": _class_mean(ct, yt, 0), "test_mean_cdr_glaucoma": _class_mean(ct, yt, 1),
        "test_metrics": None, "warnings": [],
    }
    ok = ~np.isnan(ct)
    if threshold is not None and ok.sum() >= MIN_N and len(np.unique(yt[ok])) == 2:
        m = compute_metrics(yt[ok], ct[ok], threshold)
        m["auc_ci95"] = bootstrap_auc_ci(yt[ok], ct[ok], n_boot, seed)
        info["test_metrics"] = m

    w = info["warnings"]
    auc = info["val_auc"]
    mn, mg = info["val_mean_cdr_normal"], info["val_mean_cdr_glaucoma"]
    if auc is None:
        w.append("Too few images with a cup-to-disc ratio (or only one class) to judge whether it is informative.")
    elif auc < 1 - WEAK_AUC:
        w.append(f"The cup-to-disc ratio is INVERTED on validation images: AUC {auc:.2f} means normal images got larger "
                 "ratios than glaucoma ones. That contradicts the clinical meaning of the ratio, so the predicted "
                 "outlines are wrong (or the labels are reversed). Run `python scripts/preview_cdr.py` and look.")
    elif auc < WEAK_AUC:
        w.append(f"The cup-to-disc ratio alone separates glaucoma from normal at AUC {auc:.2f} on validation images, "
                 "which is close to chance. The segmenter probably does not transfer to ACRIMA. Run "
                 "`python scripts/preview_cdr.py` and look at the outlines before relying on the fusion.")
    if (auc is None or auc >= 1 - WEAK_AUC) and mn is not None and mg is not None and mg <= mn:
        w.append(f"Glaucoma images do not get a larger mean cup-to-disc ratio than normal ones ({mg:.2f} vs {mn:.2f}). "
                 "That contradicts the clinical meaning of the ratio, so the predicted outlines are suspect.")
    for split in ("val", "test"):
        if info[f"unavailable_{split}"] > MAX_MISSING:
            w.append(f"No cup-to-disc ratio could be computed for {100 * info[f'unavailable_{split}']:.0f}% of the "
                     f"{split} images (disc or cup not found by the segmenter).")
    return info
