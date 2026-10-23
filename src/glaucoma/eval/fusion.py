"""Fusion of the autoencoder anomaly score with the cup-to-disc ratio (CDR).

Everything is fitted on the VALIDATION split only. At inference this module uses
plain numpy, so the backend does not need scikit-learn.

Definitions (also explained to users in docs/USER_GUIDE.md):
  fused score      p = sigmoid(w1 * s_norm + w2 * cdr_norm + b), fitted by logistic regression
  decision         glaucoma suspected if p >= threshold (Youden's J on validation)
  confidence score min(1, |p - threshold| / max(threshold, 1 - threshold))
  confidence level Low < 0.33 <= Medium < 0.66 <= High
This confidence is a distance-from-threshold heuristic, NOT a calibrated probability of disease.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import youden_threshold
from ..utils import load_json, save_json


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _norm(x: float, lo: float, hi: float) -> float:
    if hi - lo < 1e-12:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class FusionModel:
    def __init__(self, params: dict):
        self.p = params

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, scores, cdrs, labels, seed: int = 42) -> "FusionModel":
        from sklearn.linear_model import LogisticRegression

        s = np.asarray(scores, dtype=float)
        c = np.asarray(cdrs, dtype=float)       # NaN where CDR is unavailable
        y = np.asarray(labels).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Fusion needs both normal and glaucoma images in the validation split.")

        s_lo, s_hi = float(np.nanmin(s)), float(np.nanmax(s))
        has_c = ~np.isnan(c)
        if has_c.sum() >= 1:
            c_lo, c_hi = float(np.nanmin(c)), float(np.nanmax(c))
        else:
            c_lo, c_hi = 0.0, 1.0
        params: dict = {"s_min": s_lo, "s_max": s_hi, "c_min": c_lo, "c_max": c_hi}

        # anomaly score alone (also used when CDR is missing)
        xs = np.array([[_norm(v, s_lo, s_hi)] for v in s])
        lr_a = LogisticRegression(class_weight="balanced", random_state=seed).fit(xs, y)
        pa = lr_a.predict_proba(xs)[:, 1]
        params["anom"] = {
            "coef": lr_a.coef_[0].tolist(), "intercept": float(lr_a.intercept_[0]),
            "threshold": youden_threshold(y, pa),
        }
        params["anom_raw_threshold"] = youden_threshold(y, s)

        # CDR alone threshold (used only for the "signals agree" indicator)
        if has_c.sum() >= 10 and len(np.unique(y[has_c])) == 2:
            params["cdr_raw_threshold"] = youden_threshold(y[has_c], c[has_c])
            xb = np.array([[_norm(a, s_lo, s_hi), _norm(b, c_lo, c_hi)] for a, b in zip(s[has_c], c[has_c])])
            lr_b = LogisticRegression(class_weight="balanced", random_state=seed).fit(xb, y[has_c])
            pb = lr_b.predict_proba(xb)[:, 1]
            params["both"] = {
                "coef": lr_b.coef_[0].tolist(), "intercept": float(lr_b.intercept_[0]),
                "threshold": youden_threshold(y[has_c], pb),
            }
        else:
            params["both"] = None
            params["cdr_raw_threshold"] = None
        return cls(params)

    # ---------------------------------------------------------------- inference
    def predict(self, score: float, cdr: Optional[float]) -> dict:
        p = self.p
        s_n = _norm(score, p["s_min"], p["s_max"])
        if cdr is not None and p.get("both"):
            m = p["both"]
            c_n = _norm(cdr, p["c_min"], p["c_max"])
            z = m["coef"][0] * s_n + m["coef"][1] * c_n + m["intercept"]
            used_cdr = True
        else:
            m = p["anom"]
            z = m["coef"][0] * s_n + m["intercept"]
            used_cdr = False
        fused = _sigmoid(z)
        thr = m["threshold"]
        denom = max(thr, 1.0 - thr, 1e-9)
        conf = float(min(1.0, abs(fused - thr) / denom))
        level = "High" if conf >= 0.66 else "Medium" if conf >= 0.33 else "Low"

        anom_flag = score >= p["anom_raw_threshold"]
        agree: Optional[bool] = None
        if cdr is not None and p.get("cdr_raw_threshold") is not None:
            agree = bool(anom_flag == (cdr >= p["cdr_raw_threshold"]))
        return {
            "fused_score": fused,
            "threshold": thr,
            "margin": fused - thr,          # > 0 means glaucoma_suspected; used to report metrics exactly as the app decides
            "label": "glaucoma_suspected" if fused >= thr else "normal",
            "confidence_score": conf,
            "confidence_level": level,
            "signals_agree": agree,
            "used_cdr": used_cdr,
            "anomaly_threshold": p["anom_raw_threshold"],
            "cdr_threshold": p.get("cdr_raw_threshold"),
        }

    # -------------------------------------------------------------------- io
    def save(self, path: Path) -> None:
        save_json(self.p, Path(path))

    @classmethod
    def load(cls, path: Path) -> "FusionModel":
        return cls(load_json(Path(path)))
