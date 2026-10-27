"""TEST ONLY. Never imported by the app.

A stand-in for the trained models so the website and the response format can be tested without PyTorch or trained
weights. It invents the model outputs (an error map and disc/cup masks) and then runs the REAL post-processing from
glaucoma.inference.Analyzer: cup-to-disc ratio, fusion, confidence, annotated image, response dictionary. So it checks
everything the website depends on, except whether the neural networks themselves are any good.

Its numbers mean nothing. Its model versions say so, and that text is shown in the website's result card.
"""
from __future__ import annotations

import cv2
import numpy as np

from glaucoma.config import load_config
from glaucoma.eval.fusion import FusionModel
from glaucoma.inference import Analyzer


class StandInAnalyzer(Analyzer):
    def __init__(self, cfg: dict | None = None):          # deliberately does not call Analyzer.__init__ (needs torch)
        self.cfg = cfg or load_config()
        rng = np.random.default_rng(0)
        n = 80
        y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        scores = np.r_[rng.normal(0.05, 0.01, n), rng.normal(0.08, 0.015, n)]
        cdrs = np.r_[rng.normal(0.40, 0.08, n), rng.normal(0.65, 0.10, n)]
        self.fusion = FusionModel.fit(scores, cdrs, y, seed=0)
        self.pixel_thr = {"moderate": 0.12, "high": 0.20}
        self.versions = {"app": "test", "autoencoder_git": "stand-in (test only)", "segmenter_git": "stand-in (test only)"}
        self.trained_on = {"autoencoder": "stand-in (test only)", "segmenter": "stand-in (test only)"}
        self.test_models = True

    def _forward(self, rgb: np.ndarray):
        """Invented outputs with the same shapes as the real ones. What they depend on:
        a very dark image gets no cup (so the 'CDR unavailable' path can be tested); brighter images get a larger
        cup and stronger error blobs."""
        rng = np.random.default_rng(int(rgb.astype(np.uint32).sum()) % (2 ** 32))
        brightness = float(rgb.mean()) / 255.0
        err = rng.normal(0.045, 0.004, (64, 64)).clip(0, None).astype(np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        for cx, cy, amp in ((18, 40, 0.30 * brightness + 0.12), (46, 22, 0.16 * brightness + 0.06)):
            err += (amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5.0 ** 2)))).astype(np.float32)

        size = 256
        disc = np.zeros((size, size), np.uint8)
        cup = np.zeros((size, size), np.uint8)
        cv2.ellipse(disc, (140, 120), (44, 50), 0, 0, 360, 1, -1)
        if brightness >= 0.12:
            ratio = 0.3 + 0.45 * brightness
            cv2.ellipse(cup, (144, 120), (int(34 * ratio * 1.4), int(50 * ratio)), 0, 0, 360, 1, -1)
        return err, disc, cup
