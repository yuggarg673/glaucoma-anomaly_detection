"""Inference pipeline used by the API and the evaluation script.

Loads the autoencoder, segmenter and fusion model once and turns an RGB image into
the full result: label, anomaly score, CDR, fused score, confidence, annotated image.
PyTorch is imported lazily so the module can be imported without it.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .cdr import compute_cdr
from .config import load_config, resolve
from .eval.annotate import annotate
from .eval.fusion import FusionModel
from .eval.scoring import anomaly_score
from .preprocessing import resize_square, to_chw, to_float01
from .utils import load_json


class ModelsNotReady(RuntimeError):
    """Raised when checkpoints are missing; the message says which and how to create them."""


def describe_training(meta: dict, seg: bool = False) -> str:
    """What a checkpoint was trained on, read from the metadata saved next to it ("synthetic" for a --quick code check).
    The website shows this, so nobody mistakes a code-check model for a real one."""
    cfg = meta.get("config") or {}
    data = cfg.get("data") or {}
    if cfg.get("quick_mode") or data.get("source") == "synthetic":
        return "synthetic"
    return str(data.get("seg_source" if seg else "source", "unknown"))


class Analyzer:
    def __init__(self, cfg: dict, device: str = "auto"):
        import torch
        from .models.autoencoder import build_autoencoder
        from .models.segmenter import UNet
        from .utils import get_device

        self.cfg = cfg
        self.torch = torch
        self.device = get_device(device)
        ck = resolve(cfg, "checkpoints")
        need = {"autoencoder": ck / "autoencoder.pt", "segmenter": ck / "segmenter.pt", "fusion": ck / "fusion.json",
                "pixel_thresholds": ck / "pixel_thresholds.json"}
        missing = [f"{k} ({p.name})" for k, p in need.items() if not p.exists()]
        if missing:
            raise ModelsNotReady("Missing checkpoints: " + ", ".join(missing) + ".")
        ae_meta = load_json(ck / "autoencoder.meta.json")
        ae_cfg = ae_meta["config"]
        self.ae_size = ae_meta.get("image_size", ae_cfg["data"]["image_size"])
        self.ae = build_autoencoder(ae_cfg).to(self.device)
        self.ae.load_state_dict(torch.load(need["autoencoder"], map_location=self.device))
        self.ae.eval()
        seg_meta = load_json(ck / "segmenter.meta.json")
        self.seg_size = seg_meta.get("seg_size", seg_meta["config"]["data"]["seg_size"])
        self.seg = UNet(base=seg_meta["config"]["segmenter"]["base_channels"]).to(self.device)
        self.seg.load_state_dict(torch.load(need["segmenter"], map_location=self.device))
        self.seg.eval()
        self.fusion = FusionModel.load(need["fusion"])
        self.pixel_thr = load_json(ck / "pixel_thresholds.json")
        self.versions = {"app": __version__, "autoencoder_git": ae_meta.get("git"), "segmenter_git": seg_meta.get("git")}
        self.trained_on = {"autoencoder": describe_training(ae_meta), "segmenter": describe_training(seg_meta, seg=True)}
        # Models from a --quick run learned from synthetic drawings: their numbers mean nothing, and the app says so.
        self.test_models = "synthetic" in self.trained_on.values()

    def _tensor(self, rgb: np.ndarray, size: int):
        x = to_chw(to_float01(resize_square(rgb, size)))
        return self.torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.device)

    def _forward(self, rgb: np.ndarray):
        """The model passes (the only part that needs PyTorch).
        Returns the per-pixel error map and the disc and cup masks at the segmenter's size."""
        torch = self.torch
        with torch.no_grad():
            err = self.ae.error_map(self._tensor(rgb, self.ae_size))[0].cpu().numpy()
            seg_logits = self.seg(self._tensor(rgb, self.seg_size))[0]
            seg = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy().astype(np.uint8)
        return err, seg[0], seg[1]

    def _postprocess(self, rgb: np.ndarray, err: np.ndarray, disc_s: np.ndarray, cup_s: np.ndarray) -> dict:
        """Everything after the models (numpy only, so it can be tested without PyTorch): the cup-to-disc ratio,
        the combined score and confidence, the annotated image and the response dictionary."""
        h, w = rgb.shape[:2]
        cdr = compute_cdr(disc_s, cup_s)
        score = anomaly_score(err)   # whole-image mean, exactly as in training
        verdict = self.fusion.predict(score, cdr.value)

        a = self.cfg["annotate"]
        # The annotated picture is capped in size so a huge upload does not produce a huge response.
        s = min(1.0, int(a.get("max_output_side", 1280)) / max(h, w))
        ann_rgb = rgb if s >= 1.0 else cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))),
                                                  interpolation=cv2.INTER_AREA)
        ah, aw = ann_rgb.shape[:2]
        disc_full = cv2.resize(disc_s, (aw, ah), interpolation=cv2.INTER_NEAREST)
        cup_full = cv2.resize(cup_s, (aw, ah), interpolation=cv2.INTER_NEAREST)
        png, regions = annotate(
            ann_rgb, err, self.pixel_thr["moderate"], self.pixel_thr["high"],
            disc_full if disc_full.any() else None,
            cup_full if cup_full.any() else None, cdr.value,
            a["blur_sigma"], a["max_regions"], a["min_area_frac"],
        )
        return {
            "label": verdict["label"],
            "anomaly_score": score,
            "anomaly_threshold": verdict["anomaly_threshold"],
            "cdr": cdr.value,
            "cdr_reason": cdr.reason,
            "cdr_reference": self.cfg["cdr"]["reference_cdr"],
            "fused_score": verdict["fused_score"],
            "threshold": verdict["threshold"],
            "confidence_score": verdict["confidence_score"],
            "confidence_level": verdict["confidence_level"],
            "signals_agree": verdict["signals_agree"],
            "used_cdr": verdict["used_cdr"],
            "regions": regions,
            "annotated_image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "image_size": {"width": w, "height": h},
            "annotated_size": {"width": aw, "height": ah},   # region boxes are in this coordinate system
            "model_versions": self.versions,
            "trained_on": self.trained_on,
            "test_models": self.test_models,
        }

    def analyze(self, rgb: np.ndarray) -> dict:
        err, disc_s, cup_s = self._forward(rgb)
        return self._postprocess(rgb, err, disc_s, cup_s)


def load_analyzer(config: Optional[str] = None, device: str = "auto") -> Analyzer:
    return Analyzer(load_config(config), device)
