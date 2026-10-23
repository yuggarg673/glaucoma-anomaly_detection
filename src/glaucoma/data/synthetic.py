"""SYNTHETIC fundus-like images for smoke tests only.

These are drawings, not eye photographs. Any metric computed on them is
meaningless. They exist so the whole pipeline (and the tests) can run without
downloading the real datasets.

Layout (mirrors the real data):
  <out>/images/ImNNNN_SYN.png        normal
  <out>/images/ImNNNN_g_SYN.png      glaucoma-like (bigger cup + a pale patch)
  <out>/masks/<same stem>.png        REFUGE encoding: cup=0, disc rim=128, background=255
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _make_one(size: int, glaucoma: bool, rng: np.random.Generator):
    img = np.zeros((size, size, 3), np.float32)
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(yy - c, xx - c)
    retina = rr < size * 0.48
    base = np.stack([200 - 90 * rr / size, 95 - 40 * rr / size, 40 - 15 * rr / size], axis=-1)
    img[retina] = base[retina]
    img += rng.normal(0, 4, img.shape).astype(np.float32) * retina[..., None]

    dcx = c + rng.uniform(0.08, 0.16) * size * rng.choice([-1, 1])
    dcy = c + rng.uniform(-0.05, 0.05) * size
    r_disc = rng.uniform(0.09, 0.12) * size
    ratio = rng.uniform(0.62, 0.82) if glaucoma else rng.uniform(0.28, 0.46)
    r_cup = ratio * r_disc

    for _ in range(5):  # a few vessels
        a = rng.uniform(0, 2 * np.pi)
        p0 = (int(dcx), int(dcy))
        p1 = (int(dcx + np.cos(a) * size * 0.4), int(dcy + np.sin(a) * size * 0.4))
        cv2.line(img, p0, p1, (120, 30, 20), max(1, size // 128), cv2.LINE_AA)

    disc_m = np.hypot(yy - dcy, xx - dcx) < r_disc
    cup_m = np.hypot(yy - dcy, xx - dcx) < r_cup
    img[disc_m] = (235, 190, 120)
    img[cup_m] = (250, 235, 200)

    if glaucoma:  # pale patch next to the disc
        px, py = dcx - np.sign(dcx - c) * r_disc * 2.2, dcy + r_disc * 1.6
        patch = np.hypot(yy - py, xx - px) < r_disc * 0.9
        img[patch] = img[patch] * 0.4 + np.array([230, 215, 190]) * 0.6

    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = np.full((size, size), 255, np.uint8)
    mask[disc_m] = 128
    mask[cup_m] = 0
    return img, mask, ratio


def generate(out_dir: str | Path, n: int = 60, size: int = 128, seed: int = 42) -> dict:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_g = 0
    for i in range(n):
        g = i % 2 == 1
        img, mask, _ = _make_one(size, g, rng)
        stem = f"Im{i:04d}_g_SYN" if g else f"Im{i:04d}_SYN"
        cv2.imwrite(str(out / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        n_g += int(g)
    return {"images": n, "glaucoma": n_g, "normal": n - n_g, "dir": str(out)}
