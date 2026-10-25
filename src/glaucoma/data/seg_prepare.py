"""Turn a disc/cup annotation dataset (G1020 style: one labelme-type JSON of polygons per image) into the
format the segmenter trains on: a cropped image + a mask in the REFUGE encoding (cup=0, disc=128, background=255).

Why crop: ACRIMA images are already cropped around the optic disc, while G1020 images are full fundus photographs where
the disc is small. Training the segmenter on full photographs and then running it on ACRIMA crops would be a mismatch,
so each training image is cropped to `seg_crop_scale` x the disc size around the disc.

What is known about G1020 (from its paper, arXiv:2006.09158): JPEG images, one JSON annotation per image made with
labelme, optic disc and optic cup drawn as polygons, plus a bounding-box rectangle. What is NOT known (no download access
when this was written): the exact label strings, the folder names, and whether the JSON stores the image size. So this
code recognises several spellings (disc / disk / OD / optic_disc ...), ignores rectangles and anything labelled like a
bounding box, and reports every file it cannot use instead of silently dropping it.

No PyTorch needed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..preprocessing import load_image
from .index import is_image_file, is_junk

CUP_TOKENS = {"cup", "oc"}
DISC_TOKENS = {"disc", "disk", "od"}
BOX_TOKENS = {"bb", "bbox", "box", "bounding", "rect", "rectangle"}
MIN_CUP_INSIDE = 0.8   # at least this fraction of the cup polygon must lie inside the disc polygon


def classify_label(label: str) -> Optional[str]:
    """'cup', 'disc', 'box' (a bounding box, to be ignored) or None (unknown label)."""
    low = str(label).lower()
    tokens = {t for t in re.split(r"[^a-z0-9]+", low) if t}
    if tokens & BOX_TOKENS or "box" in low or "bounding" in low or "rect" in low:
        return "box"
    if "cup" in low or tokens & CUP_TOKENS:
        return "cup"
    if "disc" in low or "disk" in low or tokens & DISC_TOKENS:
        return "disc"
    return None


def parse_polygons(json_path: Path) -> dict:
    """Return {"height", "width", "image_path", "disc": [Nx2 arrays], "cup": [...], "other_labels": [...],
    "ignored_boxes": int, "bad_shapes": int}. Raises ValueError if the file is not a labelme-style annotation."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    shapes = data.get("shapes")
    if shapes is None:
        raise ValueError("no 'shapes' list in the JSON (not a labelme-style annotation)")
    out = {"height": data.get("imageHeight"), "width": data.get("imageWidth"), "image_path": data.get("imagePath"),
           "disc": [], "cup": [], "other_labels": [], "ignored_boxes": 0, "bad_shapes": 0}
    for s in shapes:
        kind = classify_label(s.get("label", ""))
        if str(s.get("shape_type") or "polygon").lower() == "rectangle":
            kind = "box"
        if kind == "box":
            out["ignored_boxes"] += 1
            continue
        try:
            pts = np.asarray(s.get("points", []), dtype=np.float64)
        except (ValueError, TypeError):
            out["bad_shapes"] += 1
            continue
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            out["bad_shapes"] += 1
            continue
        if kind == "cup":
            out["cup"].append(pts)
        elif kind == "disc":
            out["disc"].append(pts)
        else:
            out["other_labels"].append(str(s.get("label", "")).lower())
    return out


def rasterize(polys: dict, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Filled disc and cup masks (uint8 0/1) exactly as drawn; the cup is NOT merged into the disc here."""
    disc = np.zeros((height, width), np.uint8)
    cup = np.zeros((height, width), np.uint8)
    for p in polys["disc"]:
        cv2.fillPoly(disc, [np.round(p).astype(np.int32)], 1)
    for p in polys["cup"]:
        cv2.fillPoly(cup, [np.round(p).astype(np.int32)], 1)
    return disc, cup


def to_refuge_mask(disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """cup=0, disc rim=128, background=255. Cup pixels count as disc too when decoded (see masks_from_refuge)."""
    m = np.full(disc.shape, 255, np.uint8)
    m[disc > 0] = 128
    m[cup > 0] = 0
    return m


def crop_box(disc: np.ndarray, scale: float) -> Optional[tuple[int, int, int]]:
    """Square window (top, left, side) of `scale` x disc size around the disc centre, kept inside the image."""
    ys, xs = np.nonzero(disc)
    if len(ys) == 0:
        return None
    h, w = disc.shape
    size = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    side = int(min(max(round(size * scale), 32), h, w))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    top = int(np.clip(cy - side // 2, 0, h - side))
    left = int(np.clip(cx - side // 2, 0, w - side))
    return top, left, side


def _stem(name: str) -> str:
    """File stem of a name that may use Windows or Linux separators."""
    return Path(re.split(r"[\\/]", str(name))[-1]).stem.lower()


def _pick_image(candidates: list[Path], polys: dict) -> tuple[Optional[Path], bool]:
    """Choose the image for an annotation. When several files share the name, prefer the one whose size matches the
    annotation. Returns (path or None, ambiguous)."""
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0], False
    if polys.get("height") and polys.get("width"):
        match = [c for c in candidates if load_image(c).shape[:2] == (polys["height"], polys["width"])]
        if len(match) == 1:
            return match[0], False
        if match:
            return match[0], True
    return candidates[0], True


def prepare_json_dataset(raw_dir: Path, out_dir: Path, scale: float, name: str = "g1020",
                         max_side: Optional[int] = 512) -> dict:
    """Convert every JSON annotation under raw_dir that has a matching image. Returns a report dict.

    Crops larger than `max_side` pixels are shrunk to it (the segmenter trains at 256 anyway), which saves disk and time.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    jsons = sorted(p for p in raw_dir.rglob("*.json") if not is_junk(p))
    images: dict[str, list[Path]] = {}
    for p in sorted(raw_dir.rglob("*")):
        if is_image_file(p) and "mask" not in p.parent.name.lower():
            images.setdefault(p.stem.lower(), []).append(p)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    report = {"json_files": len(jsons), "images_found": sum(len(v) for v in images.values()), "written": 0,
              "skipped": {}, "labels_unrecognised": set(), "ignored_boxes": 0, "bad_shapes": 0, "ambiguous_names": []}

    def skip(reason, stem):
        report["skipped"].setdefault(reason, []).append(stem)

    for jp in jsons:
        try:
            polys = parse_polygons(jp)
        except Exception as e:  # unreadable or different layout
            skip(f"unreadable json ({e})", jp.stem)
            continue
        report["labels_unrecognised"].update(polys["other_labels"])
        report["ignored_boxes"] += polys["ignored_boxes"]
        report["bad_shapes"] += polys["bad_shapes"]
        candidates = list(images.get(jp.stem.lower(), []))
        if not candidates and polys["image_path"]:
            candidates = list(images.get(_stem(polys["image_path"]), []))
        ip, ambiguous = _pick_image(candidates, polys)
        if ip is None:
            skip("no image with the same file name", jp.stem)
            continue
        if ambiguous:
            report["ambiguous_names"].append(jp.stem)
        if not polys["disc"] and not polys["cup"]:
            skip("no disc or cup polygon", jp.stem)
            continue
        if not polys["disc"]:
            skip("no disc polygon", jp.stem)
            continue
        if not polys["cup"]:
            skip("no cup polygon", jp.stem)
            continue
        img = load_image(ip)
        h, w = img.shape[:2]
        if polys["height"] and polys["width"] and (polys["height"], polys["width"]) != (h, w):
            skip("annotation size differs from image size", jp.stem)
            continue
        allpts = np.concatenate(polys["disc"] + polys["cup"])
        if allpts[:, 0].min() < -2 or allpts[:, 1].min() < -2 or allpts[:, 0].max() > w + 2 or allpts[:, 1].max() > h + 2:
            skip("polygon lies outside the image (wrong image for this annotation?)", jp.stem)
            continue
        disc, cup = rasterize(polys, h, w)
        if cup.sum() == 0 or disc.sum() == 0:
            skip("empty disc or cup after rasterising", jp.stem)
            continue
        if (cup & disc).sum() < MIN_CUP_INSIDE * cup.sum():
            skip("cup polygon is not inside the disc polygon", jp.stem)
            continue
        box = crop_box(np.maximum(disc, cup), scale)
        if box is None:
            skip("empty disc after rasterising", jp.stem)
            continue
        t, l, s = box
        crop_img = img[t:t + s, l:l + s]
        crop_mask = to_refuge_mask(disc[t:t + s, l:l + s], cup[t:t + s, l:l + s])
        if max_side and s > max_side:
            crop_img = cv2.resize(crop_img, (max_side, max_side), interpolation=cv2.INTER_AREA)
            crop_mask = cv2.resize(crop_mask, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "images" / f"{name}_{jp.stem}.png"), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "masks" / f"{name}_{jp.stem}.png"), crop_mask)
        report["written"] += 1
    report["labels_unrecognised"] = sorted(report["labels_unrecognised"])
    return report
