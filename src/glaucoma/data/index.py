"""Find the dataset files on disk and build index tables (no PyTorch needed)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT, data_path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def rel(p: Path) -> str:
    """Store paths relative to the repo root (portable across machines) with forward slashes."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def absolute(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def label_from_acrima_name(name: str) -> int:
    """ACRIMA glaucoma files carry a '_g_' marker. scripts/inspect_data.py checks the 396/309 counts."""
    return 1 if "_g_" in name.lower() else 0


def classification_index(cfg: dict) -> pd.DataFrame:
    """Table with columns path,label,dataset for the dataset selected in config (data.source)."""
    source = cfg["data"]["source"]
    if source == "acrima":
        root = data_path(cfg, "acrima_dir")
        if not root.exists():
            raise FileNotFoundError(
                f"ACRIMA folder not found: {root}\n"
                "Run `python scripts/download_data.py` for instructions, or set data.source: synthetic "
                "in configs/config.yaml for a smoke test."
            )
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"No images found under {root}")
        return pd.DataFrame({
            "path": [rel(p) for p in files],
            "label": [label_from_acrima_name(p.name) for p in files],
            "dataset": "acrima",
        })
    if source == "synthetic":
        root = data_path(cfg, "synthetic_dir") / "images"
        files = sorted(root.glob("*.png"))
        if not files:
            raise FileNotFoundError(
                f"No synthetic images in {root}. Run `python scripts/make_synthetic_data.py` first."
            )
        return pd.DataFrame({
            "path": [rel(p) for p in files],
            "label": [label_from_acrima_name(p.name) for p in files],
            "dataset": "synthetic",
        })
    raise ValueError(f"Unknown data.source '{source}'. Use 'acrima' or 'synthetic'.")


def segmentation_index(cfg: dict) -> pd.DataFrame:
    """Table with columns image,mask,dataset (+ split_hint for REFUGE) for disc/cup segmentation."""
    source = cfg["data"]["source"]
    if source == "synthetic":
        root = data_path(cfg, "synthetic_dir")
        imgs = sorted((root / "images").glob("*.png"))
        if not imgs:
            raise FileNotFoundError(f"No synthetic images in {root / 'images'}.")
        return pd.DataFrame({
            "image": [rel(p) for p in imgs],
            "mask": [rel(root / "masks" / p.name) for p in imgs],
            "dataset": "synthetic", "split_hint": "",
        })

    root = data_path(cfg, "refuge_dir")
    if not root.exists():
        raise FileNotFoundError(
            f"REFUGE folder not found: {root}\nSee docs/DATA.md for how to download REFUGE "
            "(it may need a free grand-challenge.org account)."
        )
    masks = {p.stem.lower(): p for p in root.rglob("*") if p.suffix.lower() in {".bmp", ".png"}
             and ("mask" in str(p).lower() or "annotation" in str(p).lower() or "gt" in str(p).lower())}
    rows = []
    for p in sorted(root.rglob("*.jpg")):
        m = masks.get(p.stem.lower())
        if m is None:
            continue
        low = str(p).lower()
        hint = "train" if "train" in low else "val" if "valid" in low else "test" if "test" in low else ""
        rows.append({"image": rel(p), "mask": rel(m), "dataset": "refuge", "split_hint": hint})
    if not rows:
        raise FileNotFoundError(
            f"Found no REFUGE image+mask pairs under {root}. Expected .jpg images and .bmp masks with the same "
            "file name stem. Run `python scripts/inspect_data.py` to see what was found."
        )
    return pd.DataFrame(rows)
