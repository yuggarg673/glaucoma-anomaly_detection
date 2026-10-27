"""Stratified train/val/test splits with a leakage check. Saved to CSV and reused everywhere."""
from __future__ import annotations


import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import resolve
from .index import classification_index, segmentation_index


def split_frame(df: pd.DataFrame, ratios: dict, seed: int, stratify_col: str | None = "label") -> pd.DataFrame:
    """Add a `split` column. Stratified by label when possible."""
    tr, va, te = ratios["train"], ratios["val"], ratios["test"]
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {tr}+{va}+{te}")
    strat = df[stratify_col] if stratify_col and stratify_col in df and df[stratify_col].nunique() > 1 else None
    train_df, rest = train_test_split(df, train_size=tr, random_state=seed, stratify=strat)
    strat_rest = rest[stratify_col] if strat is not None else None
    val_df, test_df = train_test_split(rest, train_size=va / (va + te), random_state=seed, stratify=strat_rest)
    out = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")])
    return out.sort_values(df.columns[0]).reset_index(drop=True)


def assert_no_leakage(df: pd.DataFrame, key: str) -> None:
    """Fail loudly if any file appears in more than one split."""
    dup = df.groupby(key)["split"].nunique()
    bad = dup[dup > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} files appear in more than one split, e.g. {bad.index[:3].tolist()}")
    if df[key].duplicated().any():
        raise AssertionError("Duplicate rows in split table.")


def make_all_splits(cfg: dict) -> dict:
    out_dir = resolve(cfg, "data_splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    ratios = cfg["data"]["split"]

    cls = split_frame(classification_index(cfg), ratios, seed, "label")
    assert_no_leakage(cls, "path")
    cls.to_csv(out_dir / "classification.csv", index=False)

    seg = segmentation_index(cfg)
    if (seg["split_hint"] != "").all():
        seg = seg.assign(split=seg["split_hint"])
    else:
        seg = split_frame(seg, ratios, seed, stratify_col=None)
    assert_no_leakage(seg, "image")
    seg.to_csv(out_dir / "segmentation.csv", index=False)
    return {
        "classification": cls.groupby(["split", "label"]).size().to_dict(),
        "segmentation": seg.groupby("split").size().to_dict(),
    }


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    p = resolve(cfg, "data_splits") / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Split file missing: {p}. Run `make splits` first.")
    return pd.read_csv(p)
