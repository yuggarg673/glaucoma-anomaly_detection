"""Config loading. Everything tunable comes from configs/config.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, quick: bool = False) -> dict[str, Any]:
    """Load the YAML config. With quick=True the `quick:` section overrides it."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Use --config to point to one.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    quick_cfg = cfg.pop("quick", {}) or {}
    cfg = copy.deepcopy(cfg)
    if quick:
        _deep_update(cfg, quick_cfg)
    cfg["quick_mode"] = quick
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] relative to the repo root and create it if it is a directory."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def data_path(cfg: dict, key: str) -> Path:
    """Resolve a path under cfg['data'] relative to the repo root."""
    p = Path(cfg["data"][key])
    return p if p.is_absolute() else REPO_ROOT / p
