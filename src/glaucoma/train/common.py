"""Shared bits for the training scripts: CLI args, config, checkpoint metadata."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config, resolve
from ..utils import get_device, git_hash, save_json, set_seed, setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to a YAML config (default: configs/config.yaml)")
    p.add_argument("--quick", action="store_true", help="Tiny smoke test on synthetic data")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    return p


def init_run(args, name: str):
    """Returns (cfg, device, logger) and prints the effective config."""
    cfg = load_config(args.config, quick=args.quick)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(cfg["seed"])
    logger = setup_logging(name, resolve(cfg, "logs"))
    device = get_device(cfg["device"])
    logger.info("device=%s seed=%s quick=%s git=%s", device, cfg["seed"], cfg["quick_mode"], git_hash())
    logger.info("config: %s", {k: cfg[k] for k in ("data", "autoencoder", "baseline", "segmenter") if k in cfg})
    return cfg, device, logger


def ckpt_dir(cfg: dict) -> Path:
    d = resolve(cfg, "checkpoints")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ckpt(cfg: dict, name: str, state_dict, extra: dict | None = None) -> Path:
    """Saves <name>.pt and <name>.meta.json (config, seed, git hash) side by side."""
    import torch

    d = ckpt_dir(cfg)
    path = d / f"{name}.pt"
    torch.save(state_dict, path)
    save_json({"config": cfg, "seed": cfg["seed"], "git": git_hash(), **(extra or {})}, d / f"{name}.meta.json")
    return path
