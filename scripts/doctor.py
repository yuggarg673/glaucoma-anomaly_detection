"""Quick health check of the whole setup. Prints PASS / WARN / FAIL for each item."""
import argparse
import importlib
import json
import sys
import urllib.request

import _bootstrap  # noqa: F401
from glaucoma.config import REPO_ROOT, data_path, load_config, resolve

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def line(status, msg, fix=""):
    print(f"[{status}] {msg}" + (f"\n        fix: {fix}" if fix and status != OK else ""))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--api", default="http://localhost:5000")
    a = p.parse_args()
    cfg = load_config(a.config)
    results = []
    results.append(line(OK if sys.version_info >= (3, 10) else FAIL, f"Python {sys.version.split()[0]} (need 3.10+)",
                        "install Python 3.10 or newer"))
    for mod, why in [("numpy", "core"), ("cv2", "images"), ("sklearn", "metrics"), ("yaml", "config"),
                     ("flask", "API"), ("torch", "training and inference"), ("torchvision", "baseline model")]:
        try:
            m = importlib.import_module(mod)
            try:
                from importlib.metadata import version
                ver = version("opencv-python-headless" if mod == "cv2" else "scikit-learn" if mod == "sklearn" else "PyYAML" if mod == "yaml" else mod)
            except Exception:
                ver = getattr(m, "__version__", "") if mod != "flask" else ""
            results.append(line(OK, f"{mod} {ver}".strip()))
        except ImportError:
            results.append(line(FAIL, f"{mod} missing ({why})", "pip install -r requirements.txt"))
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        results.append(line(OK, f"torch device: {dev}"))
    except ImportError:
        pass
    for name, path, fix in [("ACRIMA data", data_path(cfg, "acrima_dir"), "python scripts/download_data.py"),
                            ("REFUGE data" if cfg["data"].get("seg_source") == "refuge" else "G1020 data",
                             data_path(cfg, "refuge_dir" if cfg["data"].get("seg_source") == "refuge" else "g1020_dir"),
                             "see docs/DATA.md")]:
        n = sum(1 for _ in path.rglob("*") if _.is_file()) if path.exists() else 0
        results.append(line(OK if n else WARN, f"{name}: {n} files in {path}", fix))
    if cfg["data"].get("seg_source") == "g1020":
        prepared = data_path(cfg, "seg_processed_dir") / "masks"
        n = len(list(prepared.glob("*.png"))) if prepared.exists() else 0
        results.append(line(OK if n >= 100 else WARN, f"prepared G1020 training masks: {n} in {prepared}"
                            + ("" if n >= 100 else " (expected several hundred)"),
                            "python scripts/prepare_segmentation_data.py"))
    sp = resolve(cfg, "data_splits")
    results.append(line(OK if (sp / "classification.csv").exists() else WARN, "split CSVs present", "python scripts/make_splits.py"))
    ck = resolve(cfg, "checkpoints")
    for f in ["autoencoder.pt", "segmenter.pt", "fusion.json", "pixel_thresholds.json"]:
        results.append(line(OK if (ck / f).exists() else WARN, f"checkpoint {f}", "python scripts/run_pipeline.py"))
    try:
        with urllib.request.urlopen(a.api + "/health", timeout=3) as r:
            h = json.load(r)
        results.append(line(OK if h.get("models_loaded") else WARN, f"API reachable, models_loaded={h.get('models_loaded')}",
                            h.get("detail", "")))
    except Exception:
        results.append(line(WARN, f"API not reachable at {a.api}", "python -m backend.app"))
    print(f"\n{results.count(OK)} passed, {results.count(WARN)} warnings, {results.count(FAIL)} failed")
    sys.exit(1 if FAIL in results else 0)


if __name__ == "__main__":
    main()
