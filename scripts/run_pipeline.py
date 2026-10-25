"""Run the whole pipeline in order. Works on Windows, macOS and Linux (no make needed).

  python scripts/run_pipeline.py --quick     # synthetic smoke test, minutes
  python scripts/run_pipeline.py             # real data (ACRIMA + G1020 must be in data/raw)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO, env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    if r.returncode != 0:
        sys.exit(f"STEP FAILED ({r.returncode}): {' '.join(cmd)}\nSee docs/TROUBLESHOOTING.md and the log in logs/.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    py, extra = sys.executable, (["--config", a.config] if a.config else [])
    q = ["--quick"] if a.quick else []
    if a.quick:
        run([py, "scripts/make_synthetic_data.py"] + extra)
    else:
        run([py, "scripts/inspect_data.py"] + extra)
        run([py, "scripts/prepare_segmentation_data.py"] + extra)
    run([py, "scripts/make_splits.py"] + q + extra)
    run([py, "-m", "glaucoma.train.train_segmenter"] + q + extra)
    run([py, "-m", "glaucoma.train.train_autoencoder"] + q + extra)
    run([py, "-m", "glaucoma.train.train_baseline"] + q + extra)
    run([py, "-m", "glaucoma.train.fit_fusion"] + q + extra)
    print("\nDone. Start the app with:  python -m backend.app   (see README.md)")


if __name__ == "__main__":
    main()
