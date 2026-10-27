"""Flask API and website host.

Puts ../src on sys.path so `python -m backend.app` works straight from a clone, without `pip install -e .` and without
setting PYTHONPATH. (The `glaucoma` library lives in src/.)
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
