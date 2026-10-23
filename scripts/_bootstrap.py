"""Lets scripts run as `python scripts/x.py` without installing the package."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
