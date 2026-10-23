"""Create stratified train/val/test CSVs (with a leakage check) from the dataset named in the config."""
import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma.config import load_config
from glaucoma.data.splits import make_all_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--quick", action="store_true", help="use the synthetic smoke-test data")
    a = p.parse_args()
    cfg = load_config(a.config, quick=a.quick)
    counts = make_all_splits(cfg)
    print(json.dumps({k: {str(kk): vv for kk, vv in v.items()} for k, v in counts.items()}, indent=2))
    print("Splits saved under data/splits/. No file appears in more than one split (checked).")


if __name__ == "__main__":
    main()
