"""Create SYNTHETIC fundus-like drawings for smoke tests (NOT real eye images)."""
import argparse

import _bootstrap  # noqa: F401
from glaucoma.config import data_path, load_config
from glaucoma.data.synthetic import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--size", type=int, default=128)
    a = p.parse_args()
    cfg = load_config(a.config)
    info = generate(data_path(cfg, "synthetic_dir"), a.n, a.size, cfg["seed"])
    print(f"Wrote {info['images']} synthetic images ({info['normal']} normal, {info['glaucoma']} glaucoma-like) to {info['dir']}")
    print("These are drawings for testing the code only. Do not report results from them.")


if __name__ == "__main__":
    main()
