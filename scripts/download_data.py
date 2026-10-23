"""Download ACRIMA if possible; print exact manual steps for anything that cannot be automated.

UNTESTED in the build environment (it had no access to figshare). If the automatic
download fails, follow the printed instructions: that is expected, not a bug.
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from glaucoma.config import data_path, load_config

FIGSHARE_API = "https://api.figshare.com/v2/articles/7613135"

MANUAL = """
==== Manual download steps ====
ACRIMA (open, no login):
  1. Open https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135
  2. Download the dataset files and unzip them so the images sit under:  {acrima}
     (Kaggle mirror, needs a free account: https://www.kaggle.com/datasets/toaharahmanratul/acrima-dataset)
REFUGE (needs a free grand-challenge.org account, or the torrent mirror):
  1. Official: https://refuge.grand-challenge.org   Mirror: https://hyper.ai/en/datasets/15858
  2. Put the unzipped training/validation images and their ground-truth masks under:  {refuge}
Then run:  python scripts/inspect_data.py   to verify counts and mask values.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    acrima, refuge = data_path(cfg, "acrima_dir"), data_path(cfg, "refuge_dir")
    acrima.mkdir(parents=True, exist_ok=True)
    refuge.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(FIGSHARE_API, timeout=30) as r:
            files = json.load(r).get("files", [])
        if not files:
            raise RuntimeError("figshare returned no files")
        for f in files:
            dest = acrima / f["name"]
            print(f"Downloading {f['name']} ({f.get('size', 0) / 1e6:.1f} MB) ...")
            urllib.request.urlretrieve(f["download_url"], dest)
            if dest.suffix.lower() == ".zip":
                with zipfile.ZipFile(dest) as z:
                    z.extractall(acrima)
        print(f"ACRIMA files are in {acrima}")
    except Exception as exc:  # noqa: BLE001  deliberate: any network problem -> manual steps
        print(f"Automatic ACRIMA download failed: {exc}", file=sys.stderr)
    print(MANUAL.format(acrima=acrima, refuge=refuge))
    print("REFUGE is never downloaded automatically (terms must be accepted by you).")


if __name__ == "__main__":
    main()
