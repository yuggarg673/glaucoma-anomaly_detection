"""Convert the G1020 disc/cup annotations into cropped images + masks the segmenter can train on.

  python scripts/prepare_segmentation_data.py

Reads data.g1020_dir, writes data.seg_processed_dir/{images,masks}. Prints how many were converted and, for every
skipped file, why. If nothing is converted, run `python scripts/inspect_data.py` and send me its output.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from glaucoma.config import data_path, load_config
from glaucoma.data.seg_prepare import prepare_json_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg["data"].get("seg_source") != "g1020":
        print(f"data.seg_source is '{cfg['data'].get('seg_source')}', nothing to prepare.")
        return
    raw = data_path(cfg, "g1020_dir")
    if not raw.exists():
        sys.exit(f"G1020 folder not found: {raw}\nDownload it first (docs/DATA.md), or change data.g1020_dir.")
    rep = prepare_json_dataset(raw, data_path(cfg, "seg_processed_dir"), cfg["data"]["seg_crop_scale"],
                               max_side=cfg["data"].get("seg_prepared_max_side", 512))
    print(f"JSON files: {rep['json_files']}   images found: {rep['images_found']}   converted: {rep['written']}")
    for reason, stems in rep["skipped"].items():
        print(f"  skipped {len(stems):4d}: {reason}   e.g. {stems[:3]}")
    if rep["labels_unrecognised"]:
        print(f"  shape labels that were neither disc nor cup (ignored): {rep['labels_unrecognised']}")
    if rep["ignored_boxes"]:
        print(f"  bounding-box rectangles ignored: {rep['ignored_boxes']}")
    if rep["bad_shapes"]:
        print(f"  malformed shapes ignored: {rep['bad_shapes']}")
    if rep["ambiguous_names"]:
        print(f"  WARNING: {len(rep['ambiguous_names'])} annotations matched several image files with the same name "
              f"(first one used), e.g. {rep['ambiguous_names'][:3]}. Check the masks look right.")
    if rep["written"] == 0:
        sys.exit("Nothing was converted. The G1020 annotation layout differs from what this script expects.\n"
                 "Run `python scripts/inspect_data.py` and send me the output so the loader can be fixed.")
    if rep["written"] < 0.9 * rep["json_files"]:
        print("WARNING: fewer than 90% of annotations converted. Check the reasons above before training.")


if __name__ == "__main__":
    main()
