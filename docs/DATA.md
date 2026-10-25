# Datasets

Datasets are **not** in this repository (size and licences). Put them under `data/raw/` as below.

| Dataset | Used for | Where to get it |
| --- | --- | --- |
| ACRIMA (705 images: 396 glaucoma, 309 normal) | Autoencoder, baseline, classification test | https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135 (open). Kaggle mirror, third party, free account: https://www.kaggle.com/datasets/toaharahmanratul/acrima-dataset |
| G1020 (1,020 images, 296 glaucoma / 724 normal, disc and cup outlines) | Trains the disc/cup segmenter. That segmenter is then run on ACRIMA to get ACRIMA's cup-to-disc ratio | Official: https://www.dfki.uni-kl.de/g1020 . Kaggle copy (third party, free account): https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets |
| REFUGE (optional alternative, 1,200 images with masks) | Same job as G1020. Set `data.seg_source: refuge` | https://refuge.grand-challenge.org (free account) or mirror https://hyper.ai/en/datasets/15858 |

**Why two datasets.** ACRIMA has only "glaucoma / normal" labels, no cup or disc outlines, so a cup-to-disc ratio cannot be
learned from it. G1020 has the outlines. The ACRIMA CDR in this project is therefore a *model estimate* from a segmenter
trained on another dataset, not an expert measurement. State this in the report. Use `python scripts/preview_cdr.py` to
look at the outlines it draws on ACRIMA images.

**What is and is not verified about G1020.** Its paper (arXiv:2006.09158) says the images are JPEG and every image has one
JSON annotation made with labelme, with the optic disc and the optic cup drawn as polygons and a separate bounding-box
rectangle. NOT verified (no download access when this was written): the folder names and the exact label strings. The
converter therefore accepts several spellings (`disc`, `disk`, `OD`, `optic_disc` ... and `cup`, `OC`, `optic_cup` ...),
ignores rectangles and anything labelled like a bounding box, rejects annotations whose cup is not inside the disc or whose
polygons do not fit the image, and prints the reason for every file it skips. If `inspect_data.py` shows something else (for
example masks as image files instead of JSON), send me its output.

Other options with outlines: RIM-ONE DL, DRISHTI-GS1 (small). Catalog: https://github.com/TheBeastCoding/glaucoma-dataset-metadata

## Folder layout

```
data/raw/acrima/    all ACRIMA images (the glaucoma ones have "_g_" in the file name)
data/raw/g1020/     G1020 images and their annotation .json files (any sub-folders)
data/processed/seg/ created by prepare_segmentation_data.py (cropped images + masks); do not edit
```

## Check them

```
python scripts/inspect_data.py              # counts, json labels; writes docs/DATA_NOTES.md
python scripts/prepare_segmentation_data.py # G1020 json -> cropped images + masks (reports every skipped file)
```

`inspect_data.py` confirms ACRIMA is 705 / 396 / 309 and shows what G1020 contains, including how each shape label is
read (disc / cup / box / unknown). The prepare step crops each G1020 photo to `data.seg_crop_scale` (default 2.5)
disc-widths around the disc, because ACRIMA images are already disc-centred crops, and shrinks crops larger than
`data.seg_prepared_max_side` (512) pixels. During training the segmenter also sees random zoom (0.75x to 1.35x), rotation,
shifts, flips and colour changes, so it is not tied to one disc size or camera. If most ACRIMA outlines look too tight or
too loose in `preview_cdr.py`, change `seg_crop_scale` and retrain. G1020 may contain both eyes of one person, so its own
segmentation scores can be slightly optimistic; this does not affect the ACRIMA classification results.
(If you use REFUGE instead, the code assumes masks cup = 0, disc = 128, background = 255; edit `masks_from_refuge` if not.)

## Citations

- Diaz-Pinto et al., "CNNs for automatic glaucoma assessment using fundus images: an extensive validation,"
  BioMedical Engineering OnLine 18, 29 (2019). (ACRIMA)
- Bajwa et al., "G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma Detection," IJCNN 2020
  (arXiv:2006.09158 — verify the citation on the dataset page before publishing). (G1020)
- Orlando et al., "REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment
  from fundus photographs," arXiv:1910.03667. (REFUGE)

Check each dataset's licence before using it, and cite the papers above.
