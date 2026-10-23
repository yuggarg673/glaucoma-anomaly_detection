# Datasets

Datasets are **not** in this repository (size and licences). Put them under `data/raw/` as below.

| Dataset | Used for | Where to get it |
| --- | --- | --- |
| ACRIMA (705 images: 396 glaucoma, 309 normal) | Autoencoder, baseline, classification test | https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135 (open). Kaggle mirror, third party, free account: https://www.kaggle.com/datasets/toaharahmanratul/acrima-dataset |
| REFUGE (1,200 images, glaucoma labels, disc/cup masks) | Disc/cup segmentation for the cup-to-disc ratio | Official: https://refuge.grand-challenge.org (free account, accept terms). Mirror "iChallenge-GON" (torrent, non-commercial): https://hyper.ai/en/datasets/15858 |

Fallback if REFUGE is unavailable: another open dataset with disc/cup masks, such as DRISHTI-GS or G1020. See the
catalog https://github.com/TheBeastCoding/glaucoma-dataset-metadata . Record the change in `docs/DECISIONS.md`.

## Folder layout

```
data/raw/acrima/    all ACRIMA images (the glaucoma ones have "_g_" in the file name)
data/raw/refuge/    REFUGE images (.jpg) and their masks (.bmp), keeping the original sub-folders
```

## Check them

```
python scripts/inspect_data.py
```

This confirms the counts (705 / 396 / 309), reads the REFUGE mask pixel values, and writes `docs/DATA_NOTES.md`.
The code assumes REFUGE masks use cup = 0, disc = 128, background = 255. **If the script shows different values,
edit `masks_from_refuge` in `src/glaucoma/cdr.py`.** Mask file matching is by file-name stem, so if no pairs are found
the error message tells you what to look at.

## Citations

- Diaz-Pinto et al., "CNNs for automatic glaucoma assessment using fundus images: an extensive validation,"
  BioMedical Engineering OnLine 18, 29 (2019). (ACRIMA)
- Orlando et al., "REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment
  from fundus photographs," arXiv:1910.03667. (REFUGE)

Check each dataset's licence before using it, and cite the papers above.
