# Troubleshooting

First step for any problem: `python scripts/doctor.py`. It prints PASS / WARN / FAIL and the fix for each item.
Logs of every training script are in `logs/`.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No module named 'torch'` | PyTorch not installed | `pip install -r requirements.txt` (CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`) |
| `ACRIMA folder not found` | Data not downloaded | `python scripts/download_data.py`, then see `docs/DATA.md` |
| `No prepared segmentation data in ...` | `prepare_segmentation_data.py` has not run or converted nothing | Put G1020 in `data/raw/g1020`, run `python scripts/inspect_data.py`, then `python scripts/prepare_segmentation_data.py` |
| `Nothing was converted` from `prepare_segmentation_data.py` | The G1020 annotation layout differs from what the converter expects | The script prints why each file was skipped. Send me `docs/DATA_NOTES.md` (made by `inspect_data.py`) |
| Many files skipped: `no image with the same file name` | Annotation and image names do not match | See `docs/DATA_NOTES.md` for the folders; the converter also tries the `imagePath` stored inside the json |
| Many files skipped: `cup polygon is not inside the disc polygon` or `no cup polygon` | Annotation errors or other label names | Look at `docs/DATA_NOTES.md` (label list). A few skipped files are normal; if most are skipped, labels are not being read |
| `Found no REFUGE image+mask pairs` (only with `data.seg_source: refuge`) | Wrong folder, or mask names differ from image names | Run `python scripts/inspect_data.py`; masks must share the image's file-name stem |
| Segmenter learns nothing (Dice near 0) | Mask pixel values differ from cup=0/disc=128/background=255 (REFUGE), or the prepared G1020 masks are empty | Check `docs/DATA_NOTES.md`; open a few files in `data/processed/seg/masks`; for REFUGE edit `masks_from_refuge` in `src/glaucoma/cdr.py` |
| `CDR CHECK:` warnings at the end of `fit_fusion` (CDR near chance, inverted, or mostly missing) | The G1020-trained segmenter does not transfer well to ACRIMA images | Run `python scripts/preview_cdr.py` and look at `results/cdr_preview/`. Try `data.seg_crop_scale` 2.0 or 3.5, retrain the segmenter, and report honestly if the CDR does not help |
| `Could not build resnet18 ... pretrained` | No internet to download ImageNet weights | Set `baseline.pretrained: false` in `configs/config.yaml` |
| `CUDA out of memory` | Batch too big | Lower `batch_size` in `configs/config.yaml`, or `--device cpu` |
| `Input size ... must be divisible by ...` | `image_size` not a multiple of 2^(number of autoencoder stages), or of 8 for the segmenter | Use 256 (or any multiple of 16) |
| Autoencoder AUC close to 0.5 | Model redraws glaucoma images as well as healthy ones | Try fewer latent channels, keep `skip_connections: false`, check image quality, then report it honestly |
| `Missing checkpoints: ...` / `/health` says degraded | Models not trained yet | `python scripts/run_pipeline.py`, then restart the API |
| `Cannot reach the analysis server` in the website | Backend not running or wrong URL | `python -m backend.app`; set `VITE_API_BASE_URL` in `frontend/.env` |
| Browser CORS error | Frontend served from a different address | Set `api.frontend_origin` in `configs/config.yaml` to the exact address (default `http://localhost:5173`) |
| `Address already in use` | Port taken | `python -m backend.app --port 5001` and update `VITE_API_BASE_URL` |
| 413 on upload | File over the size limit | Raise `api.max_upload_mb` or use a smaller image |
| 422 `unreadable_image` | Not a real JPG/PNG | Re-export the image |
| `CDR unavailable` for many images | Segmenter not good enough, or the image is a wide photo (disc small) | Check Dice in `results/metrics.json`; use images cropped around the disc; see `results/cdr_preview/` |
| `npm install` fails | No internet / old Node | Use Node 18 or newer and check your network |
| Results look too good | Possible data leakage or tiny test set | Check `data/splits/classification.csv`; report the confidence interval |

## `--quick` mode

`python scripts/run_pipeline.py --quick` uses **synthetic drawings** (not eye photos) at tiny size for two epochs.
It only proves the code runs end to end. Never report its numbers.
