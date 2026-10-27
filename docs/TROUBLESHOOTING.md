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
| `start_website.bat` says "Python 3.10 or newer was not found" | Python is missing, too old, or not on PATH (the Microsoft Store "python" shortcut does not count) | Install Python 3.10+ from python.org, tick **Add python.exe to PATH**, then double-click the file again |
| The launcher stops during "Installing the requirements" | No internet, or the large PyTorch download was interrupted | Run it again (finished downloads are reused). If it keeps failing, delete the `.venv` folder and follow the README's manual steps |
| `Cannot reach the analysis server` in the website | The server is not running, or the page was opened as a file | `python -m backend.app`, then open **http://localhost:5000** (not `web/index.html` by double-click) |
| `ModuleNotFoundError: No module named 'glaucoma'` or `'backend'` | Started from the wrong folder | Run it from the project folder: `python -m backend.app` (it finds `src/` by itself) |
| Website opens but says "models not loaded" | No checkpoints yet | `python scripts/run_pipeline.py` (or the Colab notebook), copy `checkpoints/` next to `backend/`, restart |
| Red banner "These are test models" / pill says TEST models | The checkpoints came from `--quick` (synthetic drawings) | Delete `checkpoints/`, run `python scripts/run_pipeline.py` (real data), restart |
| Page shows only "This website needs JavaScript" or stays blank | JavaScript off, or the script failed to load | Turn JavaScript on; press F12 and read the Console; hard-reload (Ctrl+Shift+R). `GET /assets/app.js` must return 200 |
| JSON `website_missing` at `/` | The `web/` folder is missing or you ran from another folder | Run from the project root; check that `web/index.html` exists (`git status`) |
| Text looks different from the screenshots | Google Fonts blocked or offline | Harmless: the page falls back to system fonts |
| Result takes very long | Large image on a slow CPU | The server analyses one image at a time; wait, or use a smaller image. The page gives up after 3 minutes |
| `Address already in use` | Port taken | `python -m backend.app --port 5001` and open `http://localhost:5001` |
| Colab: the website does not open | Pop-up blocked, or the server cell failed | Run the "show it inside the notebook" cell, or read `server.log` printed by the launch cell |
| Browser CORS error (only with the optional React app) | React dev server on a different address | Set `api.frontend_origin` in `configs/config.yaml` to the exact address (default `http://localhost:5173`) |
| 413 on upload | File over the size limit | Raise `api.max_upload_mb` or use a smaller image |
| 422 `unreadable_image` | Not a real JPG/PNG, or a picture with more than 64 million pixels (refused before decoding, for memory safety) | Re-export the image, or shrink it. The limit is `api.max_image_pixels` in `configs/config.yaml` |
| `CDR unavailable` for many images | Segmenter not good enough, or the image is a wide photo (disc small) | Check Dice in `results/metrics.json`; use images cropped around the disc; see `results/cdr_preview/` |
| `npm install` fails | Only the optional React app needs it | Ignore it and use `python -m backend.app`, or use Node 18+ and check your network |
| Results look too good | Possible data leakage or tiny test set | Check `data/splits/classification.csv`; report the confidence interval |

## `--quick` mode

`python scripts/run_pipeline.py --quick` uses **synthetic drawings** (not eye photos) at tiny size for two epochs.
It only proves the code runs end to end. Never report its numbers.
