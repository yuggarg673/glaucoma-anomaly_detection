# Troubleshooting

First step for any problem: `python scripts/doctor.py`. It prints PASS / WARN / FAIL and the fix for each item.
Logs of every training script are in `logs/`.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No module named 'torch'` | PyTorch not installed | `pip install -r requirements.txt` (CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`) |
| `ACRIMA folder not found` | Data not downloaded | `python scripts/download_data.py`, then see `docs/DATA.md` |
| `Found no REFUGE image+mask pairs` | Wrong folder, or mask names differ from image names | Run `python scripts/inspect_data.py`; masks must share the image's file-name stem |
| Segmenter learns nothing (Dice near 0) | Mask pixel values differ from cup=0/disc=128/background=255 | Check `docs/DATA_NOTES.md`, edit `masks_from_refuge` in `src/glaucoma/cdr.py` |
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
| `CDR unavailable` for many images | Segmenter not good enough | Check Dice in `results/metrics.json`; train longer or with more data |
| `npm install` fails | No internet / old Node | Use Node 18 or newer and check your network |
| Results look too good | Possible data leakage or tiny test set | Check `data/splits/classification.csv`; report the confidence interval |

## `--quick` mode

`python scripts/run_pipeline.py --quick` uses **synthetic drawings** (not eye photos) at tiny size for two epochs.
It only proves the code runs end to end. Never report its numbers.
