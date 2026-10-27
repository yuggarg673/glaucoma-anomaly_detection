# Glaucoma anomaly screening

A website that takes a retinal fundus image and returns:

- a **label** (glaucoma suspected / normal-like),
- an **anomaly score** from a convolutional autoencoder trained only on healthy images,
- the **cup-to-disc ratio (CDR)** from optic disc and cup segmentation,
- a **combined score and confidence**,
- an **annotated image** with the disc and cup outlined and the most unusual regions boxed and labelled.

It is compared against a supervised ResNet18 / EfficientNet-B0 baseline. Based on the ideas in *G-EyeNet* (Pal et al.,
ICIP 2018), changed to use reconstruction error as the diagnostic signal and to fuse it with the CDR.

> **Screening aid only.** Student research project. Not clinically validated, not a medical diagnosis.
> What the numbers mean: [docs/USER_GUIDE.md](docs/USER_GUIDE.md) (also shown in the website).

## Status of this repository (read this first)

| Part | State |
| --- | --- |
| Config, data splits, CDR, fusion, metrics, annotated image, API, history | Written **and tested** (every test that needs no PyTorch passes) |
| G1020 outlines to training masks (`prepare_segmentation_data.py`), segmenter augmentation, CDR sanity check | Written **and tested on fake G1020-style data**. The real G1020 download could not be reached, so its exact file layout is unverified (the converter reports every file it cannot use) |
| PyTorch models and training scripts (autoencoder, baseline, segmenter, fusion fitting) | Written, **not yet run**: the environment it was built in could not install PyTorch. The PyTorch tests exist and were skipped there. |
| Website (`web/`, plain HTML/CSS/JS served by the backend, no Node needed) | Written and **checked in a real browser** (headless Chromium: upload, result, errors, history, phone width, no console or security-policy errors). The browser tests use invented model outputs (`tests/standin.py`) because PyTorch was not available, so they prove the page works, not that the models are good |
| React website (`frontend/`, optional older version) | Written, **never built or opened** (no npm access there). You do not need it: `web/` replaces it |
| Real datasets | Not included. You download them ([docs/DATA.md](docs/DATA.md)) |
| Results (AUC etc.) | **None yet.** They appear in `results/` after you train on real data |

So the first thing to do on your machine is `python -m unittest discover -s tests -v` (the PyTorch tests will now
run) and `python scripts/run_pipeline.py --quick`. If something breaks, `docs/TROUBLESHOOTING.md` and
`python scripts/doctor.py` are there to help.

## Run the website

```bash
pip install -r requirements.txt
python -m backend.app --open              # website and API on http://localhost:5000
```

That is the whole setup: no Node, no build step. The site needs trained models to analyse images:

- **Without them** it still opens, and says plainly that the models are not loaded. It never invents a result.
- **After `--quick`** (a code check on synthetic drawings) it shows a red **TEST models** warning on every page and every
  result, because those numbers mean nothing.
- **With real models** (see "Real run" below, or the Colab notebook) it shows the label, anomaly score, cup-to-disc
  ratio, confidence and an annotated image.

## Quick start

Needs Python 3.10+. Node is only needed for the optional React version in `frontend/`.

```bash
git clone <your-repo-url> && cd glaucoma-anomaly
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/doctor.py                  # what is missing?
python -m unittest discover -s tests -v   # all tests, including PyTorch ones
python scripts/run_pipeline.py --quick    # smoke test on synthetic drawings (numbers meaningless)
```

Real run:

```bash
python scripts/download_data.py           # ACRIMA automatic if possible; prints manual steps for G1020
python scripts/inspect_data.py            # verify counts and mask values  (see docs/DATA.md)
python scripts/run_pipeline.py            # prepares G1020 masks, trains segmenter (on G1020), autoencoder + baseline (on ACRIMA), fits fusion
python scripts/preview_cdr.py             # LOOK at the disc/cup outlines drawn on ACRIMA images
```

**No GPU or PyTorch on your computer?** Use `notebooks/train_on_colab.ipynb` in Google Colab (free GPU). It trains on
ACRIMA + G1020 and gives you a `trained_models.zip` to extract into this project. It has not been run yet either.

Everything writes to `results/` (metrics, `tables/main_results.md`, figures) and `checkpoints/` (git-ignored).

Run the app:

```bash
python -m backend.app --open              # website and API on http://localhost:5000
python -m backend.app --port 5001         # if port 5000 is taken
python scripts/doctor.py                  # checks files, checkpoints and that the running server answers
```

On Linux/macOS, `make setup`, `make quick`, `make train`, `make test`, `make doctor`, `make website` are shortcuts.
Test the website in a real browser (skipped if Playwright is missing):
`pip install playwright && python -m playwright install chromium && python -m unittest tests.test_ui_browser -v`.

The older React version (`frontend/`, `make serve-frontend`) is optional and has never been built; `web/` does the same job.

## How it works

```
image --> autoencoder (healthy-only) --> reconstruction error --> anomaly score ----+
  |                                              |                                  +--> fusion --> label, score, confidence
  +--> U-Net (disc + cup) --> CDR = cup height / disc height -----------------------+
                                  |
                                  +--> annotated PNG (outlines, A1/A2 regions from the error map)
```

**Where the cup-to-disc ratio comes from.** ACRIMA has no cup or disc outlines, so the U-Net is trained on **G1020**
(which has them), then run on every ACRIMA image and on uploaded images. The ACRIMA ratio is therefore a *model
estimate*, not an expert measurement. `python scripts/preview_cdr.py` draws the outlines on ACRIMA images so you can
check them by eye, and the last pipeline step prints a **CDR CHECK** (is the ratio larger for glaucoma than for normal
images? does it beat chance on its own?) and warns if not.

Everything tunable is in `configs/config.yaml`. Thresholds and fusion weights are fitted on the **validation** split
only; the test split is used once for reporting. See [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

```
configs/config.yaml        all hyperparameters and paths
src/glaucoma/              library: data, models, training, eval, cdr, inference
backend/                   Flask app: serves the website and the API (app.py), SQLite history (db.py), explanations (glossary.py)
web/                       the website: index.html + assets/ (plain HTML, CSS and JavaScript; no build step)
frontend/                  optional React + Vite + Tailwind version (never built; not needed)
scripts/                   run_pipeline, download/inspect data, doctor, make_splits
tests/                     unit and API tests
docs/                      USER_GUIDE, DATA, TROUBLESHOOTING, DECISIONS
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` and `GET /assets/*` | the website |
| `GET /health` | status, whether models are loaded, whether they are test models, upload limit |
| `GET /explain` | plain-language explanations used by the website |
| `POST /predict` | multipart field `image` (JPG/PNG). Returns the result and a base64 annotated PNG |
| `GET /history` | recent results (scores only) |

Errors are JSON `{"error": {"code", "message"}}` with 400 / 413 / 415 / 422 / 503 as appropriate.

## Putting it on GitHub

```bash
git init            # already done if you got a zip with .git
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/glaucoma-anomaly.git
git push -u origin main
```

Datasets, trained weights, logs and results are git-ignored on purpose. To share trained weights, use GitHub Releases
or Git LFS rather than committing them. **Add a licence file** that matches your institution's rules before making the
repository public.

## Limitations

The ACRIMA cup-to-disc ratio is estimated by a segmenter trained on G1020 (ACRIMA has no outlines), small public datasets, no clinical validation, anomaly score also rises for poor image quality and other diseases, and
the annotated regions are where the image differs from healthy eyes, not confirmed lesions.

## References

Pal, Moorthy, Shahina. *G-EyeNet*, IEEE ICIP 2018 (https://ieeexplore.ieee.org/document/8451029/).
Diaz-Pinto et al., BioMedical Engineering OnLine 18:29, 2019 (ACRIMA). Bajwa et al., IJCNN 2020 (G1020; verify citation). Orlando et al., arXiv:1910.03667 (REFUGE, optional).
