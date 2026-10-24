# Glaucoma anomaly screening

A web app that takes a retinal fundus image and returns:

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
| Config, data splits, CDR, fusion, metrics, annotated image, API, history, tests | Written **and tested** (31 tests pass) |
| PyTorch models and training scripts (autoencoder, baseline, segmenter, fusion fitting) | Written, **not yet run**: the environment it was built in could not install PyTorch. 6 PyTorch tests exist and were skipped there. |
| React website | Written, **not yet built or opened in a browser** (no npm access there) |
| Real datasets | Not included. You download them ([docs/DATA.md](docs/DATA.md)) |
| Results (AUC etc.) | **None yet.** They appear in `results/` after you train on real data |

So the first thing to do on your machine is `python -m unittest discover -s tests -v` (the 6 PyTorch tests will now
run) and `python scripts/run_pipeline.py --quick`. If something breaks, `docs/TROUBLESHOOTING.md` and
`python scripts/doctor.py` are there to help.

## Quick start

Needs Python 3.10+ and Node 18+.

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
python scripts/download_data.py           # ACRIMA automatic if possible; prints manual steps for REFUGE
python scripts/inspect_data.py            # verify counts and mask values  (see docs/DATA.md)
python scripts/run_pipeline.py            # trains segmenter, autoencoder, baseline, fits fusion
```

**No GPU or PyTorch on your computer?** Use `notebooks/train_on_colab.ipynb` in Google Colab (free GPU). It trains on
both datasets and gives you a `trained_models.zip` to extract into this project. It has not been run yet either.

Everything writes to `results/` (metrics, `tables/main_results.md`, figures) and `checkpoints/` (git-ignored).

Run the app (two terminals):

```bash
python -m backend.app                     # API on http://localhost:5000
cd frontend && cp .env.example .env && npm install && npm run dev    # site on http://localhost:5173
```

On Linux/macOS, `make setup`, `make quick`, `make train`, `make test`, `make doctor`, `make serve-backend`,
`make serve-frontend` are shortcuts.

## How it works

```
image --> autoencoder (healthy-only) --> reconstruction error --> anomaly score ----+
  |                                              |                                  +--> fusion --> label, score, confidence
  +--> U-Net (disc + cup) --> CDR = cup height / disc height -----------------------+
                                  |
                                  +--> annotated PNG (outlines, A1/A2 regions from the error map)
```

Everything tunable is in `configs/config.yaml`. Thresholds and fusion weights are fitted on the **validation** split
only; the test split is used once for reporting. See [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

```
configs/config.yaml        all hyperparameters and paths
src/glaucoma/              library: data, models, training, eval, cdr, inference
backend/                   Flask API (app.py), SQLite history (db.py), user explanations (glossary.py)
frontend/                  React + Vite + Tailwind website
scripts/                   run_pipeline, download/inspect data, doctor, make_splits
tests/                     unit and API tests
docs/                      USER_GUIDE, DATA, TROUBLESHOOTING, DECISIONS
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | status and whether models are loaded |
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

Small public datasets, no clinical validation, anomaly score also rises for poor image quality and other diseases, and
the annotated regions are where the image differs from healthy eyes, not confirmed lesions.

## References

Pal, Moorthy, Shahina. *G-EyeNet*, IEEE ICIP 2018 (https://ieeexplore.ieee.org/document/8451029/).
Diaz-Pinto et al., BioMedical Engineering OnLine 18:29, 2019 (ACRIMA). Orlando et al., arXiv:1910.03667 (REFUGE).
