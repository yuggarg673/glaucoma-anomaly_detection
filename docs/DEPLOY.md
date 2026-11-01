# From a zip to a live Streamlit app

Three parts, in this order: **1. GitHub**, **2. train the models**, **3. Streamlit**. Part 3 needs the files that part 2 makes.

**What is tested and what is not. Read this first.**

- Tested where this was built: the code both front ends share, the Streamlit page's own logic (run against a recording
  stand-in for `streamlit`, then 16 deliberate breakages to check the tests notice them), the Flask website in a real
  browser, and a clean extract of the zip.
- **Not tested anywhere yet:** the page inside real Streamlit (it could not be installed in the build sandbox, which has
  no network), PyTorch training and inference, the Colab notebook, the real layout of the G1020 files, GitHub, and
  Streamlit Community Cloud's build. Numbers marked "about" are arithmetic from the model's layers, not measurements.
- If a step fails, copy the error text. It says more than any description.

---

## Part 1. Put the project on GitHub

**Do not upload the `.zip` file itself.** GitHub would store it as one opaque file, and Streamlit cannot run from it.
Extract it, and publish the *contents* of the folder (the one that holds `README.md`, `backend/`, `src/`).

1. On github.com click **+ > New repository**. Name it `glaucoma-anomaly`. Leave "Add a README", ".gitignore" and "license"
   **unticked** (the project has them). Private is fine for now. Click **Create repository**.
2. Install Git (git-scm.com) or GitHub Desktop. Tell Git who you are, once:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
   To keep your real address out of public commits, use the no-reply address GitHub shows under Settings > Emails.
3. Open a terminal **inside the extracted folder** and choose one:

   **A. Keep the history in the zip.** Its commits show the author "Project Author" and list Claude as co-author.
   ```
   git status                     # should say: nothing to commit, working tree clean
   git remote add origin https://github.com/<your-username>/glaucoma-anomaly.git
   git branch -M main
   git push -u origin main
   ```
   **B. Start fresh under your own name** (no earlier commits):
   ```
   rm -rf .git                    # Windows PowerShell: Remove-Item -Recurse -Force .git
   git init -b main
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/glaucoma-anomaly.git
   git push -u origin main
   ```
   GitHub Desktop instead: **File > Add local repository** (pick the folder), then **Publish repository**.
4. Sign in when asked. GitHub does not accept your account password for git: use the browser sign-in window, or a
   personal access token (Settings > Developer settings).
5. Refresh the repository page. You should see `.github`, `.streamlit`, `backend`, `configs`, `docs`, `notebooks`,
   `scripts`, `src`, `tests`, `web`, `streamlit_app.py`, `requirements.txt`. You should **not** see datasets or `checkpoints/`.

Why not drag the files into the website instead: hidden files (`.gitignore`, `.github`, `.streamlit`) are often skipped
by the file picker, and the browser uploader has limits (as I remember them: 25 MiB per file, 100 files per upload; I
could not open GitHub's help pages to confirm). Git avoids all of that.

| Message | Meaning and fix |
| --- | --- |
| `remote origin already exists` | `git remote set-url origin https://github.com/<you>/glaucoma-anomaly.git` |
| `src refspec main does not match any` | You have no commit yet. Run `git add .` then `git commit -m "Initial commit"` |
| `Please tell me who you are` | Run the two `git config --global` lines from step 2 |
| `refusing to allow a Personal Access Token to create or update workflow` | Your token lacks the `workflow` permission (the project has `.github/workflows/ci.yml`). Tick it, or sign in through the browser window instead |
| A red cross on the Actions tab | The CI workflow has never run before. Open it and read the log; it does not mean the project is broken |

**Before the repository is public:** add a `LICENSE` that your institution allows, and check that no dataset images or
patient photos were added. Datasets have their own licences, which is why `data/` and `checkpoints/` are git-ignored.

---

## Part 2. Train the models on the datasets

The training is already written (`scripts/run_pipeline.py`); the Colab notebook runs it on a free GPU.

| Dataset | Used for | Where |
| --- | --- | --- |
| **ACRIMA** (705 images: 396 glaucoma, 309 normal) | the healthy-only autoencoder, and the final test | open: https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135 (the notebook tries to fetch it itself) |
| **G1020** (1,020 images with disc and cup outlines) | the segmenter that measures the cup-to-disc ratio | official: https://www.dfki.uni-kl.de/g1020 (or the Kaggle copy in `docs/DATA.md`). You download it yourself |

1. **Put G1020 in your Google Drive** as `MyDrive/G1020` (the notebook copies it from there).
2. Go to colab.research.google.com > **File > Upload notebook** > `notebooks/train_on_colab.ipynb`.
   Then **Runtime > Change runtime type > GPU**.
3. Run the cells from the top, one at a time:
   - **1. Get the project**: upload the project zip when asked.
     *Alternative if the repository is public:* replace that cell with
     `!git clone https://github.com/<your-username>/glaucoma-anomaly.git`, `%cd glaucoma-anomaly`, and the same `!pip -q install ...` line.
   - **2. Data** and **3. Check the data**: the counts must read ACRIMA 705 / 396 / 309, and G1020 about 1,020 images with
     about 1,020 `.json` files that contain `disc` and `cup` labels. If not, **stop** and send the output of that cell:
     the converter may need adjusting to the real file layout. Do not train on data the code cannot read.
   - **4. Smoke test**, then the **real run** (`python scripts/run_pipeline.py`) and the CDR preview
     (`python scripts/preview_cdr.py --n 24`). Open the preview pictures: the outlines should sit on the optic disc.
     I could not time the run, so I cannot say how long it takes.
   - **5. Results**: these are the numbers for the paper. If the autoencoder AUC is near 0.5, see
     [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and report it honestly.
   - **6. Download**: you get `trained_models.zip` (`checkpoints/` and `results/`).
4. Unzip `trained_models.zip` **inside the project folder** so `checkpoints/` sits next to `backend/`.

**Do not run `pip install -r requirements.txt` inside Colab.** Colab already has a GPU build of PyTorch, and that file
would replace it with the CPU build.

The six files in `checkpoints/` that the app needs (all are needed):

| File | What it is |
| --- | --- |
| `autoencoder.pt`, `autoencoder.meta.json` | the healthy-only autoencoder and the settings and data it was trained with |
| `segmenter.pt`, `segmenter.meta.json` | the disc/cup U-Net and its settings |
| `fusion.json` | the weights and cut-off that combine anomaly score and cup-to-disc ratio |
| `pixel_thresholds.json` | the cut-offs that decide which regions get a box |

By my arithmetic the two `.pt` files are about 10 MB and 2 MB, far under GitHub's limits. Check the real sizes.
`results/` holds the tables and figures for your paper; the app does not need it.

Training on your own computer instead: `python scripts/download_data.py`, `python scripts/inspect_data.py`,
`python scripts/run_pipeline.py` (a GPU is strongly advised).

---

## Part 3. Put it on Streamlit Community Cloud

**3.1 Try it on your computer first.** In the project folder:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
A page opens at http://localhost:8501. It must **not** say "models not loaded" or show a red "These are test models"
banner. Upload a picture and press **Analyze image**.

**3.2 Put the trained files in the repository.** They are git-ignored on purpose, so add them by force:
```
git add -f checkpoints/autoencoder.pt checkpoints/autoencoder.meta.json checkpoints/segmenter.pt checkpoints/segmenter.meta.json checkpoints/fusion.json checkpoints/pixel_thresholds.json
git commit -m "Add trained models"
git push
```

**3.3 Deploy.** Go to **share.streamlit.io** and sign in with GitHub. Click **Create app** and choose to deploy from a
GitHub repository. Then:

- Repository: `<your-username>/glaucoma-anomaly`   (for a private repository, Streamlit asks you to allow access)
- Branch: `main`
- Main file path: `streamlit_app.py`
- **Advanced settings > Python version: 3.11 or 3.12.** As far as I know it can only be set when you deploy, so choose it now.
- Click **Deploy**. The first build takes several minutes because PyTorch is large.

(The button names may differ slightly: I could not open Streamlit's documentation from where this was written.)

**3.4 Read the build log once.** Open **Manage app** (bottom right) and look at the log:

- It should install from `requirements.txt`. If it says it used `pyproject.toml` (or `uv`) instead, rename
  `pyproject.toml` to `pyproject.toml.off` (the app does not need it), push, and reboot the app.
- PyTorch should download as a `+cpu` file of a few hundred MB. If the log shows many packages named `nvidia-...`, the
  CUDA build was picked and the app will run out of space: see the table below.
- This is also "the server log" or "the app log" that the page's error messages point to ("The analysis failed ... The
  details are in the server log"): the full error text is there, and it is what to send if you need help.

**3.5 After it is live.** Pushing to `main` redeploys the app. Free apps go to sleep when nobody uses them and wake on
the next visit (the first load is slow). Anyone who has the link can open a public app and upload pictures to it.

> **Privacy.** The page does not save pictures and keeps no history, but a deployed app is a public web page. Never
> upload identifiable patient images to it. Use pictures from the public datasets, or pictures you are allowed to share.

### If something goes wrong

| What you see | Cause | Fix |
| --- | --- | --- |
| Build log downloads `nvidia-...` packages, then fails (space or memory) | The CUDA build of PyTorch was chosen | Check that `--extra-index-url https://download.pytorch.org/whl/cpu` is the line above `torch` in `requirements.txt`. If it still happens, replace the two torch lines with `torch==2.8.0+cpu` and `torchvision==0.23.0+cpu` (pins from an example I read while preparing this and can no longer link to; unverified, so if pip says a version does not exist, use the nearest one it lists) |
| "The trained models are not loaded ... Missing checkpoints" | The files are not in the GitHub repository | Do step 3.2 (with `-f`), push, then **Reboot app** |
| Red "These are test models" | You committed models from `--quick` (synthetic drawings) | Train with the real run (part 2), replace the six files, push |
| "A required Python package is missing (...)" | A package is not in `requirements.txt` | Add it, push |
| "Error running app" while analysing, or the app restarts | Probably not enough memory on the free server (I have seen about 2.7 GB quoted for free apps, but could not confirm it) | Try a smaller picture; read the log; if it persists, use a host with more memory (for example a Hugging Face Space) or run it on a college computer |
| "File must be 10.0MB or smaller" | The upload limit (`.streamlit/config.toml` and `api.max_upload_mb`) | Shrink the picture, or raise both numbers together |
| The app is asleep | Nobody visited for a while | Click the button that wakes it |
| "Models not loaded" even though the files are there | The failure was cached for this server process | Press **Try loading the models again**, or reboot the app |

---

## Keeping things in step

- `streamlit_app.py` and the Flask website (`python -m backend.app`) use the same checking, analysis and wording code
  (`backend/service.py`, `present.py`, `advice.py`, `glossary.py`), so they accept and refuse the same pictures.
- The Streamlit page keeps no history; the Flask website keeps scores (never pictures) in `data/predictions.sqlite3`.
- Run the tests with `python -m unittest discover -s tests`. The Streamlit tests do not need Streamlit installed.
