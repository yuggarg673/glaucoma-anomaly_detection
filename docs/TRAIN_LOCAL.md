# Train the models on your own computer (VS Code, step by step)

This is the same training as the Colab notebook, run from VS Code on your own machine. It ends with a website that shows
**real** results, and the blue "Demo models" label disappears by itself.

**Read this first.**

- Written from the code, **not yet run end to end**: PyTorch and the datasets were not available where this project was
  built. If a step fails, the error names the file or step. Send me the last 30 lines of the terminal.
- **Time.** On a laptop CPU, plan for **several hours** (my estimate is roughly 2 to 6, more on an older machine). That
  is arithmetic from the size of the networks, not a measurement. With an NVIDIA GPU, or on Colab's free GPU, expect
  tens of minutes. Step 7 shows how to measure it on your machine after the first epoch.
- Needs a computer that stays awake and plugged in, about 8 GB of RAM, and several GB of free disk space.
- Whatever the numbers turn out to be, they are the results. They go in the paper as they are.

---

## Step 1. Open the project in VS Code

1. Open VS Code. **File > Open Folder...** and choose the `glaucoma-anomaly` folder (the one that contains `backend/`,
   `src/` and `scripts/`). If you have not cloned your GitHub repository yet:
   `git clone https://github.com/<your-username>/glaucoma-anomaly.git`
2. If VS Code offers to install the **Python** extension (Microsoft), accept.
3. Press **Ctrl+Shift+P** (Cmd+Shift+P on a Mac), type **Python: Select Interpreter**, and choose the one inside `.venv`.
   If there is no `.venv` yet, make it in the terminal first (next point), then select it.
4. Open a terminal inside VS Code: **Terminal > New Terminal**. Every command below is typed there, from the project
   folder. If there is no `.venv` yet:
   ```
   python -m venv .venv
   ```
   Windows: `.venv\Scripts\activate`. Mac/Linux: `source .venv/bin/activate`.
   The prompt should start with `(.venv)`. If PowerShell says "running scripts is disabled", open a Command Prompt
   terminal instead (the arrow next to the **+** in the terminal panel) and activate there.
5. Install the requirements once (a few minutes, PyTorch is large):
   ```
   pip install -r requirements.txt
   ```

## Step 2. Check what your computer has

```
python -c "import torch; print('torch', torch.__version__, '| NVIDIA GPU:', torch.cuda.is_available())"
```

- `NVIDIA GPU: False` is normal on most laptops. Training will use the CPU, slowly. Go on to step 4, or use Colab.
- `NVIDIA GPU: True`: training will print `device=cuda` and be much faster. Nothing to do.
- Apple Silicon Mac: PyTorch uses the GPU (`mps`) automatically. This path is untested here. If it stops with an error that
  mentions MPS, set `device: cpu` in `configs/config.yaml`.

Also make sure the computer will not go to sleep while it trains (change the power settings while plugged in). A sleeping
computer stops the run.

## Step 3. Only if you have an NVIDIA GPU: install the GPU build of PyTorch

`requirements.txt` installs the **CPU** build of PyTorch on purpose (it is smaller, and Streamlit Cloud needs it). With an
NVIDIA GPU, replace it:

1. `pip uninstall -y torch torchvision`
2. Open https://pytorch.org/get-started/locally/ , choose **Pip**, your system and the CUDA version it offers, and run
   the command it shows.
3. Run the check from step 2 again. It must say `NVIDIA GPU: True`.

Do not run `pip install -r requirements.txt` again afterwards: it would put the CPU build back.

## Step 4. Get the data

The datasets are not in GitHub (size and licences). Two folders, both inside the project:

```
data/raw/acrima/    all ACRIMA images (the glaucoma ones have "_g_" in the file name)
data/raw/g1020/     G1020 images and their annotation .json files (sub-folders are fine)
```

1. **ACRIMA (open).** Try the automatic download:
   ```
   python scripts/download_data.py
   ```
   It saves ACRIMA if it can, and prints the manual steps if it cannot (that is expected, not a bug). Manual: download from
   https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135 ,
   unzip, and put the images in `data/raw/acrima/`.
2. **G1020 (you download it yourself).** Official page https://www.dfki.uni-kl.de/g1020 , or the Kaggle copy listed in
   [DATA.md](DATA.md) (third party, free account). Unzip it into `data/raw/g1020/`. In VS Code you can drag the unzipped
   folder into the Explorer panel on the left, onto `data/raw/g1020`.

## Step 5. Check the data before training

```
python scripts/inspect_data.py
```

It must report:

- **ACRIMA: 705 images (396 glaucoma, 309 normal).**
- **G1020: about 1,020 images and about 1,020 `.json` files that contain `disc` and `cup` labels.**

If the counts are wrong, or G1020 shows no `disc`/`cup` labels, **stop**. Do not train on data the code cannot read.
Send me the output: the converter may need adjusting to the real file layout (I could not download G1020 to check it).

## Step 6. Optional smoke test (check the setup before a long run)

This trains tiny models on synthetic drawings. It proves that PyTorch and the code work on your machine. The script's own
note says it takes minutes; I have not timed it.

```
python scripts/run_pipeline.py --quick
```

Then **delete what it made**, so nothing from it is mixed into the real run:

- Windows PowerShell: `Remove-Item -Recurse -Force checkpoints, results\quick`
- Mac/Linux: `rm -rf checkpoints results/quick`

(If you leave it, the website would show the blue **Demo models** label, which is what that label is for.)

## Step 7. The real training

```
python scripts/run_pipeline.py
```

Run it in the terminal (not in the Python Interactive window). It runs these steps in order and stops at the first one that
fails, with `STEP FAILED` and the step's name:

| Order | Step | What it does | Makes |
| --- | --- | --- | --- |
| 1 | `inspect_data` | counts the files again | `docs/DATA_NOTES.md` |
| 2 | `prepare_segmentation_data` | turns G1020 outlines into cropped images and masks, and lists every file it skips | `data/processed/seg/` |
| 3 | `make_splits` | splits ACRIMA into train / validation / test | `data/splits/` |
| 4 | `train_segmenter` (up to 40 epochs) | learns to outline the disc and cup on G1020 | `checkpoints/segmenter.pt` + `.meta.json` |
| 5 | `train_autoencoder` (up to 60 epochs) | learns to redraw **healthy** ACRIMA images | `checkpoints/autoencoder.pt` + `.meta.json` |
| 6 | `train_baseline` (up to 25 epochs) | a supervised ResNet18 for the paper's comparison; needs internet once to download ImageNet weights | baseline metrics |
| 7 | `fit_fusion` | fits the cut-offs on the validation split, scores the test split once, writes the results table | `checkpoints/fusion.json`, `checkpoints/pixel_thresholds.json`, `results/` |

The website needs only steps 4, 5 and 7 (the baseline is for the paper's comparison table).

**What to watch.** The first lines say `device=cpu` or `device=cuda`. Then one line per epoch: the autoencoder prints
`train_l1` and `val_l1(healthy)`, the segmenter prints `val_dice disc=... cup=...`. Both should fall (or, for Dice, rise).
They stop early when the validation score stops improving, so they often finish before the maximum number of epochs.
**To estimate the total time:** note how long the first epoch took and multiply by the maximum (60 and 40), then expect
somewhat less. The full log is also in `logs/`.

To stop a run: **Ctrl+C** in the terminal. Closing VS Code or putting the computer to sleep also stops it. There is no
resume: running `run_pipeline.py` again starts from the beginning.

**If one step fails late** (for example the baseline cannot download its weights), you do not have to repeat the earlier
hours. Run the remaining steps by hand, with the `src` folder on the path:

- PowerShell: `$env:PYTHONPATH="src"` then `python -m glaucoma.train.train_baseline` then `python -m glaucoma.train.fit_fusion`
- Command Prompt: `set PYTHONPATH=src` then the same two commands
- Mac/Linux: `PYTHONPATH=src python -m glaucoma.train.train_baseline` then `PYTHONPATH=src python -m glaucoma.train.fit_fusion`

**If it says out of memory** (or the process is "Killed"): in `configs/config.yaml` lower `batch_size` under
`autoencoder:` from 16 to 8, and run again.

## Step 8. Look at the results before you trust them

```
python scripts/preview_cdr.py --n 24
```

- Open the pictures in `results/cdr_preview/` (click a `.png` in the Explorer panel). The outlines should sit on the optic
  disc and cup. ACRIMA has no outlines of its own, so its cup-to-disc ratio is a **model estimate** from the G1020 model.
  If the outlines are far off, say so in the report.
- Open `results/tables/main_results.md` (press **Ctrl+Shift+V** for the formatted view). Those are the numbers for the paper.
  They come from the test split only.
- Read the `CDR CHECK` lines the last step printed. If the autoencoder's AUC is near 0.5 it is not separating glaucoma from
  healthy images: see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and report it honestly.

## Step 9. Run the website with your own models

```
python -m backend.app --open
```

(or `streamlit run streamlit_app.py`). The top-right pill must read **Server online, models loaded** (green). If it
says **demo models** (blue), the checkpoints are from the smoke test: delete the `checkpoints` folder and repeat step 7.
If it says "models not loaded", the files are not in `checkpoints/` next to `backend/`.

## Step 10. Put the trained files on GitHub (needed for Streamlit Cloud)

The `checkpoints/` folder is git-ignored on purpose, so add the six files by force. They should be a few MB each; check the
sizes first (`dir checkpoints` on Windows, `ls -lh checkpoints` on Mac/Linux).

```
git add -f checkpoints/autoencoder.pt checkpoints/autoencoder.meta.json checkpoints/segmenter.pt checkpoints/segmenter.meta.json checkpoints/fusion.json checkpoints/pixel_thresholds.json
git commit -m "Add trained models"
git push
```

Then follow part 3 of [DEPLOY.md](DEPLOY.md). Never commit the datasets (`data/`).

---

## Quick reference

| I want to | Command |
| --- | --- |
| see whether I have a GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| check the data | `python scripts/inspect_data.py` |
| train everything | `python scripts/run_pipeline.py` |
| look at the disc/cup outlines | `python scripts/preview_cdr.py --n 24` |
| start the website | `python -m backend.app --open` |
| check files, models and the running server | `python scripts/doctor.py` |
