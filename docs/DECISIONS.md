# Design decisions

- **Healthy-only autoencoder, skip connections off.** Skip connections let the network copy anomalies to its output,
  which would hide them. A flag exists for an ablation (`autoencoder.skip_connections`).
- **L1 reconstruction loss.** The G-EyeNet paper found L1 best among the losses it tried.
- **ROI cropping off by default.** ACRIMA images are already cropped around the optic disc (`data.roi_crop: false`).
- **Fusion = logistic regression on validation data.** Inputs are min-max normalised anomaly score and CDR; threshold
  by Youden's J on validation. When CDR is missing, an anomaly-only model is used. Nothing is tuned on the test split.
- **Confidence = distance from the threshold**, not a calibrated probability. It is described that way everywhere
  (`backend/glossary.py`). Calibrating it would need a larger validation set.
- **Annotated image = reconstruction-error regions + disc/cup outlines.** Region thresholds are percentiles of the
  smoothed pixel error on healthy validation images (99th = moderate, 99.9th = high). It is a localisation aid, not a
  lesion detector.
- **Splits.** ACRIMA: stratified 70/15/15 with a fixed seed, saved to CSV, leakage-checked. REFUGE: its own
  train/validation/test folders if the folder names say so, else a seeded split.
- **API has no fake mode.** If the models are missing `/predict` returns 503 with instructions.
- **Images are never stored**, only scores and the file name.
- **No CORS package.** A small check in `backend/app.py` allows only the configured frontend origin.

- **CDR for ACRIMA (decided with the user).** The user can train on ACRIMA but REFUGE was not downloadable, and ACRIMA has
  no disc/cup masks. The segmenter is trained on G1020 (disc/cup polygons, cropped around the disc to resemble ACRIMA's
  crops) and applied to ACRIMA to get its CDR, which is fused with the autoencoder score. Consequences: the ACRIMA CDR is
  predicted, so errors in the segmenter feed into the fusion; there is a domain gap between G1020 and ACRIMA images that
  `scripts/preview_cdr.py` makes visible; the paper must say this. A classical (no-training) CDR estimate was tried and
  rejected: mean absolute error 0.24 on simple synthetic drawings (`src/glaucoma/classical_cdr.py`, not used).
- **Segmenter augmentation.** The segmenter learns on G1020 and is used on ACRIMA and on uploads, so it trains with random
  zoom (0.75x to 1.35x), rotation (+/-20 degrees), shifts, flips, and gamma/contrast/per-channel colour changes
  (`preprocessing.augment_seg`). Masks get the same geometric change, so the cup always stays inside the disc.
- **Converter is strict on purpose.** Training on one wrong mask is worse than skipping the file, so `seg_prepare.py`
  skips (and reports) annotations with no disc, no cup, a cup outside the disc, polygons outside the image, or an image of
  a different size; bounding boxes are never read as the disc.
- **CDR check (`eval/cdr_check.py`).** ACRIMA has no outlines, so the predicted CDR cannot be scored directly. The pipeline
  reports the AUC of the CDR alone, the mean CDR per class, and how often it is missing, and warns when the ratio is
  near chance or inverted. It cannot prove the outlines are right; `scripts/preview_cdr.py` is the visual check.
- **Disc-centred crops only.** The app has no disc-finding stage, so it expects photos cropped around the optic disc like
  ACRIMA's. Wide fundus photographs make the disc small and the CDR unreliable. A disc-localisation step is future work.
- **The website is plain HTML, CSS and JavaScript in `web/`, served by Flask.** The user asked for a working website. A
  React/Vite app needs Node, `npm install` and a second terminal, none of which could be tried where this was built. The
  plain version has one command (`python -m backend.app`), no build, no dependencies, and uses relative URLs so it also
  works behind Colab's proxy. The page is built with `textContent` only (never `innerHTML`) and a strict content security
  policy (`script-src 'self'`, no inline script or style), and tests enforce both. The React version stays as an optional
  extra; it was never built.
- **Models trained on synthetic data are flagged, not hidden.** `--quick` writes real-looking checkpoints from drawings. The
  analyzer reads what each checkpoint was trained on, `/health` and every result say `test_models: true`, and the website
  shows a red warning on every page and result. Without this, a code-check run could be mistaken for a working screener.
- **The website was checked with invented model outputs.** PyTorch could not be installed where the site was built, so
  `tests/standin.py` supplies an error map and disc/cup masks and runs everything after the networks for real (CDR,
  fusion, confidence, annotated image, response). It lives in `tests/` only, labels itself "stand-in (test only)", and the
  app never imports it.
