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
