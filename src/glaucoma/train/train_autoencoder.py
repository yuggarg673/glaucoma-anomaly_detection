"""Train the autoencoder on HEALTHY images only, then compute anomaly scores for val and test.

Usage: python -m glaucoma.train.train_autoencoder [--quick] [--config ...]
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import resolve
from ..data.datasets import ImageDataset
from ..data.splits import load_split
from ..eval.annotate import smooth_error
from ..eval.metrics import compute_metrics, youden_threshold, bootstrap_auc_ci
from ..eval.scoring import anomaly_score
from ..models.autoencoder import build_autoencoder
from ..utils import save_json, update_json
from .common import base_parser, init_run, save_ckpt


@torch.no_grad()
def score_split(model, df, size, device, batch_size):
    """Per-image anomaly scores for a split (no augmentation)."""
    ds = ImageDataset(df, size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    scores, labels = [], []
    for x, y, _ in loader:
        err = model.error_map(x.to(device)).cpu().numpy()
        for e in err:
            scores.append(anomaly_score(e))
        labels.extend(y.tolist())
    return np.array(scores), np.array(labels)


def main():
    args = base_parser("Train healthy-only autoencoder").parse_args()
    cfg, device, log = init_run(args, "train_autoencoder")
    a, size = cfg["autoencoder"], cfg["data"]["image_size"]

    train_df = load_split(cfg, "classification")
    healthy_train = train_df[(train_df.split == "train") & (train_df.label == 0)]
    val_df = train_df[train_df.split == "val"]
    test_df = train_df[train_df.split == "test"]
    if len(healthy_train) == 0:
        raise RuntimeError("No healthy training images found. Check the dataset labels (see docs/DATA.md).")
    assert (healthy_train.label == 0).all(), "Autoencoder training set must contain only healthy images"
    log.info("healthy train=%d  val=%d (normal=%d glaucoma=%d)  test=%d", len(healthy_train), len(val_df),
             int((val_df.label == 0).sum()), int((val_df.label == 1).sum()), len(test_df))

    model = build_autoencoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a["lr"])
    loss_fn = torch.nn.L1Loss()
    tr_loader = DataLoader(ImageDataset(healthy_train, size, train=True, seed=cfg["seed"]),
                           batch_size=a["batch_size"], shuffle=True, drop_last=len(healthy_train) > a["batch_size"])
    val_healthy = val_df[val_df.label == 0]
    va_loader = DataLoader(ImageDataset(val_healthy, size, train=False), batch_size=a["batch_size"])

    best, bad, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, a["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for x, _, _ in tr_loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        model.eval()
        with torch.no_grad():
            v = [loss_fn(model(x.to(device)), x.to(device)).item() * len(x) for x, _, _ in va_loader]
        v_loss = sum(v) / max(len(val_healthy), 1)
        history.append({"epoch": epoch, "train_l1": tot / n, "val_l1": v_loss})
        log.info("epoch %d  train_l1=%.4f  val_l1(healthy)=%.4f", epoch, tot / n, v_loss)
        if v_loss < best - 1e-5:
            best, bad = v_loss, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a["patience"]:
                log.info("early stopping at epoch %d", epoch)
                break
    model.load_state_dict(best_state)

    s_val, y_val = score_split(model, val_df, size, device, a["batch_size"])
    s_test, y_test = score_split(model, test_df, size, device, a["batch_size"])
    res = {"history": history,
           "val_mean_error": {"normal": float(s_val[y_val == 0].mean()), "glaucoma": float(s_val[y_val == 1].mean())},
           "test_mean_error": {"normal": float(s_test[y_test == 0].mean()), "glaucoma": float(s_test[y_test == 1].mean())}}
    thr = youden_threshold(y_val, s_val)          # threshold from VALIDATION only
    m = compute_metrics(y_test, s_test, thr)
    m["auc_ci95"] = bootstrap_auc_ci(y_test, s_test, cfg["fusion"]["bootstrap"], cfg["seed"])
    res["test_metrics"] = m
    log.info("TEST (anomaly score only): %s", {k: m[k] for k in ("accuracy", "sensitivity", "specificity", "auc")})
    if cfg["quick_mode"]:
        log.warning("--quick uses SYNTHETIC images: these numbers are meaningless and only prove the code runs.")

    # healthy-validation pixel error statistics -> thresholds for the annotated image
    vh_loader = DataLoader(ImageDataset(val_healthy, size), batch_size=a["batch_size"])
    an = cfg["annotate"]
    pix = []
    for x, _, _ in vh_loader:
        for e in model.error_map(x.to(device)).cpu().numpy():
            pix.append(smooth_error(e, an["blur_sigma"]).ravel())   # same smoothing annotate() applies
    pix = np.concatenate(pix)
    thresholds = {"moderate": float(np.percentile(pix, an["moderate_percentile"])),
                  "high": float(np.percentile(pix, an["high_percentile"]))}

    save_ckpt(cfg, "autoencoder", model.state_dict(), {"image_size": size})
    out = resolve(cfg, "results") / ("quick" if cfg["quick_mode"] else "")
    save_json({"val_scores": s_val.tolist(), "val_labels": y_val.tolist(), "test_scores": s_test.tolist(),
               "test_labels": y_test.tolist(), "pixel_thresholds": thresholds}, out / "autoencoder_scores.json")
    update_json(out / "metrics.json", "autoencoder", res)
    log.info("saved checkpoints/autoencoder.pt and %s", out / "metrics.json")


if __name__ == "__main__":
    main()
