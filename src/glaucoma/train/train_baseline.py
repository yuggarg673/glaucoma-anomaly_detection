"""Train the supervised baseline classifier (ResNet18 / EfficientNet-B0)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import resolve
from ..data.datasets import ImageDataset
from ..data.splits import load_split
from ..eval.metrics import bootstrap_auc_ci, compute_metrics, youden_threshold
from ..models.baseline import build_baseline
from ..utils import update_json
from .common import base_parser, init_run, save_ckpt


@torch.no_grad()
def predict(model, df, size, device, batch_size):
    loader = DataLoader(ImageDataset(df, size, imagenet_norm=True), batch_size=batch_size)
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        probs.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().tolist())
        labels.extend(y.tolist())
    return np.array(probs), np.array(labels)


def main():
    args = base_parser("Train supervised baseline").parse_args()
    cfg, device, log = init_run(args, "train_baseline")
    b, size = cfg["baseline"], cfg["data"]["image_size"]
    df = load_split(cfg, "classification")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    model = build_baseline(b["backbone"], b["pretrained"]).to(device)
    counts = np.bincount(tr.label.values, minlength=2).astype(float)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    loader = DataLoader(ImageDataset(tr, size, train=True, imagenet_norm=True, seed=cfg["seed"]),
                        batch_size=b["batch_size"], shuffle=True)
    best_auc, bad, best_state = -1.0, 0, None
    from sklearn.metrics import roc_auc_score
    for epoch in range(1, b["epochs"] + 1):
        model.train()
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        p, l = predict(model, va, size, device, b["batch_size"])
        auc = roc_auc_score(l, p) if len(np.unique(l)) > 1 else float("nan")
        log.info("epoch %d  loss=%.4f  val_auc=%.4f", epoch, loss.item(), auc)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= b["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pv, lv = predict(model, va, size, device, b["batch_size"])
    pt, lt = predict(model, te, size, device, b["batch_size"])
    thr = youden_threshold(lv, pv)
    m = compute_metrics(lt, pt, thr)
    m["auc_ci95"] = bootstrap_auc_ci(lt, pt, cfg["fusion"]["bootstrap"], cfg["seed"])
    log.info("TEST (baseline): %s", {k: m[k] for k in ("accuracy", "sensitivity", "specificity", "auc")})
    if cfg["quick_mode"]:
        log.warning("--quick uses SYNTHETIC images: these numbers are meaningless and only prove the code runs.")
    save_ckpt(cfg, "baseline", model.state_dict(), {"backbone": b["backbone"], "threshold": thr})
    out = resolve(cfg, "results") / ("quick" if cfg["quick_mode"] else "")
    update_json(out / "metrics.json", "baseline", {"test_metrics": m})


if __name__ == "__main__":
    main()
