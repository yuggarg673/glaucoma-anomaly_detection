"""Train the optic disc / cup U-Net and report Dice."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..config import resolve
from ..data.datasets import SegDataset
from ..data.splits import load_split
from ..models.segmenter import UNet, dice_bce_loss, dice_score
from ..utils import update_json
from .common import base_parser, init_run, save_ckpt


@torch.no_grad()
def eval_dice(model, df, size, device, bs):
    loader = DataLoader(SegDataset(df, size), batch_size=bs)
    model.eval()
    tot = torch.zeros(2); n = 0
    for x, y in loader:
        d = dice_score(model(x.to(device)), y.to(device)).cpu()
        tot += d * len(x); n += len(x)
    return (tot / max(n, 1)).tolist()


def main():
    args = base_parser("Train disc/cup segmenter").parse_args()
    cfg, device, log = init_run(args, "train_segmenter")
    s, size = cfg["segmenter"], cfg["data"]["seg_size"]
    df = load_split(cfg, "segmentation")
    tr, va, te = (df[df.split == k] for k in ("train", "val", "test"))
    if len(va) == 0:  # REFUGE ships train/val/test, but fall back gracefully
        va = te
    model = UNet(base=s["base_channels"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    loader = DataLoader(SegDataset(tr, size, train=True, seed=cfg["seed"]), batch_size=s["batch_size"], shuffle=True)
    best, bad, best_state = -1.0, 0, None
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = dice_bce_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        d = eval_dice(model, va, size, device, s["batch_size"])
        mean_d = sum(d) / 2
        log.info("epoch %d  loss=%.4f  val_dice disc=%.3f cup=%.3f", epoch, loss.item(), d[0], d[1])
        if mean_d > best:
            best, bad = mean_d, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s["patience"]:
                break
    model.load_state_dict(best_state)
    d = eval_dice(model, te if len(te) else va, size, device, s["batch_size"])
    log.info("TEST dice disc=%.3f cup=%.3f", d[0], d[1])
    if cfg["quick_mode"]:
        log.warning("--quick uses SYNTHETIC images: these numbers are meaningless and only prove the code runs.")
    save_ckpt(cfg, "segmenter", model.state_dict(), {"seg_size": size})
    out = resolve(cfg, "results") / ("quick" if cfg["quick_mode"] else "")
    update_json(out / "metrics.json", "segmenter", {"test_dice_disc": d[0], "test_dice_cup": d[1]})


if __name__ == "__main__":
    main()
