"""See what the trained segmenter does on ACRIMA images (there is no ground truth, so LOOK at them).

  python scripts/preview_cdr.py --n 24

Saves results/cdr_preview/*.png with the predicted disc (teal) and cup (white) outlines and the CDR. Also prints the
CDR for glaucoma vs normal ACRIMA images. If outlines are clearly wrong on many images, the CDR feature is unreliable
for ACRIMA and the report must say so.
"""
import argparse

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch

from glaucoma.cdr import compute_cdr
from glaucoma.config import load_config, resolve
from glaucoma.data.index import absolute, classification_index
from glaucoma.models.segmenter import UNet
from glaucoma.preprocessing import load_image, resize_square, to_chw, to_float01
from glaucoma.utils import get_device, load_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=24)
    a = p.parse_args()
    cfg = load_config(a.config)
    ck = resolve(cfg, "checkpoints")
    meta = load_json(ck / "segmenter.meta.json")
    dev = get_device(cfg["device"])
    seg = UNet(base=meta["config"]["segmenter"]["base_channels"]).to(dev)
    seg.load_state_dict(torch.load(ck / "segmenter.pt", map_location=dev)); seg.eval()
    size = meta["seg_size"]
    df = classification_index(cfg).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    out = resolve(cfg, "results") / "cdr_preview"
    out.mkdir(parents=True, exist_ok=True)
    vals = {0: [], 1: []}
    missing = 0
    for i, row in df.iterrows():
        rgb = resize_square(load_image(absolute(row["path"])), size)
        x = torch.from_numpy(np.ascontiguousarray(to_chw(to_float01(rgb)))).unsqueeze(0).to(dev)
        with torch.no_grad():
            m = (torch.sigmoid(seg(x))[0] > 0.5).cpu().numpy().astype(np.uint8)
        r = compute_cdr(m[0], m[1])
        if r.value is None:
            missing += 1
        else:
            vals[int(row["label"])].append(r.value)
        if i < a.n:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            for mask, col in ((m[0], (106, 111, 27)), (m[1], (255, 255, 255))):   # BGR: teal disc, white cup
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cs, -1, col, 1)
            txt = f"CDR {r.value:.2f}" if r.value is not None else f"CDR n/a: {r.reason}"
            cv2.putText(vis, f"{'G' if row['label'] else 'N'} {txt}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imwrite(str(out / f"{i:03d}.png"), cv2.resize(vis, (size * 2, size * 2)))
    for k, name in ((0, "normal"), (1, "glaucoma")):
        v = vals[k]
        print(f"{name:9s} n={len(v):3d}  mean CDR={np.mean(v) if v else float('nan'):.3f}")
    print(f"CDR unavailable for {missing}/{len(df)} images ({100 * missing / len(df):.1f}%)")
    print(f"Overlays saved in {out}. Open several and check the outlines sit on the disc and cup.")


if __name__ == "__main__":
    main()
