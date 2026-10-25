"""Fit the fusion model on the VALIDATION split, then evaluate all systems on TEST.

Needs the trained autoencoder and segmenter. Writes checkpoints/fusion.json,
checkpoints/pixel_thresholds.json and results/tables/main_results.md.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader  # noqa: F401  (kept for parity with other scripts)

from ..cdr import compute_cdr
from ..config import resolve
from ..data.index import absolute
from ..data.splits import load_split
from ..eval.cdr_check import cdr_sanity
from ..eval.fusion import FusionModel
from sklearn.metrics import roc_auc_score

from ..eval.metrics import bootstrap_auc_ci, compute_metrics
from ..eval.scoring import anomaly_score
from ..models.autoencoder import build_autoencoder
from ..models.segmenter import UNet
from ..preprocessing import load_image, resize_square, to_chw, to_float01
from ..utils import load_json, save_json, update_json
from .common import base_parser, ckpt_dir, init_run


def _t(rgb, size, device):
    return torch.from_numpy(np.ascontiguousarray(to_chw(to_float01(resize_square(rgb, size))))).unsqueeze(0).to(device)


@torch.no_grad()
def collect(df, ae, seg, ae_size, seg_size, device):
    """Anomaly score and CDR (NaN when unavailable) for every image in df."""
    scores, cdrs, labels = [], [], []
    for _, row in df.iterrows():
        rgb = load_image(absolute(row["path"]))
        err = ae.error_map(_t(rgb, ae_size, device))[0].cpu().numpy()
        s = (torch.sigmoid(seg(_t(rgb, seg_size, device))[0]) > 0.5).cpu().numpy().astype(np.uint8)
        c = compute_cdr(s[0], s[1])
        scores.append(anomaly_score(err)); cdrs.append(np.nan if c.value is None else c.value)
        labels.append(int(row["label"]))
    return np.array(scores), np.array(cdrs), np.array(labels)


def main():
    args = base_parser("Fit fusion and evaluate").parse_args()
    cfg, device, log = init_run(args, "fit_fusion")
    ck = ckpt_dir(cfg)
    ae_meta, seg_meta = load_json(ck / "autoencoder.meta.json"), load_json(ck / "segmenter.meta.json")
    ae = build_autoencoder(ae_meta["config"]).to(device)
    ae.load_state_dict(torch.load(ck / "autoencoder.pt", map_location=device)); ae.eval()
    seg = UNet(base=seg_meta["config"]["segmenter"]["base_channels"]).to(device)
    seg.load_state_dict(torch.load(ck / "segmenter.pt", map_location=device)); seg.eval()
    ae_size, seg_size = ae_meta["image_size"], seg_meta["seg_size"]

    df = load_split(cfg, "classification")
    val, test = df[df.split == "val"], df[df.split == "test"]
    sv, cv, yv = collect(val, ae, seg, ae_size, seg_size, device)
    st, ct, yt = collect(test, ae, seg, ae_size, seg_size, device)
    log.info("CDR unavailable: val %.1f%%  test %.1f%%", 100 * np.isnan(cv).mean(), 100 * np.isnan(ct).mean())

    fm = FusionModel.fit(sv, cv, yv, cfg["seed"])
    fm.save(ck / "fusion.json")
    scores_file = resolve(cfg, "results") / ("quick" if cfg["quick_mode"] else "") / "autoencoder_scores.json"
    save_json(load_json(scores_file)["pixel_thresholds"], ck / "pixel_thresholds.json")

    preds_t = [fm.predict(s, None if np.isnan(c) else c) for s, c in zip(st, ct)]
    fused_t = np.array([p["fused_score"] for p in preds_t])
    margin_t = np.array([p["margin"] for p in preds_t])
    # accuracy / sensitivity / specificity exactly as the app decides (margin >= 0 means "glaucoma suspected");
    # AUC from the fused probabilities themselves.
    m_fused = compute_metrics(yt, margin_t, 0.0)
    m_fused["auc"] = float(roc_auc_score(yt, fused_t))
    m_fused["threshold"] = "per-case cut-off from the fusion model (see checkpoints/fusion.json)"
    m_fused["auc_ci95"] = bootstrap_auc_ci(yt, fused_t, cfg["fusion"]["bootstrap"], cfg["seed"])
    m_fused["cdr_unavailable_test"] = float(np.isnan(ct).mean())
    m_ae = compute_metrics(yt, st, fm.p["anom_raw_threshold"])
    m_ae["auc_ci95"] = bootstrap_auc_ci(yt, st, cfg["fusion"]["bootstrap"], cfg["seed"])

    # Is the predicted CDR informative on this data? (ACRIMA has no outlines to check it against.)
    cdr_info = cdr_sanity(cv, yv, ct, yt, fm.p.get("cdr_raw_threshold"), cfg["fusion"]["bootstrap"], cfg["seed"])
    for msg in cdr_info["warnings"]:
        log.warning("CDR CHECK: %s", msg)
    log.info("CDR check: validation AUC %s, mean CDR normal %s vs glaucoma %s",
             cdr_info["val_auc"], cdr_info["val_mean_cdr_normal"], cdr_info["val_mean_cdr_glaucoma"])

    out = resolve(cfg, "results") / ("quick" if cfg["quick_mode"] else "")
    update_json(out / "metrics.json", "autoencoder_only", {"test_metrics": m_ae})
    update_json(out / "metrics.json", "autoencoder_plus_cdr", {"test_metrics": m_fused})
    update_json(out / "metrics.json", "cdr_check", cdr_info)
    all_m = load_json(out / "metrics.json")
    rows = [("Supervised baseline", all_m.get("baseline", {}).get("test_metrics")),
            ("Autoencoder anomaly score only", m_ae),
            (f"CDR only (segmenter trained on {cfg['data'].get('seg_source', 'refuge')}; images with a CDR)",
             cdr_info["test_metrics"]),
            ("Autoencoder + CDR fusion", m_fused)]
    lines = ["| Method | Accuracy | Sensitivity | Specificity | AUC-ROC (95% CI) |", "| --- | --- | --- | --- | --- |"]
    for name, m in rows:
        if not m:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a |"); continue
        lo, hi = m["auc_ci95"]
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | "
                     f"{m['auc']:.3f} ({lo:.3f} to {hi:.3f}) |")
    note = "\n\n> SYNTHETIC smoke-test data: numbers are meaningless.\n" if cfg["quick_mode"] else "\n"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "main_results.md").write_text("\n".join(lines) + note, encoding="utf-8")
    log.info("wrote %s", out / "tables" / "main_results.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
