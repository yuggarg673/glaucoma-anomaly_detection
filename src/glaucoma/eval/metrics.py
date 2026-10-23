"""Evaluation metrics: accuracy, sensitivity, specificity, AUC-ROC, bootstrap CI."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _check_two_classes(y_true: np.ndarray) -> None:
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Need both classes (normal and glaucoma) in y_true to compute AUC/threshold. "
            f"Got classes: {np.unique(y_true).tolist()}"
        )


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1. Choose on validation data only."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    _check_two_classes(y_true)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = thr[best]
    if not np.isfinite(t):  # roc_curve's first threshold can be inf
        t = float(y_score.max()) + 1e-9
    return float(t)


def compute_metrics(y_true, y_score, threshold: float) -> dict:
    """Metrics at a fixed threshold (prediction = score >= threshold)."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    _check_two_classes(y_true)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": sens,
        "specificity": spec,
        "auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": int(len(y_true)),
    }


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI for AUC. Resamples that miss a class are skipped."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    _check_two_classes(y_true)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
