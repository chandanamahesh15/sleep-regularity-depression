"""Screening-oriented metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    two_classes = len(np.unique(y)) == 2
    return {
        "auroc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI for AUROC and balanced accuracy, resampling *users*."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    au, ba = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        au.append(roc_auc_score(y[idx], p[idx]))
        ba.append(balanced_accuracy_score(y[idx], (p[idx] >= 0.5).astype(int)))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "auroc_ci": [float(v) for v in np.percentile(au, q)] if au else [float("nan")] * 2,
        "balanced_accuracy_ci": [float(v) for v in np.percentile(ba, q)] if ba else [float("nan")] * 2,
    }


def summarise(runs: list[dict]) -> dict:
    """Mean and std of each metric across CV repeats."""
    keys = [k for k in runs[0] if k != "n"]
    return {
        k: {"mean": float(np.nanmean([r[k] for r in runs])), "std": float(np.nanstd([r[k] for r in runs]))}
        for k in keys
    }
