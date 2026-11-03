"""Small metric helpers used by probes and imputation evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score as _roc_auc_score


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(_roc_auc_score(y_true, y_score))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = float(np.sqrt((yt**2).sum() * (yp**2).sum()))
    return float((yt * yp).sum() / denom) if denom > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (low, high, mean) percentile bootstrap CI."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high), float(values.mean())
