"""Frozen-encoder linear probe.

Protocol (paper M.3.4):
    person-level mean+std of embeddings -> PCA-50 -> linear/logistic head
    -> 5-fold person-independent CV.

This module only implements the last two steps. Extracting and aggregating
embeddings is the caller's job; `person_aggregate` is a helper for the common
case.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import mae, pearson, roc_auc


def person_aggregate(day_embeddings: np.ndarray, subject_id: np.ndarray) -> np.ndarray:
    """[N, d] day embeddings + [N] subject ids -> [n_subjects, 2d] mean+std."""
    day_embeddings = np.asarray(day_embeddings)
    subject_id = np.asarray(subject_id)
    subjects = np.unique(subject_id)
    out = np.empty((subjects.size, day_embeddings.shape[1] * 2), dtype=np.float64)
    for i, s in enumerate(subjects):
        rows = day_embeddings[subject_id == s]
        out[i] = np.concatenate([rows.mean(axis=0), rows.std(axis=0)])
    return out


def linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    task_type: str = "classification",
    n_components: int = 50,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Run a PCA + linear head with out-of-fold predictions.

    embeddings: [n_samples, d]
    labels:     [n_samples]
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    n_components = int(min(n_components, X.shape[1], max(1, X.shape[0] - 1)))

    if task_type == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components),
            LogisticRegression(max_iter=2000),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return {"roc_auc": roc_auc(y, proba)}

    if task_type == "regression":
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components),
            Ridge(alpha=1.0),
        )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(pipe, X, y, cv=cv)
        return {"pearson": pearson(y, pred), "mae": mae(y, pred)}

    raise ValueError(f"unknown task_type {task_type!r}")
