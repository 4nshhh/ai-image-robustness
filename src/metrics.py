"""Reusable binary-classification metrics.

Task labels (fixed, do not change):
    0 = Real
    1 = AI-generated

All functions accept lists, NumPy arrays, or compatible array-likes holding
binary labels. ROC-AUC always uses prediction probabilities/scores for the
AI class, never hard predictions. If no probabilities are supplied, ROC-AUC
is reported as unavailable (``None``) — never invented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "LABEL_REAL",
    "LABEL_AI",
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_roc_auc",
    "compute_confusion_matrix",
    "compute_classification_metrics",
]

LABEL_REAL: int = 0
LABEL_AI: int = 1


def _as_int_array(values: Any, name: str) -> np.ndarray:
    """Convert array-like binary labels to a 1-D int NumPy array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr.astype(int)


def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convert array-like scores to a 1-D float NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def compute_accuracy(y_true: Any, y_pred: Any) -> float:
    """Fraction of correct predictions. Uses ``sklearn.metrics.accuracy_score``."""
    return float(accuracy_score(_as_int_array(y_true, "y_true"), _as_int_array(y_pred, "y_pred")))


def compute_precision(y_true: Any, y_pred: Any) -> float:
    """Precision of the AI class (label 1). Returns 0.0 when undefined."""
    return float(
        precision_score(
            _as_int_array(y_true, "y_true"),
            _as_int_array(y_pred, "y_pred"),
            pos_label=LABEL_AI,
            zero_division=0,
        )
    )


def compute_recall(y_true: Any, y_pred: Any) -> float:
    """Recall of the AI class (label 1), i.e. AI recall. Returns 0.0 when undefined."""
    return float(
        recall_score(
            _as_int_array(y_true, "y_true"),
            _as_int_array(y_pred, "y_pred"),
            pos_label=LABEL_AI,
            zero_division=0,
        )
    )


def compute_f1(y_true: Any, y_pred: Any) -> float:
    """F1 score of the AI class (label 1). Returns 0.0 when undefined."""
    return float(
        f1_score(
            _as_int_array(y_true, "y_true"),
            _as_int_array(y_pred, "y_pred"),
            pos_label=LABEL_AI,
            zero_division=0,
        )
    )


def compute_roc_auc(y_true: Any, y_prob: Any) -> float:
    """ROC-AUC from AI-class probabilities. Returns NaN when undefined.

    Undefined means ``y_true`` contains only one class; there is no valid
    ROC-AUC in that case, so NaN is returned instead of inventing a value.
    """
    yt = _as_int_array(y_true, "y_true")
    yp = _as_float_array(y_prob, "y_prob")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_prob {yp.shape}")
    if np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, yp))


def compute_confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """Confusion matrix with fixed label order ``[Real(0), AI(1)]``.

    Returns:
        2x2 int array ``[[TN, FP], [FN, TP]]`` where rows are true labels
        and columns are predicted labels.
    """
    return confusion_matrix(
        _as_int_array(y_true, "y_true"),
        _as_int_array(y_pred, "y_pred"),
        labels=[LABEL_REAL, LABEL_AI],
    )


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, Any]:
    """Compute the standard metric set for one evaluation.

    Args:
        y_true: True binary labels (0 = Real, 1 = AI).
        y_pred: Predicted binary labels.
        y_prob: Optional AI-class probabilities/scores for ROC-AUC.
            When omitted, ``roc_auc`` is ``None`` (unavailable).

    Returns:
        Dictionary with accuracy, precision, recall, f1, ai_recall,
        ai_false_negative_rate, real_recall, false_positive_rate,
        roc_auc (or None), confusion_matrix (2x2 int array),
        tn/fp/fn/tp counts, n_samples, and n_positive/n_negative support.
    """
    yt = _as_int_array(y_true, "y_true")
    yp = _as_int_array(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    cm = compute_confusion_matrix(yt, yp)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    ai_recall = compute_recall(yt, yp)
    real_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "accuracy": compute_accuracy(yt, yp),
        "precision": compute_precision(yt, yp),
        "recall": compute_recall(yt, yp),
        "f1": compute_f1(yt, yp),
        "ai_recall": ai_recall,
        "ai_false_negative_rate": float(1.0 - ai_recall),
        "real_recall": real_recall,
        "false_positive_rate": float(1.0 - real_recall),
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_samples": int(yt.size),
        "n_positive": int(np.sum(yt == LABEL_AI)),
        "n_negative": int(np.sum(yt == LABEL_REAL)),
    }
    metrics["roc_auc"] = compute_roc_auc(yt, y_prob) if y_prob is not None else None
    return metrics
