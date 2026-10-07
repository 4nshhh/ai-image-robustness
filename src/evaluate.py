"""Reusable evaluation for binary Real-vs-AI image classifiers.

This module only evaluates. It never trains, never modifies model weights,
never modifies the dataset, and never applies image transformations — the
caller supplies a DataLoader that already yields the (possibly transformed)
test data.

Label mapping (fixed, do not change):
    0 = Real
    1 = AI-generated
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from src.metrics import compute_classification_metrics
except ImportError:  # allow `sys.path`-based use of the src/ directory
    from metrics import compute_classification_metrics

__all__ = ["evaluate_model", "DEFAULT_THRESHOLD"]

#: Default decision threshold on the AI-class probability.
DEFAULT_THRESHOLD: float = 0.5


def _ai_probability(outputs: torch.Tensor) -> torch.Tensor:
    """Map raw model outputs to AI-class (label 1) probabilities.

    Supported output shapes per batch of size B:
    - ``[B]`` or ``[B, 1]``: single logit -> ``sigmoid``.
    - ``[B, 2]``: two-class logits -> ``softmax``, take column 1.

    Raises:
        ValueError: For any other output shape.
    """
    if outputs.ndim == 1:  # [B] single logits
        return torch.sigmoid(outputs)
    if outputs.ndim == 2 and outputs.shape[1] == 1:  # [B, 1] single logits
        return torch.sigmoid(outputs.squeeze(1))
    if outputs.ndim == 2 and outputs.shape[1] == 2:  # [B, 2] class logits
        return torch.softmax(outputs, dim=1)[:, 1]
    raise ValueError(
        "Unsupported model output shape "
        f"{tuple(outputs.shape)}; expected [B], [B, 1], or [B, 2]."
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate a binary classifier on a DataLoader.

    Args:
        model: PyTorch binary classifier with one of the output shapes
            documented in :func:`_ai_probability`. Any architecture is
            accepted; no checkpoint is loaded here.
        dataloader: Yields ``(inputs, targets)`` batches where targets are
            binary labels (0 = Real, 1 = AI). Any dataset implementation is
            accepted; batches are not transformed here.
        device: Torch device (or device string). Defaults to CUDA when
            available, else CPU.
        threshold: Decision threshold on the AI-class probability for the
            predicted label (default 0.5).

    Returns:
        Dictionary with:
        - ``metrics``: output of ``compute_classification_metrics``
          (includes ROC-AUC, since probabilities are always collected).
        - ``y_true`` / ``y_pred``: int NumPy arrays.
        - ``y_prob``: float NumPy array of AI-class probabilities.
        - ``threshold`` and ``n_samples``.

    The model is set to eval mode under ``torch.no_grad()``; its original
    training/eval state is restored afterwards and its parameters are never
    modified.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    was_training = model.training
    model.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = _ai_probability(outputs).detach().cpu().numpy().astype(float)
            preds = (probs >= threshold).astype(int)
            all_true.append(np.asarray(targets).reshape(-1).astype(int))
            all_pred.append(preds.reshape(-1))
            all_prob.append(probs.reshape(-1))

    model.train(was_training)

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
    y_prob = np.concatenate(all_prob) if all_prob else np.array([], dtype=float)

    return {
        "metrics": compute_classification_metrics(y_true, y_pred, y_prob),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
    }
