"""Reusable orchestration for configured evaluation experiments.

This module connects the existing infrastructure — ``experiment_config``,
``evaluate``, ``metrics`` (via ``evaluate``), and ``results`` — so that a
single call executes one configured evaluation once Person 1 provides a
real model and DataLoader.

Infrastructure only: nothing here trains, loads data, applies
project-level image transformations by itself, or produces project
results. The caller supplies everything the run needs.

Transformation boundary (read carefully):
    Image-level (PIL) transformations such as the ones in
    ``src/transformations.py`` operate before tensor conversion, i.e.
    inside the dataset. That dataset is Person 1's component and is NOT
    modified here. Therefore, for a transformed condition the caller
    provides an already-transformed DataLoader. Optionally, the caller
    may also supply a tensor-level ``batch_transform`` applied to cloned
    input batches inside the run (the dataset itself is never mutated),
    or a custom ``evaluate_fn`` with the same contract as
    ``evaluate_model`` for transformation-aware evaluation strategies
    defined later. No transformation qualities, scales, passes, or
    combinations are hardcoded here — they belong to the
    ``ExperimentConfig`` / caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from src.evaluate import DEFAULT_THRESHOLD, evaluate_model
    from src.experiment_config import ExperimentConfig
    from src.results import ExperimentResult
except ImportError:  # allow `sys.path`-based use of the src/ directory
    from evaluate import DEFAULT_THRESHOLD, evaluate_model
    from experiment_config import ExperimentConfig
    from results import ExperimentResult

__all__ = ["ExperimentOutcome", "run_experiment"]

#: Tensor-level transform applied to cloned input batches (dataset untouched).
BatchTransform = Callable[[torch.Tensor], torch.Tensor]
#: Evaluation callable with the same contract as ``evaluate_model``.
EvaluateFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ExperimentOutcome:
    """Everything one configured run produced.

    ``result`` carries the stored outcome (config metadata + copied
    metrics); ``y_true``/``y_pred``/``y_prob`` keep the raw prediction
    information accessible for later analysis.
    """

    config: ExperimentConfig
    result: ExperimentResult
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    threshold: float
    n_samples: int


def _iter_batches(
    dataloader: Iterable,
    batch_transform: BatchTransform | None,
) -> Iterable:
    """Yield ``(inputs, targets)`` batches, optionally tensor-transformed.

    When ``batch_transform`` is given it is applied to a clone of each
    tensor input batch, so the underlying dataset is never modified
    in place. Non-tensor inputs are rejected because a tensor-level
    transform cannot be applied to them safely.
    """
    for inputs, targets in dataloader:
        if batch_transform is None:
            yield inputs, targets
        else:
            if not torch.is_tensor(inputs):
                raise TypeError(
                    "batch_transform requires tensor inputs, "
                    f"got {type(inputs).__name__}"
                )
            yield batch_transform(inputs.clone()), targets


def run_experiment(
    config: ExperimentConfig,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    batch_transform: BatchTransform | None = None,
    evaluate_fn: EvaluateFn | None = None,
) -> ExperimentOutcome:
    """Execute one configured evaluation experiment.

    Args:
        config: What to call this run (model/split/transformation/
            condition metadata preserved into the result).
        model: PyTorch binary classifier (any architecture supported by
            ``evaluate_model``). Weights are never modified.
        dataloader: Test batches. For transformed conditions this must
            be an already-transformed DataLoader supplied by the caller.
        device: Torch device (or string); defaults to CUDA when
            available, else CPU.
        threshold: Decision threshold on the AI-class probability.
        batch_transform: Optional caller-supplied tensor-level transform
            for input batches (applied to clones; dataset untouched).
            Image-level PIL transformations do NOT belong here — they
            belong in the dataset behind an already-transformed
            DataLoader.
        evaluate_fn: Optional override with the ``evaluate_model``
            contract, for transformation-aware evaluation strategies.
            Defaults to ``evaluate_model``. No metric logic lives here
            either way — metrics come from the existing layer and are
            copied into the result via ``ExperimentResult.from_metrics``.

    Returns:
        ``ExperimentOutcome`` with the config, the ``ExperimentResult``,
        and the raw ``y_true``/``y_pred``/``y_prob`` arrays.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"config must be an ExperimentConfig, got {type(config).__name__}")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if batch_transform is not None and not callable(batch_transform):
        raise TypeError("batch_transform must be callable or None")

    evaluate = evaluate_fn if evaluate_fn is not None else evaluate_model
    batches = dataloader if batch_transform is None else _iter_batches(dataloader, batch_transform)
    evaluation = evaluate(model, batches, device, threshold)
    result = ExperimentResult.from_metrics(config, evaluation["metrics"])
    return ExperimentOutcome(
        config=config,
        result=result,
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        y_prob=evaluation["y_prob"],
        threshold=float(evaluation.get("threshold", threshold)),
        n_samples=int(evaluation.get("n_samples", evaluation["y_true"].size)),
    )
