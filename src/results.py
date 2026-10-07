"""Reusable representation of experiment outcomes.

This layer only organizes and stores results produced elsewhere (e.g. by
``src.evaluate`` / ``src.metrics``). It performs no metric computation of
its own. No values here are project results — every instance is built
from data the caller supplies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from src.experiment_config import ExperimentConfig
except ImportError:  # allow `sys.path`-based use of the src/ directory
    from experiment_config import ExperimentConfig

__all__ = ["ExperimentResult"]

#: Metric keys copied verbatim from a ``compute_classification_metrics`` dict.
_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "ai_recall",
    "ai_false_negative_rate",
    "roc_auc",
)


def _confusion_to_tuples(cm: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Normalize a 2x2 confusion matrix to immutable int tuples.

    Accepts NumPy arrays, nested lists, or nested tuples with layout
    ``[[TN, FP], [FN, TP]]``. Returns None when ``cm`` is None.
    """
    if cm is None:
        return None
    rows = [[int(v) for v in row] for row in cm]
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError(f"confusion matrix must be 2x2, got {rows!r}")
    return ((rows[0][0], rows[0][1]), (rows[1][0], rows[1][1]))


@dataclass(frozen=True)
class ExperimentResult:
    """One stored experiment outcome.

    Scalar metrics are floats in [0, 1] (or None when unavailable, e.g.
    ROC-AUC without probabilities); the confusion matrix is kept as
    immutable ``((TN, FP), (FN, TP))`` tuples so nested information stays
    usable after serialization round-trips.
    """

    experiment_name: str
    model_name: str
    transformation: str = "none"
    transformation_params: Mapping[str, Any] = field(default_factory=dict)
    condition: str = "clean"
    split: str = "test"
    training_condition: str | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    ai_recall: float | None = None
    ai_false_negative_rate: float | None = None
    roc_auc: float | None = None
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]] | None = None
    n_samples: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name:
            raise ValueError("experiment_name must be a non-empty string")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.transformation_params, Mapping):
            raise TypeError("transformation_params must be a mapping")
        if self.n_samples is not None:
            if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int):
                raise TypeError("n_samples must be an int or None")
            if self.n_samples < 0:
                raise ValueError("n_samples must be non-negative")

    @classmethod
    def from_metrics(
        cls,
        config: ExperimentConfig,
        metrics: Mapping[str, Any],
    ) -> ExperimentResult:
        """Build a result from a config plus a metrics dict.

        The ``metrics`` mapping is expected to be the output of
        ``compute_classification_metrics`` (as returned inside
        ``evaluate_model``), but any mapping with the same keys works.
        Metric values are copied, never recomputed.
        """
        if not isinstance(config, ExperimentConfig):
            raise TypeError(f"config must be an ExperimentConfig, got {type(config).__name__}")
        values = {key: metrics.get(key) for key in _METRIC_KEYS}
        return cls(
            experiment_name=config.experiment_name,
            model_name=config.model_name,
            transformation=config.transformation,
            transformation_params=dict(config.transformation_params),
            condition=config.condition,
            split=config.split,
            training_condition=config.training_condition,
            confusion_matrix=_confusion_to_tuples(metrics.get("confusion_matrix")),
            n_samples=metrics.get("n_samples"),
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Nested dict view with JSON-safe types.

        The confusion matrix becomes a list of lists; transformation
        parameters become a plain dict. Suitable for ``json.dump``.
        """
        cm = self.confusion_matrix
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "transformation": self.transformation,
            "transformation_params": dict(self.transformation_params),
            "condition": self.condition,
            "split": self.split,
            "training_condition": self.training_condition,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ai_recall": self.ai_recall,
            "ai_false_negative_rate": self.ai_false_negative_rate,
            "roc_auc": self.roc_auc,
            "confusion_matrix": [list(row) for row in cm] if cm is not None else None,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentResult:
        """Rebuild a result previously produced by :meth:`to_dict`."""
        payload = dict(data)
        payload["confusion_matrix"] = _confusion_to_tuples(payload.get("confusion_matrix"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_record(self) -> dict[str, Any]:
        """Flat single-level dict for one pandas DataFrame row / CSV line.

        The confusion matrix is expanded to ``tn``/``fp``/``fn``/``tp``
        columns and each transformation parameter becomes a ``param_<name>``
        column, so records from different experiments stay comparable.
        """
        record: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "transformation": self.transformation,
            "condition": self.condition,
            "split": self.split,
            "training_condition": self.training_condition,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ai_recall": self.ai_recall,
            "ai_false_negative_rate": self.ai_false_negative_rate,
            "roc_auc": self.roc_auc,
            "n_samples": self.n_samples,
        }
        cm = self.confusion_matrix
        record["tn"], record["fp"], record["fn"], record["tp"] = (
            (cm[0][0], cm[0][1], cm[1][0], cm[1][1]) if cm is not None else (None, None, None, None)
        )
        for name, value in self.transformation_params.items():
            record[f"param_{name}"] = value
        return record
