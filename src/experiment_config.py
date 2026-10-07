"""Experiment configuration structures.

Infrastructure only: these structures describe robustness experiments so
that later baseline/robustness/robust-training runs share one vocabulary.
They contain no results and trigger no dataset, training, or evaluation
work on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CONDITION_CLEAN",
    "CONDITION_TRANSFORMED",
    "TRANSFORMATION_NONE",
    "DEFAULT_SEED",
    "ExperimentConfig",
]

#: Experiment run on untransformed data.
CONDITION_CLEAN: str = "clean"
#: Experiment run on transformed data.
CONDITION_TRANSFORMED: str = "transformed"
#: Transformation name used when no transformation is applied.
TRANSFORMATION_NONE: str = "none"
#: Default random seed, consistent with the project convention.
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Declarative description of one robustness experiment.

    Example:
        ExperimentConfig(
            model_name="resnet50",
            experiment_name="jpeg_q50",
            split="test",
            transformation="jpeg",
            transformation_params={"quality": 50},
            condition="transformed",
        )

    A clean experiment must use ``transformation="none"`` with empty
    ``transformation_params``; a transformed experiment must name an
    actual transformation. This keeps "clean vs transformed" unambiguous.
    """

    model_name: str
    experiment_name: str
    split: str = "test"
    transformation: str = TRANSFORMATION_NONE
    transformation_params: dict[str, Any] = field(default_factory=dict)
    condition: str = CONDITION_CLEAN
    training_condition: str | None = None
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.experiment_name, str) or not self.experiment_name:
            raise ValueError("experiment_name must be a non-empty string")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string")
        if not isinstance(self.transformation, str) or not self.transformation:
            raise ValueError("transformation must be a non-empty string")
        if not isinstance(self.transformation_params, dict):
            raise TypeError("transformation_params must be a dict")
        if self.condition not in (CONDITION_CLEAN, CONDITION_TRANSFORMED):
            raise ValueError(
                f"condition must be {CONDITION_CLEAN!r} or {CONDITION_TRANSFORMED!r}, "
                f"got {self.condition!r}"
            )
        if self.condition == CONDITION_CLEAN:
            if self.transformation != TRANSFORMATION_NONE or self.transformation_params:
                raise ValueError(
                    "clean experiments must use transformation='none' "
                    "with empty transformation_params"
                )
        elif self.transformation == TRANSFORMATION_NONE:
            raise ValueError(
                "transformed experiments must name an actual transformation "
                "(not 'none')"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")

    @property
    def is_clean(self) -> bool:
        """True when this experiment uses untransformed data."""
        return self.condition == CONDITION_CLEAN

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (JSON-safe) of this configuration."""
        return {
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "split": self.split,
            "transformation": self.transformation,
            "transformation_params": dict(self.transformation_params),
            "condition": self.condition,
            "training_condition": self.training_condition,
            "seed": self.seed,
        }
