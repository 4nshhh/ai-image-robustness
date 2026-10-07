"""Baseline model constructors: ResNet-50 and EfficientNet-B0.

Both models start from ImageNet-pretrained weights (transfer learning) and
have their original classification head replaced with a 2-class linear
layer, so outputs are logits of shape ``[B, 2]`` suitable for
``CrossEntropyLoss`` with targets ``0 = Real`` / ``1 = AI``.

Model-specific notes:

- ResNet-50: ``model.fc`` replaced with ``nn.Linear(fc.in_features, 2)``.
- EfficientNet-B0: ``model.classifier[1]`` (the final linear layer)
  replaced with ``nn.Linear(in_features, 2)``; the dropout in
  ``classifier[0]`` is kept.
- Both models take the ``224 x 224`` RGB input produced by
  ``src.preprocessing``; no model-specific preprocessing is needed.

No custom architectures are introduced. Nothing here trains or evaluates;
this module only builds (and optionally saves/loads) models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn
from torchvision import models

__all__ = [
    "MODEL_NAMES",
    "NUM_CLASSES",
    "build_model",
    "build_resnet50",
    "build_efficientnet_b0",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
]

#: Supported model names for :func:`build_model`.
MODEL_NAMES: tuple[str, ...] = ("resnet50", "efficientnet_b0")

#: Binary task: Real (0) vs AI-generated (1).
NUM_CLASSES: int = 2

PathLike = Union[str, Path]


def _resolve_weights_arg(weights_enum, pretrained: bool):
    """Return kwargs for the torchvision factory for pretrained control."""
    if pretrained:
        try:
            # Modern API: default ImageNet weights for this architecture.
            default = weights_enum.DEFAULT
            return {"weights": default}
        except AttributeError:
            return {"pretrained": True}
    try:
        # Explicitly request random initialization on the modern API.
        weights_enum.DEFAULT  # validated to exist; value unused
        return {"weights": None}
    except AttributeError:
        return {"pretrained": False}


def build_resnet50(pretrained: bool = True) -> nn.Module:
    """Build ResNet-50 with a 2-class head.

    Args:
        pretrained: If True, initialize from ImageNet weights;
            otherwise random initialization (useful for smoke tests
            without downloading weights).

    Returns:
        ResNet-50 outputting ``[B, 2]`` logits.
    """
    kwargs = _resolve_weights_arg(models.ResNet50_Weights, pretrained)
    model = models.resnet50(**kwargs)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with a 2-class head.

    Args:
        pretrained: If True, initialize from ImageNet weights;
            otherwise random initialization.

    Returns:
        EfficientNet-B0 outputting ``[B, 2]`` logits.
    """
    kwargs = _resolve_weights_arg(models.EfficientNet_B0_Weights, pretrained)
    model = models.efficientnet_b0(**kwargs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


_BUILDERS = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """Build a baseline classifier by explicit name (no duplicated logic).

    Args:
        model_name: One of ``"resnet50"`` / ``"efficientnet_b0"``.
        pretrained: ImageNet initialization switch (see builders).

    Returns:
        Model outputting ``[B, 2]`` logits.

    Raises:
        ValueError: For unknown model names.
    """
    key = model_name.lower()
    if key not in _BUILDERS:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}."
        )
    return _BUILDERS[key](pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters (useful for model-selection notes)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    model_name: str,
    extra: dict | None = None,
) -> None:
    """Save a checkpoint bundle: architecture name, weights, extras.

    Args:
        path: Destination ``.pth`` path (parent dirs are created).
        model: Model whose ``state_dict`` is stored.
        model_name: Architecture name (needed to rebuild before loading).
        extra: Optional extra state (epoch, history, config, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict = {
        "model_name": model_name,
        "num_classes": NUM_CLASSES,
        "label_map": {"real": 0, "ai": 1},
        "state_dict": model.state_dict(),
    }
    if extra:
        bundle["extra"] = extra
    torch.save(bundle, path)


def load_checkpoint(
    path: PathLike,
    model_name: str | None = None,
    pretrained: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Rebuild a model and load checkpoint weights into it.

    Args:
        path: Checkpoint ``.pth`` path.
        model_name: Architecture name. If ``None``, it is read from the
            bundle (checkpoints saved by :func:`save_checkpoint`).
        pretrained: Passed to the builder before weights are overwritten;
            ``False`` (default) avoids a weights download at load time.
        map_location: Device mapping for ``torch.load``.

    Returns:
        ``(model, bundle)`` with loaded weights in eval-safe state
        (caller decides train/eval mode).
    """
    bundle = torch.load(Path(path), map_location=map_location)
    name = model_name or bundle.get("model_name")
    if name is None:
        raise ValueError("Checkpoint has no model_name and none was given.")
    model = build_model(name, pretrained=pretrained)
    model.load_state_dict(bundle["state_dict"])
    return model, bundle
