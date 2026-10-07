"""Reusable clean-baseline training pipeline.

One training implementation serves every baseline model (ResNet-50,
EfficientNet-B0): the caller picks the architecture, this module runs the
same loop, the same loss, and the same checkpoint discipline.

Methodology rules enforced here:

- Best-checkpoint selection uses VALIDATION accuracy only. The test split
  is never loaded, never evaluated, and never influences training.
- Loss is ``CrossEntropyLoss`` over the 2-class logits (0 = Real, 1 = AI).
- History is recorded, never invented: only epoch aggregates computed
  from real batches are stored.

Reproducibility: :func:`set_seed` seeds Python/NumPy/PyTorch and requests
deterministic cuDNN. Exact bit-reproducibility across machines (in
particular on CUDA) is NOT guaranteed; nondeterminism sources are
documented in :func:`set_seed`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from src.dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, GenImageDataset
    from src.model import MODEL_NAMES, build_model
    from src.preprocessing import IMAGE_SIZE, get_transform
except ImportError:  # allow `sys.path`-based use of the src/ directory
    from dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, GenImageDataset
    from model import MODEL_NAMES, build_model
    from preprocessing import IMAGE_SIZE, get_transform

__all__ = [
    "SEED",
    "TrainConfig",
    "set_seed",
    "get_device",
    "make_dataloaders",
    "train_one_epoch",
    "validate",
    "fit",
]

#: Project-wide default seed.
SEED: int = 42

PathLike = Union[str, Path]


@dataclass
class TrainConfig:
    """All knobs for one baseline training run (single source of truth)."""

    model_name: str = "resnet50"
    pretrained: bool = True
    manifest_path: PathLike = DEFAULT_MANIFEST
    data_root: PathLike = DEFAULT_ROOT
    image_size: int = IMAGE_SIZE
    batch_size: int = 32
    num_workers: int = 0
    augment: bool = False
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    seed: int = SEED
    device: str | None = None  # None -> auto (CUDA if available else CPU)
    checkpoint_dir: PathLike = field(
        default_factory=lambda: Path("models/baseline")
    )
    # Optional index subsets for lightweight smoke tests. Full runs MUST
    # leave these as None (train on all train/val rows). Test data can
    # never be selected here: make_dataloaders has no test path at all.
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (Path objects rendered as strings)."""
        data = asdict(self)
        for key in ("manifest_path", "data_root", "checkpoint_dir"):
            data[key] = str(data[key])
        return data


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch; request deterministic cuDNN.

    Known nondeterminism limits (documented, not hidden):

    - DataLoader workers are seeded separately in :func:`make_dataloaders`.
    - On CUDA, some operations have no deterministic implementation; exact
      bit-reproducibility across GPU models/drivers is not guaranteed even
      with these flags. CPU runs are the reproducibility reference.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str | None = None) -> torch.device:
    """Resolve the compute device: CUDA when available, else CPU.

    Args:
        preference: Explicit ``"cuda"`` / ``"cpu"`` request, or ``None``
            for automatic selection.

    Raises:
        ValueError: For unknown preferences, or ``"cuda"`` when no GPU
            is available.
    """
    if preference is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preference = preference.lower()
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")
    if preference not in ("cuda", "cpu"):
        raise ValueError(f"Unknown device {preference!r}; use 'cuda' or 'cpu'.")
    return torch.device(preference)


def _seeded_worker_init(worker_seed: int):
    """Worker init fn: derive NumPy/Python seeds from the worker id."""

    def _init(worker_id: int) -> None:
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)

    return _init


def make_dataloaders(
    manifest_path: PathLike = DEFAULT_MANIFEST,
    data_root: PathLike = DEFAULT_ROOT,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = False,
    seed: int = SEED,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders over the finalized manifest.

    Only train/val splits are ever constructed here. There is deliberately
    NO test DataLoader builder: test data must stay out of the training
    module entirely.

    Args:
        train_indices / val_indices: Optional index subsets (used by
            lightweight smoke tests; full runs pass ``None``).

    Returns:
        ``(train_loader, val_loader)``; train shuffles, val does not.
    """
    train_ds = GenImageDataset(
        split="train",
        transform=get_transform("train", image_size, augment=augment),
        manifest_path=manifest_path,
        root=data_root,
    )
    val_ds = GenImageDataset(
        split="val",
        transform=get_transform("val", image_size),
        manifest_path=manifest_path,
        root=data_root,
    )
    if train_indices is not None:
        train_ds = Subset(train_ds, train_indices)
    if val_indices is not None:
        val_ds = Subset(val_ds, val_indices)

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        worker_init_fn=_seeded_worker_init(seed),
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def _epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch; return ``(mean_loss, accuracy)``.

    When ``training`` is True the model trains (gradients + optimizer
    step); otherwise it evaluates under ``torch.no_grad()``.
    """
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                assert optimizer is not None
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch; return ``(mean_loss, accuracy)``."""
    return _epoch_metrics(model, loader, criterion, device, True, optimizer)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch; return ``(mean_loss, accuracy)``."""
    return _epoch_metrics(model, loader, criterion, device, False)


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build Adam / AdamW / SGD by name (explicit, no duplicated loops)."""
    name = name.lower()
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer {name!r}; use 'adam', 'adamw', or 'sgd'.")


def fit(config: TrainConfig) -> dict[str, Any]:
    """Run a full baseline training job from one config.

    Selection rule: the checkpoint saved as ``best_model.pth`` is the
    epoch with the highest VALIDATION accuracy. Test data plays no role.

    Artifacts written to ``config.checkpoint_dir``:

    - ``<model>_best.pth``: best-validation weights + config snapshot.
    - ``<model>_last.pth``: final-epoch weights.
    - ``<model>_history.json``: per-epoch train/val loss/accuracy.

    Args:
        config: Training configuration.

    Returns:
        History dict with ``train_loss`` / ``train_acc`` / ``val_loss`` /
        ``val_acc`` lists plus ``best_epoch`` and ``best_val_acc``.
    """
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"[train] device={device} model={config.model_name} "
          f"pretrained={config.pretrained}")

    train_loader, val_loader = make_dataloaders(
        manifest_path=config.manifest_path,
        data_root=config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=config.augment,
        seed=config.seed,
        train_indices=config.train_indices,
        val_indices=config.val_indices,
    )
    model = build_model(config.model_name, pretrained=config.pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        name=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{config.model_name}_best.pth"
    last_path = checkpoint_dir / f"{config.model_name}_last.pth"

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": -1,
        "best_val_acc": -1.0,
        "config": config.to_dict(),
    }
    best_val_acc = -1.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[train] epoch {epoch}/{config.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_epoch"] = epoch
            history["best_val_acc"] = val_acc
            torch.save(
                {
                    "model_name": config.model_name,
                    "num_classes": 2,
                    "label_map": {"real": 0, "ai": 1},
                    "epoch": epoch,
                    "config": config.to_dict(),
                    "state_dict": model.state_dict(),
                },
                best_path,
            )
    torch.save(
        {
            "model_name": config.model_name,
            "num_classes": 2,
            "label_map": {"real": 0, "ai": 1},
            "epoch": config.epochs,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
        },
        last_path,
    )
    with open(checkpoint_dir / f"{config.model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] best epoch={history['best_epoch']} "
          f"best_val_acc={history['best_val_acc']:.4f}")
    return history


def _parse_args() -> TrainConfig:
    """CLI entry: every TrainConfig field is overridable by flag."""
    parser = argparse.ArgumentParser(description="Clean baseline training.")
    parser.add_argument("--model", default="resnet50", choices=list(MODEL_NAMES))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default="models/baseline")
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        augment=args.augment,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    fit(_parse_args())
