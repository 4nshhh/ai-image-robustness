"""PyTorch dataset over the finalized manifest (7,986 images).

The manifest is the single source of truth for split membership, labels,
and generator metadata:

- ``split`` column selects train/val/test rows.
- ``label`` column (``"real"`` / ``"ai"``) is the ONLY label source.
  Labels are never inferred from filenames or directory names.
- ``generator`` column is carried as metadata only; it is never used as
  an input feature or a label.

Label mapping (fixed, do not change):
    0 = Real
    1 = AI-generated

Finalized dataset composition (committed 7,986-image manifest):
    train: 5,600 = 2,800 Real + 2,800 AI
    val:   1,200 =   600 Real +   600 AI
    test:  1,186 =   600 Real +   586 AI
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, NamedTuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from src.preprocessing import load_image
except ImportError:  # allow `sys.path`-based use of the src/ directory
    from preprocessing import load_image

__all__ = [
    "LABEL_REAL",
    "LABEL_AI",
    "LABEL_MAP",
    "DEFAULT_ROOT",
    "DEFAULT_MANIFEST",
    "ManifestRow",
    "load_manifest",
    "GenImageDataset",
]

#: Label for Real images. Must stay consistent across the whole project.
LABEL_REAL: int = 0

#: Label for AI-generated images. Must stay consistent everywhere.
LABEL_AI: int = 1

#: Manifest ``label`` strings -> integer targets. Anything else raises.
LABEL_MAP: dict[str, int] = {"real": LABEL_REAL, "ai": LABEL_AI}

#: Default dataset root (split/label/filename live underneath it).
DEFAULT_ROOT: Path = Path("data/splits/genimage_8000_split_v4")

#: Default manifest path.
DEFAULT_MANIFEST: Path = DEFAULT_ROOT / "manifest.csv"

PathLike = Union[str, Path]


class ManifestRow(NamedTuple):
    """One manifest entry with its resolved local file path."""

    filename: str
    split: str
    label: int
    generator: str
    path: Path


def _resolve_path(root: Path, split: str, label: str, filename: str) -> Path:
    """Resolve ``root / split / label / filename``.

    The manifest's own ``path`` column carries a ``/kaggle/working/...``
    absolute prefix from the machine that built the split, so it is NOT
    portable and is ignored here. Local layout mirrors the split
    directories, which is what we resolve against.
    """
    return root / split / label / filename


def load_manifest(
    manifest_path: PathLike = DEFAULT_MANIFEST,
    root: PathLike = DEFAULT_ROOT,
    split: str | None = None,
) -> list[ManifestRow]:
    """Load and validate manifest rows, optionally filtered to one split.

    Args:
        manifest_path: Path to ``manifest.csv``.
        root: Dataset root used to resolve local image paths.
        split: Optional ``"train"`` / ``"val"`` / ``"test"`` filter
            (case-insensitive). ``None`` loads all rows.

    Returns:
        Manifest rows in file order.

    Raises:
        FileNotFoundError: If the manifest file is missing.
        ValueError: For unknown split filters, unknown label strings,
            or rows whose image file does not exist on disk.
    """
    manifest_path = Path(manifest_path)
    root = Path(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if split is not None:
        split = split.lower()
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split {split!r}; expected 'train', 'val', or 'test'."
            )

    rows: list[ManifestRow] = []
    with open(manifest_path, newline="") as f:
        for record in csv.DictReader(f):
            row_split = record["split"].lower()
            if split is not None and row_split != split:
                continue
            label_str = record["label"].lower()
            if label_str not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label {record['label']!r} for "
                    f"{record['filename']!r}; refusing to guess."
                )
            local_path = _resolve_path(
                root, row_split, label_str, record["filename"]
            )
            if not local_path.is_file():
                raise ValueError(
                    f"Manifest references missing file: {local_path}"
                )
            rows.append(
                ManifestRow(
                    filename=record["filename"],
                    split=row_split,
                    label=LABEL_MAP[label_str],
                    generator=record["generator"],
                    path=local_path,
                )
            )
    return rows


class GenImageDataset(Dataset):
    """Real-vs-AI image dataset backed by the finalized manifest.

    Args:
        split: ``"train"`` / ``"val"`` / ``"test"``.
        transform: Callable mapping a PIL RGB image to an input tensor
            (see ``src.preprocessing.get_transform``). If ``None``,
            images are returned as PIL RGB images.
        manifest_path: Path to ``manifest.csv``.
        root: Dataset root for resolving image files.
        return_metadata: If True, ``__getitem__`` returns
            ``(image, label, metadata)`` where metadata holds
            ``filename`` / ``split`` / ``generator``. Otherwise returns
            ``(image, label)``.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Callable[[Image.Image], Any] | None = None,
        manifest_path: PathLike = DEFAULT_MANIFEST,
        root: PathLike = DEFAULT_ROOT,
        return_metadata: bool = False,
    ) -> None:
        self.split = split.lower()
        self.transform = transform
        self.return_metadata = return_metadata
        self.rows: list[ManifestRow] = load_manifest(
            manifest_path=manifest_path, root=root, split=self.split
        )
        if not self.rows:
            raise ValueError(f"No manifest rows for split {split!r}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        row = self.rows[index]
        image = load_image(row.path)  # always RGB
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row.label, dtype=torch.long)
        if self.return_metadata:
            return image, label, {
                "filename": row.filename,
                "split": row.split,
                "generator": row.generator,
            }
        return image, label

    def class_counts(self) -> dict[int, int]:
        """Count rows per integer label (useful for sanity checks)."""
        counts = {LABEL_REAL: 0, LABEL_AI: 0}
        for row in self.rows:
            counts[row.label] += 1
        return counts
