"""Shared image preprocessing for the Real-vs-AI baseline pipeline.

Design notes (methodology-relevant):

- The SAME preprocessing applies to Real and AI images. There is no
  class-specific logic anywhere in this module.
- This is ordinary model-input preprocessing only (RGB conversion, resize,
  tensor conversion, ImageNet normalization). Research robustness
  transformations (JPEG compression, resizing-as-degradation,
  recompression, combined pipelines) live in ``src/transformations.py``
  and MUST NOT be added here: the clean baseline has to stay clean.
- No preprocessing statistics are fitted on project data. Normalization
  uses fixed ImageNet mean/std because both baseline models start from
  ImageNet-pretrained weights. Nothing here reads the validation or test
  sets, so there is no leakage surface.
- ``IMAGE_SIZE = 224`` is the documented standard input size for both
  ResNet-50 and EfficientNet-B0.

Label mapping (fixed, do not change):
    0 = Real
    1 = AI-generated
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image
from torchvision import transforms

__all__ = [
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "load_image",
    "build_train_transform",
    "build_eval_transform",
    "get_transform",
]

#: Model input size (square) used for both ResNet-50 and EfficientNet-B0.
IMAGE_SIZE: int = 224

#: Fixed ImageNet channel mean for normalization (pretrained weights).
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]

#: Fixed ImageNet channel std for normalization (pretrained weights).
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

PathLike = Union[str, Path]


def load_image(path: PathLike) -> Image.Image:
    """Load an image file and convert it to RGB.

    Handles JPEG/PNG as well as grayscale (L), paletted (P), and
    alpha-channel (RGBA/LA) inputs by converting everything to RGB, so
    downstream transforms always see 3 channels.

    Args:
        path: Filesystem path to the image.

    Returns:
        A new PIL image in RGB mode.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
) -> transforms.Compose:
    """Build the training-time preprocessing pipeline.

    Deterministic core: direct resize to ``(image_size, image_size)``.
    Direct resize (instead of resize + center crop) keeps train and eval
    geometry identical, which keeps the clean baseline easy to interpret.

    Optional standard augmentation (``augment=True``): a single random
    horizontal flip. Kept deliberately mild so the baseline experiment
    stays interpretable; stronger augmentation belongs to later,
    explicitly-named experiments, not to the default pipeline.

    Args:
        image_size: Square model input size.
        augment: If True, append ``RandomHorizontalFlip(p=0.5)``.

    Returns:
        A torchvision transform mapping PIL image -> normalized tensor
        of shape ``[3, image_size, image_size]``.
    """
    steps: list = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the validation/test preprocessing pipeline.

    Fully deterministic: resize -> tensor -> ImageNet normalization.
    Contains NO random augmentation by construction.

    Args:
        image_size: Square model input size.

    Returns:
        A torchvision transform mapping PIL image -> normalized tensor
        of shape ``[3, image_size, image_size]``.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_transform(
    split: str,
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
) -> transforms.Compose:
    """Return the preprocessing pipeline for a dataset split.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        image_size: Square model input size.
        augment: Optional mild training augmentation. Only allowed for
            ``split="train"``; requesting augmentation for ``val``/``test``
            raises ``ValueError`` so eval pipelines can never silently
            contain randomness.

    Returns:
        The train or eval transform as appropriate.

    Raises:
        ValueError: For unknown splits, or augmentation on val/test.
    """
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(
            f"Unknown split {split!r}; expected 'train', 'val', or 'test'."
        )
    if split in ("val", "test") and augment:
        raise ValueError(
            f"augment=True is not allowed for split {split!r}: "
            "validation/test preprocessing must be deterministic."
        )
    if split == "train":
        return build_train_transform(image_size=image_size, augment=augment)
    return build_eval_transform(image_size=image_size)
