"""Social-media-style image transformations.

This module implements controlled, reusable transformations used to study
detector robustness. Transformations apply identically to Real and
AI-generated images; no class-specific logic belongs here.

Label mapping (unchanged): 0 = Real, 1 = AI-generated.

Functions accept PIL Images and return new PIL Images. Inputs are never
modified in place.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from typing import Final

from PIL import Image

__all__ = [
    "JPEG_QUALITY_LEVELS",
    "RESIZE_SCALES",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_RECOMPRESSION_PASSES",
    "apply_jpeg_compression",
    "apply_resize",
    "apply_recompression",
    "apply_pipeline",
]

# Suggested experiment levels from AGENTS.md. These are defaults for
# callers to reference; pass explicit values to the functions instead of
# relying on these inside experiment code.
JPEG_QUALITY_LEVELS: Final[list[int]] = [90, 70, 50, 30]
RESIZE_SCALES: Final[list[float]] = [0.75, 0.50, 0.25]
DEFAULT_JPEG_QUALITY: Final[int] = 90
DEFAULT_RECOMPRESSION_PASSES: Final[int] = 2


def _ensure_saveable_as_jpeg(image: Image.Image) -> Image.Image:
    """Return a copy of ``image`` that can be saved as JPEG.

    JPEG does not support alpha channels or palette mode, so RGBA/LA/PA/P
    images are converted to RGB. RGB, L, and CMYK images are returned as a
    copy unchanged. The input image is never modified.
    """
    if image.mode in ("RGB", "L", "CMYK"):
        return image.copy()
    return image.convert("RGB")


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply a single JPEG compression at the given quality.

    Args:
        image: Input PIL image (any mode; RGBA handled safely).
        quality: JPEG quality in 1..100 (e.g. 90, 70, 50, 30).

    Returns:
        New RGB/RGB-converted PIL image after JPEG encode/decode.

    Raises:
        TypeError: If ``quality`` is not an int.
        ValueError: If ``quality`` is outside 1..100.
    """
    if not isinstance(quality, int) or isinstance(quality, bool):
        raise TypeError(f"quality must be int in 1..100, got {type(quality).__name__}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")

    saveable = _ensure_saveable_as_jpeg(image)
    buffer = io.BytesIO()
    saveable.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        result = decoded.convert("RGB") if decoded.mode != "RGB" else decoded.copy()
    return result


def apply_resize(
    image: Image.Image,
    scale: float,
    resample: int = Image.BILINEAR,
) -> Image.Image:
    """Resize ``image`` by ``scale`` while preserving aspect ratio.

    Args:
        image: Input PIL image.
        scale: Multiplicative scale factor (e.g. 0.5 halves each side).
            Must be positive and finite.
        resample: PIL resampling filter (default: ``Image.BILINEAR``).

    Returns:
        New resized PIL image with ``round(w * scale)`` x ``round(h * scale)``.

    Raises:
        TypeError: If ``scale`` is not a number.
        ValueError: If ``scale`` is not positive/finite or yields a
            zero-pixel dimension.
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be a number, got {type(scale).__name__}")
    if not (0 < float(scale) < float("inf")):
        raise ValueError(f"scale must be positive and finite, got {scale}")

    new_width = max(1, round(image.width * float(scale)))
    new_height = max(1, round(image.height * float(scale)))
    if new_width < 1 or new_height < 1:
        raise ValueError(f"scale {scale} yields invalid size for image {image.size}")
    return image.resize((new_width, new_height), resample=resample)


def apply_recompression(
    image: Image.Image,
    quality: int,
    passes: int = DEFAULT_RECOMPRESSION_PASSES,
) -> Image.Image:
    """Apply JPEG compression ``passes`` times in sequence.

    Recompression (save -> reload -> save again) differs from a single
    JPEG compression and is kept as a separate function intentionally.

    Args:
        image: Input PIL image.
        quality: JPEG quality in 1..100, used for every pass.
        passes: Number of compression passes (>= 1).

    Returns:
        New PIL image after ``passes`` sequential JPEG compressions.

    Raises:
        TypeError: If ``passes`` is not an int.
        ValueError: If ``passes`` < 1 (quality is validated per pass).
    """
    if not isinstance(passes, int) or isinstance(passes, bool):
        raise TypeError(f"passes must be int >= 1, got {type(passes).__name__}")
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")

    result = image
    for _ in range(passes):
        result = apply_jpeg_compression(result, quality)
    return result


# A single transformation step: takes a PIL image, returns a PIL image.
TransformStep = Callable[[Image.Image], Image.Image]


def apply_pipeline(
    image: Image.Image, transformations: Sequence[TransformStep]
) -> Image.Image:
    """Apply a sequence of transformations in order.

    Example:
        combined = apply_pipeline(img, [
            lambda im: apply_resize(im, 0.5),
            lambda im: apply_jpeg_compression(im, 70),
        ])

    Transformation order is preserved and may matter (e.g. Resize->JPEG
    vs JPEG->Resize), so callers should document the order used.

    Args:
        image: Input PIL image (never modified; a copy is transformed).
        transformations: Ordered sequence of callables, each accepting
            and returning a PIL image. An empty sequence returns a copy
            of the input.

    Returns:
        New PIL image after all steps.

    Raises:
        TypeError: If ``transformations`` is not a sequence or contains
            non-callable entries.
    """
    if not isinstance(transformations, Sequence) or isinstance(
        transformations, (str, bytes)
    ):
        raise TypeError("transformations must be a sequence of callables")
    result = image.copy()
    for step in transformations:
        if not callable(step):
            raise TypeError(f"pipeline step must be callable, got {step!r}")
        result = step(result)
    return result
