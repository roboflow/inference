from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from inference_models.models.auto_loaders.models_registry import (
    OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
)


def build_random_rgb_images(
    batch_size: int,
    *,
    height: int,
    width: int,
) -> List[np.ndarray]:
    """Build deterministic random RGB images for profiling runs.

    Uses a fixed NumPy seed so repeated runs see the same pixel values.

    Args:
        batch_size: Number of images in the batch.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        List of ``uint8`` arrays shaped ``(height, width, 3)``.
    """
    rng = np.random.default_rng(0)
    images = []

    for _ in range(batch_size):
        image = rng.integers(
            0,
            255,
            size=(height, width, 3),
            dtype=np.uint8,
        )
        images.append(
            image,
        )

    return images


def default_infer_kwargs_for_task(task_type: str | None) -> Dict[str, Any]:
    """Return minimal kwargs so ``infer`` can run for specialized task types.

    Args:
        task_type: Registry task type string, or ``None`` when unknown.

    Returns:
        Keyword arguments merged into profiling ``infer`` calls.
    """
    if task_type == OPEN_VOCABULARY_OBJECT_DETECTION_TASK:
        return {"classes": ["object"]}

    return {}


def merge_infer_kwargs(
    *,
    task_type: str | None,
    user: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Merge task defaults with user-supplied infer kwargs.

    Args:
        task_type: Registry task type string, or ``None`` when unknown.
        user: Extra kwargs from the CLI; overrides defaults on key collision.

    Returns:
        Combined keyword argument dict for the profiling ``infer`` call.
    """
    merged = default_infer_kwargs_for_task(task_type)

    if user:
        merged.update(user)

    return merged


def describe_shape_signature(
    batch_size: int,
    *,
    height: int,
    width: int,
    infer_kwargs: Dict[str, Any],
) -> str:
    """Build a stable text fingerprint for the synthetic input regime.

    Args:
        batch_size: Batch dimension used for profiling.
        height: Image height in pixels.
        width: Image width in pixels.
        infer_kwargs: Effective kwargs passed to ``infer``.

    Returns:
        Human-readable signature stored in profiling JSON results.
    """
    keys = sorted(infer_kwargs.keys())
    kw_repr = ",".join(f"{k}={infer_kwargs[k]!r}" for k in keys)
    shape_signature = f"b{batch_size}x{height}x{width};{kw_repr}"

    return shape_signature
