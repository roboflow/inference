from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from inference_models.models.auto_loaders.models_registry import (
    OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
)


def build_random_rgb_images(
    batch_size: int, height: int, width: int
) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    images = []
    for _ in range(batch_size):
        images.append(
            rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        )
    return images


def default_infer_kwargs_for_task(task_type: str | None) -> Dict[str, Any]:
    """Minimal kwargs so ``infer`` can run for task types that need extra parameters."""
    if task_type == OPEN_VOCABULARY_OBJECT_DETECTION_TASK:
        return {"classes": ["object"]}
    return {}


def merge_infer_kwargs(
    task_type: str | None, user: Dict[str, Any] | None
) -> Dict[str, Any]:
    merged = default_infer_kwargs_for_task(task_type)
    if user:
        merged.update(user)
    return merged


def describe_shape_signature(
    batch_size: int, height: int, width: int, infer_kwargs: Dict[str, Any]
) -> str:
    """Stable text fingerprint for the synthetic input regime."""
    keys = sorted(infer_kwargs.keys())
    kw_repr = ",".join(f"{k}={infer_kwargs[k]!r}" for k in keys)
    return f"b{batch_size}x{height}x{width};{kw_repr}"
