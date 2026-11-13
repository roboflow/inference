"""Request validators for model registry.

Each validator: (kwargs: dict) → dict (validated kwargs).
Raises ValueError with clear, actionable message on bad input.
"""

from __future__ import annotations

import math


def validate_images_required(kwargs: dict) -> dict:
    if "images" not in kwargs:
        raise ValueError("'images' param required for this action")
    return kwargs


def validate_images_and_classes(kwargs: dict) -> dict:
    if "images" not in kwargs:
        raise ValueError("'images' param required")
    if "classes" not in kwargs:
        raise ValueError("'classes' param required for open-vocabulary detection")
    return kwargs


def validate_texts_required(kwargs: dict) -> dict:
    if "texts" not in kwargs:
        raise ValueError("'texts' param required for text embedding")
    return kwargs


def validate_images_and_prompt(kwargs: dict) -> dict:
    if "images" not in kwargs:
        raise ValueError("'images' param required")
    if "prompt" not in kwargs:
        raise ValueError("'prompt' param required for this action")
    return kwargs


def validate_prompt_only(kwargs: dict) -> dict:
    if "prompt" not in kwargs:
        raise ValueError("'prompt' param required")
    return kwargs


def validate_frames_and_fps(kwargs: dict) -> dict:
    frames = kwargs.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            "'frames' param required as a non-empty list for action recognition"
        )
    fps = kwargs.get("fps")
    if fps is None:
        raise ValueError("'fps' param required for action recognition")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("'fps' param must be a finite number greater than 0")
    return kwargs


def validate_passthrough(kwargs: dict) -> dict:
    """No validation — accept any kwargs."""
    return kwargs


def validate_sam_segment(kwargs: dict) -> dict:
    def _is_empty(key: str) -> bool:
        value = kwargs.get(key)
        return value is None or (hasattr(value, "__len__") and len(value) == 0)

    if all(_is_empty(k) for k in ("images", "embeddings", "image_hashes")):
        raise ValueError(
            "one of 'images', 'embeddings', 'image_hashes' required for segmentation"
        )
    return kwargs
