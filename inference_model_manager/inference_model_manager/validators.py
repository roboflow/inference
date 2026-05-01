"""Request validators for model registry.

Each validator: (kwargs: dict) → dict (validated kwargs).
Raises ValueError with clear, actionable message on bad input.
"""

from __future__ import annotations


def validate_images_required(kwargs: dict) -> dict:
    if "images" not in kwargs:
        raise ValueError("'images' param required for this task")
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
        raise ValueError("'prompt' param required for this task")
    return kwargs


def validate_prompt_only(kwargs: dict) -> dict:
    if "prompt" not in kwargs:
        raise ValueError("'prompt' param required")
    return kwargs


def validate_passthrough(kwargs: dict) -> dict:
    """No validation — accept any kwargs."""
    return kwargs
