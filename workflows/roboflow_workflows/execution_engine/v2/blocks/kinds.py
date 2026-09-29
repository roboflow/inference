"""Payload kinds understood by the native V2 CPU image blocks.

The generic V2 engine never inspects payload types. Every kind used by the
native catalogue is declared here together with its validator, and the
validators are registered on the explicit native registry only. Kind names
are plain identifiers compared by the compiler; the validators run at the
engine boundary when a payload enters or leaves a block.

Image payloads are NumPy ``uint8`` arrays with shape ``(height, width, 3)``
in RGB channel order. Ownership of that convention stays in this plugin.
"""

from typing import Any, Mapping

import numpy as np

IMAGE_KIND = "image"
BOOLEAN_KIND = "boolean"
INTEGER_KIND = "integer"
CROP_SUMMARY_KIND = "crop_summary"

IMAGE_CHANNELS = 3


def is_image_payload(payload: Any) -> bool:
    """Check that a payload is a non-empty RGB ``uint8`` NumPy image.

    Args:
        payload: Candidate payload supplied by the engine or a direct caller.

    Returns:
        True when the payload is a ``numpy.ndarray`` with dtype ``uint8``,
        three dimensions, three channels and a positive height and width.
    """
    if not isinstance(payload, np.ndarray):
        return False

    if payload.dtype != np.uint8 or payload.ndim != 3:
        return False

    height, width, channels = payload.shape
    valid = channels == IMAGE_CHANNELS and height > 0 and width > 0

    return valid


def is_boolean_payload(payload: Any) -> bool:
    """Check that a payload is an actual Python ``bool``.

    NumPy booleans and integers are rejected so that gate decisions produced by
    native blocks are always plain ``bool`` values, as required by the engine.

    Args:
        payload: Candidate payload.

    Returns:
        True only for ``True`` or ``False``.
    """
    valid = isinstance(payload, bool)

    return valid


def is_integer_payload(payload: Any) -> bool:
    """Check that a payload is a plain non-boolean Python ``int``.

    Args:
        payload: Candidate payload.

    Returns:
        True for ``int`` instances that are not ``bool``.
    """
    valid = isinstance(payload, int) and not isinstance(payload, bool)

    return valid


def is_crop_summary_payload(payload: Any) -> bool:
    """Check that a payload is a crop summary mapping.

    A crop summary is the per-parent description emitted by ``v2/crop`` next to
    its child crops. It must contain the crop count and the parent dimensions.

    Args:
        payload: Candidate payload.

    Returns:
        True when the payload is a mapping containing integer ``crop_count``,
        ``image_height`` and ``image_width`` entries.
    """
    if not isinstance(payload, Mapping):
        return False

    required_keys = ("crop_count", "image_height", "image_width")
    valid = all(is_integer_payload(payload.get(key)) for key in required_keys)

    return valid
