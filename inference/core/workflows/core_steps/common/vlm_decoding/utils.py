"""Small shared helpers used across the VLM decoding modules."""

import math
from typing import Dict, List


def create_classes_index(classes: List[str]) -> Dict[str, int]:
    """Map class names onto their positional class ids.

    Args:
        classes: Class names in the order the caller declared them.

    Returns:
        Mapping of class name to zero-based class id.
    """
    return {class_name: idx for idx, class_name in enumerate(classes)}


def scale_confidence(value: float) -> float:
    """Clamp a model-provided confidence into the ``[0.0, 1.0]`` range.

    Args:
        value: Raw confidence value.

    Returns:
        The value clamped into ``[0.0, 1.0]``.

    Raises:
        ValueError: If the value is not a finite number (``json.loads``
            accepts ``NaN`` / ``Infinity``).
    """
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError(f"confidence is not a finite number: {value!r}")
    return min(max(confidence, 0.0), 1.0)
