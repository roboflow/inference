"""Compatibility checks shared by RF-DETR reference preprocessors."""

from typing import Any, List

import numpy as np
import torch

from inference_models.models.optimization.contracts import CompatibilityResult
from inference_models.models.rfdetr.optimization.contracts import PreprocessRequest


def _request_items(images) -> List[Any]:
    """Expand a supported image or image batch into individual items.

    Args:
        images: Single image or batch supplied to RF-DETR preprocessing.

    Returns:
        Individual batch items. Unsupported values are retained for validation.
    """
    if isinstance(images, list):
        items = list(images)
    elif isinstance(images, torch.Tensor) and images.ndim == 4:
        items = list(images.unbind(0))
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        items = list(images)
    else:
        items = [images]

    return items


def check_base_request_compatibility(
    request: PreprocessRequest,
) -> CompatibilityResult:
    """Check the broad request constraints of the reference preprocessor.

    Args:
        request: Typed preprocessing request.

    Returns:
        Compatibility result. Detailed shape validation remains in the base path.
    """
    items = _request_items(request.images)
    reasons = []
    if not items:
        reasons.append("empty image batch")
    unsupported_types = sorted(
        {
            type(item).__name__
            for item in items
            if not isinstance(item, (np.ndarray, torch.Tensor))
        }
    )
    if unsupported_types:
        reasons.append(f"unsupported input types: {unsupported_types}")
    if reasons:
        result = CompatibilityResult.incompatible(*reasons)
    else:
        result = CompatibilityResult.compatible()

    return result


def check_threaded_request_compatibility(
    request: PreprocessRequest,
) -> CompatibilityResult:
    """Check the exact NumPy batch constraints of threaded preprocessing.

    Args:
        request: Typed preprocessing request.

    Returns:
        Compatibility result with all discovered limitations.
    """
    base_compatibility = check_base_request_compatibility(request=request)
    if not base_compatibility.supported:
        return base_compatibility

    items = _request_items(request.images)
    unsupported = [
        type(item).__name__ for item in items if not isinstance(item, np.ndarray)
    ]
    invalid = [
        (str(item.dtype), tuple(item.shape))
        for item in items
        if isinstance(item, np.ndarray)
        and (item.dtype != np.uint8 or item.ndim != 3 or item.shape[-1] != 3)
    ]
    reasons = []
    if unsupported:
        reasons.append(f"threaded preprocessing requires NumPy inputs: {unsupported}")
    if invalid:
        reasons.append(f"threaded preprocessing requires uint8 HWC images: {invalid}")
    if reasons:
        result = CompatibilityResult.incompatible(*reasons)
    else:
        result = CompatibilityResult.compatible()

    return result
