"""Instance-segmentation inputs of the region blocks: predictions or a view.

The region blocks accept both instance-segmentation kinds (decision 018). What
they read from a ``SegmentationView`` depends on whether they need mask pixels:

    need                   InstanceDetections        SegmentationView
    boxes only             used as given             view.detections (no masks,
                                                     nothing materialized)
    mask pixels            used as given             reference masks:
                                                     mask_provider_of(view)
                                                     .binary(fidelity="reference")

``v2/dynamic_crop@v1`` needs boxes, and masks only for its fade.
``v2/detections_stitch@v1`` and ``v2/bounding_rect@v1`` need mask pixels.

Reference masks are never an approximation. They cost what the view's
representation costs: nothing for a dense-fallback view, one full-resolution
``(N, H, W)`` upscale for a grid-backed view, computed once per view and
shared. Rows, metadata and the row record are the view's own in both cases.
"""

from typing import Any

from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.masks.providers import mask_provider_of

__all__ = [
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "boxes_of_view",
    "instance_predictions_of",
]


def boxes_of_view(payload: Any) -> Any:
    """Return a view's box-only detections; other payloads unchanged.

    Args:
        payload: A native prediction, a ``SegmentationView`` or ``None``.

    Returns:
        ``view.detections`` for a view, without masks and without
        materializing any; ``payload`` itself otherwise.
    """
    if not isinstance(payload, SegmentationView):
        return payload

    detections = payload.detections

    return detections


def instance_predictions_of(payload: Any) -> Any:
    """Return predictions with reference masks; other payloads unchanged.

    Args:
        payload: A native prediction, a ``SegmentationView`` or ``None``.

    Returns:
        ``InstanceDetections`` with dense reference masks for a view;
        ``payload`` itself otherwise.
    """
    if not isinstance(payload, SegmentationView):
        return payload

    masks = mask_provider_of(payload).binary(fidelity="reference")
    predictions = masks.predictions

    return predictions
