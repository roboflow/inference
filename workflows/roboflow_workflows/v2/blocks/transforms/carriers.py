"""Prediction carriers that the transform blocks accept and return.

Every transform returns the carrier type it received::

    object_detection_prediction            Detections
    instance_segmentation_prediction       InstanceDetections, dense mask
    rle_instance_segmentation_prediction   InstanceDetections, RLE mask
    instance_segmentation_view             SegmentationView (compact masks)
    keypoint_detection_prediction          (KeyPoints, Detections)
    prediction_rows                        PredictionRows
"""

from typing import Any

from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND, PredictionRows

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    PREDICTION_ROWS_KIND,
)


def carrier_name(value: Any) -> str:
    """Name a prediction's carrier type for errors and same-type checks.

    Args:
        value: Any value.

    Returns:
        ``"Detections"``, ``"InstanceDetections"``, ``"SegmentationView"``,
        ``"(KeyPoints, Detections)"``, ``"PredictionRows"`` or the Python
        type name of anything else.
    """
    if isinstance(value, PredictionRows):
        return "PredictionRows"
    if isinstance(value, (InstanceDetections, Detections, SegmentationView)):
        return type(value).__name__
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], KeyPoints):
        return "(KeyPoints, Detections)"

    return type(value).__name__
