from typing import List, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    DETECTION_MAX_EDGE_PIXELS,
    attach_parents_coordinates_to_sv_detections,
    scale_dimensions_to_max_edge,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.gemini_detection_parsing import (
    create_classes_index,
    get_gemini_detection_class_name,
    scale_confidence,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def convert_openai_detection_to_pixel_xyxy(
    detection: dict,
    image_height: int,
    image_width: int,
) -> List[float]:
    """Convert a ``box_2d`` entry into pixel coordinates of the original image.

    The OpenAI block downscales images so their longest edge does not exceed
    ``DETECTION_MAX_EDGE_PIXELS`` before upload, so returned coordinates refer
    to the uploaded image. This helper recomputes the uploaded dimensions
    deterministically from the original ones, clamps coordinates to them, and
    rescales the box back onto the original image.

    Args:
        detection: Detection entry with a ``box_2d`` field holding
            ``[x_min, y_min, x_max, y_max]`` in absolute pixel coordinates
            of the uploaded image.
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.

    Returns:
        ``[x_min, y_min, x_max, y_max]`` in pixel coordinates of the
        original image.
    """
    uploaded_width, uploaded_height = scale_dimensions_to_max_edge(
        image_width, image_height, DETECTION_MAX_EDGE_PIXELS
    )
    scale_x = image_width / uploaded_width
    scale_y = image_height / uploaded_height
    x_min, y_min, x_max, y_max = detection["box_2d"]
    x_min = min(max(float(x_min), 0.0), uploaded_width)
    x_max = min(max(float(x_max), 0.0), uploaded_width)
    y_min = min(max(float(y_min), 0.0), uploaded_height)
    y_max = min(max(float(y_max), 0.0), uploaded_height)
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def parse_openai_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    """Parse OpenAI block (v5+) object-detection output into detections.

    The model returns ``box_2d`` entries as ``[x_min, y_min, x_max, y_max]``
    in absolute pixel coordinates of the uploaded (possibly downscaled)
    image; see ``convert_openai_detection_to_pixel_xyxy`` for the coordinate
    contract.

    Args:
        image: Workflow image the detections refer to (original resolution).
        parsed_data: JSON list of detection entries produced by the model.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed detection.

    Returns:
        Parsed detections in the original image's coordinate space.

    Raises:
        ValueError: If the response is not a JSON list.
    """
    if not isinstance(parsed_data, list):
        raise ValueError("Unexpected OpenAI object detection response format")
    if len(parsed_data) == 0:
        return sv.Detections.empty()

    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]

    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in parsed_data:
        xyxy.append(
            convert_openai_detection_to_pixel_xyxy(
                detection=detection,
                image_height=image_height,
                image_width=image_width,
            )
        )
        label = get_gemini_detection_class_name(detection=detection)
        class_id.append(class_name2id.get(label, -1))
        class_name.append(label)
        confidence.append(scale_confidence(detection.get("confidence", 1.0)))

    xyxy = np.array(xyxy).round(0) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detection_ids = np.array([str(uuid4()) for _ in range(len(xyxy))])
    dimensions = np.array([[image_height, image_width]] * len(xyxy))
    inference_ids = np.array([inference_id] * len(xyxy))
    prediction_type = np.array(["object-detection"] * len(xyxy))
    data = {
        CLASS_NAME_DATA_FIELD: class_name,
        IMAGE_DIMENSIONS_KEY: dimensions,
        INFERENCE_ID_KEY: inference_ids,
        DETECTION_ID_KEY: detection_ids,
        PREDICTION_TYPE_KEY: prediction_type,
    }
    detections_result = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=None,
        tracker_id=None,
        data=data,
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections_result,
        image=image,
    )
