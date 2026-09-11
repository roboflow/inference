from typing import List, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
    empty_detections_with_image_metadata,
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

SPACEXAI_BOX_COORDINATE_SCALE = 100.0


def extract_spacexai_detection_entries(
    parsed_data: Union[dict, list],
) -> List[dict]:
    """Extract detection entries from SpaceXAI object-detection JSON.

    Args:
        parsed_data: JSON payload extracted from the VLM output.

    Returns:
        List of detection dictionaries.

    Raises:
        ValueError: If the response is not a JSON list or a
            ``{"detections": [...]}`` wrapper.
    """
    if isinstance(parsed_data, list):
        return parsed_data
    if isinstance(parsed_data, dict) and "detections" in parsed_data:
        return parsed_data["detections"]
    raise ValueError("Unexpected SpaceXAI object detection response format")


def convert_spacexai_detection_to_pixel_xyxy(
    detection: dict,
    image_height: int,
    image_width: int,
) -> List[float]:
    """Convert a percent ``box_2d`` entry into original-image pixel coordinates.

    SpaceXAI Grok detection prompts ask for ``[x_min, y_min, x_max, y_max]`` as
    percentages of image width and height (floats 0-100). Coordinates are
    clamped to ``[0, 100]`` before scaling.

    Args:
        detection: Detection entry with a ``box_2d`` field.
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.

    Returns:
        ``[x_min, y_min, x_max, y_max]`` in pixel coordinates of the original
        image.
    """
    x_min, y_min, x_max, y_max = detection["box_2d"]
    scale = SPACEXAI_BOX_COORDINATE_SCALE
    x_min = min(max(float(x_min), 0.0), scale)
    x_max = min(max(float(x_max), 0.0), scale)
    y_min = min(max(float(y_min), 0.0), scale)
    y_max = min(max(float(y_max), 0.0), scale)
    return [
        x_min / scale * image_width,
        y_min / scale * image_height,
        x_max / scale * image_width,
        y_max / scale * image_height,
    ]


def parse_spacexai_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    """Parse SpaceXAI Grok object-detection output into detections.

    Args:
        image: Workflow image the detections refer to.
        parsed_data: JSON list of detection entries produced by the model.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed detection.

    Returns:
        Parsed detections in the original image's coordinate space.
    """
    detections = extract_spacexai_detection_entries(parsed_data=parsed_data)
    if len(detections) == 0:
        image_height, image_width = image.numpy_image.shape[:2]
        return empty_detections_with_image_metadata(
            image_height=image_height,
            image_width=image_width,
        )

    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]

    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in detections:
        xyxy.append(
            convert_spacexai_detection_to_pixel_xyxy(
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
