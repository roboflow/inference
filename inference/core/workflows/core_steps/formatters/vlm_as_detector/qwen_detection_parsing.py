from typing import List, Optional, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
    empty_detections_with_image_metadata,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.gemini_detection_parsing import (
    create_classes_index,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

QWEN_BOX_COORDINATE_SCALE = 1000.0

# Qwen models occasionally drift between the prompted keys and their native
# grounding vocabulary; accept both, matching the vlm-exam parser.
_BOX_KEYS = ("box_2d", "bbox_2d")
_LABEL_KEYS = ("label", "description", "class_name", "class")


def extract_qwen_detection_entries(
    parsed_data: Union[dict, list],
) -> List[dict]:
    """Extract the list of detection entries from parsed Qwen JSON output.

    The Qwen prompt asks for a bare JSON list, but some responses wrap the
    entries in a ``{"detections": [...]}`` object; both shapes are accepted.

    Args:
        parsed_data: JSON payload extracted from the VLM output.

    Returns:
        List of raw detection entry dicts.

    Raises:
        ValueError: If the payload matches neither accepted shape.
    """
    if isinstance(parsed_data, list):
        return parsed_data
    if isinstance(parsed_data, dict) and isinstance(
        parsed_data.get("detections"), list
    ):
        return parsed_data["detections"]
    raise ValueError("Unexpected Qwen object detection response format")


def get_qwen_detection_box(detection: dict) -> Optional[List[float]]:
    """Read a valid bounding box from a Qwen detection entry.

    Args:
        detection: Raw detection entry.

    Returns:
        ``[x_min, y_min, x_max, y_max]`` floats in the 0-1000 normalized
        space, or ``None`` when the entry carries no well-formed box.
    """
    for key in _BOX_KEYS:
        box = detection.get(key)
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in box
            )
        ):
            return [float(value) for value in box]

    return None


def get_qwen_detection_class_name(detection: dict) -> str:
    """Resolve the class label of a Qwen detection entry.

    Args:
        detection: Raw detection entry.

    Returns:
        First present label under the accepted key aliases, or ``"unknown"``.
    """
    for key in _LABEL_KEYS:
        value = detection.get(key)
        if value is not None:
            return str(value)

    return "unknown"


def convert_qwen_detection_to_pixel_xyxy(
    box: List[float],
    image_height: int,
    image_width: int,
) -> List[float]:
    """Convert a 0-1000-normalized ``box_2d`` into original-image pixels.

    Coordinates are clamped to the 0-1000 range before scaling so slightly
    out-of-range model output stays inside the image.

    Args:
        box: ``[x_min, y_min, x_max, y_max]`` normalized to 0-1000.
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.

    Returns:
        ``[x_min, y_min, x_max, y_max]`` in pixel coordinates of the
        original image.
    """
    x_min, y_min, x_max, y_max = (
        min(max(value, 0.0), QWEN_BOX_COORDINATE_SCALE) for value in box
    )
    scale_x = image_width / QWEN_BOX_COORDINATE_SCALE
    scale_y = image_height / QWEN_BOX_COORDINATE_SCALE

    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def parse_qwen_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    """Parse Qwen block object-detection output into detections.

    The Qwen block prompts for a JSON list of entries with a ``box_2d``
    field holding ``[x_min, y_min, x_max, y_max]`` integers normalized to
    0-1000 and a ``label`` field. Entries without a well-formed box are
    skipped (with a debug log) rather than failing the whole response;
    labels outside ``classes`` are kept with ``class_id == -1``.
    Confidence is hardcoded to 1.0 — VLMs do not produce calibrated
    detection confidences and the prompt does not ask for one.

    Args:
        image: Workflow image the detections refer to.
        parsed_data: JSON payload extracted from the VLM output.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed detection.

    Returns:
        Parsed detections in the original image's coordinate space.

    Raises:
        ValueError: If the response is neither a JSON list nor a
            ``{"detections": [...]}`` object.
    """
    entries = extract_qwen_detection_entries(parsed_data=parsed_data)
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]

    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in entries:
        if not isinstance(detection, dict):
            logger.debug("Skipping non-dict Qwen detection entry: %r", detection)
            continue
        box = get_qwen_detection_box(detection=detection)
        if box is None:
            logger.debug(
                "Skipping Qwen detection entry without a well-formed box: %r",
                detection,
            )
            continue

        xyxy.append(
            convert_qwen_detection_to_pixel_xyxy(
                box=box,
                image_height=image_height,
                image_width=image_width,
            )
        )
        label = get_qwen_detection_class_name(detection=detection)
        class_id.append(class_name2id.get(label, -1))
        class_name.append(label)
        confidence.append(1.0)

    if not xyxy:
        return empty_detections_with_image_metadata(
            image_height=image_height,
            image_width=image_width,
        )

    xyxy = np.array(xyxy).round(0)
    confidence = np.array(confidence)
    class_id = np.array(class_id).astype(int)
    class_name = np.array(class_name)
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
