import json
from typing import List, Optional, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
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

MUSE_BOX_COORDINATE_SCALE = 1000.0
_BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")


def extract_flat_object_entries(prediction: str) -> List[dict]:
    decoder = json.JSONDecoder()
    entries: List[dict] = []
    index = 0
    while True:
        start = prediction.find("{", index)
        if start == -1:
            break
        try:
            entry, end = decoder.raw_decode(prediction, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        index = end
    return entries


def extract_muse_detection_entries(
    parsed_data: Union[dict, list],
) -> List[dict]:
    if isinstance(parsed_data, list):
        return [entry for entry in parsed_data if isinstance(entry, dict)]
    if isinstance(parsed_data, dict):
        detections = parsed_data.get("detections")
        if isinstance(detections, list):
            return [entry for entry in detections if isinstance(entry, dict)]
        if all(field in parsed_data for field in _BOX_FIELDS):
            return [parsed_data]
    raise ValueError("Unexpected Muse object detection response format")


def get_muse_detection_box(detection: dict) -> Optional[List[float]]:
    try:
        box = [float(detection[field]) for field in _BOX_FIELDS]
    except (KeyError, TypeError, ValueError):
        return None
    if any(isinstance(detection.get(field), bool) for field in _BOX_FIELDS):
        return None
    return box


def convert_muse_detection_to_pixel_xyxy(
    box: List[float],
    image_height: int,
    image_width: int,
) -> List[float]:
    x_min, y_min, x_max, y_max = (
        min(max(value, 0.0), MUSE_BOX_COORDINATE_SCALE) for value in box
    )
    scale_x = image_width / MUSE_BOX_COORDINATE_SCALE
    scale_y = image_height / MUSE_BOX_COORDINATE_SCALE
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def parse_muse_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    entries = extract_muse_detection_entries(parsed_data=parsed_data)
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]

    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in entries:
        box = get_muse_detection_box(detection=detection)
        if box is None:
            logger.debug(
                "Skipping Muse detection entry without named 0-1000 fields: %r",
                detection,
            )
            continue
        xyxy.append(
            convert_muse_detection_to_pixel_xyxy(
                box=box,
                image_height=image_height,
                image_width=image_width,
            )
        )
        label = str(detection.get("label") or "unknown")
        class_id.append(class_name2id.get(label, -1))
        class_name.append(label)
        confidence.append(1.0)

    if not xyxy:
        return sv.Detections.empty()

    xyxy = np.array(xyxy).round(0)
    detection_ids = np.array([str(uuid4()) for _ in range(len(xyxy))])
    dimensions = np.array([[image_height, image_width]] * len(xyxy))
    detections_result = sv.Detections(
        xyxy=xyxy,
        confidence=np.array(confidence),
        class_id=np.array(class_id).astype(int),
        mask=None,
        tracker_id=None,
        data={
            CLASS_NAME_DATA_FIELD: np.array(class_name),
            IMAGE_DIMENSIONS_KEY: dimensions,
            INFERENCE_ID_KEY: np.array([inference_id] * len(xyxy)),
            DETECTION_ID_KEY: detection_ids,
            PREDICTION_TYPE_KEY: np.array(["object-detection"] * len(xyxy)),
        },
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections_result,
        image=image,
    )
