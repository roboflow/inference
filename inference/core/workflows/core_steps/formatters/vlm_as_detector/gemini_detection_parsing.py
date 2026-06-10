from typing import List, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

GEMINI_NATIVE_BOX_COORDINATE_SCALE = 1000.0


def extract_gemini_detection_entries(
    parsed_data: Union[dict, list],
) -> List[dict]:
    if isinstance(parsed_data, list):
        return parsed_data
    if isinstance(parsed_data, dict) and "detections" in parsed_data:
        return parsed_data["detections"]
    raise ValueError("Unexpected Gemini object detection response format")


def get_gemini_detection_class_name(detection: dict) -> str:
    for key in ("class_name", "label", "class"):
        value = detection.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def parse_gemini_detection_xyxy(
    detection: dict,
    image_height: int,
    image_width: int,
) -> List[float]:
    if "box_2d" in detection:
        y_min, x_min, y_max, x_max = detection["box_2d"]
        scale = GEMINI_NATIVE_BOX_COORDINATE_SCALE
        return [
            x_min / scale * image_width,
            y_min / scale * image_height,
            x_max / scale * image_width,
            y_max / scale * image_height,
        ]
    return [
        detection["x_min"] * image_width,
        detection["y_min"] * image_height,
        detection["x_max"] * image_width,
        detection["y_max"] * image_height,
    ]


def scale_confidence(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def create_classes_index(classes: List[str]) -> dict:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def parse_gemini_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]
    detections = extract_gemini_detection_entries(parsed_data=parsed_data)
    if len(detections) == 0:
        return sv.Detections.empty()

    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in detections:
        xyxy.append(
            parse_gemini_detection_xyxy(
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
