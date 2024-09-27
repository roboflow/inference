from typing import Any, Dict

import numpy as np
import supervision as sv

from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CONFIDENCE_KEY,
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_ID_KEY,
    PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY,
    TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
    TRACKER_ID_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def serialise_sv_detections(detections: sv.Detections) -> dict:
    serialized_detections = []
    image_dimensions = None
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        detection_dict = {}
        image_dimensions = data.get(IMAGE_DIMENSIONS_KEY)
        if isinstance(xyxy, np.ndarray):
            xyxy = xyxy.astype(float).tolist()
        x1, y1, x2, y2 = xyxy
        detection_dict[WIDTH_KEY] = abs(x2 - x1)
        detection_dict[HEIGHT_KEY] = abs(y2 - y1)
        detection_dict[X_KEY] = x1 + detection_dict[WIDTH_KEY] / 2
        detection_dict[Y_KEY] = y1 + detection_dict[HEIGHT_KEY] / 2

        detection_dict[CONFIDENCE_KEY] = float(confidence)
        detection_dict[CLASS_ID_KEY] = int(class_id)
        if mask is not None:
            polygon = sv.mask_to_polygons(mask=mask)
            detection_dict[POLYGON_KEY] = []
            for x, y in polygon[0]:
                detection_dict[POLYGON_KEY].append(
                    {
                        X_KEY: float(x),
                        Y_KEY: float(y),
                    }
                )
        if tracker_id is not None:
            detection_dict[TRACKER_ID_KEY] = int(tracker_id)
        detection_dict[CLASS_NAME_KEY] = str(data["class_name"])
        detection_dict[DETECTION_ID_KEY] = str(data[DETECTION_ID_KEY])
        if PATH_DEVIATION_KEY_IN_SV_DETECTIONS in data:
            detection_dict[PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE] = data[
                PATH_DEVIATION_KEY_IN_SV_DETECTIONS
            ]
        if TIME_IN_ZONE_KEY_IN_SV_DETECTIONS in data:
            detection_dict[TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE] = data[
                TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
            ]
        if (
            BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS in data
        ):
            detection_dict[BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
            ]
        if PARENT_ID_KEY in data:
            detection_dict[PARENT_ID_KEY] = str(data[PARENT_ID_KEY])
        if (
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in data
        ):
            kp_class_id = data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS]
            kp_class_name = data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS]
            kp_confidence = data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS]
            kp_xy = data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
            detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE] = []
            for (
                keypoint_class_id,
                keypoint_class_name,
                keypoint_confidence,
                (x, y),
            ) in zip(kp_class_id, kp_class_name, kp_confidence, kp_xy):
                detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE].append(
                    {
                        "class_id": int(keypoint_class_id),
                        "class": str(keypoint_class_name),
                        "confidence": float(keypoint_confidence),
                        "x": float(x),
                        "y": float(y),
                    }
                )
        if DETECTED_CODE_KEY in data:
            detection_dict[DETECTED_CODE_KEY] = data[DETECTED_CODE_KEY]
        serialized_detections.append(detection_dict)
    image_metadata = {
        "width": None,
        "height": None,
    }  # TODO: this breaks the contract of
    # standard inference, but to fix that problem, we would need sv.Detections to provide
    # detection-level metadata.
    if image_dimensions is not None:
        image_metadata = {
            "width": image_dimensions[1].item(),
            "height": image_dimensions[0].item(),
        }
    return {"image": image_metadata, "predictions": serialized_detections}


def serialise_image(image: WorkflowImageData) -> Dict[str, Any]:
    return {
        "type": "base64",
        "value": image.base64_image,
    }
