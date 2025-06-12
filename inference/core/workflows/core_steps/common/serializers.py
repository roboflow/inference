from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

from inference.core import logger
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
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
    TRACKER_ID_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImageData,
)

MIN_SECRET_LENGTH_TO_REVEAL_PREFIX = 8
MIN_POLYGON_POINT_COUNT = 3


def serialise_sv_detections(detections: sv.Detections) -> dict:
    serialized_detections = []
    image_dimensions = None

    for detection in detections:
        xyxy, mask, confidence, class_id, tracker_id, data = detection
        detection_dict = {
            WIDTH_KEY: abs(xyxy[2] - xyxy[0]),
            HEIGHT_KEY: abs(xyxy[3] - xyxy[1]),
            X_KEY: xyxy[0] + abs(xyxy[2] - xyxy[0]) / 2,
            Y_KEY: xyxy[1] + abs(xyxy[3] - xyxy[1]) / 2,
            CONFIDENCE_KEY: float(confidence),
            CLASS_ID_KEY: int(class_id),
            CLASS_NAME_KEY: str(data["class_name"]),
            DETECTION_ID_KEY: str(data[DETECTION_ID_KEY]),
        }

        image_dimensions = data.get(IMAGE_DIMENSIONS_KEY)

        if mask is not None:
            if (
                POLYGON_KEY_IN_SV_DETECTIONS in data
                and data[POLYGON_KEY_IN_SV_DETECTIONS] is not None
                and len(data[POLYGON_KEY_IN_SV_DETECTIONS]) > 2
            ):
                polygon = data[POLYGON_KEY_IN_SV_DETECTIONS]
            else:
                polygon = mask_to_polygon(mask=mask)
            if polygon is None:
                # ignoring the whole instance
                continue
            detection_dict[POLYGON_KEY] = [
                {X_KEY: float(x), Y_KEY: float(y)}
                for x, y in sv.mask_to_polygons(mask)[0]
            ]

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
        if POLYGON_KEY_IN_SV_DETECTIONS in data:
            detection_dict[POLYGON_KEY_IN_INFERENCE_RESPONSE] = (
                data[POLYGON_KEY_IN_SV_DETECTIONS]
                .astype(float)
                .round()
                .astype(int)
                .tolist()
            )

        bounding_rect_keys = [
            BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
            BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
            BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
            BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
        ]

        if all(key in data for key in bounding_rect_keys):
            bounding_rect_mapping = {
                BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE: BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
                BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE: BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
                BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE: BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
                BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE: BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
            }
            for infer_key, sv_key in bounding_rect_mapping.items():
                detection_dict[infer_key] = data[sv_key]

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

            detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE] = [
                {
                    "class_id": int(kpc_id),
                    "class": str(kpc_name),
                    "confidence": float(kpc_conf),
                    "x": float(x),
                    "y": float(y),
                }
                for kpc_id, kpc_name, kpc_conf, (x, y) in zip(
                    kp_class_id, kp_class_name, kp_confidence, kp_xy
                )
            ]

        serialized_detections.append(detection_dict)

    image_metadata = {
        "width": image_dimensions[1].item() if image_dimensions is not None else None,
        "height": image_dimensions[0].item() if image_dimensions is not None else None,
    }

    return {"image": image_metadata, "predictions": serialized_detections}


def mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    # masks here should be predicted by instance segmentation
    # model and in theory can only present SINGLE!!! shape
    # our response schema for InstanceSegmentationPrediction
    # (see `inference.core.entities.responses.inference.InstanceSegmentationPrediction`)
    # FROM THE BEGINNING were allowing to serialise single polygon that belongs to mask,
    # no hierarchy respected and multiple polygons are not taken into account
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        # nothing found - instance should be rejected
        return None
    if len(contours) > 1:
        logger.warning(
            f"Detected instance segmentation that has {len(contours)} in the mask, which by convention "
            "should not happen. We are taking the first polygon to avoid exceptions, but if you see this "
            "warning - it may indicate that some model producing instance segmentation result works under "
            "different assumptions that models historically added into inference and that should be a signal "
            "to figure out more generic representation for instance seg predictions."
        )
    contour = np.squeeze(contours[0], axis=1)
    contour_padding = max(MIN_POLYGON_POINT_COUNT - contour.shape[0], 0)
    if contour_padding > 0:
        padding = np.repeat(
            np.expand_dims(contour[-1], axis=0), repeats=contour_padding, axis=0
        )
        return np.append(contour, padding, axis=0)
    return contour


def serialise_image(image: WorkflowImageData) -> Dict[str, Any]:
    return {
        "type": "base64",
        "value": image.base64_image,
        "video_metadata": image.video_metadata.dict(),
    }


def serialize_video_metadata_kind(video_metadata: VideoMetadata) -> dict:
    return video_metadata.dict()


def serialize_wildcard_kind(value: Any) -> Any:
    if isinstance(value, WorkflowImageData):
        value = serialise_image(image=value)
    elif isinstance(value, dict):
        value = serialize_dict(elements=value)
    elif isinstance(value, list):
        value = serialize_list(elements=value)
    elif isinstance(value, sv.Detections):
        value = serialise_sv_detections(detections=value)
    elif isinstance(value, datetime):
        value = serialize_timestamp(timestamp=value)
    return value


def serialize_list(elements: List[Any]) -> List[Any]:
    result = []
    for element in elements:
        element = serialize_wildcard_kind(value=element)
        result.append(element)
    return result


def serialize_dict(elements: Dict[str, Any]) -> Dict[str, Any]:
    serialized_result = {}
    for key, value in elements.items():
        value = serialize_wildcard_kind(value=value)
        serialized_result[key] = value
    return serialized_result


def serialize_secret(secret: str) -> str:
    if len(secret) < MIN_SECRET_LENGTH_TO_REVEAL_PREFIX:
        return "*" * MIN_SECRET_LENGTH_TO_REVEAL_PREFIX
    prefix = secret[:2]
    infix = "*" * MIN_SECRET_LENGTH_TO_REVEAL_PREFIX
    suffix = secret[-2:]
    return f"{prefix}{infix}{suffix}"


def serialize_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat()
