from typing import Any, Dict, List
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
            detection_dict[POLYGON_KEY] = [
                {X_KEY: float(x), Y_KEY: float(y)}
                for x, y in sv.mask_to_polygons(mask)[0]
            ]

        if tracker_id is not None:
            detection_dict[TRACKER_ID_KEY] = int(tracker_id)

        optional_keys = [
            PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
            TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
            POLYGON_KEY_IN_SV_DETECTIONS,
            PARENT_ID_KEY,
            DETECTED_CODE_KEY,
        ]

        for key in optional_keys:
            if key in data:
                detection_dict[key] = data[key]

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
        "width": image_dimensions[1].item() if image_dimensions else None,
        "height": image_dimensions[0].item() if image_dimensions else None,
    }

    return {"image": image_metadata, "predictions": serialized_detections}


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
