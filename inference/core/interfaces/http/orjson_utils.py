import base64
from typing import Any, Dict, List, Optional, Union

import numpy as np
import orjson
import supervision as sv
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.utils.image_utils import ImageType, encode_image_to_jpeg_bytes
from inference.core.workflows.constants import (
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CONFIDENCE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    KEYPOINTS_CLASS_ID_KEY,
    KEYPOINTS_CLASS_NAME_KEY,
    KEYPOINTS_CONFIDENCE_KEY,
    KEYPOINTS_KEY,
    KEYPOINTS_XY_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY,
    TRACKER_ID_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)


class ORJSONResponseBytes(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(
            content,
            default=default,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )


JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]


def default(obj: Any) -> JSON:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    return obj


def orjson_response(
    response: Union[List[InferenceResponse], InferenceResponse, BaseModel]
) -> ORJSONResponseBytes:
    if isinstance(response, list):
        content = [r.model_dump(by_alias=True, exclude_none=True) for r in response]
    else:
        content = response.model_dump(by_alias=True, exclude_none=True)
    return ORJSONResponseBytes(content=content)


def serialise_workflow_result(
    result: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if excluded_fields is None:
        excluded_fields = []
    excluded_fields = set(excluded_fields)
    serialised_result = {}
    for key, value in result.items():
        if key in excluded_fields:
            continue
        if contains_image(element=value):
            value = serialise_image(image=value)
        elif isinstance(value, dict):
            value = serialise_dict(elements=value)
        elif isinstance(value, list):
            value = serialise_list(elements=value)
        elif isinstance(value, sv.Detections):
            value = serialise_sv_detections(detections=value)
        serialised_result[key] = value
    return serialised_result


def serialise_list(elements: List[Any]) -> List[Any]:
    result = []
    for element in elements:
        if contains_image(element=element):
            element = serialise_image(image=element)
        elif isinstance(element, dict):
            element = serialise_dict(elements=element)
        elif isinstance(element, list):
            element = serialise_list(elements=element)
        elif isinstance(element, sv.Detections):
            element = serialise_sv_detections(detections=element)
        result.append(element)
    return result


def serialise_dict(elements: Dict[str, Any]) -> Dict[str, Any]:
    serialised_result = {}
    for key, value in elements.items():
        if contains_image(element=value):
            value = serialise_image(image=value)
        elif isinstance(value, dict):
            value = serialise_dict(elements=value)
        elif isinstance(value, list):
            value = serialise_list(elements=value)
        elif isinstance(value, sv.Detections):
            value = serialise_sv_detections(detections=value)
        serialised_result[key] = value
    return serialised_result


def contains_image(element: Any) -> bool:
    return (
        isinstance(element, dict)
        and element.get("type") == ImageType.NUMPY_OBJECT.value
    )


def serialise_image(image: Dict[str, Any]) -> Dict[str, Any]:
    image["type"] = "base64"
    image["value"] = base64.b64encode(
        encode_image_to_jpeg_bytes(image["value"])
    ).decode("ascii")
    return image


def serialise_sv_detections(detections: sv.Detections) -> List[Dict[str, Any]]:
    serialized_detections = []
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        detection_dict = {}

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
        if PARENT_ID_KEY in data:
            detection_dict[PARENT_ID_KEY] = str(data[PARENT_ID_KEY])
        if (
            KEYPOINTS_CLASS_ID_KEY in data
            and KEYPOINTS_CLASS_NAME_KEY in data
            and KEYPOINTS_CONFIDENCE_KEY in data
            and KEYPOINTS_XY_KEY in data
        ):
            kp_class_id = data[KEYPOINTS_CLASS_ID_KEY]
            kp_class_name = data[KEYPOINTS_CLASS_NAME_KEY]
            kp_confidence = data[KEYPOINTS_CONFIDENCE_KEY]
            kp_xy = data[KEYPOINTS_XY_KEY]
            detection_dict[KEYPOINTS_KEY] = []
            for (
                keypoint_class_id,
                keypoint_class_name,
                keypoint_confidence,
                (x, y),
            ) in zip(kp_class_id, kp_class_name, kp_confidence, kp_xy):
                detection_dict[KEYPOINTS_KEY].append(
                    {
                        KEYPOINTS_CLASS_ID_KEY: int(keypoint_class_id),
                        KEYPOINTS_CLASS_NAME_KEY: str(keypoint_class_name),
                        KEYPOINTS_CONFIDENCE_KEY: float(keypoint_confidence),
                        X_KEY: float(x),
                        Y_KEY: float(y),
                    }
                )
        serialized_detections.append(detection_dict)
    return serialized_detections
