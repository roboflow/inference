import os
from typing import Any, List
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
from pydantic import ValidationError

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    load_image_from_url,
)
from inference.core.workflows.core_steps.common.utils import (
    add_inference_keypoints_to_sv_detections,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    PARENT_ID_KEY,
    PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def deserialize_image_kind(
    parameter: str,
    image: Any,
    prevent_local_images_loading: bool = False,
) -> WorkflowImageData:
    if isinstance(image, WorkflowImageData):
        return image
    video_metadata = None
    if isinstance(image, dict) and "video_metadata" in image:
        video_metadata = deserialize_video_metadata_kind(
            parameter=parameter, video_metadata=image["video_metadata"]
        )
    if isinstance(image, dict) and isinstance(image.get("value"), np.ndarray):
        image = image["value"]
    if isinstance(image, np.ndarray):
        parent_metadata = ImageParentMetadata(parent_id=parameter)
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            numpy_image=image,
            video_metadata=video_metadata,
        )
    try:
        if isinstance(image, dict):
            image = image["value"]
        if isinstance(image, str):
            base64_image = None
            image_reference = None
            if image.startswith("http://") or image.startswith("https://"):
                image_reference = image
                image = load_image_from_url(value=image)
            elif not prevent_local_images_loading and os.path.exists(image):
                # prevent_local_images_loading is introduced to eliminate
                # server vulnerability - namely it prevents local server
                # file system from being exploited.
                image_reference = image
                image = cv2.imread(image)
            else:
                base64_image = image
                image = attempt_loading_image_from_string(image)[0]
            parent_metadata = ImageParentMetadata(parent_id=parameter)
            return WorkflowImageData(
                parent_metadata=parent_metadata,
                numpy_image=image,
                base64_image=base64_image,
                image_reference=image_reference,
                video_metadata=video_metadata,
            )
    except Exception as error:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as `WorkflowImage` "
            f"that is invalid. Failed on input validation. Details: {error}",
            context="workflow_execution | runtime_input_validation",
        ) from error
    raise RuntimeInputError(
        public_message=f"Detected runtime parameter `{parameter}` defined as `WorkflowImage` "
        f"with type {type(image)} that is invalid. Workflows accept only np.arrays, `WorkflowImageData` "
        f"and dicts with keys `type` and `value` compatible with `inference` (or list of them).",
        context="workflow_execution | runtime_input_validation",
    )


def deserialize_video_metadata_kind(
    parameter: str,
    video_metadata: Any,
) -> VideoMetadata:
    if isinstance(video_metadata, VideoMetadata):
        return video_metadata
    if not isinstance(video_metadata, dict):
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` holding "
            f"`WorkflowVideoMetadata`, but provided value is not a dict.",
            context="workflow_execution | runtime_input_validation",
        )
    try:
        return VideoMetadata.model_validate(video_metadata)
    except ValidationError as error:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` holding "
            f"`WorkflowVideoMetadata`, but provided value is malformed. "
            f"See details in inner error.",
            context="workflow_execution | runtime_input_validation",
            inner_error=error,
        )


def deserialize_detections_kind(
    parameter: str,
    detections: Any,
) -> sv.Detections:
    if isinstance(detections, sv.Detections):
        return detections
    if not isinstance(detections, dict):
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` declared to hold "
            f"detections, but invalid type of data found.",
            context="workflow_execution | runtime_input_validation",
        )
    if "predictions" not in detections or "image" not in detections:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` declared to hold "
            f"detections, but dictionary misses required keys.",
            context="workflow_execution | runtime_input_validation",
        )
    parsed_detections = sv.Detections.from_inference(detections)
    if len(parsed_detections) == 0:
        return parsed_detections
    height, width = detections["image"]["height"], detections["image"]["width"]
    image_metadata = np.array([[height, width]] * len(parsed_detections))
    parsed_detections.data[IMAGE_DIMENSIONS_KEY] = image_metadata
    detection_ids = [
        detection.get(DETECTION_ID_KEY, str(uuid4()))
        for detection in detections["predictions"]
    ]
    parsed_detections.data[DETECTION_ID_KEY] = np.array(detection_ids)
    parent_ids = [
        detection.get(PARENT_ID_KEY, parameter)
        for detection in detections["predictions"]
    ]
    detections[PARENT_ID_KEY] = np.array(parent_ids)
    optional_elements_keys = [
        (PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE, PATH_DEVIATION_KEY_IN_SV_DETECTIONS),
        (TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE, TIME_IN_ZONE_KEY_IN_SV_DETECTIONS),
        (
            BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE,
            BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
        ),
        (
            BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE,
            BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
        ),
        (
            BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE,
            BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
        ),
        (
            BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE,
            BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
        ),
        (DETECTED_CODE_KEY, DETECTED_CODE_KEY),
    ]
    for raw_detection_key, parsed_detection_key in optional_elements_keys:
        parsed_detections = _attach_optional_detection_element(
            raw_detections=detections["predictions"],
            parsed_detections=parsed_detections,
            raw_detection_key=raw_detection_key,
            parsed_detection_key=parsed_detection_key,
        )
    return _attach_optional_key_points_detections(
        raw_detections=detections["predictions"],
        parsed_detections=parsed_detections,
    )


def _attach_optional_detection_element(
    raw_detections: List[dict],
    parsed_detections: sv.Detections,
    raw_detection_key: str,
    parsed_detection_key: str,
) -> sv.Detections:
    if raw_detection_key not in raw_detections[0]:
        return parsed_detections
    result = []
    for detection in raw_detections:
        result.append(detection[raw_detection_key])
    parsed_detections.data[parsed_detection_key] = np.array(result)
    return parsed_detections


def _attach_optional_key_points_detections(
    raw_detections: List[dict],
    parsed_detections: sv.Detections,
) -> sv.Detections:
    if KEYPOINTS_KEY_IN_INFERENCE_RESPONSE not in raw_detections[0]:
        return parsed_detections
    return add_inference_keypoints_to_sv_detections(
        inference_prediction=raw_detections,
        detections=parsed_detections,
    )


def deserialize_numpy_array(parameter: str, raw_array: Any) -> np.ndarray:
    if isinstance(raw_array, np.ndarray):
        return raw_array
    return np.array(raw_array)
