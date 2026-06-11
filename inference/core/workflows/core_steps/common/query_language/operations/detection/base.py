from typing import Any, Optional

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.analytics.line_counter.v2 import (
    DETECTIONS_IN_OUT_PARAM,
)
from inference.core.workflows.core_steps.analytics.velocity.v1 import (
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_SV_DETECTIONS,
    VELOCITY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    CLASS_NAMES_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)

DETECTION_PROPERTY_EXTRACTION = {
    DetectionsProperty.X_MIN: lambda x: x[0][0].item(),
    DetectionsProperty.Y_MIN: lambda x: x[0][1].item(),
    DetectionsProperty.X_MAX: lambda x: x[0][2].item(),
    DetectionsProperty.Y_MAX: lambda x: x[0][3].item(),
    DetectionsProperty.CONFIDENCE: lambda x: x[2].item(),
    DetectionsProperty.CLASS_ID: lambda x: x[3].item(),
    DetectionsProperty.CLASS_NAME: lambda x: x[5].get("class_name").item(),
    DetectionsProperty.SIZE: lambda x: (
        (x[0][3] - x[0][1]) * (x[0][2] - x[0][0])
    ).item(),
    DetectionsProperty.CENTER: lambda x: (
        x[0][0] + (x[0][2] - x[0][0]) / 2,
        x[0][1] + (x[0][3] - x[0][1]) / 2,
    ),
    DetectionsProperty.TOP_LEFT: lambda x: (x[0][0].item(), x[0][1].item()),
    DetectionsProperty.TOP_RIGHT: lambda x: (x[0][2].item(), x[0][1].item()),
    DetectionsProperty.BOTTOM_LEFT: lambda x: (x[0][0].item(), x[0][3].item()),
    DetectionsProperty.BOTTOM_RIGHT: lambda x: (x[0][2].item(), x[0][3].item()),
    DetectionsProperty.IN_OUT: lambda x: x[5].get(DETECTIONS_IN_OUT_PARAM),
    DetectionsProperty.PATH_DEVIATION: lambda x: x[5].get(
        PATH_DEVIATION_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.POLYGON: lambda x: x[5].get(POLYGON_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.TIME_IN_ZONE: lambda x: x[5].get(
        TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.TRACKER_ID: lambda x: x[5].get("tracker_id"),
    DetectionsProperty.VELOCITY: lambda x: x[5].get(VELOCITY_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.SPEED: lambda x: x[5].get(SPEED_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.SMOOTHED_VELOCITY: lambda x: x[5].get(
        SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.SMOOTHED_SPEED: lambda x: x[5].get(
        SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.DIMENSIONS: lambda x: x[5].get(IMAGE_DIMENSIONS_KEY),
    DetectionsProperty.PREDICTION_TYPE: lambda x: x[5].get(PREDICTION_TYPE_KEY),
    DetectionsProperty.KEYPOINTS_XY: lambda x: x[5].get(
        KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT: lambda x: x[5].get(
        BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_WIDTH: lambda x: x[5].get(
        BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_HEIGHT: lambda x: x[5].get(
        BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_ANGLE: lambda x: x[5].get(
        BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.AREA: lambda x: x[5].get(AREA_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.AREA_CONVERTED: lambda x: x[5].get(
        AREA_CONVERTED_KEY_IN_SV_DETECTIONS
    ),
}


def _extract_detection_property(
    value: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> Any:
    if not isinstance(value, tuple) or len(value) != 6:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing extract_detection_property(...) operation in context {execution_context} "
            f"it was expected to get 6-elements tuple representing single element. Got value: "
            f"{value_as_str} of type {type(value)} ",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return DETECTION_PROPERTY_EXTRACTION[property_name](value)


def _extract_class_name_tensor_native(detection: tuple) -> str:
    class_names = detection[6].get(CLASS_NAMES_KEY)
    if class_names is None:
        raise OperationError(
            public_message=(
                "Executing extract_detection_property_tensor_native(...) for property "
                f"`class_name`, but `metadata['{CLASS_NAMES_KEY}']` is missing — the "
                "producer block must attach the class_id → name mapping."
            ),
            context="step_execution | roboflow_query_language_evaluation",
        )
    if detection[2] is None:
        raise OperationError(
            public_message=(
                "Executing extract_detection_property_tensor_native(...) for property "
                "`class_name`, but the detection carries no class_id."
            ),
            context="step_execution | roboflow_query_language_evaluation",
        )
    class_id = int(detection[2])
    class_name = class_names.get(class_id)
    if class_name is None:
        raise OperationError(
            public_message=(
                "Executing extract_detection_property_tensor_native(...) for property "
                f"`class_name`, class_id={class_id} is missing from the class_names "
                f"mapping (keys present: {sorted(class_names.keys())})."
            ),
            context="step_execution | roboflow_query_language_evaluation",
        )
    return class_name


def _extract_tracker_id_tensor_native(detection: tuple) -> Optional[int]:
    if detection[4] is not None:
        return int(detection[4])
    tracker_id = detection[5].get("tracker_id")
    if tracker_id is None:
        return None
    return int(tracker_id)


DETECTION_PROPERTY_EXTRACTION_TENSOR_NATIVE = {
    DetectionsProperty.X_MIN: lambda x: float(x[0][0]),
    DetectionsProperty.Y_MIN: lambda x: float(x[0][1]),
    DetectionsProperty.X_MAX: lambda x: float(x[0][2]),
    DetectionsProperty.Y_MAX: lambda x: float(x[0][3]),
    DetectionsProperty.CONFIDENCE: lambda x: (
        None if x[3] is None else float(x[3])
    ),
    DetectionsProperty.CLASS_ID: lambda x: None if x[2] is None else int(x[2]),
    DetectionsProperty.CLASS_NAME: _extract_class_name_tensor_native,
    DetectionsProperty.SIZE: lambda x: float(
        (x[0][3] - x[0][1]) * (x[0][2] - x[0][0])
    ),
    DetectionsProperty.CENTER: lambda x: (
        float(x[0][0] + (x[0][2] - x[0][0]) / 2),
        float(x[0][1] + (x[0][3] - x[0][1]) / 2),
    ),
    DetectionsProperty.TOP_LEFT: lambda x: (float(x[0][0]), float(x[0][1])),
    DetectionsProperty.TOP_RIGHT: lambda x: (float(x[0][2]), float(x[0][1])),
    DetectionsProperty.BOTTOM_LEFT: lambda x: (float(x[0][0]), float(x[0][3])),
    DetectionsProperty.BOTTOM_RIGHT: lambda x: (float(x[0][2]), float(x[0][3])),
    DetectionsProperty.IN_OUT: lambda x: x[5].get(DETECTIONS_IN_OUT_PARAM),
    DetectionsProperty.PATH_DEVIATION: lambda x: x[5].get(
        PATH_DEVIATION_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.POLYGON: lambda x: x[5].get(POLYGON_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.TIME_IN_ZONE: lambda x: x[5].get(
        TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.TRACKER_ID: _extract_tracker_id_tensor_native,
    DetectionsProperty.VELOCITY: lambda x: x[5].get(VELOCITY_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.SPEED: lambda x: x[5].get(SPEED_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.SMOOTHED_VELOCITY: lambda x: x[5].get(
        SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.SMOOTHED_SPEED: lambda x: x[5].get(
        SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.DIMENSIONS: lambda x: x[6].get(IMAGE_DIMENSIONS_KEY),
    DetectionsProperty.PREDICTION_TYPE: lambda x: x[6].get(PREDICTION_TYPE_KEY),
    DetectionsProperty.KEYPOINTS_XY: lambda x: x[5].get(
        KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT: lambda x: x[5].get(
        BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_WIDTH: lambda x: x[5].get(
        BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_HEIGHT: lambda x: x[5].get(
        BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.BOUNDING_RECT_ANGLE: lambda x: x[5].get(
        BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS
    ),
    DetectionsProperty.AREA: lambda x: x[5].get(AREA_KEY_IN_SV_DETECTIONS),
    DetectionsProperty.AREA_CONVERTED: lambda x: x[5].get(
        AREA_CONVERTED_KEY_IN_SV_DETECTIONS
    ),
}


def _extract_detection_property_tensor_native(
    value: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> Any:
    if not isinstance(value, tuple) or len(value) != 7:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=(
                "While executing extract_detection_property_tensor_native(...) "
                f"operation in context {execution_context} it was expected to get "
                "a 7-element tuple (xyxy, mask, class_id, confidence, tracker_id, "
                "data, metadata) representing a single tensor-native detection. "
                f"Got value: {value_as_str} of type {type(value)}"
            ),
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return DETECTION_PROPERTY_EXTRACTION_TENSOR_NATIVE[property_name](value)


def extract_detection_property(
    value: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> Any:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _extract_detection_property_tensor_native(
            value=value,
            property_name=property_name,
            execution_context=execution_context,
        )
    return _extract_detection_property(
        value=value,
        property_name=property_name,
        execution_context=execution_context,
    )
