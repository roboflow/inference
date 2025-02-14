from typing import Any

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
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)
from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
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
    DetectionsProperty.TOP_LEFT: lambda xyxy: (xyxy[0], xyxy[1]),
    DetectionsProperty.TOP_RIGHT: lambda xyxy: (xyxy[2], xyxy[1]),
    DetectionsProperty.BOTTOM_LEFT: lambda xyxy: (xyxy[0], xyxy[3]),
    DetectionsProperty.BOTTOM_RIGHT: lambda xyxy: (xyxy[2], xyxy[3]),
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
}


def extract_detection_property(
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
