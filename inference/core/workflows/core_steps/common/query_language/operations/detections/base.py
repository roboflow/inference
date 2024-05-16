from copy import copy, deepcopy
from typing import Any, Callable, Dict, List

import supervision as sv

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DEFAULT_OPERAND_NAME,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)

PROPERTIES_EXTRACTORS = {
    DetectionsProperty.CONFIDENCE: lambda detections: detections.confidence.tolist(),
    DetectionsProperty.CLASS_NAME: lambda detections: detections.data[
        "class_name"
    ].tolist(),
    DetectionsProperty.X_MIN: lambda detections: detections.xyxy[:, 0].tolist(),
    DetectionsProperty.Y_MIN: lambda detections: detections.xyxy[:, 1].tolist(),
    DetectionsProperty.X_MAX: lambda detections: detections.xyxy[:, 2].tolist(),
    DetectionsProperty.Y_MAX: lambda detections: detections.xyxy[:, 3].tolist(),
    DetectionsProperty.CLASS_ID: lambda detections: detections.class_id.tolist(),
    DetectionsProperty.SIZE: lambda detections: detections.box_area.tolist(),
}


def extract_detections_property(
    value: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> List[Any]:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_detections_property(...) in context {execution_context}, "
            f"expected sv.Detections object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return PROPERTIES_EXTRACTORS[property_name](value)


def filter_detections(
    value: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing filter_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    local_parameters = copy(global_parameters)
    result = []
    for detection in value:
        local_parameters[DEFAULT_OPERAND_NAME] = detection
        should_stay = filtering_fun(local_parameters)
        result.append(should_stay)
    return value[result]


def offset_detections(
    value: Any, offset_x: int, offset_y: int, **kwargs
) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing offset_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    width_change = offset_x / 2
    height_change = offset_y / 2
    detections_copy = deepcopy(value)
    detections_copy.xyxy[:, 0] -= width_change
    detections_copy.xyxy[:, 2] += width_change
    detections_copy.xyxy[:, 1] -= height_change
    detections_copy.xyxy[:, 3] += height_change
    return detections_copy


def shift_detections(value: Any, shift_x: int, shift_y: int, **kwargs) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing shift_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    detections_copy = deepcopy(value)
    detections_copy.xyxy[:, 0] += shift_x
    detections_copy.xyxy[:, 2] += shift_x
    detections_copy.xyxy[:, 1] += shift_y
    detections_copy.xyxy[:, 3] += shift_y
    return detections_copy
