from typing import Any, Callable, Dict, List

import supervision as sv

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def extract_detections_property(
    value: Any,
    property_name: DetectionsProperty,
) -> List[Any]:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Expected sv.Detections object as value, got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if property_name == "class_name":
        return value.data["class_name"].tolist()
    return value.confidence.tolist()


def filter_detections(
    value: Any, filtering_fun: Callable[[Dict[str, Any]], bool]
) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Expected sv.Detections object as value, got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    result = []
    for detection in value:
        should_stay = filtering_fun({"_": detection})
        result.append(should_stay)
    return value[result]
