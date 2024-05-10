from typing import Any

import supervision as sv

from inference.core.workflows.core_steps.common.query_language.entities.enums import DetectionsProperty, \
    SequenceUnwrapMethod
from inference.core.workflows.core_steps.common.query_language.errors import InvalidInputTypeError
from inference.core.workflows.core_steps.common.query_language.operations.utils import safe_stringify

DETECTION_PROPERTY_EXTRACTION = {
    DetectionsProperty.X_MIN: lambda x: x[0][0].item(),
    DetectionsProperty.Y_MIN: lambda x: x[0][1].item(),
    DetectionsProperty.X_MAX: lambda x: x[0][2].item(),
    DetectionsProperty.Y_MAX: lambda x: x[0][3].item(),
    DetectionsProperty.CONFIDENCE: lambda x: x[2],
    DetectionsProperty.CLASS_ID: lambda x: x[3],
    DetectionsProperty.CLASS_NAME: lambda x: x[5].get("class_name"),
    DetectionsProperty.SIZE: lambda x: (x[0][3] - x[0][1]) * (x[0][2] - x[0][0])
}


def extract_detection_property(
    value: Any,
    property_name: DetectionsProperty,
    unwrap_method: SequenceUnwrapMethod,
) -> Any:
    if isinstance(value, sv.Detections):
        if unwrap_method is SequenceUnwrapMethod.FIRST:
            value = next(value.__iter__())
        else:
            value = list(value.__iter__())[-1]
    if not isinstance(value, tuple) or len(value) != 6:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing extract_detection_property(...) operation it was "
                           f"expected to get 6-elements tuple representing single element. Got value: "
                           f"{value_as_str} of type {type(value)} ",
            context="step_execution | roboflow_query_language_evaluation",
        )
    return DETECTION_PROPERTY_EXTRACTION[property_name](value)
