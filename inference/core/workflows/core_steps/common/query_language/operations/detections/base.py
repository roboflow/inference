from copy import copy, deepcopy
from typing import Any, Callable, Dict, List

import numpy as np
import supervision as sv
from supervision import Position

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
    DetectionsSelectionMode,
    DetectionsSortProperties,
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
    detections: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> List[Any]:
    if not isinstance(detections, sv.Detections):
        value_as_str = safe_stringify(value=detections)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_detections_property(...) in context {execution_context}, "
            f"expected sv.Detections object as value, got {value_as_str} of type {type(detections)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return PROPERTIES_EXTRACTORS[property_name](detections)


def filter_detections(
    detections: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> sv.Detections:
    if not isinstance(detections, sv.Detections):
        value_as_str = safe_stringify(value=detections)
        raise InvalidInputTypeError(
            public_message=f"Executing filter_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(detections)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    local_parameters = copy(global_parameters)
    result = []
    for detection in detections:
        local_parameters[DEFAULT_OPERAND_NAME] = detection
        should_stay = filtering_fun(local_parameters)
        result.append(should_stay)
    return detections[result]


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
    detections_copy = deepcopy(value)
    detections_copy.xyxy += [-offset_x / 2, -offset_y / 2, offset_x / 2, offset_y / 2]
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
    detections_copy.xyxy += [shift_x, shift_y, shift_x, shift_y]
    return detections_copy


def select_leftmost_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return deepcopy(detections)
    centers_x = detections.get_anchors_coordinates(anchor=Position.CENTER)[:, 0]
    min_value = centers_x.min()
    index = np.argwhere(centers_x == min_value)[0].item()
    return detections[index]


def select_rightmost_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return deepcopy(detections)
    centers_x = detections.get_anchors_coordinates(anchor=Position.CENTER)[:, 0]
    max_value = centers_x.max()
    index = np.argwhere(centers_x == max_value)[-1].item()
    return detections[index]


DETECTIONS_SELECTORS = {
    DetectionsSelectionMode.LEFT_MOST: select_leftmost_detection,
    DetectionsSelectionMode.RIGHT_MOST: select_rightmost_detection,
}


def select_detections(
    value: Any, mode: DetectionsSelectionMode, **kwargs
) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing select_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if mode not in DETECTIONS_SELECTORS:
        InvalidInputTypeError(
            public_message=f"Executing select_detections(...), expected mode to be one of {DETECTIONS_SELECTORS.values()}, "
            f"got {mode}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    return DETECTIONS_SELECTORS[mode](value)


def extract_x_coordinate_of_detections_center(detections: sv.Detections) -> np.ndarray:
    return detections.xyxy[:, 0] + (detections.xyxy[:, 2] - detections.xyxy[:, 0]) / 2


def extract_y_coordinate_of_detections_center(detections: sv.Detections) -> np.ndarray:
    return detections.xyxy[:, 1] + (detections.xyxy[:, 3] - detections.xyxy[:, 1]) / 2


SORT_PROPERTIES_EXTRACT = {
    DetectionsSortProperties.CONFIDENCE: lambda detections: detections.confidence,
    DetectionsSortProperties.X_MIN: lambda detections: detections.xyxy[:, 0],
    DetectionsSortProperties.X_MAX: lambda detections: detections.xyxy[:, 2],
    DetectionsSortProperties.Y_MIN: lambda detections: detections.xyxy[:, 1],
    DetectionsSortProperties.Y_MAX: lambda detections: detections.xyxy[:, 3],
    DetectionsSortProperties.SIZE: lambda detections: detections.box_area,
    DetectionsSortProperties.CENTER_X: extract_x_coordinate_of_detections_center,
    DetectionsSortProperties.CENTER_Y: extract_y_coordinate_of_detections_center,
}


def sort_detections(
    value: Any, mode: DetectionsSortProperties, ascending: bool, **kwargs
) -> sv.Detections:
    if not isinstance(value, sv.Detections):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing sort_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if mode not in SORT_PROPERTIES_EXTRACT:
        InvalidInputTypeError(
            public_message=f"Executing sort_detections(...), expected mode to be one of "
            f"{SORT_PROPERTIES_EXTRACT.values()}, got {mode}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if len(value) == 0:
        return value
    extracted_property = SORT_PROPERTIES_EXTRACT[mode](value)
    if extracted_property is None:
        # property may not be set, as sv.Detections declares Optional[...]
        return value
    sorted_indices = np.argsort(extracted_property)
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    return value[sorted_indices]
