from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Union

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
    OperationError,
    UndeclaredSymbolError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)

PROPERTIES_EXTRACTORS = {
    DetectionsProperty.CONFIDENCE: lambda detections: detections.confidence.tolist(),
    DetectionsProperty.CLASS_NAME: lambda detections: detections.data.get(
        "class_name", np.array([], dtype=str)
    ).tolist(),
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


def select_top_confidence_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return deepcopy(detections)
    confidence = detections.confidence
    max_value = confidence.max()
    index = np.argwhere(confidence == max_value)[0].item()
    return detections[index]


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
    DetectionsSelectionMode.TOP_CONFIDENCE: select_top_confidence_detection,
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
        raise InvalidInputTypeError(
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


def rename_detections(
    detections: Any,
    class_map: Union[Dict[str, str], str],
    strict: Union[bool, str],
    new_classes_id_offset: int,
    global_parameters: Dict[str, Any],
    **kwargs,
) -> sv.Detections:
    if not isinstance(detections, sv.Detections):
        value_as_str = safe_stringify(value=detections)
        raise InvalidInputTypeError(
            public_message=f"Executing rename_detections(...), expected sv.Detections object as value, "
            f"got {value_as_str} of type {type(detections)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if isinstance(class_map, str):
        if class_map not in global_parameters:
            raise UndeclaredSymbolError(
                public_message=f"Attempted to retrieve variable `{class_map}` that was expected to hold "
                f"class mapping of rename_detections(...), but that turned out not to be registered.",
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_map = global_parameters[class_map]
    if not isinstance(class_map, dict):
        value_as_str = safe_stringify(value=class_map)
        raise InvalidInputTypeError(
            public_message=f"Executing rename_detections(...), expected dictionary to be given as class map, "
            f"got {value_as_str} of type {type(class_map)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if isinstance(strict, str):
        if strict not in global_parameters:
            raise UndeclaredSymbolError(
                public_message=f"Attempted to retrieve variable `{strict}` that was expected to hold "
                f"parameter for `strict` flag of rename_detections(...), but that turned out not "
                f"to be registered.",
                context="step_execution | roboflow_query_language_evaluation",
            )
        strict = global_parameters[strict]
    if not isinstance(strict, bool):
        value_as_str = safe_stringify(value=strict)
        raise InvalidInputTypeError(
            public_message=f"Executing rename_detections(...), expected dictionary to be given as `strict` flag, "
            f"got {value_as_str} of type {type(strict)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    detections_copy = deepcopy(detections)
    original_class_names = detections_copy.data.get("class_name", []).tolist()
    original_class_ids = detections_copy.class_id.tolist()
    new_class_names = []
    new_class_ids = []
    if strict:
        _ensure_all_classes_covered_in_new_mapping(
            original_class_names=original_class_names,
            class_map=class_map,
        )
        new_class_mapping = {
            class_name: class_id
            for class_id, class_name in enumerate(sorted(set(class_map.values())))
        }
    else:
        new_class_mapping = _build_non_strict_class_to_id_mapping(
            original_class_names=original_class_names,
            original_class_ids=original_class_ids,
            class_map=class_map,
            new_classes_id_offset=new_classes_id_offset,
        )
    for class_name in original_class_names:
        new_class_name = class_map.get(class_name, class_name)
        new_class_id = new_class_mapping[new_class_name]
        new_class_names.append(new_class_name)
        new_class_ids.append(new_class_id)
    detections_copy.data["class_name"] = np.array(new_class_names, dtype=object)
    detections_copy.class_id = np.array(new_class_ids, dtype=int)
    return detections_copy


def _ensure_all_classes_covered_in_new_mapping(
    original_class_names: List[str],
    class_map: Dict[str, str],
) -> None:
    for original_class in original_class_names:
        if original_class not in class_map:
            raise OperationError(
                public_message=f"Class '{original_class}' not found in class_map.",
                context="step_execution | roboflow_query_language_evaluation",
            )


def _build_non_strict_class_to_id_mapping(
    original_class_names: List[str],
    original_class_ids: List[int],
    class_map: Dict[str, str],
    new_classes_id_offset: int,
) -> Dict[str, int]:
    original_mapping = {
        class_name: class_id
        for class_name, class_id in zip(original_class_names, original_class_ids)
    }
    new_target_classes = {
        new_class_name
        for new_class_name in class_map.values()
        if new_class_name not in original_mapping
    }
    new_class_id = new_classes_id_offset
    for new_target_class in sorted(new_target_classes):
        original_mapping[new_target_class] = new_class_id
        new_class_id += 1
    return original_mapping
