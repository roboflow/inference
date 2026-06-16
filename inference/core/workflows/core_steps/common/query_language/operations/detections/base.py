from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import supervision as sv
import torch
from supervision import Position

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
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
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections as serialise_tensor_native_detections,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY


def detections_anchor_coordinates(
    detections: sv.Detections, anchor: Position
) -> np.ndarray:
    return (
        detections.get_anchors_coordinates(anchor=anchor).round().astype(int).tolist()
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
    DetectionsProperty.CENTER: lambda detections: detections_anchor_coordinates(
        detections=detections, anchor=Position.CENTER
    ),
    DetectionsProperty.TOP_LEFT: lambda detections: detections_anchor_coordinates(
        detections=detections, anchor=Position.TOP_LEFT
    ),
    DetectionsProperty.TOP_RIGHT: lambda detections: detections_anchor_coordinates(
        detections=detections, anchor=Position.TOP_RIGHT
    ),
    DetectionsProperty.BOTTOM_LEFT: lambda detections: detections_anchor_coordinates(
        detections=detections, anchor=Position.BOTTOM_LEFT
    ),
    DetectionsProperty.BOTTOM_RIGHT: lambda detections: detections_anchor_coordinates(
        detections=detections, anchor=Position.BOTTOM_RIGHT
    ),
}


def _extract_detections_property(
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
    if (
        property_name.value not in PROPERTIES_EXTRACTORS
        and property_name.value in detections.data
    ):
        return detections.data[property_name.value].tolist()
    return PROPERTIES_EXTRACTORS[property_name](detections)


def _filter_detections(
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


def _offset_detections(
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


def _shift_detections(value: Any, shift_x: int, shift_y: int, **kwargs) -> sv.Detections:
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
        return detections  # Directly return the original empty detections if empty

    centers_x = detections.get_anchors_coordinates(anchor=Position.CENTER)[:, 0]
    index = np.argmin(centers_x)
    return detections[int(index)]


def select_rightmost_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return detections

    centers_x = detections.get_anchors_coordinates(anchor=Position.CENTER)[:, 0]
    index = centers_x.argmax()
    return detections[index]


def select_first_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return deepcopy(detections)
    return detections[0]


def select_last_detection(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return deepcopy(detections)
    return detections[-1]


DETECTIONS_SELECTORS = {
    DetectionsSelectionMode.FIRST: select_first_detection,
    DetectionsSelectionMode.LAST: select_last_detection,
    DetectionsSelectionMode.LEFT_MOST: select_leftmost_detection,
    DetectionsSelectionMode.RIGHT_MOST: select_rightmost_detection,
    DetectionsSelectionMode.TOP_CONFIDENCE: select_top_confidence_detection,
}


def _select_detections(
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
    return (detections.xyxy[:, 0] + detections.xyxy[:, 2]) * 0.5


def extract_y_coordinate_of_detections_center(detections: sv.Detections) -> np.ndarray:
    return (detections.xyxy[:, 1] + detections.xyxy[:, 3]) * 0.5


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


def _sort_detections(
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


def _rename_detections(
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


def _detections_to_dictionary(
    detections: Any,
    execution_context: str,
    **kwargs,
) -> dict:
    if not isinstance(detections, sv.Detections):
        value_as_str = safe_stringify(value=detections)
        raise InvalidInputTypeError(
            public_message=f"Executing detections_to_dictionary(...) in context {execution_context}, "
            f"expected sv.Detections object as value, got {value_as_str} of type {type(detections)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        return serialise_sv_detections(detections=detections)
    except Exception as error:
        raise OperationError(
            public_message=f"While Using operation detections_to_dictionary(...) in context {execution_context} "
            f"encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        )


def _pick_detections_by_parent_class(
    detections: Any,
    parent_class: str,
    execution_context: str,
    **kwargs,
) -> sv.Detections:
    if not isinstance(detections, sv.Detections):
        value_as_str = safe_stringify(value=detections)
        raise InvalidInputTypeError(
            public_message=f"Executing pick_detections_by_parent_class(...) in context {execution_context}, "
            f"expected sv.Detections object as value, got {value_as_str} of type {type(detections)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        return _pick_detections_by_parent_class_impl(
            detections=detections, parent_class=parent_class
        )
    except Exception as error:
        raise OperationError(
            public_message=f"While Using operation pick_detections_by_parent_class(...) in context {execution_context} "
            f"encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        )


def _pick_detections_by_parent_class_impl(
    detections: sv.Detections,
    parent_class: str,
) -> sv.Detections:
    class_names = detections.data.get("class_name")
    if class_names is None or len(class_names) == 0:
        return sv.Detections.empty()
    if not isinstance(class_names, np.ndarray):
        class_names = np.array(class_names)
    parent_mask = class_names == parent_class
    parent_detections = detections[parent_mask]
    if len(parent_detections) == 0:
        return sv.Detections.empty()
    dependent_detections = detections[~parent_mask]
    dependent_detections_anchors = dependent_detections.get_anchors_coordinates(
        anchor=Position.CENTER
    )
    dependent_detections_to_keep = set()
    for detection_idx, anchor in enumerate(dependent_detections_anchors):
        for parent_detection_box in parent_detections.xyxy:
            if _is_point_within_box(point=anchor, box=parent_detection_box):
                dependent_detections_to_keep.add(detection_idx)
                continue
    detections_to_keep_list = sorted(list(dependent_detections_to_keep))
    filtered_dependent_detections = dependent_detections[detections_to_keep_list]
    return sv.Detections.merge([parent_detections, filtered_dependent_detections])


def _is_point_within_box(point: np.ndarray, box: np.ndarray) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


TensorNativeDetections = Union[Detections, InstanceDetections]
TENSOR_NATIVE_DETECTIONS_TYPES = (Detections, InstanceDetections)
# A keypoint-detection prediction kind is carried as a 2-tuple
# (KeyPoints, Optional[Detections]); the bounding-box component is operated on by
# the UQL detections ops and the KeyPoints component is sliced / shifted to match.
KeyPointPrediction = tuple
TensorNativePrediction = Union[Detections, InstanceDetections, "KeyPointPrediction"]


def _is_key_point_prediction(value: Any) -> bool:
    """True for the keypoint-detection tuple ``(KeyPoints, Optional[Detections])``."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], KeyPoints)
        and (value[1] is None or isinstance(value[1], TENSOR_NATIVE_DETECTIONS_TYPES))
    )


def _split_key_point_prediction(
    value: tuple,
    operation_name: str,
    execution_context: Optional[str] = None,
) -> tuple:
    """Return ``(key_points, detections)`` from a keypoint-detection tuple, raising
    if the bbox component is missing (the UQL ops operate on bounding boxes)."""
    key_points, detections = value
    if detections is None:
        context = "step_execution | roboflow_query_language_evaluation"
        in_context = ""
        if execution_context is not None:
            context = f"{context} | {execution_context}"
            in_context = f" in context {execution_context}"
        raise InvalidInputTypeError(
            public_message=f"Executing {operation_name}(...){in_context}, the keypoint "
            f"prediction is missing the bounding-box `inference_models.Detections` "
            f"component required by this operation.",
            context=context,
        )
    return key_points, detections


def _ensure_tensor_native_detections(
    value: Any,
    operation_name: str,
    execution_context: Optional[str] = None,
) -> None:
    if isinstance(value, TENSOR_NATIVE_DETECTIONS_TYPES) or _is_key_point_prediction(
        value
    ):
        return
    value_as_str = safe_stringify(value=value)
    context = "step_execution | roboflow_query_language_evaluation"
    in_context = ""
    if execution_context is not None:
        context = f"{context} | {execution_context}"
        in_context = f" in context {execution_context}"
    raise InvalidInputTypeError(
        public_message=f"Executing {operation_name}(...){in_context}, expected "
        f"`inference_models.Detections`, `inference_models.InstanceDetections` or a "
        f"keypoint-detection `(KeyPoints, Detections)` tuple as value, got "
        f"{value_as_str} of type {type(value)}",
        context=context,
    )


def _take_key_points(key_points: KeyPoints, indices: List[int]) -> KeyPoints:
    """Slice a ``KeyPoints`` along the instance dimension by index list, carrying
    per-instance ``key_points_metadata`` and sharing ``image_metadata`` as-is."""
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=key_points.xy.device
    )
    key_points_metadata = None
    if key_points.key_points_metadata is not None:
        key_points_metadata = [key_points.key_points_metadata[i] for i in indices]
    return KeyPoints(
        xy=key_points.xy[index_tensor],
        class_id=key_points.class_id[index_tensor],
        confidence=key_points.confidence[index_tensor],
        image_metadata=key_points.image_metadata,
        key_points_metadata=key_points_metadata,
    )


def _detections_count(detections: TensorNativeDetections) -> int:
    return int(detections.xyxy.shape[0])


def _bboxes_metadata_list(detections: TensorNativeDetections) -> List[dict]:
    if detections.bboxes_metadata is not None:
        return detections.bboxes_metadata
    return [{} for _ in range(_detections_count(detections))]


def _class_names_lookup(
    detections: TensorNativeDetections, operation_name: str
) -> Dict[int, str]:
    image_metadata = detections.image_metadata or {}
    class_names = image_metadata.get(CLASS_NAMES_KEY)
    if class_names is None:
        raise OperationError(
            public_message=f"Executing {operation_name}(...), but "
            f"`image_metadata['{CLASS_NAMES_KEY}']` is missing — the producer block "
            f"must attach the class_id → name mapping.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    return class_names


def _resolve_class_names(
    detections: TensorNativeDetections, operation_name: str
) -> List[str]:
    if _detections_count(detections) == 0:
        return []
    class_names = _class_names_lookup(detections, operation_name=operation_name)
    result = []
    for class_id_scalar in detections.class_id.tolist():
        class_id = int(class_id_scalar)
        class_name = class_names.get(class_id)
        if class_name is None:
            raise OperationError(
                public_message=f"Executing {operation_name}(...), class_id={class_id} "
                f"is missing from the class_names mapping "
                f"(keys present: {sorted(class_names.keys())}).",
                context="step_execution | roboflow_query_language_evaluation",
            )
        result.append(class_name)
    return result


def _take_mask(
    mask: Union[torch.Tensor, InstancesRLEMasks], indices: List[int]
) -> Union[torch.Tensor, InstancesRLEMasks]:
    if isinstance(mask, InstancesRLEMasks):
        return InstancesRLEMasks(
            image_size=mask.image_size,
            masks=[mask.masks[index] for index in indices],
        )
    return mask[torch.as_tensor(indices, dtype=torch.long, device=mask.device)]


def _take_detections(
    detections: TensorNativeDetections, indices: List[int]
) -> TensorNativeDetections:
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=detections.xyxy.device
    )
    bboxes_metadata = None
    if detections.bboxes_metadata is not None:
        bboxes_metadata = [detections.bboxes_metadata[index] for index in indices]
    if isinstance(detections, InstanceDetections):
        return InstanceDetections(
            xyxy=detections.xyxy[index_tensor],
            class_id=detections.class_id[index_tensor],
            confidence=detections.confidence[index_tensor],
            mask=_take_mask(detections.mask, indices),
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=detections.xyxy[index_tensor],
        class_id=detections.class_id[index_tensor],
        confidence=detections.confidence[index_tensor],
        image_metadata=detections.image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _copy_detections(detections: TensorNativeDetections) -> TensorNativeDetections:
    if isinstance(detections, InstanceDetections):
        mask = detections.mask
        if isinstance(mask, InstancesRLEMasks):
            mask = InstancesRLEMasks(
                image_size=mask.image_size, masks=list(mask.masks)
            )
        else:
            mask = mask.clone()
        return InstanceDetections(
            xyxy=detections.xyxy.clone(),
            class_id=detections.class_id.clone(),
            confidence=detections.confidence.clone(),
            mask=mask,
            image_metadata=deepcopy(detections.image_metadata),
            bboxes_metadata=deepcopy(detections.bboxes_metadata),
        )
    return Detections(
        xyxy=detections.xyxy.clone(),
        class_id=detections.class_id.clone(),
        confidence=detections.confidence.clone(),
        image_metadata=deepcopy(detections.image_metadata),
        bboxes_metadata=deepcopy(detections.bboxes_metadata),
    )


def _concatenate_detections(
    first: TensorNativeDetections, second: TensorNativeDetections
) -> TensorNativeDetections:
    bboxes_metadata = None
    if first.bboxes_metadata is not None or second.bboxes_metadata is not None:
        bboxes_metadata = _bboxes_metadata_list(first) + _bboxes_metadata_list(second)
    if isinstance(first, InstanceDetections):
        if isinstance(first.mask, InstancesRLEMasks) != isinstance(
            second.mask, InstancesRLEMasks
        ):
            raise OperationError(
                public_message="Cannot concatenate InstanceDetections with mixed mask "
                "representations (dense tensor vs RLE).",
                context="step_execution | roboflow_query_language_evaluation",
            )
        if isinstance(first.mask, InstancesRLEMasks):
            mask = InstancesRLEMasks(
                image_size=first.mask.image_size,
                masks=first.mask.masks + second.mask.masks,
            )
        else:
            mask = torch.cat([first.mask, second.mask], dim=0)
        return InstanceDetections(
            xyxy=torch.cat([first.xyxy, second.xyxy], dim=0),
            class_id=torch.cat([first.class_id, second.class_id], dim=0),
            confidence=torch.cat([first.confidence, second.confidence], dim=0),
            mask=mask,
            image_metadata=first.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=torch.cat([first.xyxy, second.xyxy], dim=0),
        class_id=torch.cat([first.class_id, second.class_id], dim=0),
        confidence=torch.cat([first.confidence, second.confidence], dim=0),
        image_metadata=first.image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _detections_anchor_coordinates_tensor_native(
    detections: TensorNativeDetections, anchor: Position
) -> List[List[int]]:
    xyxy = detections.xyxy
    if anchor is Position.CENTER:
        xs = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        ys = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    elif anchor is Position.TOP_LEFT:
        xs, ys = xyxy[:, 0], xyxy[:, 1]
    elif anchor is Position.TOP_RIGHT:
        xs, ys = xyxy[:, 2], xyxy[:, 1]
    elif anchor is Position.BOTTOM_LEFT:
        xs, ys = xyxy[:, 0], xyxy[:, 3]
    else:
        xs, ys = xyxy[:, 2], xyxy[:, 3]
    return torch.stack([xs, ys], dim=1).round().long().tolist()


PROPERTIES_EXTRACTORS_TENSOR_NATIVE = {
    DetectionsProperty.CONFIDENCE: lambda detections: detections.confidence.tolist(),
    DetectionsProperty.CLASS_NAME: lambda detections: _resolve_class_names(
        detections, operation_name="extract_detections_property"
    ),
    DetectionsProperty.X_MIN: lambda detections: detections.xyxy[:, 0].tolist(),
    DetectionsProperty.Y_MIN: lambda detections: detections.xyxy[:, 1].tolist(),
    DetectionsProperty.X_MAX: lambda detections: detections.xyxy[:, 2].tolist(),
    DetectionsProperty.Y_MAX: lambda detections: detections.xyxy[:, 3].tolist(),
    DetectionsProperty.CLASS_ID: lambda detections: [
        int(value) for value in detections.class_id.tolist()
    ],
    DetectionsProperty.SIZE: lambda detections: (
        (detections.xyxy[:, 2] - detections.xyxy[:, 0])
        * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    ).tolist(),
    DetectionsProperty.CENTER: lambda detections: (
        _detections_anchor_coordinates_tensor_native(
            detections=detections, anchor=Position.CENTER
        )
    ),
    DetectionsProperty.TOP_LEFT: lambda detections: (
        _detections_anchor_coordinates_tensor_native(
            detections=detections, anchor=Position.TOP_LEFT
        )
    ),
    DetectionsProperty.TOP_RIGHT: lambda detections: (
        _detections_anchor_coordinates_tensor_native(
            detections=detections, anchor=Position.TOP_RIGHT
        )
    ),
    DetectionsProperty.BOTTOM_LEFT: lambda detections: (
        _detections_anchor_coordinates_tensor_native(
            detections=detections, anchor=Position.BOTTOM_LEFT
        )
    ),
    DetectionsProperty.BOTTOM_RIGHT: lambda detections: (
        _detections_anchor_coordinates_tensor_native(
            detections=detections, anchor=Position.BOTTOM_RIGHT
        )
    ),
}


def _extract_detections_property_tensor_native(
    detections: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> List[Any]:
    _ensure_tensor_native_detections(
        detections,
        operation_name="extract_detections_property",
        execution_context=execution_context,
    )
    if _is_key_point_prediction(detections):
        # Property extraction reads the bounding-box component (mirrors the numpy
        # path where keypoints ride alongside boxes in the same sv.Detections).
        _, detections = _split_key_point_prediction(
            detections,
            operation_name="extract_detections_property",
            execution_context=execution_context,
        )
    if property_name not in PROPERTIES_EXTRACTORS_TENSOR_NATIVE:
        bboxes_metadata = _bboxes_metadata_list(detections)
        if any(property_name.value in data for data in bboxes_metadata):
            return [data.get(property_name.value) for data in bboxes_metadata]
        raise OperationError(
            public_message=f"Executing extract_detections_property(...) in context "
            f"{execution_context}, property `{property_name.value}` is neither "
            f"natively supported nor present in `bboxes_metadata` of the detections.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return PROPERTIES_EXTRACTORS_TENSOR_NATIVE[property_name](detections)


def _filter_detections_indices_tensor_native(
    detections: TensorNativeDetections,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> List[int]:
    local_parameters = copy(global_parameters)
    indices_to_keep = []
    for index, detection in enumerate(detections):
        local_parameters[DEFAULT_OPERAND_NAME] = detection
        if filtering_fun(local_parameters):
            indices_to_keep.append(index)
    return indices_to_keep


def _filter_detections_tensor_native(
    detections: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(detections, operation_name="filter_detections")
    if _is_key_point_prediction(detections):
        key_points, bboxes = _split_key_point_prediction(
            detections, operation_name="filter_detections"
        )
        indices = _filter_detections_indices_tensor_native(
            bboxes, filtering_fun=filtering_fun, global_parameters=global_parameters
        )
        return _take_key_points(key_points, indices), _take_detections(bboxes, indices)
    indices = _filter_detections_indices_tensor_native(
        detections, filtering_fun=filtering_fun, global_parameters=global_parameters
    )
    return _take_detections(detections, indices)


def _offset_detections_impl_tensor_native(
    detections: TensorNativeDetections, offset_x: int, offset_y: int
) -> TensorNativeDetections:
    detections_copy = _copy_detections(detections)
    detections_copy.xyxy = detections_copy.xyxy + torch.tensor(
        [-offset_x / 2, -offset_y / 2, offset_x / 2, offset_y / 2],
        dtype=detections_copy.xyxy.dtype,
        device=detections_copy.xyxy.device,
    )
    return detections_copy


def _offset_detections_tensor_native(
    value: Any, offset_x: int, offset_y: int, **kwargs
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(value, operation_name="offset_detections")
    # Mirrors the numpy path: only bbox `xyxy` is offset; the keypoint `xy`
    # coordinates (carried in sv `.data` in numpy mode) are left untouched.
    if _is_key_point_prediction(value):
        key_points, bboxes = _split_key_point_prediction(
            value, operation_name="offset_detections"
        )
        return key_points, _offset_detections_impl_tensor_native(
            bboxes, offset_x=offset_x, offset_y=offset_y
        )
    return _offset_detections_impl_tensor_native(
        value, offset_x=offset_x, offset_y=offset_y
    )


def _shift_detections_impl_tensor_native(
    detections: TensorNativeDetections, shift_x: int, shift_y: int
) -> TensorNativeDetections:
    detections_copy = _copy_detections(detections)
    detections_copy.xyxy = detections_copy.xyxy + torch.tensor(
        [shift_x, shift_y, shift_x, shift_y],
        dtype=detections_copy.xyxy.dtype,
        device=detections_copy.xyxy.device,
    )
    return detections_copy


def _shift_detections_tensor_native(
    value: Any, shift_x: int, shift_y: int, **kwargs
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(value, operation_name="shift_detections")
    # Mirrors the numpy path: only bbox `xyxy` is shifted; the keypoint `xy`
    # coordinates (carried in sv `.data` in numpy mode) are left untouched.
    if _is_key_point_prediction(value):
        key_points, bboxes = _split_key_point_prediction(
            value, operation_name="shift_detections"
        )
        return key_points, _shift_detections_impl_tensor_native(
            bboxes, shift_x=shift_x, shift_y=shift_y
        )
    return _shift_detections_impl_tensor_native(
        value, shift_x=shift_x, shift_y=shift_y
    )


def _select_top_confidence_index_tensor_native(
    detections: TensorNativeDetections,
) -> Optional[int]:
    if _detections_count(detections) == 0:
        return None
    return int(torch.argmax(detections.confidence))


def _select_leftmost_index_tensor_native(
    detections: TensorNativeDetections,
) -> Optional[int]:
    if _detections_count(detections) == 0:
        return None
    centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) * 0.5
    return int(torch.argmin(centers_x))


def _select_rightmost_index_tensor_native(
    detections: TensorNativeDetections,
) -> Optional[int]:
    if _detections_count(detections) == 0:
        return None
    centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) * 0.5
    return int(torch.argmax(centers_x))


def _select_first_index_tensor_native(
    detections: TensorNativeDetections,
) -> Optional[int]:
    if _detections_count(detections) == 0:
        return None
    return 0


def _select_last_index_tensor_native(
    detections: TensorNativeDetections,
) -> Optional[int]:
    count = _detections_count(detections)
    if count == 0:
        return None
    return count - 1


def _select_top_confidence_detection_tensor_native(
    detections: TensorNativeDetections,
) -> TensorNativeDetections:
    index = _select_top_confidence_index_tensor_native(detections)
    if index is None:
        return _copy_detections(detections)
    return _take_detections(detections, [index])


def _select_leftmost_detection_tensor_native(
    detections: TensorNativeDetections,
) -> TensorNativeDetections:
    index = _select_leftmost_index_tensor_native(detections)
    if index is None:
        return detections
    return _take_detections(detections, [index])


def _select_rightmost_detection_tensor_native(
    detections: TensorNativeDetections,
) -> TensorNativeDetections:
    index = _select_rightmost_index_tensor_native(detections)
    if index is None:
        return detections
    return _take_detections(detections, [index])


def _select_first_detection_tensor_native(
    detections: TensorNativeDetections,
) -> TensorNativeDetections:
    index = _select_first_index_tensor_native(detections)
    if index is None:
        return _copy_detections(detections)
    return _take_detections(detections, [index])


def _select_last_detection_tensor_native(
    detections: TensorNativeDetections,
) -> TensorNativeDetections:
    index = _select_last_index_tensor_native(detections)
    if index is None:
        return _copy_detections(detections)
    return _take_detections(detections, [index])


DETECTIONS_SELECTORS_TENSOR_NATIVE = {
    DetectionsSelectionMode.FIRST: _select_first_detection_tensor_native,
    DetectionsSelectionMode.LAST: _select_last_detection_tensor_native,
    DetectionsSelectionMode.LEFT_MOST: _select_leftmost_detection_tensor_native,
    DetectionsSelectionMode.RIGHT_MOST: _select_rightmost_detection_tensor_native,
    DetectionsSelectionMode.TOP_CONFIDENCE: _select_top_confidence_detection_tensor_native,
}

DETECTIONS_SELECTOR_INDICES_TENSOR_NATIVE = {
    DetectionsSelectionMode.FIRST: _select_first_index_tensor_native,
    DetectionsSelectionMode.LAST: _select_last_index_tensor_native,
    DetectionsSelectionMode.LEFT_MOST: _select_leftmost_index_tensor_native,
    DetectionsSelectionMode.RIGHT_MOST: _select_rightmost_index_tensor_native,
    DetectionsSelectionMode.TOP_CONFIDENCE: _select_top_confidence_index_tensor_native,
}


def _select_detections_tensor_native(
    value: Any, mode: DetectionsSelectionMode, **kwargs
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(value, operation_name="select_detections")
    if mode not in DETECTIONS_SELECTORS_TENSOR_NATIVE:
        raise InvalidInputTypeError(
            public_message=f"Executing select_detections(...), expected mode to be one "
            f"of {list(DETECTIONS_SELECTORS_TENSOR_NATIVE.keys())}, got {mode}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if _is_key_point_prediction(value):
        key_points, bboxes = _split_key_point_prediction(
            value, operation_name="select_detections"
        )
        index = DETECTIONS_SELECTOR_INDICES_TENSOR_NATIVE[mode](bboxes)
        indices = [] if index is None else [index]
        return _take_key_points(key_points, indices), _take_detections(bboxes, indices)
    return DETECTIONS_SELECTORS_TENSOR_NATIVE[mode](value)


SORT_PROPERTIES_EXTRACT_TENSOR_NATIVE = {
    DetectionsSortProperties.CONFIDENCE: lambda detections: detections.confidence,
    DetectionsSortProperties.X_MIN: lambda detections: detections.xyxy[:, 0],
    DetectionsSortProperties.X_MAX: lambda detections: detections.xyxy[:, 2],
    DetectionsSortProperties.Y_MIN: lambda detections: detections.xyxy[:, 1],
    DetectionsSortProperties.Y_MAX: lambda detections: detections.xyxy[:, 3],
    DetectionsSortProperties.SIZE: lambda detections: (
        (detections.xyxy[:, 2] - detections.xyxy[:, 0])
        * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    ),
    DetectionsSortProperties.CENTER_X: lambda detections: (
        (detections.xyxy[:, 0] + detections.xyxy[:, 2]) * 0.5
    ),
    DetectionsSortProperties.CENTER_Y: lambda detections: (
        (detections.xyxy[:, 1] + detections.xyxy[:, 3]) * 0.5
    ),
}


def _sort_detections_indices_tensor_native(
    detections: TensorNativeDetections,
    mode: DetectionsSortProperties,
    ascending: bool,
) -> List[int]:
    extracted_property = SORT_PROPERTIES_EXTRACT_TENSOR_NATIVE[mode](detections)
    sorted_indices = torch.argsort(extracted_property, descending=not ascending)
    return [int(index) for index in sorted_indices.tolist()]


def _sort_detections_tensor_native(
    value: Any, mode: DetectionsSortProperties, ascending: bool, **kwargs
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(value, operation_name="sort_detections")
    if mode not in SORT_PROPERTIES_EXTRACT_TENSOR_NATIVE:
        raise InvalidInputTypeError(
            public_message=f"Executing sort_detections(...), expected mode to be one of "
            f"{list(SORT_PROPERTIES_EXTRACT_TENSOR_NATIVE.keys())}, got {mode}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if _is_key_point_prediction(value):
        key_points, bboxes = _split_key_point_prediction(
            value, operation_name="sort_detections"
        )
        if _detections_count(bboxes) == 0:
            return value
        indices = _sort_detections_indices_tensor_native(
            bboxes, mode=mode, ascending=ascending
        )
        return _take_key_points(key_points, indices), _take_detections(bboxes, indices)
    if _detections_count(value) == 0:
        return value
    indices = _sort_detections_indices_tensor_native(
        value, mode=mode, ascending=ascending
    )
    return _take_detections(value, indices)


def _rename_detections_tensor_native(
    detections: Any,
    class_map: Union[Dict[str, str], str],
    strict: Union[bool, str],
    new_classes_id_offset: int,
    global_parameters: Dict[str, Any],
    **kwargs,
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(detections, operation_name="rename_detections")
    # Rename only affects the bounding-box `Detections` class ids / names (mirrors
    # the numpy path, which touches `data["class_name"]` / `class_id`). The
    # keypoint component is preserved unchanged in the re-wrapped tuple.
    if _is_key_point_prediction(detections):
        key_points, bboxes = _split_key_point_prediction(
            detections, operation_name="rename_detections"
        )
        renamed_bboxes = _rename_detections_tensor_native(
            detections=bboxes,
            class_map=class_map,
            strict=strict,
            new_classes_id_offset=new_classes_id_offset,
            global_parameters=global_parameters,
            **kwargs,
        )
        return key_points, renamed_bboxes
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
    original_class_names = _resolve_class_names(
        detections, operation_name="rename_detections"
    )
    original_class_ids = [int(value) for value in detections.class_id.tolist()]
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
    new_class_ids = [
        new_class_mapping[class_map.get(class_name, class_name)]
        for class_name in original_class_names
    ]
    detections_copy = _copy_detections(detections)
    detections_copy.class_id = torch.tensor(
        new_class_ids,
        dtype=detections.class_id.dtype,
        device=detections.class_id.device,
    )
    allowed_names = {class_map.get(name, name) for name in original_class_names}
    allowed_names.update(class_map.values())
    image_metadata = detections_copy.image_metadata or {}
    image_metadata[CLASS_NAMES_KEY] = {
        class_id: class_name
        for class_name, class_id in new_class_mapping.items()
        if class_name in allowed_names
    }
    detections_copy.image_metadata = image_metadata
    return detections_copy


def _detections_to_dictionary_tensor_native(
    detections: Any,
    execution_context: str,
    **kwargs,
) -> dict:
    _ensure_tensor_native_detections(
        detections,
        operation_name="detections_to_dictionary",
        execution_context=execution_context,
    )
    if _is_key_point_prediction(detections):
        # The op-level numpy path serialises the (box-carrying) sv.Detections; here
        # we serialise the bounding-box component (keypoint details are not part of
        # this dict-serialisation, matching the established kind serialisation).
        _, detections = _split_key_point_prediction(
            detections,
            operation_name="detections_to_dictionary",
            execution_context=execution_context,
        )
    try:
        return serialise_tensor_native_detections(detections=detections)
    except Exception as error:
        raise OperationError(
            public_message=f"While Using operation detections_to_dictionary(...) in context {execution_context} "
            f"encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        )


def _pick_detections_by_parent_class_tensor_native(
    detections: Any,
    parent_class: str,
    execution_context: str,
    **kwargs,
) -> TensorNativePrediction:
    _ensure_tensor_native_detections(
        detections,
        operation_name="pick_detections_by_parent_class",
        execution_context=execution_context,
    )
    try:
        if _is_key_point_prediction(detections):
            key_points, bboxes = _split_key_point_prediction(
                detections,
                operation_name="pick_detections_by_parent_class",
                execution_context=execution_context,
            )
            indices = _pick_detections_by_parent_class_indices_tensor_native(
                detections=bboxes, parent_class=parent_class
            )
            return (
                _take_key_points(key_points, indices),
                _take_detections(bboxes, indices),
            )
        return _pick_detections_by_parent_class_tensor_native_impl(
            detections=detections, parent_class=parent_class
        )
    except Exception as error:
        raise OperationError(
            public_message=f"While Using operation pick_detections_by_parent_class(...) in context {execution_context} "
            f"encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        )


def _pick_detections_by_parent_class_indices_tensor_native(
    detections: TensorNativeDetections,
    parent_class: str,
) -> List[int]:
    """Absolute index order produced by ``pick_detections_by_parent_class``: every
    parent detection first (in original order), then the dependent detections whose
    center lies inside any parent (mirrors the ``_concatenate_detections`` ordering
    in the impl). Returned so the keypoint component can be sliced consistently."""
    if _detections_count(detections) == 0:
        return []
    class_names = _resolve_class_names(
        detections, operation_name="pick_detections_by_parent_class"
    )
    parent_indices = [
        index for index, name in enumerate(class_names) if name == parent_class
    ]
    if not parent_indices:
        return []
    dependent_indices = [
        index for index, name in enumerate(class_names) if name != parent_class
    ]
    dependent_detections = _take_detections(detections, dependent_indices)
    parent_detections = _take_detections(detections, parent_indices)
    centers_x = (
        (dependent_detections.xyxy[:, 0] + dependent_detections.xyxy[:, 2]) * 0.5
    ).unsqueeze(1)
    centers_y = (
        (dependent_detections.xyxy[:, 1] + dependent_detections.xyxy[:, 3]) * 0.5
    ).unsqueeze(1)
    parents_x1 = parent_detections.xyxy[:, 0].unsqueeze(0)
    parents_y1 = parent_detections.xyxy[:, 1].unsqueeze(0)
    parents_x2 = parent_detections.xyxy[:, 2].unsqueeze(0)
    parents_y2 = parent_detections.xyxy[:, 3].unsqueeze(0)
    inside_any_parent = (
        (centers_x >= parents_x1)
        & (centers_x <= parents_x2)
        & (centers_y >= parents_y1)
        & (centers_y <= parents_y2)
    ).any(dim=1)
    kept_dependent_indices = [
        dependent_indices[position]
        for position, keep in enumerate(inside_any_parent.tolist())
        if keep
    ]
    return parent_indices + kept_dependent_indices


def _pick_detections_by_parent_class_tensor_native_impl(
    detections: TensorNativeDetections,
    parent_class: str,
) -> TensorNativeDetections:
    if _detections_count(detections) == 0:
        return _take_detections(detections, [])
    class_names = _resolve_class_names(
        detections, operation_name="pick_detections_by_parent_class"
    )
    parent_indices = [
        index for index, name in enumerate(class_names) if name == parent_class
    ]
    if not parent_indices:
        return _take_detections(detections, [])
    dependent_indices = [
        index for index, name in enumerate(class_names) if name != parent_class
    ]
    parent_detections = _take_detections(detections, parent_indices)
    dependent_detections = _take_detections(detections, dependent_indices)
    centers_x = (
        (dependent_detections.xyxy[:, 0] + dependent_detections.xyxy[:, 2]) * 0.5
    ).unsqueeze(1)
    centers_y = (
        (dependent_detections.xyxy[:, 1] + dependent_detections.xyxy[:, 3]) * 0.5
    ).unsqueeze(1)
    parents_x1 = parent_detections.xyxy[:, 0].unsqueeze(0)
    parents_y1 = parent_detections.xyxy[:, 1].unsqueeze(0)
    parents_x2 = parent_detections.xyxy[:, 2].unsqueeze(0)
    parents_y2 = parent_detections.xyxy[:, 3].unsqueeze(0)
    inside_any_parent = (
        (centers_x >= parents_x1)
        & (centers_x <= parents_x2)
        & (centers_y >= parents_y1)
        & (centers_y <= parents_y2)
    ).any(dim=1)
    dependent_detections_to_keep = [
        index for index, keep in enumerate(inside_any_parent.tolist()) if keep
    ]
    filtered_dependent_detections = _take_detections(
        dependent_detections, dependent_detections_to_keep
    )
    return _concatenate_detections(parent_detections, filtered_dependent_detections)


def extract_detections_property(
    detections: Any,
    property_name: DetectionsProperty,
    execution_context: str,
    **kwargs,
) -> List[Any]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _extract_detections_property_tensor_native(
            detections=detections,
            property_name=property_name,
            execution_context=execution_context,
            **kwargs,
        )
    return _extract_detections_property(
        detections=detections,
        property_name=property_name,
        execution_context=execution_context,
        **kwargs,
    )


def filter_detections(
    detections: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _filter_detections_tensor_native(
            detections=detections,
            filtering_fun=filtering_fun,
            global_parameters=global_parameters,
        )
    return _filter_detections(
        detections=detections,
        filtering_fun=filtering_fun,
        global_parameters=global_parameters,
    )


def offset_detections(
    value: Any, offset_x: int, offset_y: int, **kwargs
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _offset_detections_tensor_native(
            value=value, offset_x=offset_x, offset_y=offset_y, **kwargs
        )
    return _offset_detections(value=value, offset_x=offset_x, offset_y=offset_y, **kwargs)


def shift_detections(
    value: Any, shift_x: int, shift_y: int, **kwargs
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _shift_detections_tensor_native(
            value=value, shift_x=shift_x, shift_y=shift_y, **kwargs
        )
    return _shift_detections(value=value, shift_x=shift_x, shift_y=shift_y, **kwargs)


def select_detections(
    value: Any, mode: DetectionsSelectionMode, **kwargs
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _select_detections_tensor_native(value=value, mode=mode, **kwargs)
    return _select_detections(value=value, mode=mode, **kwargs)


def sort_detections(
    value: Any, mode: DetectionsSortProperties, ascending: bool, **kwargs
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _sort_detections_tensor_native(
            value=value, mode=mode, ascending=ascending, **kwargs
        )
    return _sort_detections(value=value, mode=mode, ascending=ascending, **kwargs)


def rename_detections(
    detections: Any,
    class_map: Union[Dict[str, str], str],
    strict: Union[bool, str],
    new_classes_id_offset: int,
    global_parameters: Dict[str, Any],
    **kwargs,
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _rename_detections_tensor_native(
            detections=detections,
            class_map=class_map,
            strict=strict,
            new_classes_id_offset=new_classes_id_offset,
            global_parameters=global_parameters,
            **kwargs,
        )
    return _rename_detections(
        detections=detections,
        class_map=class_map,
        strict=strict,
        new_classes_id_offset=new_classes_id_offset,
        global_parameters=global_parameters,
        **kwargs,
    )


def detections_to_dictionary(
    detections: Any,
    execution_context: str,
    **kwargs,
) -> dict:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _detections_to_dictionary_tensor_native(
            detections=detections, execution_context=execution_context, **kwargs
        )
    return _detections_to_dictionary(
        detections=detections, execution_context=execution_context, **kwargs
    )


def pick_detections_by_parent_class(
    detections: Any,
    parent_class: str,
    execution_context: str,
    **kwargs,
) -> Union[sv.Detections, TensorNativeDetections]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return _pick_detections_by_parent_class_tensor_native(
            detections=detections,
            parent_class=parent_class,
            execution_context=execution_context,
            **kwargs,
        )
    return _pick_detections_by_parent_class(
        detections=detections,
        parent_class=parent_class,
        execution_context=execution_context,
        **kwargs,
    )
