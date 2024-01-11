import itertools
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import ImageType, load_image
from inference.enterprise.deployments.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.deployments.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.deployments.complier.utils import construct_step_selector
from inference.enterprise.deployments.entities.steps import (
    AbsoluteStaticCrop,
    BinaryOperator,
    CompoundDetectionFilterDefinition,
    Condition,
    Crop,
    DetectionFilter,
    DetectionFilterDefinition,
    DetectionOffset,
    Operator,
    RelativeStaticCrop,
)
from inference.enterprise.deployments.errors import ExecutionGraphError

OPERATORS = {
    Operator.EQUAL: lambda a, b: a == b,
    Operator.NOT_EQUAL: lambda a, b: a != b,
    Operator.LOWER_THAN: lambda a, b: a < b,
    Operator.GREATER_THAN: lambda a, b: a > b,
    Operator.LOWER_OR_EQUAL_THAN: lambda a, b: a <= b,
    Operator.GREATER_OR_EQUAL_THAN: lambda a, b: a >= b,
    Operator.IN: lambda a, b: a in b,
}

BINARY_OPERATORS = {
    BinaryOperator.AND: lambda a, b: a and b,
    BinaryOperator.OR: lambda a, b: a or b,
}


async def run_crop_step(
    step: Crop,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    detections = resolve_parameter(
        selector_or_value=step.detections,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if issubclass(type(image), list):
        decoded_image = [load_image(e) for e in image]
        decoded_image = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_image
        ]
        crops = list(
            itertools.chain.from_iterable(
                crop_image(image=i, detections=d)
                for i, d in zip(decoded_image, detections)
            )
        )
    else:
        decoded_image, is_bgr = load_image(image)
        if not is_bgr:
            decoded_image = decoded_image[:, :, ::-1]
        crops = crop_image(image=decoded_image, detections=detections)
    parent_ids = [c["parent_id"] for c in crops]
    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "crops": crops,
        "parent_id": parent_ids,
    }
    return None, outputs_lookup


def crop_image(
    image: np.ndarray,
    detections: List[dict],
) -> List[Dict[str, Union[str, np.ndarray]]]:
    crops = []
    for detection in detections:
        x_min = round(detection["x"] - detection["width"] / 2)
        y_min = round(detection["y"] - detection["height"] / 2)
        x_max = round(x_min + detection["width"])
        y_max = round(y_min + detection["height"])
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                "type": ImageType.NUMPY_OBJECT.value,
                "value": cropped_image,
                "parent_id": detection["detection_id"],
            }
        )
    return crops


async def run_condition_step(
    step: Condition,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    left_value = resolve_parameter(
        selector_or_value=step.left,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    right_value = resolve_parameter(
        selector_or_value=step.right,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    evaluation_result = OPERATORS[step.operator](left_value, right_value)
    next_step = step.step_if_true if evaluation_result else step.step_if_false
    return next_step, outputs_lookup


async def run_detection_filter(
    step: DetectionFilter,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    predictions = resolve_parameter(
        selector_or_value=step.predictions,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    filter_callable = build_filter_callable(definition=step.filter_definition)
    result_detections, result_parent_id = [], []
    nested = False
    for prediction in predictions:
        if issubclass(type(prediction), list):
            nested = True  # assuming that we either have all nested or none
            filtered_predictions = [
                deepcopy(p) for p in prediction if filter_callable(p)
            ]
            result_detections.append(filtered_predictions)
            result_parent_id.append([p["parent_id"] for p in filtered_predictions])
        elif filter_callable(prediction):
            result_detections.append(deepcopy(prediction))
            result_parent_id.append(prediction["parent_id"])
    step_selector = construct_step_selector(step_name=step.name)
    if nested:
        outputs_lookup[step_selector] = [
            {"predictions": d, "parent_id": p}
            for d, p in zip(result_detections, result_parent_id)
        ]
    else:
        outputs_lookup[step_selector] = {
            "predictions": result_detections,
            "parent_id": result_parent_id,
        }
    return None, outputs_lookup


def build_filter_callable(
    definition: Union[DetectionFilterDefinition, CompoundDetectionFilterDefinition],
) -> Callable[[dict], bool]:
    if definition.type == "CompoundDetectionFilterDefinition":
        left_callable = build_filter_callable(definition=definition.left)
        right_callable = build_filter_callable(definition=definition.right)
        binary_operator = BINARY_OPERATORS[definition.operator]
        return lambda e: binary_operator(left_callable(e), right_callable(e))
    if definition.type == "DetectionFilterDefinition":
        operator = OPERATORS[definition.operator]
        return lambda e: operator(e[definition.field_name], definition.reference_value)
    raise ExecutionGraphError(
        f"Detected filter definition of type {definition.type} which is unknown"
    )


async def run_detection_offset_step(
    step: DetectionOffset,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    detections = resolve_parameter(
        selector_or_value=step.predictions,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    offset_x = resolve_parameter(
        selector_or_value=step.offset_x,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    offset_y = resolve_parameter(
        selector_or_value=step.offset_y,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    result_detections, result_parent_id = [], []
    nested = False
    for detection in detections:
        if issubclass(type(detection), list):
            nested = True  # assuming that we either have all nested or none
            offset_detections = [
                offset_detection(detection=d, offset_x=offset_x, offset_y=offset_y)
                for d in detection
            ]
            result_detections.append(offset_detections)
            result_parent_id.append([d["parent_id"] for d in offset_detections])
        else:
            result_detections.append(
                offset_detection(
                    detection=detection, offset_x=offset_x, offset_y=offset_y
                )
            )
            result_parent_id.append(detection["parent_id"])
    step_selector = construct_step_selector(step_name=step.name)
    if nested:
        outputs_lookup[step_selector] = [
            {"predictions": d, "parent_id": p}
            for d, p in zip(result_detections, result_parent_id)
        ]
    else:
        outputs_lookup[step_selector] = {
            "predictions": result_detections,
            "parent_id": result_parent_id,
        }
    return None, outputs_lookup


def offset_detection(
    detection: Dict[str, Any], offset_x: int, offset_y: int
) -> Dict[str, Any]:
    detection_copy = deepcopy(detection)
    detection_copy["width"] += round(offset_x)
    detection_copy["height"] += round(offset_y)
    return detection_copy


async def run_static_crop_step(
    step: Union[AbsoluteStaticCrop, RelativeStaticCrop],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )

    if issubclass(type(image), list):
        decoded_image = [load_image(e) for e in image]
        decoded_image = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_image
        ]
        crops = [
            take_static_crop(
                image=i,
                crop=step,
                runtime_parameters=runtime_parameters,
                outputs_lookup=outputs_lookup,
            )
            for i in decoded_image
        ]
    else:
        decoded_image, is_bgr = load_image(image)
        if not is_bgr:
            decoded_image = decoded_image[:, :, ::-1]
        crops = take_static_crop(
            image=decoded_image,
            crop=step,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
        )
    parent_ids = [c["parent_id"] for c in crops]
    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "crops": crops,
        "parent_id": parent_ids,
    }
    return None, outputs_lookup


def take_static_crop(
    image: np.ndarray,
    crop: Union[AbsoluteStaticCrop, RelativeStaticCrop],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> Dict[str, Union[str, np.ndarray]]:
    resolve_parameter_closure = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    x_center = resolve_parameter_closure(crop.x_center)
    y_center = resolve_parameter_closure(crop.y_center)
    width = resolve_parameter_closure(crop.width)
    height = resolve_parameter_closure(crop.height)
    if crop.type == "RelativeStaticCrop":
        x_center = round(image.shape[1] * x_center)
        y_center = round(image.shape[0] * y_center)
        width = round(image.shape[1] * width)
        height = round(image.shape[0] * height)
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return {
        "type": ImageType.NUMPY_OBJECT.value,
        "value": cropped_image,
        "parent_id": f"$steps.{crop.name}",
    }
