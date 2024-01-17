import itertools
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import ImageType, load_image
from inference.enterprise.deployments.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.enterprise.deployments.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.deployments.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.deployments.complier.utils import (
    construct_selector_pointing_step_output,
    construct_step_selector,
)
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
    if not issubclass(type(image), list):
        image = [image]
        detections = [detections]
    decoded_images = [load_image(e) for e in image]
    decoded_images = [
        i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
    ]
    origin_image_shape = extract_origin_size_from_images(
        input_images=image,
        decoded_images=decoded_images,
    )
    crops = list(
        itertools.chain.from_iterable(
            crop_image(image=i, detections=d, origin_size=o)
            for i, d, o in zip(decoded_images, detections, origin_image_shape)
        )
    )
    parent_ids = [c[PARENT_ID_KEY] for c in crops]
    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "crops": crops,
        PARENT_ID_KEY: parent_ids,
    }
    return None, outputs_lookup


def crop_image(
    image: np.ndarray,
    detections: List[dict],
    origin_size: dict,
) -> List[Dict[str, Union[str, np.ndarray]]]:
    crops = []
    for detection in detections:
        x_min = round(detection["x"] - detection[WIDTH_KEY] / 2)
        y_min = round(detection["y"] - detection[HEIGHT_KEY] / 2)
        x_max = round(x_min + detection[WIDTH_KEY])
        y_max = round(y_min + detection[HEIGHT_KEY])
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
                IMAGE_VALUE_KEY: cropped_image,
                PARENT_ID_KEY: detection[DETECTION_ID_KEY],
                ORIGIN_COORDINATES_KEY: {
                    CENTER_X_KEY: detection["x"],
                    CENTER_Y_KEY: detection["y"],
                    ORIGIN_SIZE_KEY: origin_size,
                },
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
    images_meta_selector = construct_selector_pointing_step_output(
        selector=step.predictions,
        new_output="image",
    )
    images_meta = resolve_parameter(
        selector_or_value=images_meta_selector,
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
            result_parent_id.append([p[PARENT_ID_KEY] for p in filtered_predictions])
        elif filter_callable(prediction):
            result_detections.append(deepcopy(prediction))
            result_parent_id.append(prediction[PARENT_ID_KEY])
    step_selector = construct_step_selector(step_name=step.name)
    if nested:
        outputs_lookup[step_selector] = [
            {"predictions": d, PARENT_ID_KEY: p, "image": i}
            for d, p, i in zip(result_detections, result_parent_id, images_meta)
        ]
    else:
        outputs_lookup[step_selector] = {
            "predictions": result_detections,
            PARENT_ID_KEY: result_parent_id,
            "image": images_meta,
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
    images_meta_selector = construct_selector_pointing_step_output(
        selector=step.predictions,
        new_output="image",
    )
    images_meta = resolve_parameter(
        selector_or_value=images_meta_selector,
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
            result_parent_id.append([d[PARENT_ID_KEY] for d in offset_detections])
        else:
            result_detections.append(
                offset_detection(
                    detection=detection, offset_x=offset_x, offset_y=offset_y
                )
            )
            result_parent_id.append(detection[PARENT_ID_KEY])
    step_selector = construct_step_selector(step_name=step.name)
    if nested:
        outputs_lookup[step_selector] = [
            {"predictions": d, PARENT_ID_KEY: p, "image": i}
            for d, p, i in zip(result_detections, result_parent_id, images_meta)
        ]
    else:
        outputs_lookup[step_selector] = {
            "predictions": result_detections,
            PARENT_ID_KEY: result_parent_id,
            "image": images_meta,
        }
    return None, outputs_lookup


def offset_detection(
    detection: Dict[str, Any], offset_x: int, offset_y: int
) -> Dict[str, Any]:
    detection_copy = deepcopy(detection)
    detection_copy[WIDTH_KEY] += round(offset_x)
    detection_copy[HEIGHT_KEY] += round(offset_y)
    detection_copy[PARENT_ID_KEY] = detection_copy[DETECTION_ID_KEY]
    detection_copy[DETECTION_ID_KEY] = str(uuid4())
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

    if not issubclass(type(image), list):
        image = [image]
    decoded_images = [load_image(e) for e in image]
    decoded_images = [
        i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
    ]
    origin_image_shape = extract_origin_size_from_images(
        input_images=image,
        decoded_images=decoded_images,
    )
    crops = [
        take_static_crop(
            image=i,
            crop=step,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            origin_size=size,
        )
        for i, size in zip(decoded_images, origin_image_shape)
    ]
    parent_ids = [c[PARENT_ID_KEY] for c in crops]
    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "crops": crops,
        PARENT_ID_KEY: parent_ids,
    }
    return None, outputs_lookup


def extract_origin_size_from_images(
    input_images: List[Union[dict, np.ndarray]],
    decoded_images: List[np.ndarray],
) -> List[Dict[str, int]]:
    result = []
    for input_image, decoded_image in zip(input_images, decoded_images):
        if (
            issubclass(type(input_image), dict)
            and ORIGIN_COORDINATES_KEY in input_image
        ):
            result.append(input_image[ORIGIN_COORDINATES_KEY][ORIGIN_SIZE_KEY])
        else:
            result.append(
                {HEIGHT_KEY: decoded_image.shape[0], WIDTH_KEY: decoded_image.shape[1]}
            )
    return result


def take_static_crop(
    image: np.ndarray,
    crop: Union[AbsoluteStaticCrop, RelativeStaticCrop],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    origin_size: dict,
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
        IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
        IMAGE_VALUE_KEY: cropped_image,
        PARENT_ID_KEY: f"$steps.{crop.name}",
        ORIGIN_COORDINATES_KEY: {
            CENTER_X_KEY: x_center,
            CENTER_Y_KEY: y_center,
            ORIGIN_SIZE_KEY: origin_size,
        },
    }
