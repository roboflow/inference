import itertools
import statistics
from collections import Counter
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
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
    DetectionsConsensus,
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
        x_min, y_min, x_max, y_max = detection_to_xyxy(detection=detection)
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


async def run_detections_consensus_step(
    step: DetectionsConsensus,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    resolve_parameter_closure = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    all_predictions = [resolve_parameter_closure(p) for p in step.predictions]
    if len(all_predictions) < 2:
        raise ExecutionGraphError(
            f"Consensus step requires at least two sources of predictions."
        )
    batch_sizes = get_predictions_batch_sizes(all_predictions=all_predictions)
    if not all_batch_sizes_equal(batch_sizes=batch_sizes):
        raise ExecutionGraphError(
            f"Detected missmatch of input dimensions in step: {step.name}"
        )
    images_meta_selector = construct_selector_pointing_step_output(
        selector=step.predictions[0],
        new_output="image",
    )
    images_meta = resolve_parameter_closure(images_meta_selector)
    batch_size = batch_sizes[0]
    if batch_size == 1:
        all_predictions = [[e] for e in all_predictions]
        images_meta = [images_meta]
    results = []
    for batch_index in range(len(all_predictions)):
        batch_predictions = [e[batch_index] for e in all_predictions]
        parent_id, consensus_detections, consensus_reached = resolve_batch_consensus(
            predictions=batch_predictions,
            required_votes=step.required_votes,
            class_aware=step.class_aware,
            iou_threshold=step.iou_threshold,
            confidence=step.confidence,
            classes_to_consider=step.classes_to_consider,
            required_objects=step.required_objects,
        )
        results.append(
            {
                "predictions": consensus_detections,
                "parent_id": parent_id,
                "consensus": consensus_reached,
                "image": images_meta[batch_index],
            }
        )
    if batch_size == 1:
        results = results[0]
    outputs_lookup[construct_step_selector(step_name=step.name)] = results
    return None, outputs_lookup


def get_predictions_batch_sizes(
    all_predictions: List[Union[List[dict], List[List[dict]]]]
) -> List[int]:
    return [get_batch_size(predictions=predictions) for predictions in all_predictions]


def get_batch_size(predictions: Union[List[dict], List[List[dict]]]) -> int:
    if len(predictions) == 0 or issubclass(type(predictions[0]), dict):
        return 1
    return len(predictions)


def all_batch_sizes_equal(batch_sizes: List[int]) -> bool:
    if len(batch_sizes) == 0:
        return True
    reference = batch_sizes[0]
    return all(e == reference for e in batch_sizes)


def resolve_batch_consensus(
    predictions: List[List[dict]],
    required_votes: int,
    class_aware: bool,
    iou_threshold: float,
    confidence: float,
    classes_to_consider: Optional[List[str]],
    required_objects: Optional[Union[int, Dict[str, int]]],
) -> Tuple[str, List[dict], Optional[bool]]:
    encountered_parent_ids = {
        p[PARENT_ID_KEY] for prediction_source in predictions for p in prediction_source
    }
    if len(encountered_parent_ids) > 1:
        raise ExecutionGraphError(
            f"Missmatch in predictions - while executing consensus step, "
            f"in equivalent batches, detections are assigned different parent "
            f"identifiers, whereas consensus can only be applied for predictions "
            f"made against the same input."
        )
    parent_id = list(encountered_parent_ids)[0]
    predictions = filter_predictions(
        predictions=predictions,
        confidence=confidence,
        classes_to_consider=classes_to_consider,
    )
    detections_already_considered = set()
    consensus_detections = []
    for source_id, detection in enumerate_detections(predictions=predictions):
        detections_with_max_overlap = (
            get_detections_from_different_sources_with_max_overlap(
                detection=detection,
                source=source_id,
                predictions=predictions,
                iou_threshold=iou_threshold,
                class_aware=class_aware,
                detections_already_considered=detections_already_considered,
            )
        )
        if len(detections_with_max_overlap) >= required_votes:
            merged_detection = merge_detections(
                detections=[detection]
                + [
                    matched_value[0]
                    for matched_value in detections_with_max_overlap.values()
                ]
            )
            consensus_detections.append(merged_detection)
            detections_already_considered.add(detection[DETECTION_ID_KEY])
            for matched_value in detections_with_max_overlap.values():
                detections_already_considered.add(matched_value[0][DETECTION_ID_KEY])
    if required_objects is None:
        return parent_id, consensus_detections, None
    if issubclass(type(required_objects), int):
        return (
            parent_id,
            consensus_detections,
            len(consensus_detections) > required_objects,
        )
    consensus_classes = Counter([d["class_name"] for d in consensus_detections])
    consensus_reached = all(
        consensus_classes[class_name] >= consensus_value
        for class_name, consensus_value in required_objects.items()
    )
    return parent_id, consensus_detections, consensus_reached


def filter_predictions(
    predictions: List[List[dict]],
    confidence: float,
    classes_to_consider: Optional[List[str]],
) -> List[List[dict]]:
    if classes_to_consider is not None:
        detection_matches = partial(
            confidence_and_class_match,
            confidence_threshold=confidence,
            classes=set(classes_to_consider),
        )
    else:
        detection_matches = partial(
            confidence_matches,
            confidence_threshold=confidence,
        )
    return [
        [detection for detection in detections if detection_matches(detection)]
        for detections in predictions
    ]


def confidence_and_class_match(
    detection: dict,
    confidence_threshold: float,
    classes: Set[str],
) -> bool:
    return (
        confidence_matches(
            detection=detection, confidence_threshold=confidence_threshold
        )
        and detection["class_name"] in classes
    )


def confidence_matches(detection: dict, confidence_threshold: float) -> bool:
    return detection["confidence"] > confidence_threshold


def get_detections_from_different_sources_with_max_overlap(
    detection: dict,
    source: int,
    predictions: List[List[dict]],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[dict, float]]:
    current_max_overlap = {}
    for other_source, other_detection in enumerate_detections(
        predictions=predictions,
        excluded_source=source,
    ):
        if other_detection[DETECTION_ID_KEY] in detections_already_considered:
            continue
        if class_aware and detection["class_name"] != other_detection["class_name"]:
            continue
        iou_value = calculate_iou(
            detection_a=detection,
            detection_b=other_detection,
        )
        if iou_value <= iou_threshold:
            continue
        if current_max_overlap.get(other_source) is None:
            current_max_overlap[other_source] = (other_detection, iou_value)
        if current_max_overlap[other_source][1] < iou_value:
            current_max_overlap[other_source] = (other_detection, iou_value)
    return current_max_overlap


def enumerate_detections(
    predictions: List[List[dict]],
    excluded_source: Optional[int] = None,
) -> Generator[Tuple[int, dict], None, None]:
    for source_id, detections in enumerate(predictions):
        if excluded_source is not None and excluded_source == source_id:
            continue
        for detection in detections:
            yield source_id, detection


def calculate_iou(detection_a: dict, detection_b: dict) -> float:
    box_a = detection_to_xyxy(detection=detection_a)
    box_b = detection_to_xyxy(detection=detection_b)
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    intersection = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    bbox_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    bbox_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return intersection / float(bbox_a_area + bbox_b_area - intersection)


def detection_to_xyxy(detection: dict) -> Tuple[int, int, int, int]:
    x_min = round(detection["x"] - detection[WIDTH_KEY] / 2)
    y_min = round(detection["y"] - detection[HEIGHT_KEY] / 2)
    x_max = round(x_min + detection[WIDTH_KEY])
    y_max = round(y_min + detection[HEIGHT_KEY])
    return x_min, y_min, x_max, y_max


def merge_detections(detections: List[dict]) -> dict:
    class_name, class_id = get_majority_class(detections=detections)
    return {
        PARENT_ID_KEY: detections[0][PARENT_ID_KEY],
        DETECTION_ID_KEY: f"{uuid4()}",
        "class_name": class_name,
        "class_id": class_id,
        "confidence": average_field_values(detections=detections, field="confidence"),
        "x": round(average_field_values(detections=detections, field="x")),
        "y": round(average_field_values(detections=detections, field="y")),
        "width": round(average_field_values(detections=detections, field="width")),
        "height": round(average_field_values(detections=detections, field="height")),
    }


def get_majority_class(detections: List[dict]) -> Tuple[str, int]:
    class_counts = Counter(d["class_name"] for d in detections)
    most_common_class_name = class_counts.most_common(1)[0]
    class_id = [
        d["class_id"] for d in detections if d["class_name"] == most_common_class_name
    ][0]
    return most_common_class_name, class_id


def average_field_values(detections: List[dict], field: str) -> float:
    return statistics.mean([d[field] for d in detections])
