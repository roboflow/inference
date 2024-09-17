import math
import statistics
from collections import Counter
from enum import Enum
from typing import Dict, Generator, List, Literal, Optional, Set, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    FloatZeroToOne,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class AggregationMode(Enum):
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


LONG_DESCRIPTION = """
Combine detections from multiple detection-based models based on a majority vote 
strategy.

This block is useful if you have multiple specialized models that you want to consult 
to determine whether a certain object is present in an image.

See the table below to explore the values you can use to configure the consensus block.
"""

SHORT_DESCRIPTION = (
    "Combine predictions from multiple detections models to make a "
    "decision about object presence."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Consensus",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal["roboflow_core/detections_consensus@v1", "DetectionsConsensus"]
    predictions_batches: List[
        StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        ),
    ] = Field(
        min_items=1,
        description="Reference to detection-like model predictions made against single image to agree on model consensus",
        examples=[["$steps.a.predictions", "$steps.b.predictions"]],
        validation_alias=AliasChoices("predictions_batches", "predictions"),
    )
    required_votes: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Required number of votes for single detection from different models to accept detection as output detection",
        examples=[2, "$inputs.required_votes"],
    )
    class_aware: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to decide if merging detections is class-aware or only bounding boxes aware",
        examples=[True, "$inputs.class_aware"],
    )
    iou_threshold: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.3,
        description="IoU threshold to consider detections from different models as matching (increasing votes for region)",
        examples=[0.3, "$inputs.iou_threshold"],
    )
    confidence: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.0,
        description="Confidence threshold for merged detections",
        examples=[0.1, "$inputs.confidence"],
    )
    classes_to_consider: Optional[
        Union[List[str], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        description="Optional list of classes to consider in consensus procedure.",
        examples=[["a", "b"], "$inputs.classes_to_consider"],
    )
    required_objects: Optional[
        Union[
            PositiveInt,
            Dict[str, PositiveInt],
            WorkflowParameterSelector(kind=[INTEGER_KIND, DICTIONARY_KIND]),
        ]
    ] = Field(
        default=None,
        description="If given, it holds the number of objects that must be present in merged results, to assume that object presence is reached. Can be selector to `InferenceParameter`, integer value or dictionary with mapping of class name into minimal number of merged detections of given class to assume consensus.",
        examples=[3, {"a": 7, "b": 2}, "$inputs.required_objects"],
    )
    presence_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.MAX,
        description="Mode dictating aggregation of confidence scores and classes both in case of object presence deduction procedure.",
        examples=["max", "min"],
    )
    detections_merge_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of confidence scores and classes both in case of boxes consensus procedure. One of `average`, `max`, `min`. Default: `average`. While using for merging overlapping boxes, against classes - `average` equals to majority vote, `max` - for the class of detection with max confidence, `min` - for the class of detection with min confidence.",
        examples=["min", "max"],
    )
    detections_merge_coordinates_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of bounding boxes. One of `average`, `max`, `min`. Default: `average`. `average` means taking mean from all boxes coordinates, `min` - taking smallest box, `max` - taking largest box.",
        examples=["min", "max"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="object_present", kind=[BOOLEAN_KIND, DICTIONARY_KIND]
            ),
            OutputDefinition(
                name="presence_confidence",
                kind=[FLOAT_ZERO_TO_ONE_KIND, DICTIONARY_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DetectionsConsensusBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions_batches: List[Batch[sv.Detections]],
        required_votes: int,
        class_aware: bool,
        iou_threshold: float,
        confidence: float,
        classes_to_consider: Optional[List[str]],
        required_objects: Optional[Union[int, Dict[str, int]]],
        presence_confidence_aggregation: AggregationMode,
        detections_merge_confidence_aggregation: AggregationMode,
        detections_merge_coordinates_aggregation: AggregationMode,
    ) -> BlockResult:
        if len(predictions_batches) < 1:
            raise ValueError(
                f"Consensus step requires at least one source of predictions."
            )
        results = []
        for detections_from_sources in zip(*predictions_batches):
            (
                parent_id,
                object_present,
                presence_confidence,
                consensus_detections,
            ) = agree_on_consensus_for_all_detections_sources(
                detections_from_sources=list(detections_from_sources),
                required_votes=required_votes,
                class_aware=class_aware,
                iou_threshold=iou_threshold,
                confidence=confidence,
                classes_to_consider=classes_to_consider,
                required_objects=required_objects,
                presence_confidence_aggregation=presence_confidence_aggregation,
                detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
                detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            )
            results.append(
                {
                    "predictions": consensus_detections,
                    "object_present": object_present,
                    "presence_confidence": presence_confidence,
                }
            )
        return results


def does_not_detect_objects_in_any_source(
    detections_from_sources: List[sv.Detections],
) -> bool:
    return all(len(p) == 0 for p in detections_from_sources)


def get_parent_id_of_detections_from_sources(
    detections_from_sources: List[sv.Detections],
) -> str:
    encountered_parent_ids = set(
        np.concatenate(
            [
                detections[PARENT_ID_KEY]
                for detections in detections_from_sources
                if PARENT_ID_KEY in detections.data
            ]
        ).tolist()
    )
    if len(encountered_parent_ids) != 1:
        raise ValueError(
            "Missmatch in predictions - while executing consensus step, "
            "in equivalent batches, detections are assigned different parent "
            "identifiers, whereas consensus can only be applied for predictions "
            "made against the same input."
        )
    return next(iter(encountered_parent_ids))


def filter_predictions(
    predictions: List[sv.Detections],
    classes_to_consider: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_consider:
        return predictions
    return [
        detections[np.isin(detections["class_name"], classes_to_consider)]
        for detections in predictions
        if "class_name" in detections.data
    ]


def get_detections_from_different_sources_with_max_overlap(
    detection: sv.Detections,
    source: int,
    detections_from_sources: List[sv.Detections],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[sv.Detections, float]]:
    current_max_overlap = {}
    for other_source, other_detection in enumerate_detections(
        detections_from_sources=detections_from_sources,
        excluded_source_id=source,
    ):
        if other_detection[DETECTION_ID_KEY][0] in detections_already_considered:
            continue
        if (
            class_aware
            and detection["class_name"][0] != other_detection["class_name"][0]
        ):
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
    detections_from_sources: List[sv.Detections],
    excluded_source_id: Optional[int] = None,
) -> Generator[Tuple[int, sv.Detections], None, None]:
    for source_id, detections in enumerate(detections_from_sources):
        if excluded_source_id == source_id:
            continue
        for i in range(len(detections)):
            yield source_id, detections[i]


def calculate_iou(detection_a: sv.Detections, detection_b: sv.Detections) -> float:
    iou = float(sv.box_iou_batch(detection_a.xyxy, detection_b.xyxy)[0][0])
    if math.isnan(iou):
        iou = 0
    return iou


def agree_on_consensus_for_all_detections_sources(
    detections_from_sources: List[sv.Detections],
    required_votes: int,
    class_aware: bool,
    iou_threshold: float,
    confidence: float,
    classes_to_consider: Optional[List[str]],
    required_objects: Optional[Union[int, Dict[str, int]]],
    presence_confidence_aggregation: AggregationMode,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
) -> Tuple[str, bool, Dict[str, float], sv.Detections]:
    if does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources
    ):
        return "undefined", False, {}, sv.Detections.empty()
    parent_id = get_parent_id_of_detections_from_sources(
        detections_from_sources=detections_from_sources,
    )
    detections_from_sources = filter_predictions(
        predictions=detections_from_sources,
        classes_to_consider=classes_to_consider,
    )
    detections_already_considered = set()
    consensus_detections = []
    for source_id, detection in enumerate_detections(
        detections_from_sources=detections_from_sources
    ):
        (
            consensus_detections_update,
            detections_already_considered,
        ) = get_consensus_for_single_detection(
            detection=detection,
            source_id=source_id,
            detections_from_sources=detections_from_sources,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            required_votes=required_votes,
            confidence=confidence,
            detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
            detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            detections_already_considered=detections_already_considered,
        )
        consensus_detections += consensus_detections_update
    consensus_detections = sv.Detections.merge(consensus_detections)
    (
        object_present,
        presence_confidence,
    ) = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        aggregation_mode=presence_confidence_aggregation,
        class_aware=class_aware,
        required_objects=required_objects,
    )
    return (
        parent_id,
        object_present,
        presence_confidence,
        consensus_detections,
    )


def get_consensus_for_single_detection(
    detection: sv.Detections,
    source_id: int,
    detections_from_sources: List[sv.Detections],
    iou_threshold: float,
    class_aware: bool,
    required_votes: int,
    confidence: float,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
    detections_already_considered: Set[str],
) -> Tuple[List[sv.Detections], Set[str]]:
    if detection and detection["detection_id"][0] in detections_already_considered:
        return [], detections_already_considered
    consensus_detections = []
    detections_with_max_overlap = (
        get_detections_from_different_sources_with_max_overlap(
            detection=detection,
            source=source_id,
            detections_from_sources=detections_from_sources,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            detections_already_considered=detections_already_considered,
        )
    )

    if len(detections_with_max_overlap) < (required_votes - 1):
        # Returning empty sv.Detections
        return consensus_detections, detections_already_considered
    detections_to_merge = sv.Detections.merge(
        [detection]
        + [matched_value[0] for matched_value in detections_with_max_overlap.values()]
    )
    merged_detection = merge_detections(
        detections=detections_to_merge,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
    )
    if merged_detection.confidence[0] < confidence:
        # Returning empty sv.Detections
        return consensus_detections, detections_already_considered
    consensus_detections.append(merged_detection)
    detections_already_considered.add(detection[DETECTION_ID_KEY][0])
    for matched_value in detections_with_max_overlap.values():
        detections_already_considered.add(matched_value[0][DETECTION_ID_KEY][0])
    return consensus_detections, detections_already_considered


def check_objects_presence_in_consensus_detections(
    consensus_detections: sv.Detections,
    class_aware: bool,
    aggregation_mode: AggregationMode,
    required_objects: Optional[Union[int, Dict[str, int]]],
) -> Tuple[bool, Dict[str, float]]:
    if not consensus_detections:
        return False, {}
    if required_objects is None:
        required_objects = 0
    if isinstance(required_objects, dict) and not class_aware:
        required_objects = sum(required_objects.values())
    if (
        isinstance(required_objects, int)
        and len(consensus_detections) < required_objects
    ):
        return False, {}
    if not class_aware:
        aggregated_confidence = aggregate_field_values(
            detections=consensus_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        return True, {"any_object": aggregated_confidence}
    class2detections = {}
    for class_name in set(consensus_detections["class_name"]):
        class2detections[class_name] = consensus_detections[
            consensus_detections["class_name"] == class_name
        ]
    if isinstance(required_objects, dict):
        for requested_class, required_objects_count in required_objects.items():
            if (
                requested_class not in class2detections
                or len(class2detections[requested_class]) < required_objects_count
            ):
                return False, {}
    class2confidence = {
        class_name: aggregate_field_values(
            detections=class_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        for class_name, class_detections in class2detections.items()
    }
    return True, class2confidence


def merge_detections(
    detections: sv.Detections,
    confidence_aggregation_mode: AggregationMode,
    boxes_aggregation_mode: AggregationMode,
) -> sv.Detections:
    class_name, class_id = AGGREGATION_MODE2CLASS_SELECTOR[confidence_aggregation_mode](
        detections
    )
    x1, y1, x2, y2 = AGGREGATION_MODE2BOXES_AGGREGATOR[boxes_aggregation_mode](
        detections
    )
    data = {
        "class_name": np.array([class_name]),
        PARENT_ID_KEY: np.array([detections[PARENT_ID_KEY][0]]),
        DETECTION_ID_KEY: np.array([str(uuid4())]),
        PREDICTION_TYPE_KEY: np.array(["object-detection"]),
        PARENT_COORDINATES_KEY: np.array([detections[PARENT_COORDINATES_KEY][0]]),
        PARENT_DIMENSIONS_KEY: np.array([detections[PARENT_DIMENSIONS_KEY][0]]),
        ROOT_PARENT_ID_KEY: np.array([detections[ROOT_PARENT_ID_KEY][0]]),
        ROOT_PARENT_COORDINATES_KEY: np.array(
            [detections[ROOT_PARENT_COORDINATES_KEY][0]]
        ),
        ROOT_PARENT_DIMENSIONS_KEY: np.array(
            [detections[ROOT_PARENT_DIMENSIONS_KEY][0]]
        ),
        IMAGE_DIMENSIONS_KEY: np.array([detections[IMAGE_DIMENSIONS_KEY][0]]),
    }
    if SCALING_RELATIVE_TO_PARENT_KEY in detections.data:
        data[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
            [detections[SCALING_RELATIVE_TO_PARENT_KEY][0]]
        )
    else:
        data[SCALING_RELATIVE_TO_PARENT_KEY] = np.array([1.0])
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections.data:
        data[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
            [detections[SCALING_RELATIVE_TO_ROOT_PARENT_KEY][0]]
        )
    else:
        data[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array([1.0])
    return sv.Detections(
        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float64),
        class_id=np.array([class_id]),
        confidence=np.array(
            [
                aggregate_field_values(
                    detections=detections,
                    field="confidence",
                    aggregation_mode=confidence_aggregation_mode,
                )
            ],
            dtype=np.float64,
        ),
        data=data,
    )


def get_majority_class(detections: sv.Detections) -> Tuple[str, int]:
    class_counts = Counter(
        [
            (str(class_name), int(class_id))
            for class_name, class_id in zip(
                detections["class_name"], detections.class_id
            )
        ]
    )
    return class_counts.most_common(1)[0][0]


def get_class_of_most_confident_detection(detections: sv.Detections) -> Tuple[str, int]:
    confidences: List[float] = detections.confidence.astype(float).tolist()
    max_confidence_index = confidences.index(max(confidences))
    max_confidence_detection = detections[max_confidence_index]
    return (
        max_confidence_detection["class_name"][0],
        max_confidence_detection.class_id[0],
    )


def get_class_of_least_confident_detection(
    detections: sv.Detections,
) -> Tuple[str, int]:
    confidences: List[float] = detections.confidence.astype(float).tolist()
    min_confidence_index = confidences.index(min(confidences))
    min_confidence_detection = detections[min_confidence_index]
    return (
        min_confidence_detection["class_name"][0],
        min_confidence_detection.class_id[0],
    )


AGGREGATION_MODE2CLASS_SELECTOR = {
    AggregationMode.MAX: get_class_of_most_confident_detection,
    AggregationMode.MIN: get_class_of_least_confident_detection,
    AggregationMode.AVERAGE: get_majority_class,
}


def get_average_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    avg_xyxy: np.ndarray = sum(detections.xyxy) / len(detections)
    return tuple(avg_xyxy.astype(float))


def get_smallest_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    areas: List[float] = detections.area.astype(float).tolist()
    min_area = min(areas)
    min_area_index = areas.index(min_area)
    return tuple(detections[min_area_index].xyxy[0])


def get_largest_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    areas: List[float] = detections.area.astype(float).tolist()
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    return tuple(detections[max_area_index].xyxy[0])


AGGREGATION_MODE2BOXES_AGGREGATOR = {
    AggregationMode.MAX: get_largest_bounding_box,
    AggregationMode.MIN: get_smallest_bounding_box,
    AggregationMode.AVERAGE: get_average_bounding_box,
}

AGGREGATION_MODE2FIELD_AGGREGATOR = {
    AggregationMode.MAX: max,
    AggregationMode.MIN: min,
    AggregationMode.AVERAGE: statistics.mean,
}


def aggregate_field_values(
    detections: sv.Detections,
    field: str,
    aggregation_mode: AggregationMode = AggregationMode.AVERAGE,
) -> float:
    values = []
    if hasattr(detections, field):
        values = getattr(detections, field)
        if isinstance(values, np.ndarray):
            values = values.astype(float).tolist()
    elif hasattr(detections, "data") and field in detections.data:
        values = detections[field]
        if isinstance(values, np.ndarray):
            values = values.astype(float).tolist()
    return AGGREGATION_MODE2FIELD_AGGREGATOR[aggregation_mode](values)
