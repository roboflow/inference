import math
import statistics
from collections import Counter
from enum import Enum
from functools import lru_cache
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
    Selector,
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


class MaskAggregationMode(Enum):
    INTERSECTION = "intersection"
    UNION = "union"
    MAX = "max"
    MIN = "min"


LONG_DESCRIPTION = """
Combine detection predictions from multiple models using a majority vote consensus strategy, merging overlapping detections that receive sufficient votes from different models and aggregating their properties (confidence scores, bounding boxes, masks) into unified consensus detections with improved accuracy and reliability.

## How This Block Works

This block fuses predictions from multiple detection models by requiring agreement (consensus) among models before accepting detections. The block:

1. Takes detection predictions from multiple model sources (object detection, instance segmentation, or keypoint detection) as input
2. Matches detections from different models that overlap spatially by calculating Intersection over Union (IoU) between bounding boxes
3. Compares overlapping detections against an IoU threshold to determine if they represent the same object
4. Counts "votes" for each detection by finding matching detections from other models (subject to class-awareness if enabled)
5. Requires a minimum number of votes (`required_votes`) before accepting a detection as part of the consensus output
6. Aggregates properties of matching detections using configurable modes:
   - **Confidence aggregation**: Combines confidence scores using average, max, or min
   - **Coordinates aggregation**: Merges bounding boxes using average (mean coordinates), max (largest box), or min (smallest box)
   - **Mask aggregation** (for instance segmentation): Combines masks using union, intersection, max, or min
   - **Class selection**: Chooses class name based on majority vote (average), highest confidence (max), or lowest confidence (min)
7. Filters detections based on optional criteria (specific classes to consider, minimum confidence threshold)
8. Determines object presence by checking if the required number of objects (per class or total) are present in consensus results
9. Returns merged consensus detections, object presence indicators, and presence confidence scores

The block enables class-aware or class-agnostic matching: when `class_aware` is true, only detections with matching class names are considered for voting; when false, any overlapping detections (regardless of class) contribute votes. The consensus mechanism helps reduce false positives (detections seen by only one model) and improves reliability by requiring multiple models to agree on object presence. Aggregation modes allow flexibility in how overlapping detections are combined, balancing between conservative (intersection, min) and inclusive (union, max) strategies.

## Common Use Cases

- **Multi-Model Ensemble**: Combine predictions from multiple specialized models (e.g., one optimized for people, another for vehicles) to improve overall detection accuracy, leveraging strengths of different models while filtering out detections that only one model sees
- **Reducing False Positives**: Require consensus from multiple models before accepting detections (e.g., require 2 out of 3 models to detect an object), reducing false positives by filtering out detections seen by only one model
- **Improving Detection Reliability**: Use majority voting to increase confidence in detections (e.g., merge overlapping detections from 3 models, keeping only those with 2+ votes), ensuring only high-confidence, multi-model-agreed detections are retained
- **Object Presence Detection**: Determine if specific objects are present based on consensus (e.g., check if at least 2 "person" detections exist across models, use aggregated confidence to determine presence), enabling robust object presence checking with configurable thresholds
- **Class-Specific Consensus**: Apply different consensus requirements per class (e.g., require 3 votes for "car" but only 2 for "person"), allowing stricter criteria for critical objects while being more lenient for common detections
- **Specialized Model Fusion**: Combine general-purpose and specialized models (e.g., general object detector + specialized license plate detector), creating a unified detection system that benefits from both broad coverage and specific expertise

## Connecting to Other Blocks

The consensus predictions from this block can be connected to:

- **Multiple detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions from different models that are fused into consensus detections based on majority voting and spatial overlap matching
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display the merged consensus detections, showing unified results from multiple models with improved accuracy
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze consensus detections, providing more reliable metrics based on multi-model agreement
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit consensus detection results, storing fused predictions that represent multi-model agreement
- **Flow control blocks** (e.g., Continue If) to conditionally trigger downstream processing based on `object_present` indicators or `presence_confidence` scores, enabling workflows that respond to multi-model consensus on object presence
- **Filtering blocks** (e.g., Detections Filter) to further refine consensus detections based on additional criteria, enabling multi-stage filtering after consensus fusion
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
            "ui_manifest": {
                "section": "flow_control",
                "icon": "fak fa-circles-overlap",
                "blockPriority": 4,
            },
        }
    )
    type: Literal["roboflow_core/detections_consensus@v1", "DetectionsConsensus"]
    predictions_batches: List[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        ),
    ] = Field(
        min_items=1,
        description="List of references to detection predictions from multiple models. Each model's predictions must be made against the same input image. Predictions can be from object detection, instance segmentation, or keypoint detection models. The block matches overlapping detections across models and requires a minimum number of votes (required_votes) before accepting detections in the consensus output. Requires at least one prediction source. Supports batch processing.",
        examples=[["$steps.a.predictions", "$steps.b.predictions"]],
        validation_alias=AliasChoices("predictions_batches", "predictions"),
    )
    required_votes: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Minimum number of votes (matching detections from different models) required to accept a detection in the consensus output. Detections that receive fewer votes than this threshold are filtered out. For example, if set to 2, at least 2 models must detect an overlapping object (above IoU threshold) for it to appear in the consensus results. Higher values create stricter consensus requirements, reducing false positives but potentially missing detections seen by fewer models.",
        examples=[2, "$inputs.required_votes"],
    )
    class_aware: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If true, only detections with matching class names from different models are considered as votes for the same object. If false, any overlapping detections (regardless of class) contribute votes. Class-aware mode is more conservative and ensures class consistency in consensus, while class-agnostic mode allows voting across different classes but may merge detections of different object types.",
        examples=[True, "$inputs.class_aware"],
    )
    iou_threshold: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = (
        Field(
            default=0.3,
            description="Intersection over Union (IoU) threshold for considering detections from different models as matching the same object. Detections with IoU above this threshold are considered overlapping and contribute votes to each other. Lower values (e.g., 0.2) are more lenient and match detections with less overlap, while higher values (e.g., 0.5) require stronger spatial overlap for matching. Typical values range from 0.2 to 0.5.",
            examples=[0.3, "$inputs.iou_threshold"],
        )
    )
    confidence: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.0,
        description="Confidence threshold applied to merged consensus detections. Only detections with aggregated confidence scores above this threshold are included in the output. Set to 0.0 to disable confidence filtering. Higher values filter out low-confidence consensus detections, improving output quality at the cost of potentially removing valid but lower-confidence detections.",
        examples=[0.1, "$inputs.confidence"],
    )
    classes_to_consider: Optional[
        Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        description="Optional list of class names to include in the consensus procedure. If provided, only detections of these classes are considered for voting and merging; all other classes are filtered out before consensus matching. Use this to focus consensus on specific object types while ignoring irrelevant detections. If None, all classes participate in consensus.",
        examples=[["a", "b"], "$inputs.classes_to_consider"],
    )
    required_objects: Optional[
        Union[
            PositiveInt,
            Dict[str, PositiveInt],
            Selector(kind=[INTEGER_KIND, DICTIONARY_KIND]),
        ]
    ] = Field(
        default=None,
        description="Optional minimum number of objects required to determine object presence. Can be an integer (total objects across all classes) or a dictionary mapping class names to per-class minimum counts. Used in conjunction with object_present output to determine if sufficient objects of each class are detected. For example, 3 means at least 3 total objects must be present, while {'person': 2, 'car': 1} requires at least 2 persons and 1 car. If None, object presence is determined solely by whether any consensus detections exist.",
        examples=[3, {"a": 7, "b": 2}, "$inputs.required_objects"],
    )
    presence_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.MAX,
        description="Aggregation mode for calculating presence confidence scores. Determines how confidence values are combined when computing object presence confidence: 'average' (mean confidence), 'max' (highest confidence), or 'min' (lowest confidence). This mode applies to the presence_confidence output which indicates confidence that required objects are present.",
        examples=["max", "min"],
    )
    detections_merge_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Aggregation mode for merging confidence scores of overlapping detections. 'average' computes mean confidence (majority vote approach), 'max' uses the highest confidence among matching detections, 'min' uses the lowest confidence. For class selection, 'average' represents majority vote (most common class), 'max' selects class from detection with highest confidence, 'min' selects class from detection with lowest confidence.",
        examples=["min", "max"],
    )
    detections_merge_coordinates_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Aggregation mode for merging bounding box coordinates of overlapping detections. 'average' computes mean coordinates from all matching boxes (balanced approach), 'max' takes the largest box (most inclusive), 'min' takes the smallest box (most conservative). This mode only applies to bounding boxes; mask aggregation uses detections_merge_mask_aggregation instead.",
        examples=["min", "max"],
    )
    detections_merge_mask_aggregation: MaskAggregationMode = Field(
        default=MaskAggregationMode.UNION,
        description="Aggregation mode for merging segmentation masks of overlapping detections. 'union' combines all masks into the largest possible area (most inclusive), 'intersection' takes only the overlapping region (most conservative), 'max' selects the largest mask, 'min' selects the smallest mask. This mode applies only to instance segmentation detections with masks; bounding box detections use detections_merge_coordinates_aggregation instead.",
        examples=["union", "intersection"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions_batches"]

    @classmethod
    @lru_cache(maxsize=None)
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
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
        return ">=1.3.0,<2.0.0"


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
        detections_merge_mask_aggregation: MaskAggregationMode,
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
                detections_merge_mask_aggregation=detections_merge_mask_aggregation,
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
    detections_merge_mask_aggregation: MaskAggregationMode,
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
            detections_merge_mask_aggregation=detections_merge_mask_aggregation,
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
    detections_merge_mask_aggregation: MaskAggregationMode,
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
    if detection.mask is not None:
        for matched_value in detections_with_max_overlap.values():
            if matched_value[0].mask is None:
                matched_value[0].mask = np.zeros(detection.mask.shape)
    else:
        shape = None
        for d in detections_with_max_overlap.values():
            if d[0].mask is not None:
                shape = d[0].mask.shape
                break
        if shape:
            for d in detections_with_max_overlap.values():
                if d[0].mask is None:
                    d[0].mask = np.zeros(shape)
            detection.mask = np.zeros(shape)
    detections_to_merge = sv.Detections.merge(
        [detection]
        + [matched_value[0] for matched_value in detections_with_max_overlap.values()]
    )
    merged_detection = merge_detections(
        detections=detections_to_merge,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
        mask_aggregation_mode=detections_merge_mask_aggregation,
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
    mask_aggregation_mode: MaskAggregationMode,
) -> sv.Detections:
    class_name, class_id = AGGREGATION_MODE2CLASS_SELECTOR[confidence_aggregation_mode](
        detections
    )
    if detections.mask is not None:
        mask = np.array(
            [AGGREGATION_MODE2MASKS_AGGREGATOR[mask_aggregation_mode](detections)]
        )
        x1, y1, x2, y2 = sv.mask_to_xyxy(mask)[0]
    else:
        mask = None
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
        mask=mask,
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
    if len(detections) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    avg_xyxy = np.mean(detections.xyxy, axis=0)
    return tuple(avg_xyxy)


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


def get_intersection_mask(detections: sv.Detections) -> np.ndarray:
    return np.all(detections.mask, axis=0)


def get_union_mask(detections: sv.Detections) -> np.ndarray:
    return np.sum(detections.mask, axis=0)


def get_smallest_mask(detections: sv.Detections) -> np.ndarray:
    areas: List[float] = detections.area.astype(float).tolist()
    min_area = min(areas)
    min_area_index = areas.index(min_area)
    return detections[min_area_index].mask[0]


def get_largest_mask(detections: sv.Detections) -> np.ndarray:
    areas: List[float] = detections.area.astype(float).tolist()
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    return detections[max_area_index].mask[0]


AGGREGATION_MODE2MASKS_AGGREGATOR = {
    MaskAggregationMode.MAX: get_largest_mask,
    MaskAggregationMode.MIN: get_smallest_mask,
    MaskAggregationMode.UNION: get_union_mask,
    MaskAggregationMode.INTERSECTION: get_intersection_mask,
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
