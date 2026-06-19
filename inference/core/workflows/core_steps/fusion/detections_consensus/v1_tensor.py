import math
import statistics
from collections import Counter
from enum import Enum
from functools import lru_cache
from typing import Dict, Generator, List, Literal, Optional, Set, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import torch
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.core_steps.common.tensor_native import (
    instance_mask_to_numpy,
    take_prediction_by_indices,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
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
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

# Tensor-native detections handled by this block. The consensus pipeline only
# ever needs the bounding-box component, so keypoint predictions (which arrive
# as a `(KeyPoints, Detections)` tuple) are reduced to their `Detections` part
# up-front and the consensus output is always a plain `Detections` /
# `InstanceDetections` (the output kind never includes keypoints, matching the
# numpy block).
TensorNativeDetections = Union[Detections, InstanceDetections]


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
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
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
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
        predictions_batches: List[Batch[TensorNativeDetections]],
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
                detections_from_sources=[
                    _to_bbox_component(detections)
                    for detections in detections_from_sources
                ],
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


def _to_bbox_component(
    detections: Union[TensorNativeDetections, Tuple],
) -> TensorNativeDetections:
    # Keypoint predictions arrive as a (KeyPoints, Detections) tuple; consensus
    # only operates on bounding boxes so the bbox component is used directly.
    if isinstance(detections, tuple):
        _key_points, bbox_detections = detections
        if bbox_detections is None:
            raise ValueError(
                "Keypoint prediction is missing the bounding-box component "
                "required by the consensus step."
            )
        return bbox_detections
    return detections


def _resolve_class_name(detection: TensorNativeDetections, index: int = 0) -> str:
    # class_name of detection i = image_metadata["class_names"][int(class_id[i])]
    # (fallback f"class_{id}") - mirrors the per-detection class name that the
    # numpy block read from sv.Detections.data["class_name"].
    class_id = int(detection.class_id[index])
    class_names = (detection.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    return class_names.get(class_id, f"class_{class_id}")


def _resolve_detection_id(detection: TensorNativeDetections, index: int = 0) -> str:
    bboxes_metadata = detection.bboxes_metadata or [{}]
    return bboxes_metadata[index][DETECTION_ID_KEY]


def _xyxy_as_numpy(detection: TensorNativeDetections) -> np.ndarray:
    return detection.xyxy.detach().to("cpu").numpy().astype(float)


def _native_areas(detections: TensorNativeDetections) -> np.ndarray:
    xyxy = _xyxy_as_numpy(detections)
    return (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])


def _empty_native_detections() -> Detections:
    return Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        image_metadata=None,
        bboxes_metadata=None,
    )


def _concat_metadata(
    detections_list: List[TensorNativeDetections],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict], Optional[List[dict]]
]:
    # Concatenate xyxy / class_id / confidence / bboxes_metadata of several
    # predictions (masks handled separately). image_metadata is shared within one
    # image so the first non-empty one is carried over as the base, BUT the
    # class_names map is unioned across all sources: each consensus row supplies
    # its own {class_id: class_name} pair, so taking only the first row's map
    # would lose names for every other class in a multi-class consensus output.
    image_metadata = None
    merged_class_names: dict = {}
    for d in detections_list:
        if d.image_metadata is None:
            continue
        if image_metadata is None:
            image_metadata = d.image_metadata
        source_class_names = d.image_metadata.get(CLASS_NAMES_KEY)
        if source_class_names:
            for class_id_key, class_name in source_class_names.items():
                merged_class_names[int(class_id_key)] = class_name
    if image_metadata is not None and merged_class_names:
        image_metadata = dict(image_metadata)
        image_metadata[CLASS_NAMES_KEY] = merged_class_names
    bboxes_metadata: Optional[List[dict]] = None
    if any(d.bboxes_metadata is not None for d in detections_list):
        bboxes_metadata = []
        for d in detections_list:
            if d.bboxes_metadata is not None:
                bboxes_metadata.extend(d.bboxes_metadata)
            else:
                bboxes_metadata.extend({} for _ in range(len(d)))
    xyxy = torch.cat([d.xyxy for d in detections_list], dim=0)
    class_id = torch.cat([d.class_id for d in detections_list], dim=0)
    confidence = torch.cat([d.confidence for d in detections_list], dim=0)
    return xyxy, class_id, confidence, image_metadata, bboxes_metadata


def _merge_native_detections(
    detections_list: List[TensorNativeDetections],
) -> TensorNativeDetections:
    # Native counterpart of sv.Detections.merge(list): concatenate several
    # predictions into one native object. When any source carries a mask the
    # result is an InstanceDetections whose dense masks are padded to a common
    # (H, W) (mask-less / object-detection rows become zero masks, mirroring the
    # numpy block's zero padding before sv.Detections.merge).
    # SHARED-HELPER CANDIDATE: this inline native merge mirrors the
    # _concatenate_detections helper in query_language and should be consolidated.
    if not detections_list:
        return _empty_native_detections()
    xyxy, class_id, confidence, image_metadata, bboxes_metadata = _concat_metadata(
        detections_list
    )
    has_masks = any(
        isinstance(d, InstanceDetections) and d.mask is not None
        for d in detections_list
    )
    if not has_masks:
        return Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    mask_height, mask_width = 0, 0
    for d in detections_list:
        if isinstance(d, InstanceDetections) and d.mask is not None and len(d) > 0:
            sample = instance_mask_to_numpy(d, 0)
            mask_height, mask_width = sample.shape
            break
    mask_rows = []
    for d in detections_list:
        if isinstance(d, InstanceDetections) and d.mask is not None:
            for i in range(len(d)):
                mask_rows.append(instance_mask_to_numpy(d, i))
        else:
            for _ in range(len(d)):
                mask_rows.append(np.zeros((mask_height, mask_width), dtype=bool))
    mask = torch.from_numpy(np.stack(mask_rows, axis=0)).to(torch.bool)
    return InstanceDetections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def does_not_detect_objects_in_any_source(
    detections_from_sources: List[TensorNativeDetections],
) -> bool:
    return all(len(p) == 0 for p in detections_from_sources)


def get_parent_id_of_detections_from_sources(
    detections_from_sources: List[TensorNativeDetections],
) -> str:
    # Per-image parent_id lives in image_metadata (the numpy block read it from
    # the per-detection sv.Detections.data[PARENT_ID_KEY]); a source contributes
    # one parent_id per non-empty prediction.
    encountered_parent_ids = set()
    for detections in detections_from_sources:
        image_metadata = detections.image_metadata or {}
        if PARENT_ID_KEY in image_metadata and len(detections) > 0:
            encountered_parent_ids.add(image_metadata[PARENT_ID_KEY])
    if len(encountered_parent_ids) != 1:
        raise ValueError(
            "Missmatch in predictions - while executing consensus step, "
            "in equivalent batches, detections are assigned different parent "
            "identifiers, whereas consensus can only be applied for predictions "
            "made against the same input."
        )
    return next(iter(encountered_parent_ids))


def filter_predictions(
    predictions: List[TensorNativeDetections],
    classes_to_consider: Optional[List[str]],
) -> List[TensorNativeDetections]:
    if not classes_to_consider:
        return predictions
    filtered = []
    for detections in predictions:
        class_names = (detections.image_metadata or {}).get(CLASS_NAMES_KEY)
        if class_names is None:
            continue
        mask = [
            _resolve_class_name(detections, index) in classes_to_consider
            for index in range(len(detections))
        ]
        indices = [index for index, keep in enumerate(mask) if keep]
        filtered.append(take_prediction_by_indices(detections, indices))
    return filtered


def get_detections_from_different_sources_with_max_overlap(
    detection: TensorNativeDetections,
    source: int,
    detections_from_sources: List[TensorNativeDetections],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[TensorNativeDetections, float]]:
    current_max_overlap = {}
    for other_source, other_detection in enumerate_detections(
        detections_from_sources=detections_from_sources,
        excluded_source_id=source,
    ):
        if _resolve_detection_id(other_detection) in detections_already_considered:
            continue
        if class_aware and _resolve_class_name(detection) != _resolve_class_name(
            other_detection
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
    detections_from_sources: List[TensorNativeDetections],
    excluded_source_id: Optional[int] = None,
) -> Generator[Tuple[int, TensorNativeDetections], None, None]:
    for source_id, detections in enumerate(detections_from_sources):
        if excluded_source_id == source_id:
            continue
        for i in range(len(detections)):
            yield source_id, take_prediction_by_indices(detections, [i])


def calculate_iou(
    detection_a: TensorNativeDetections, detection_b: TensorNativeDetections
) -> float:
    # sv.box_iou_batch is used here purely as the IoU algorithm over numpy boxes
    # (no sv.Detections involved); the xyxy tensors are read out as float arrays.
    iou = float(
        sv.box_iou_batch(_xyxy_as_numpy(detection_a), _xyxy_as_numpy(detection_b))[0][0]
    )
    if math.isnan(iou):
        iou = 0
    return iou


def agree_on_consensus_for_all_detections_sources(
    detections_from_sources: List[TensorNativeDetections],
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
) -> Tuple[str, bool, Dict[str, float], TensorNativeDetections]:
    if does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources
    ):
        return "undefined", False, {}, _empty_native_detections()
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
    consensus_detections = _merge_native_detections(consensus_detections)
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
    detection: TensorNativeDetections,
    source_id: int,
    detections_from_sources: List[TensorNativeDetections],
    iou_threshold: float,
    class_aware: bool,
    required_votes: int,
    confidence: float,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
    detections_merge_mask_aggregation: MaskAggregationMode,
    detections_already_considered: Set[str],
) -> Tuple[List[TensorNativeDetections], Set[str]]:
    if (
        len(detection)
        and _resolve_detection_id(detection) in detections_already_considered
    ):
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
        # Returning empty list of detections
        return consensus_detections, detections_already_considered
    # Mask handling: object-detection rows have no mask while instance-
    # segmentation rows do. The numpy block padded any missing mask with zeros so
    # the group could stack; here the per-row masks are materialised to numpy and
    # padded to a common shape before aggregation, never round-tripping sv.
    detection_mask = _single_mask(detection)
    overlap_masks = {
        other_source: _single_mask(matched_value[0])
        for other_source, matched_value in detections_with_max_overlap.items()
    }
    if detection_mask is not None:
        for other_source, matched_mask in overlap_masks.items():
            if matched_mask is None:
                overlap_masks[other_source] = np.zeros(detection_mask.shape, dtype=bool)
    else:
        shape = None
        for matched_mask in overlap_masks.values():
            if matched_mask is not None:
                shape = matched_mask.shape
                break
        if shape:
            for other_source, matched_mask in overlap_masks.items():
                if matched_mask is None:
                    overlap_masks[other_source] = np.zeros(shape, dtype=bool)
            detection_mask = np.zeros(shape, dtype=bool)
    group_detections = [detection] + [
        matched_value[0] for matched_value in detections_with_max_overlap.values()
    ]
    group_masks = (
        [detection_mask]
        + [overlap_masks[other_source] for other_source in detections_with_max_overlap]
        if detection_mask is not None
        else None
    )
    merged_detection = merge_detections(
        detections=group_detections,
        masks=group_masks,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
        mask_aggregation_mode=detections_merge_mask_aggregation,
    )
    if float(merged_detection.confidence[0]) < confidence:
        # Returning empty list of detections
        return consensus_detections, detections_already_considered
    consensus_detections.append(merged_detection)
    detections_already_considered.add(_resolve_detection_id(detection))
    for matched_value in detections_with_max_overlap.values():
        detections_already_considered.add(_resolve_detection_id(matched_value[0]))
    return consensus_detections, detections_already_considered


def _single_mask(detection: TensorNativeDetections) -> Optional[np.ndarray]:
    # One detection's mask as a numpy (H, W) bool array, or None for object
    # detection. `detection` is always a single-row prediction here.
    if not isinstance(detection, InstanceDetections) or detection.mask is None:
        return None
    if len(detection) == 0:
        return None
    return instance_mask_to_numpy(detection, 0)


def check_objects_presence_in_consensus_detections(
    consensus_detections: TensorNativeDetections,
    class_aware: bool,
    aggregation_mode: AggregationMode,
    required_objects: Optional[Union[int, Dict[str, int]]],
) -> Tuple[bool, Dict[str, float]]:
    if len(consensus_detections) == 0:
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
    class_names = [
        _resolve_class_name(consensus_detections, index)
        for index in range(len(consensus_detections))
    ]
    class2detections = {}
    for class_name in set(class_names):
        indices = [
            index for index, name in enumerate(class_names) if name == class_name
        ]
        class2detections[class_name] = take_prediction_by_indices(
            consensus_detections, indices
        )
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
    detections: List[TensorNativeDetections],
    masks: Optional[List[np.ndarray]],
    confidence_aggregation_mode: AggregationMode,
    boxes_aggregation_mode: AggregationMode,
    mask_aggregation_mode: MaskAggregationMode,
) -> TensorNativeDetections:
    # `detections` is the native group of overlapping detections (one single-row
    # prediction per voting source). `masks` is the parallel list of numpy masks
    # (already padded to a common shape) or None when no source carried a mask.
    # The aggregation group only needs xyxy / class_id / confidence / metadata
    # (masks come from the `masks` argument), so it is concatenated mask-free.
    group_xyxy, group_class_id, group_confidence, group_metadata, _ = _concat_metadata(
        detections
    )
    group = Detections(
        xyxy=group_xyxy,
        class_id=group_class_id,
        confidence=group_confidence,
        image_metadata=group_metadata,
    )
    class_name, class_id = AGGREGATION_MODE2CLASS_SELECTOR[confidence_aggregation_mode](
        group
    )
    if masks is not None:
        mask_stack = np.stack(masks, axis=0)
        aggregated_mask = np.array(
            [AGGREGATION_MODE2MASKS_AGGREGATOR[mask_aggregation_mode](mask_stack)]
        )
        x1, y1, x2, y2 = sv.mask_to_xyxy(aggregated_mask)[0]
        output_mask = torch.from_numpy(aggregated_mask.astype(bool))
    else:
        output_mask = None
        x1, y1, x2, y2 = AGGREGATION_MODE2BOXES_AGGREGATOR[boxes_aggregation_mode](
            group
        )
    # Per-image state is carried in image_metadata; the merged consensus row is a
    # fresh detection so it gets a new detection_id in bboxes_metadata while the
    # class_names map carries the chosen class. The source image_metadata supplies
    # parent/root coordinates which are shared across the same input image.
    source_metadata = group.image_metadata or {}
    image_metadata = {
        CLASS_NAMES_KEY: {int(class_id): class_name},
        PARENT_ID_KEY: source_metadata.get(PARENT_ID_KEY),
        PREDICTION_TYPE_KEY: "object-detection",
        PARENT_COORDINATES_KEY: source_metadata.get(PARENT_COORDINATES_KEY),
        PARENT_DIMENSIONS_KEY: source_metadata.get(PARENT_DIMENSIONS_KEY),
        ROOT_PARENT_ID_KEY: source_metadata.get(ROOT_PARENT_ID_KEY),
        ROOT_PARENT_COORDINATES_KEY: source_metadata.get(ROOT_PARENT_COORDINATES_KEY),
        ROOT_PARENT_DIMENSIONS_KEY: source_metadata.get(ROOT_PARENT_DIMENSIONS_KEY),
        IMAGE_DIMENSIONS_KEY: source_metadata.get(IMAGE_DIMENSIONS_KEY),
    }
    if SCALING_RELATIVE_TO_PARENT_KEY in source_metadata:
        image_metadata[SCALING_RELATIVE_TO_PARENT_KEY] = source_metadata[
            SCALING_RELATIVE_TO_PARENT_KEY
        ]
    else:
        image_metadata[SCALING_RELATIVE_TO_PARENT_KEY] = 1.0
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in source_metadata:
        image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = source_metadata[
            SCALING_RELATIVE_TO_ROOT_PARENT_KEY
        ]
    else:
        image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 1.0
    xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    class_id_tensor = torch.tensor([class_id], dtype=torch.long)
    confidence_tensor = torch.tensor(
        [
            aggregate_field_values(
                detections=group,
                field="confidence",
                aggregation_mode=confidence_aggregation_mode,
            )
        ],
        dtype=torch.float32,
    )
    bboxes_metadata = [{DETECTION_ID_KEY: str(uuid4())}]
    if output_mask is not None:
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id_tensor,
            confidence=confidence_tensor,
            mask=output_mask,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=xyxy,
        class_id=class_id_tensor,
        confidence=confidence_tensor,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def get_majority_class(detections: TensorNativeDetections) -> Tuple[str, int]:
    class_counts = Counter(
        [
            (_resolve_class_name(detections, index), int(detections.class_id[index]))
            for index in range(len(detections))
        ]
    )
    return class_counts.most_common(1)[0][0]


def get_class_of_most_confident_detection(
    detections: TensorNativeDetections,
) -> Tuple[str, int]:
    confidences: List[float] = (
        detections.confidence.detach().to("cpu").numpy().astype(float).tolist()
    )
    max_confidence_index = confidences.index(max(confidences))
    return (
        _resolve_class_name(detections, max_confidence_index),
        int(detections.class_id[max_confidence_index]),
    )


def get_class_of_least_confident_detection(
    detections: TensorNativeDetections,
) -> Tuple[str, int]:
    confidences: List[float] = (
        detections.confidence.detach().to("cpu").numpy().astype(float).tolist()
    )
    min_confidence_index = confidences.index(min(confidences))
    return (
        _resolve_class_name(detections, min_confidence_index),
        int(detections.class_id[min_confidence_index]),
    )


AGGREGATION_MODE2CLASS_SELECTOR = {
    AggregationMode.MAX: get_class_of_most_confident_detection,
    AggregationMode.MIN: get_class_of_least_confident_detection,
    AggregationMode.AVERAGE: get_majority_class,
}


def get_average_bounding_box(
    detections: TensorNativeDetections,
) -> Tuple[int, int, int, int]:
    if len(detections) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    avg_xyxy = np.mean(_xyxy_as_numpy(detections), axis=0)
    return tuple(avg_xyxy)


def get_smallest_bounding_box(
    detections: TensorNativeDetections,
) -> Tuple[int, int, int, int]:
    areas: List[float] = _native_areas(detections).astype(float).tolist()
    min_area = min(areas)
    min_area_index = areas.index(min_area)
    return tuple(_xyxy_as_numpy(detections)[min_area_index])


def get_largest_bounding_box(
    detections: TensorNativeDetections,
) -> Tuple[int, int, int, int]:
    areas: List[float] = _native_areas(detections).astype(float).tolist()
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    return tuple(_xyxy_as_numpy(detections)[max_area_index])


AGGREGATION_MODE2BOXES_AGGREGATOR = {
    AggregationMode.MAX: get_largest_bounding_box,
    AggregationMode.MIN: get_smallest_bounding_box,
    AggregationMode.AVERAGE: get_average_bounding_box,
}


def get_intersection_mask(mask: np.ndarray) -> np.ndarray:
    return np.all(mask, axis=0)


def get_union_mask(mask: np.ndarray) -> np.ndarray:
    return np.sum(mask, axis=0)


def get_smallest_mask(mask: np.ndarray) -> np.ndarray:
    areas: List[float] = [float(m.sum()) for m in mask]
    min_area = min(areas)
    min_area_index = areas.index(min_area)
    return mask[min_area_index]


def get_largest_mask(mask: np.ndarray) -> np.ndarray:
    areas: List[float] = [float(m.sum()) for m in mask]
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    return mask[max_area_index]


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
    detections: TensorNativeDetections,
    field: str,
    aggregation_mode: AggregationMode = AggregationMode.AVERAGE,
) -> float:
    values = []
    if hasattr(detections, field):
        values = getattr(detections, field)
        if isinstance(values, torch.Tensor):
            values = values.detach().to("cpu").numpy().astype(float).tolist()
        elif isinstance(values, np.ndarray):
            values = values.astype(float).tolist()
    return AGGREGATION_MODE2FIELD_AGGREGATOR[aggregation_mode](values)
