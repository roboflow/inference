from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
    take_prediction_by_mask,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
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
    DICTIONARY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Filter detections by applying a per-class confidence threshold."

LONG_DESCRIPTION = """
Filter detection predictions by applying a different confidence threshold to each class, keeping only detections whose confidence meets or exceeds the threshold configured for their class (with a configurable fallback threshold for classes that are not listed).

## How This Block Works

This block applies class-aware confidence filtering to detection predictions, enabling precise control over which detections are retained based on per-class quality requirements. The block:

1. Takes detection predictions (object detection, instance segmentation, or keypoint detection) and a dictionary mapping class names to confidence thresholds
2. Iterates through each detection, looking up the threshold associated with the detection's class name
3. If the class is not present in the dictionary, falls back to the configurable `default_threshold` value
4. Keeps only the detections whose confidence is greater than or equal to the resolved threshold
5. Returns the filtered detections while preserving all original metadata (class ids, masks, keypoints, tracker ids, etc.)

Unlike a single global confidence threshold, this block lets you demand high-confidence predictions for classes that are prone to false positives while keeping a more permissive threshold for classes that are harder to detect. Unlike the generic detections filter, it exposes a purpose-built dictionary input that maps cleanly to a simple `{"class_name": threshold}` JSON object.

## Common Use Cases

- **Noise-prone classes**: Demand very high confidence (e.g. 0.9) for classes that frequently produce false positives, while accepting lower confidence for well-behaved classes
- **Hard-to-detect classes**: Lower the threshold for classes that the model rarely detects with high confidence so that they are not filtered out entirely
- **Production-grade filtering**: Apply domain-specific thresholds tuned during evaluation so that downstream analytics, alerts, or counting blocks only see detections that meet the project's quality bar
- **Multi-class pipelines**: Combine with object detection models that predict many classes at once when a single global confidence threshold is too coarse

## Connecting to Other Blocks

The filtered predictions from this block can be connected to:

- **Visualization blocks** (Bounding Box Visualization, Label Visualization, Polygon Visualization) to render only detections that cleared their per-class threshold
- **Counting and analytics blocks** (Line Counter, Time in Zone, Velocity) so that metrics reflect only high-quality detections
- **Tracking blocks** (Byte Tracker) so that tracker associations are not polluted by low-confidence noise
- **Storage or sink blocks** (Roboflow Dataset Upload, Webhook Sink, CSV Formatter) so that only detections meeting the quality bar are persisted or transmitted
- **Downstream transformation blocks** (Dynamic Crop, Detection Offset) for subsequent processing on the filtered subset
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Per-Class Confidence Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-filter",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/per_class_confidence_filter@v1"]
    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions to filter. Each detection is kept only if its confidence is greater than or equal to the threshold configured for its class (with a fallback to default_threshold for classes that are not listed in class_thresholds).",
        examples=["$steps.object_detection_model.predictions"],
    )
    class_thresholds: Union[
        Dict[str, float],
        Selector(kind=[DICTIONARY_KIND]),
    ] = Field(
        description="Mapping of class name to minimum confidence threshold. Detections whose class name is present in this dictionary are kept only if their confidence is at least the corresponding threshold. Classes not present fall back to default_threshold. Thresholds should be in the [0.0, 1.0] range.",
        examples=[{"person": 0.98, "car": 0.5}, "$inputs.class_thresholds"],
    )
    default_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.3,
        description="Confidence threshold applied to detections whose class name is not listed in class_thresholds. Must be in the [0.0, 1.0] range.",
        examples=[0.3, "$inputs.default_threshold"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class PerClassConfidenceFilterBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[TensorNativePrediction],
        class_thresholds: Dict[str, Any],
        default_threshold: float = 0.3,
    ) -> BlockResult:
        return [
            {
                "predictions": filter_detections_by_class_confidence(
                    prediction=prediction,
                    class_thresholds=class_thresholds,
                    default_threshold=default_threshold,
                )
            }
            for prediction in predictions
        ]


def filter_detections_by_class_confidence(
    prediction: TensorNativePrediction,
    class_thresholds: Dict[str, Any],
    default_threshold: float = 0.3,
) -> TensorNativePrediction:
    if prediction is None:
        return prediction
    # The keypoint-detection kind is a (KeyPoints, Optional[Detections]) tuple; the
    # bbox confidences live on the Detections component. Standalone Detections /
    # InstanceDetections carry confidence directly.
    if isinstance(prediction, tuple):
        _, detections = prediction
    else:
        detections = prediction
    if detections is None:
        return prediction
    confidences = detections.confidence
    if confidences is None or int(confidences.shape[0]) == 0:
        return prediction
    class_names = _resolve_class_names(detections)
    thresholds = {str(k): float(v) for k, v in (class_thresholds or {}).items()}
    default = float(default_threshold)
    # Per-row thresholds are host data (class names live in image_metadata), so
    # they are assembled on host in ONE pass and shipped to the confidence
    # tensor's device in ONE transfer; the comparison then runs vectorised
    # on-device and the resulting torch mask feeds take_prediction_by_mask's
    # device-native selection - the previous per-row `float(confidences[i])`
    # loop synced device->host once per detection.
    per_row_thresholds = [
        thresholds.get(str(class_names[i] if i < len(class_names) else None), default)
        for i in range(int(confidences.shape[0]))
    ]
    keep = confidences >= torch.as_tensor(
        per_row_thresholds, dtype=confidences.dtype, device=confidences.device
    )
    if bool(keep.all()):
        return prediction
    return take_prediction_by_mask(prediction, keep)


def _resolve_class_names(
    detections: TensorNativeDetections,
) -> List[Optional[str]]:
    """Per-box class NAME for each detection, preferring the canonical
    ``image_metadata[CLASS_NAMES_KEY]`` (class_id -> name) map and falling back
    to ``bboxes_metadata[i]["class"]`` when the map is absent or lacks the id."""
    number_of_detections = int(detections.confidence.shape[0])
    name_by_class_id: Dict[int, str] = {}
    if detections.image_metadata:
        name_by_class_id = detections.image_metadata.get(CLASS_NAMES_KEY) or {}
    bboxes_metadata = detections.bboxes_metadata or []
    class_ids = detections.class_id
    resolved: List[Optional[str]] = []
    for i in range(number_of_detections):
        class_name: Optional[str] = None
        if class_ids is not None and i < int(class_ids.shape[0]):
            class_name = name_by_class_id.get(int(class_ids[i]))
        if class_name is None and i < len(bboxes_metadata):
            entry = bboxes_metadata[i] or {}
            class_name = entry.get(CLASS_NAME_KEY)
        resolved.append(class_name)
    return resolved
