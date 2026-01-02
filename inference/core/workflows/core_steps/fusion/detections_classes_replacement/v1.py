import sys
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Replace class labels of detection bounding boxes with classes predicted by a classification model applied to cropped regions, combining generic detection results with specialized classification predictions to enable two-stage detection workflows, fine-grained classification, and class refinement workflows where generic detections are refined with specific class labels from specialized classifiers.

## How This Block Works

This block combines results from a detection model (with bounding boxes and generic classes) with classification predictions (from a specialized classifier applied to cropped regions) to replace generic class labels with specific ones. The block:

1. Receives two inputs with different dimensionality levels:
   - `object_detection_predictions`: Detection results (dimensionality level 1) containing bounding boxes with generic classes (e.g., "dog", "person", "vehicle")
   - `classification_predictions`: Classification results (dimensionality level 2) from a classifier applied to cropped regions of each detection (e.g., "Golden Retriever", "Labrador" for dog detections)
2. Matches classifications to detections:
   - Uses `PARENT_ID_KEY` (detection_id) in classification predictions to link each classification result to its source detection
   - Creates a mapping from detection IDs to classification results
3. Extracts leading class from each classification prediction:

   **For single-label classifications:**
   - Uses the "top" class (predicted class) from the classification result
   - Extracts class name, class ID, and confidence from the classification prediction

   **For multi-label classifications:**
   - Finds the class with the highest confidence score
   - Uses the most confident label as the replacement class
   - Extracts class name, class ID, and confidence from the highest-confidence prediction

4. Handles missing classifications:
   - Detections without corresponding classification predictions are discarded by default
   - If `fallback_class_name` is provided, detections without classifications use the fallback class instead of being discarded
   - Fallback class ID is set to the provided value, or `sys.maxsize` if not specified or negative
5. Filters detections:
   - Keeps only detections that have classification results (or fallback if specified)
   - Removes detections that cannot be matched to classification predictions
6. Replaces class information:
   - Replaces class names in detections with classification class names
   - Replaces class IDs in detections with classification class IDs
   - Replaces confidence scores in detections with classification confidence scores
   - Updates all detection metadata to reflect the new class information
7. Generates new detection IDs:
   - Creates new unique detection IDs for updated detections (prevents ID conflicts)
   - Ensures detection IDs are unique after class replacement
8. Returns updated detections:
   - Outputs detections with replaced classes, maintaining bounding box coordinates and other properties
   - Output dimensionality matches input detection predictions (dimensionality level 1)

The block enables two-stage detection workflows where a generic detection model locates objects and a specialized classification model provides fine-grained labels. This is useful when you need generic localization (e.g., "dog") combined with specific classification (e.g., "Golden Retriever", "German Shepherd") without losing spatial information.

## Common Use Cases

- **Two-Stage Detection and Classification**: Combine generic detection with specialized classification for fine-grained labeling (e.g., detect "dog" then classify breed, detect "vehicle" then classify type, detect "person" then classify age group), enabling two-stage detection workflows
- **Class Refinement**: Refine generic class labels with specific classifications from specialized models (e.g., refine "animal" to specific species, refine "vehicle" to specific models, refine "food" to specific dishes), enabling class refinement workflows
- **Multi-Model Workflows**: Combine detection and classification models to leverage the strengths of both (e.g., use generic detector for localization and specialist classifier for identification, combine coarse and fine-grained models, leverage specialized classifiers with general detectors), enabling multi-model workflows
- **Hierarchical Classification**: Apply hierarchical classification where detection provides high-level classes and classification provides detailed sub-classes (e.g., detect "mammal" then classify species, detect "plant" then classify variety, detect "structure" then classify type), enabling hierarchical classification workflows
- **Crop-Based Classification**: Use classification results from cropped regions to enhance detection results (e.g., classify crops to improve detection labels, apply specialized classifiers to detected regions, refine detections with crop classifications), enabling crop-based classification workflows
- **Fine-Grained Object Recognition**: Enable fine-grained recognition by combining localization and detailed classification (e.g., recognize specific product models, identify specific animal breeds, classify specific vehicle types), enabling fine-grained recognition workflows

## Connecting to Other Blocks

This block receives detection and classification predictions and produces detections with replaced classes:

- **After detection and classification model blocks** to combine generic detection with specialized classification (e.g., object detection + classification to refined detections, detection model + classifier to labeled detections), enabling detection-classification fusion workflows
- **After crop blocks** that create crops from detections for classification (e.g., crop detections then classify crops, create crops for classification then replace classes), enabling crop-classification workflows
- **Before visualization blocks** to display detections with refined classes (e.g., visualize refined detections, display detections with specific labels, show classification-enhanced detections), enabling refined detection visualization workflows
- **Before filtering blocks** to filter detections with refined classes (e.g., filter by specific classes, filter refined detections, apply filters to classified detections), enabling refined detection filtering workflows
- **Before analytics blocks** to perform analytics on refined detections (e.g., analyze specific classes, perform analytics on classified detections, track refined detection metrics), enabling refined detection analytics workflows
- **In workflow outputs** to provide refined detections as final output (e.g., two-stage detection outputs, classification-enhanced detection outputs, refined detection results), enabling refined detection output workflows

## Requirements

This block requires object detection predictions (with bounding boxes) and classification predictions from crops of those bounding boxes. The classification predictions must have `PARENT_ID_KEY` (detection_id) to link classifications to their source detections. The block accepts different dimensionality levels: detection predictions at level 1 and classification predictions at level 2 (from crops). For single-label classifications, the "top" class is used. For multi-label classifications, the most confident class is selected. Detections without classification results are discarded unless `fallback_class_name` is provided. The block outputs detections with replaced classes, class IDs, and confidences, with new detection IDs generated. Output dimensionality matches input detection predictions (level 1).
"""

SHORT_DESCRIPTION = "Replace classes of detections with classes predicted by a chained classification model."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Classes Replacement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-arrow-right-arrow-left",
                "blockPriority": 5,
            },
        }
    )
    type: Literal[
        "roboflow_core/detections_classes_replacement@v1",
        "DetectionsClassesReplacement",
    ]
    object_detection_predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="Detection predictions (object detection, instance segmentation, or keypoint detection) containing bounding boxes with generic class labels that will be replaced with classification results. These detections should correspond to the regions that were cropped and classified. Detections must have detection IDs that match the PARENT_ID_KEY in classification predictions. Detections at dimensionality level 1.",
        examples=["$steps.object_detection_model.predictions", "$steps.instance_segmentation_model.predictions"],
    )
    classification_predictions: Selector(kind=[CLASSIFICATION_PREDICTION_KIND]) = Field(
        title="Classification results for crops",
        description="Classification predictions from a classifier applied to cropped regions of the detections. Each classification result must have PARENT_ID_KEY (detection_id) linking it to its source detection. Supports both single-label (uses 'top' class) and multi-label (uses most confident class) classifications. Classification results at dimensionality level 2 (one classification per crop/detection).",
        examples=["$steps.classification_model.predictions", "$steps.breed_classifier.predictions"],
    )
    fallback_class_name: Union[Optional[str], Selector(kind=[STRING_KIND])] = Field(
        default=None,
        title="Fallback class name",
        description="Optional class name to use for detections that don't have corresponding classification predictions. If not provided (default None), detections without classifications are discarded. If provided, detections without classifications use this fallback class name instead of being removed. Useful for preserving detections when classification fails or is unavailable.",
        examples=[None, "unknown", "unclassified"],
    )
    fallback_class_id: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=None,
        title="Fallback class id",
        description="Optional class ID to use with fallback_class_name for detections without classification predictions. If not specified or negative, the class ID is set to sys.maxsize. Only used when fallback_class_name is provided. Should match the class ID mapping used in your model.",
        examples=[None, 77, 999],
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "object_detection_predictions": 0,
            "classification_predictions": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "object_detection_predictions"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsClassesReplacementBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        object_detection_predictions: Optional[sv.Detections],
        classification_predictions: Optional[Batch[Optional[dict]]],
        fallback_class_name: Optional[str],
        fallback_class_id: Optional[int],
    ) -> BlockResult:
        if object_detection_predictions is None:
            return {"predictions": None}
        if not classification_predictions:
            return {"predictions": sv.Detections.empty()}
        if all(
            p is None or "top" in p and not p["top"] or "predictions" not in p
            for p in classification_predictions
        ):
            return {"predictions": sv.Detections.empty()}
        detection_id_by_class: Dict[str, Optional[Tuple[str, int]]] = {
            prediction[PARENT_ID_KEY]: extract_leading_class_from_prediction(
                prediction=prediction,
                fallback_class_name=fallback_class_name,
                fallback_class_id=fallback_class_id,
            )
            for prediction in classification_predictions
            if prediction is not None
        }
        detections_to_remain_mask = [
            detection_id_by_class.get(detection_id) is not None
            for detection_id in object_detection_predictions.data[DETECTION_ID_KEY]
        ]
        selected_object_detection_predictions = object_detection_predictions[
            detections_to_remain_mask
        ]
        replaced_class_names = np.array(
            [
                detection_id_by_class[detection_id][0]
                for detection_id in selected_object_detection_predictions.data[
                    DETECTION_ID_KEY
                ]
            ]
        )
        replaced_class_ids = np.array(
            [
                detection_id_by_class[detection_id][1]
                for detection_id in selected_object_detection_predictions.data[
                    DETECTION_ID_KEY
                ]
            ]
        )
        replaced_confidences = np.array(
            [
                detection_id_by_class[detection_id][2]
                for detection_id in selected_object_detection_predictions.data[
                    DETECTION_ID_KEY
                ]
            ]
        )
        selected_object_detection_predictions.class_id = replaced_class_ids
        selected_object_detection_predictions.confidence = replaced_confidences
        selected_object_detection_predictions.data[CLASS_NAME_DATA_FIELD] = (
            replaced_class_names
        )
        selected_object_detection_predictions.data[DETECTION_ID_KEY] = np.array(
            [f"{uuid4()}" for _ in range(len(selected_object_detection_predictions))]
        )
        return {"predictions": selected_object_detection_predictions}


def extract_leading_class_from_prediction(
    prediction: dict,
    fallback_class_name: Optional[str] = None,
    fallback_class_id: Optional[int] = None,
) -> Optional[Tuple[str, int, float]]:
    if "top" in prediction:
        if not prediction.get("predictions") and not fallback_class_name:
            return None
        elif not prediction.get("predictions") and fallback_class_name:
            try:
                fallback_class_id = int(fallback_class_id)
            except ValueError:
                fallback_class_id = None
            if fallback_class_id is None or fallback_class_id < 0:
                fallback_class_id = sys.maxsize
            return fallback_class_name, fallback_class_id, 0
        class_name = prediction["top"]
        matching_class_ids = [
            (p["class_id"], p["confidence"])
            for p in prediction["predictions"]
            if p["class"] == class_name
        ]
        if len(matching_class_ids) != 1:
            raise ValueError(f"Could not resolve class id for prediction: {prediction}")
        return class_name, matching_class_ids[0][0], matching_class_ids[0][1]
    predicted_classes = prediction.get("predicted_classes", [])
    if not predicted_classes:
        return None
    max_confidence, max_confidence_class_name, max_confidence_class_id = (
        None,
        None,
        None,
    )
    for class_name, prediction_details in prediction["predictions"].items():
        current_class_confidence = prediction_details["confidence"]
        current_class_id = prediction_details["class_id"]
        if max_confidence is None:
            max_confidence = current_class_confidence
            max_confidence_class_name = class_name
            max_confidence_class_id = current_class_id
            continue
        if max_confidence < current_class_confidence:
            max_confidence = current_class_confidence
            max_confidence_class_name = class_name
            max_confidence_class_id = current_class_id
    if not max_confidence:
        return None
    return max_confidence_class_name, max_confidence_class_id, max_confidence
