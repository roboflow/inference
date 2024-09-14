from typing import Dict, List, Literal, Optional, Tuple, Type
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
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Combine results of detection model with classification results performed separately for 
each and every bounding box. 

Bounding boxes without top class predicted by classification model are discarded, 
for multi-label classification results, most confident label is taken as bounding box
class.  
"""

SHORT_DESCRIPTION = "Replaces classes of detections with classes predicted by a chained classification model"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Classes Replacement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal[
        "roboflow_core/detections_classes_replacement@v1",
        "DetectionsClassesReplacement",
    ]
    object_detection_predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="The output of a detection model describing the bounding boxes that will have classes replaced.",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    classification_predictions: StepOutputSelector(
        kind=[CLASSIFICATION_PREDICTION_KIND]
    ) = Field(
        title="Classification results for crops",
        description="The output of classification model for crops taken based on RoIs pointed as the other parameter",
        examples=["$steps.my_classification_model.predictions"],
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
        return ">=1.0.0,<2.0.0"


class DetectionsClassesReplacementBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        object_detection_predictions: Optional[sv.Detections],
        classification_predictions: Optional[Batch[Optional[dict]]],
    ) -> BlockResult:
        if object_detection_predictions is None:
            return {"predictions": None}
        if classification_predictions is None:
            return {"predictions": sv.Detections.empty()}
        detection_id_by_class: Dict[str, Optional[Tuple[str, int]]] = {
            prediction[PARENT_ID_KEY]: extract_leading_class_from_prediction(
                prediction=prediction
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
) -> Optional[Tuple[str, int, float]]:
    if "top" in prediction:
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
