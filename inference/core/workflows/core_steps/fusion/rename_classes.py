from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.constants import DETECTION_ID_KEY, PARENT_ID_KEY
from inference.core.workflows.entities.base import Batch, OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
TODO
"""

SHORT_DESCRIPTION = "TODO"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal[f"RenameClasses"]
    object_detection_predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Predictions",
        description="Predictions",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    class_name_to_replace: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        Field(
            title="Class Name to Replace",
            description="The class name to replace in the detections.",
            examples=["new_class", "$input.class_name_to_replace"],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            )
        ]


class RenameClassesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        object_detection_predictions: Optional[sv.Detections],
        class_name_to_replace: str,
    ) -> BlockResult:

        class_names = object_detection_predictions.data["class_name"]

        # Create a new numpy array with all class names replaced
        new_class_names = [class_name_to_replace for _ in range(len(class_names))]
        new_class_names_array = np.array(new_class_names, dtype=object)

        # Update the class_name field in the detections data
        object_detection_predictions.data["class_name"] = new_class_names_array

        return {"predictions": object_detection_predictions}
