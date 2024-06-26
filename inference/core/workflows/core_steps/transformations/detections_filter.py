from typing import Any, Dict, List, Literal, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
    OperationDefinition,
)
from inference.core.workflows.core_steps.transformations.detections_transformation import (
    execute_transformation,
)
from inference.core.workflows.entities.base import Batch, OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Filters out unwanted Bounding Boxes based on conditions specified"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": SHORT_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["DetectionsFilter"]
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType]
    operations_parameters: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image"],
        default_factory=lambda: {},
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

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


class DetectionsFilterBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        predictions: Batch[sv.Detections],
        operations: List[OperationDefinition],
        operations_parameters: Dict[str, Any],
    ) -> BlockResult:
        return execute_transformation(
            predictions=predictions,
            operations=operations,
            operations_parameters=operations_parameters,
        )
