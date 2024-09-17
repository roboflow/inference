from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
    OperationDefinition,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
    execute_transformation,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Conditionally filter out model predictions."

OPERATIONS_EXAMPLE = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "classes",
                    },
                },
            ],
        },
    }
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": SHORT_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/detections_filter@v1", "DetectionsFilter"]
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="Definition of filtering operations", examples=[OPERATIONS_EXAMPLE]
    )
    operations_parameters: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=[
            {
                "classes": "$inputs.classes",
            }
        ],
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
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DetectionsFilterBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
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
