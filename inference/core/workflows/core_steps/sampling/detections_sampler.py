from copy import copy
from typing import Any, Dict, List, Literal, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DEFAULT_OPERAND_NAME,
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    FlowControl,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
"""

SHORT_DESCRIPTION = ""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sampling",
        }
    )
    type: Literal["DetectionsSampler"]
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
    sampling_statement: StatementGroup
    operations_parameters: Dict[
        str, Union[WorkflowImageSelector, WorkflowParameterSelector()]
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image"],
        default_factory=lambda: {},
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


class DetectionsSamplerBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        predictions: List[sv.Detections],
        sampling_statement: StatementGroup,
        operations_parameters: Dict[str, Any],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if DEFAULT_OPERAND_NAME in operations_parameters:
            raise ValueError(
                f"Detected reserved parameter name: {DEFAULT_OPERAND_NAME} declared in `operations_parameters` "
                f"of `DetectionsTransformation` block."
            )
        sampling_function = build_eval_function(definition=sampling_statement)
        operations_parameters_copy = copy(operations_parameters)
        result = []
        for detections in predictions:
            operations_parameters_copy[DEFAULT_OPERAND_NAME] = detections
            should_stay = sampling_function(operations_parameters_copy)
            result.append({"predictions": detections if should_stay else None})
        return result
