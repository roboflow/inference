from copy import copy
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DEFAULT_OPERAND_NAME,
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.core_steps.common.utils import (
    grab_batch_parameters,
    grab_non_batch_parameters,
)
from inference.core.workflows.entities.base import Batch, OutputDefinition
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

SHORT_DESCRIPTION = (
    "Exclude detections from further processing based on configurable conditions"
)

LONG_DESCRIPTION = """
Block let user to define wide range of rules dictating which predictions should be passed to further
processing and which should be rejected.

Using this block in workflow may be beneficial in pair with sinks to avoid extensive data transfers. 
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sampling",
        }
    )
    type: Literal["DetectionsRateLimiter"]
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
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
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


class DetectionsRateLimiterBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        predictions: Batch[Optional[sv.Detections]],
        sampling_statement: StatementGroup,
        operations_parameters: Dict[str, Any],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if DEFAULT_OPERAND_NAME in operations_parameters:
            raise ValueError(
                f"Detected reserved parameter name: {DEFAULT_OPERAND_NAME} declared in `operations_parameters` "
                f"of `DetectionsTransformation` block."
            )
        sampling_function = build_eval_function(definition=sampling_statement)
        batch_parameters = grab_batch_parameters(
            operations_parameters=operations_parameters,
            main_batch_size=len(predictions),
        )
        non_batch_parameters = grab_non_batch_parameters(
            operations_parameters=operations_parameters,
        )
        batch_parameters_keys = list(batch_parameters.keys())
        batches_to_align = [predictions] + [
            batch_parameters[k] for k in batch_parameters_keys
        ]
        results = []
        for payload in Batch.zip_nonempty(batches=batches_to_align):
            operations_parameters_copy = copy(non_batch_parameters)
            operations_parameters_copy[DEFAULT_OPERAND_NAME] = payload[0]
            for key, value in zip(batch_parameters_keys, payload[1:]):
                operations_parameters_copy[key] = value
            should_stay = sampling_function(operations_parameters_copy)
            results.append({"predictions": payload[0] if should_stay else None})
        return Batch.align_batches_results(
            batches=batches_to_align,
            results=results,
            null_element={"predictions": None},
        )
