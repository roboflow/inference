"""
This is just example, test implementation, please do not assume it being fully functional.
This is extremely unsafe block - be aware for injected code execution!
"""

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class PythonCodeBlock(BaseModel):
    type: Literal["PythonBlock"]
    code: str


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "",
        }
    )
    type: Literal["TakeFirstNonEmpty"]
    inputs: List[
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=dict,
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class TakeFirstNonEmptyBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    async def run(
        self,
        inputs: List[Any],
    ) -> BlockResult:
        result = None
        for input_element in inputs:
            if input_element is not None:
                result = input_element
        return {"output": result}
