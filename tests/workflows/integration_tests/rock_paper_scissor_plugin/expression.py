"""
This is just example, test implementation, please do not assume it being fully functional.
This is extremely unsafe block - be aware for injected code execution!
"""

from copy import deepcopy
from typing import List, Literal, Optional, Type, Union, Dict, Any

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, BaseModel

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
    WorkflowImageSelector, WorkflowParameterSelector,
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
    type: Literal["Expression"]
    data: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=dict,
    )
    output: Union[str, PythonCodeBlock]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class ExpressionBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    async def run(
        self,
        data: Dict[str, Any],
        output: Union[str, PythonCodeBlock]
    ) -> BlockResult:
        if isinstance(output, str):
            return {"output": output}
        results = {}
        params = ", ".join(f"{k}={k}" for k in data)
        code = output.code + f"\n\nresult = function({params})"
        exec(code, data, results)
        return {"output": results["result"]}

