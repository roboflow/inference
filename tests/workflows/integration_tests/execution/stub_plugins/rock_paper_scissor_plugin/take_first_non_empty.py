"""
This is just example, test implementation, please do not assume it being fully functional.
This is extremely unsafe block - be aware for injected code execution!
"""

from typing import Any, List, Literal, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


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

    def run(
        self,
        inputs: List[Any],
    ) -> BlockResult:
        result = None
        for input_element in inputs:
            if input_element is not None:
                result = input_element
                break
        return {"output": result}
