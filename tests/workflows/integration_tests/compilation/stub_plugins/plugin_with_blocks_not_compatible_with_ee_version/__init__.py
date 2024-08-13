from typing import List, Literal, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    type: Literal["IncompatibleBlock"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="some")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return "==0.0.0"


class BlockBlock(WorkflowBlock):

    def __init__(self, required_but_not_declared_parameter: str):
        self._required_but_not_declared_parameter = required_but_not_declared_parameter

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self) -> BlockResult:
        pass


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [BlockBlock]
