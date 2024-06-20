from typing import List, Literal, Type

from pydantic import Field

from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ABTestManifest(WorkflowBlockManifest):
    type: Literal["FaultyInit"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class FaultyInitBlock(WorkflowBlock):

    def __init__(self, required_but_not_declared_parameter: str):
        self._required_but_not_declared_parameter = required_but_not_declared_parameter

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ABTestManifest

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    async def run(self) -> BlockResult:
        pass


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [FaultyInitBlock]
