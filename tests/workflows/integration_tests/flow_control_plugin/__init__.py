import random
from typing import Any, Dict, List, Literal, Tuple, Type, Union

from pydantic import Field

from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import FlowControl, StepSelector
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ABTestManifest(WorkflowBlockManifest):
    type: Literal["ABTest"]
    name: str = Field(description="name field")
    a_step: StepSelector
    b_step: StepSelector

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ABTestBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ABTestManifest

    async def run_locally(
        self,
        a_step: StepSelector,
        b_step: StepSelector,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        choice = a_step
        if random.random() > 0.5:
            choice = b_step
        return [], FlowControl(mode="select_step", context=choice)


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [ABTestBlock]
