from typing import Any, Dict, List, Literal, Tuple, Type, Union

from pydantic import Field

from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    FlowControl,
    Kind,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

MY_KIND_1 = Kind(name="1")
MY_KIND_2 = Kind(name="2")
MY_KIND_3 = Kind(name="3")


class Block1Manifest(WorkflowBlockManifest):
    type: Literal["BlockManifest"]
    name: str = Field(description="name field")
    field_1: Union[bool, WorkflowParameterSelector(kind=[MY_KIND_1])]
    field_2: Union[str, StepOutputSelector(kind=[MY_KIND_2])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output_1", kind=[MY_KIND_1])]


class Block1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Block1Manifest

    async def run_locally(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class Block2Manifest(WorkflowBlockManifest):
    type: Literal["BlockManifest"]
    name: str = Field(description="name field")
    field_1: List[StepOutputSelector(kind=[MY_KIND_1])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output_1", kind=[MY_KIND_3]),
            OutputDefinition(name="output_2", kind=[MY_KIND_2]),
        ]


class Block2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Block2Manifest

    async def run_locally(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        Block1,
        Block2,
    ]
