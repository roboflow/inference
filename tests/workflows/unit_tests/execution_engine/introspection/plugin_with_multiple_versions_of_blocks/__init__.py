from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    Kind,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

MY_KIND_1 = Kind(name="1")
MY_KIND_2 = Kind(name="2")
MY_KIND_3 = Kind(name="3")


class Block1V1Manifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Block1",
            "version": "v1",
        }
    )
    type: Literal["Block1Manifest@v1"]
    name: str = Field(description="name field")
    field_1: Union[bool, WorkflowParameterSelector(kind=[MY_KIND_1])]
    field_2: Union[str, StepOutputSelector(kind=[MY_KIND_2])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output_1", kind=[MY_KIND_1])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class Block1V1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Block1V1Manifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class Block1V2Manifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Block1",
            "version": "v2",
        }
    )
    type: Literal["Block1Manifest@v2"]
    name: str = Field(description="name field")
    field_1: Union[bool, WorkflowParameterSelector(kind=[MY_KIND_1])]
    field_2: Union[str, StepOutputSelector(kind=[MY_KIND_2])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output_1", kind=[MY_KIND_1])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class Block1V2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Block1V2Manifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class Block2Manifest(WorkflowBlockManifest):
    type: Literal["Block2Manifest"]
    name: str = Field(description="name field")
    field_1: List[StepOutputSelector(kind=[MY_KIND_1])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output_1", kind=[MY_KIND_3]),
            OutputDefinition(name="output_2", kind=[MY_KIND_2]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class Block2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Block2Manifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        Block1V1,
        Block1V2,
        Block2,
    ]
