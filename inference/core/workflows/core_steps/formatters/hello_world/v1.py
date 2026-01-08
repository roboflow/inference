from typing import List, Literal, Optional, Type

from pydantic import ConfigDict

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Hello World",
            "version": "v1",
            "short_description": "Prints hello world and returns a greeting message.",
            "long_description": "A simple block that prints 'hello world' to the console and returns a greeting message as output.",
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-hand-wave",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/hello_world@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="message")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class HelloWorldBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self) -> BlockResult:
        print("hello world")
        return {"message": "hello world"}
