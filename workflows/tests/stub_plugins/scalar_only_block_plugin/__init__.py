"""Package-owned scalar-only stub block plugin.

Copy of the root integration stub, rewritten for standalone imports. Use via
`get_plugin_modules` monkeypatch: `["tests.stub_plugins.scalar_only_block_plugin"]`.
"""

from typing import List, Literal, Optional, Type, Union

from pydantic import Field

from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import Selector
from roboflow_workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ScalarOnlyEchoBlockManifest(WorkflowBlockManifest):
    type: Literal["scalar_only_echo"]
    value: Union[Selector(), str] = Field(
        default="foobar",
        description="Scalar value to echo (e.g. from $inputs.param or another scalar step).",
        examples=["$inputs.my_param"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ScalarOnlyEchoBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ScalarOnlyEchoBlockManifest

    def run(self, value: str = "foobar") -> BlockResult:
        return {"output": value}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [ScalarOnlyEchoBlock]
