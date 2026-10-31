"""
Stub plugin exposing a workflow block that accepts only scalar (non-batch) inputs.

Use in tests by patching blocks_loader.get_plugin_modules to return
["tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin"],
then reference the step in your workflow JSON with "type": "scalar_only_echo".
"""

from typing import List, Literal, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ScalarOnlyEchoBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that accepts only scalar inputs (no batch parameters)."""

    type: Literal["scalar_only_echo"]
    # Union allows runtime validator to set resolved input value (e.g. "hello") on this field
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

    # No get_parameters_accepting_batches() / get_parameters_accepting_batches_and_scalars()
    # → accepts_batch_input() is False → scalar-only block


class ScalarOnlyEchoBlock(WorkflowBlock):
    """Block that accepts only scalars and echoes the input value."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ScalarOnlyEchoBlockManifest

    def run(self, value: str = "foobar") -> BlockResult:
        return {"output": value}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [ScalarOnlyEchoBlock]
