"""
Stub plugin exposing a workflow block that accepts only one optional batch input (no scalars).

Use in tests by patching blocks_loader.get_plugin_modules to return
["tests.workflows.integration_tests.execution.stub_plugins.batch_only_block_plugin"],
then reference the step in your workflow JSON with "type": "batch_only_echo".
"""

from typing import List, Literal, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BatchOnlyEchoBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that accepts only one optional batch input (no scalars)."""

    type: Literal["batch_only_echo"]
    items: Optional[
        Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]
    ] = Field(
        default=None,
        description="Optional batch of values to echo (e.g. from $inputs.names).",
        examples=["$inputs.names"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["items"]

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BatchOnlyEchoBlock(WorkflowBlock):
    """Block that accepts only one optional batch input and echoes each element."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BatchOnlyEchoBlockManifest

    def run(
        self,
        items: Optional[List[str]] = None,
    ) -> BlockResult:
        if not items:
            return [{"output": "no_batch"}]
        return [{"output": x} for x in items]


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [BatchOnlyEchoBlock]
