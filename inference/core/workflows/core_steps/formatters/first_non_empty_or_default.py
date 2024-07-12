from typing import Any, List, Literal, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.entities.base import Batch, OutputDefinition
from inference.core.workflows.entities.types import StepOutputSelector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Takes input data which may not be present due to filtering or conditional execution and
fills with default value to make it compliant with further processing.
"""

SHORT_DESCRIPTION = "Takes first non-empty data element or default"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal["FirstNonEmptyOrDefault"]
    data: List[StepOutputSelector()] = Field(
        description="Reference data to replace empty values",
        examples=["$steps.my_step.predictions"],
        min_items=1,
    )
    default: Any = Field(
        description="Default value that will be placed whenever there is no data found",
        examples=["empty"],
        default=None,
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class FirstNonEmptyOrDefaultBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        data: Batch[Any],
        default: Any,
    ) -> BlockResult:
        result = default
        for data_element in data:
            if data_element is not None:
                return {"output": data_element}
        return {"output": result}
