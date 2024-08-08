from typing import Any, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Takes multiple step outputs at data depth level n, concatenate them into list and reduce dimensionality
to level n-1.

Useful in scenarios like:
* aggregation of classification results for dynamically cropped images
* aggregation of OCR results for dynamically cropped images
"""

SHORT_DESCRIPTION = (
    "Collapses dimensionality level by aggregation of nested data into list"
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dimension Collapse",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal["roboflow_core/dimension_collapse@v1", "DimensionCollapse"]
    data: StepOutputSelector() = Field(
        description="Reference to step outputs at depth level n to be concatenated and moved into level n-1.",
        examples=["$steps.ocr_step.results"],
    )

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return -1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output",
                kind=[LIST_OF_VALUES_KIND],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DimensionCollapseBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, data: Batch[Any]) -> BlockResult:
        return {"output": [e for e in data]}
