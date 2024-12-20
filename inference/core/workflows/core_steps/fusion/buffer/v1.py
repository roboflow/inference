from typing import Any, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Returns an array of the last `length` values passed to it. The newest
elements are added to the beginning of the array.

Useful for keeping a sliding window of images or detections for
later processing, visualization, or comparison.
"""

SHORT_DESCRIPTION = "Returns an array of the last `length` values passed to it."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Buffer",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-layer-group",
            },
        }
    )
    type: Literal["roboflow_core/buffer@v1", "Buffer"]
    data: Selector(
        kind=[WILDCARD_KIND, LIST_OF_VALUES_KIND, IMAGE_KIND],
    ) = Field(
        description="Reference to step outputs at depth level n to be concatenated and moved into level n-1.",
        examples=["$steps.visualization"],
    )
    length: int = Field(
        description="The number of elements to keep in the buffer. Older elements will be removed.",
        examples=[5],
    )
    pad: bool = Field(
        description="If True, the end of the buffer will be padded with `None` values so its size is always exactly `length`.",
        default=False,
        examples=[True],
    )

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
        return ">=1.3.0,<2.0.0"


class BufferBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, data: Any, length: int, pad: bool) -> BlockResult:
        self.buffer.insert(0, data)
        if len(self.buffer) > length:
            self.buffer = self.buffer[:length]

        if pad:
            while len(self.buffer) < length:
                self.buffer.append(None)

        return {"output": self.buffer}
