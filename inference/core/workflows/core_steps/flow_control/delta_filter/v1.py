from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    WILDCARD_KIND,
    Selector,
    StepSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Allow the execution of workflow to proceed if the input value has changed."
)
LONG_DESCRIPTION: str = """
The Delta Filter is a flow control block that triggers workflow steps only when an input value changes.
It avoids redundant processing and optimizes system efficiency.

    +----------------+      (value changes)       +----------------+
    | Previous Value |  ----------------------->  |   Next Steps    |
    +----------------+                           +----------------+

Key Features:

Change Detection: Tracks input values and only proceeds when a change is detected.
Dynamic Value Support: Handles various input types (e.g., numbers, strings).
Context-Aware Caching: Tracks changes on a per-video basis using video_identifier.

Usage Instructions:
Input Configuration: Set "Input Value" to reference the value to monitor (e.g., counter).
Next Steps Setup: Define steps to execute on value change.

Example Use Case:

A video analysis workflow counts people in the zone. When the count changes, Delta Filter triggers downstream steps (e.g., setting variable in OPC), minimizing redundant processing.
"""


class DeltaFilterManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/delta_filter@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Delta Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-arrow-progress",
                "blockPriority": 3,
            },
        }
    )
    image: WorkflowImageSelector
    value: Selector(kind=[WILDCARD_KIND]) = Field(
        title="Input Value",
        description="Flow will be allowed to continue only if this value changes between frames.",
        examples=["$steps.line_counter.count_in"],
    )
    next_steps: List[StepSelector] = Field(
        description="Steps to execute when the value changes.",
        examples=["$steps.write_to_csv", "$steps.write_to_opc"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class DeltaFilterBlockV1(WorkflowBlock):
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    @classmethod
    def get_manifest(cls) -> Type[DeltaFilterManifest]:
        return DeltaFilterManifest

    def run(
        self,
        image: WorkflowImageData,
        value: Any,
        next_steps: List[StepSelector],
    ) -> BlockResult:
        metadata = image.video_metadata
        video_identifier = metadata.video_identifier
        if self.cache.get(video_identifier) != value:
            self.cache[video_identifier] = value
            return FlowControl(mode="select_step", context=next_steps)
        return FlowControl(mode="terminate_branch")
