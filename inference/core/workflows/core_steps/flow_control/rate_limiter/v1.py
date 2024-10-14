from typing import List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import StepSelector, WorkflowImageSelector
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

class RateLimiterManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/rate_limiter@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Rate Limiter",
            "version": "v1",
            "short_description": "Limits the rate at which a branch of the Workflow will fire.",
            "long_description": "This block only continues to execute the next steps once every `seconds` seconds. Otherwise, it terminates the branch.",
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    image: WorkflowImageSelector = Field(
        description="The input image for this step.",
        examples=["$inputs.image"],
    )
    seconds: float = Field(
        description="The minimum number of seconds between allowed executions.",
        examples=[1.0],
        default=1.0,
    )
    next_steps: List[StepSelector] = Field(
        description="Reference to steps which shall be executed if rate limit allows.",
        examples=[["$steps.upload"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

class RateLimiterBlockV1(WorkflowBlock):
    def __init__(self):
        super().__init__()
        self._last_executed_at = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RateLimiterManifest

    def run(
        self,
        image: WorkflowImageData,
        seconds: float,
        next_steps: List[StepSelector],
        **kwargs,
    ) -> BlockResult:
        import time

        current_time = time.time()

        # Extract video identifier; use 'default' if not available
        metadata = image.video_metadata
        video_id = metadata.video_identifier if metadata and metadata.video_identifier else "default"

        last_executed_at = self._last_executed_at.get(video_id)

        if last_executed_at is None or (current_time - last_executed_at) >= seconds:
            # Update the last executed time for this video
            self._last_executed_at[video_id] = current_time
            # Proceed to next steps
            return FlowControl(mode="select_step", context=next_steps)
        else:
            # Terminate the branch
            return FlowControl(mode="terminate_branch")
