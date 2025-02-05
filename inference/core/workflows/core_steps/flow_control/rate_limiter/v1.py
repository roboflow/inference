from datetime import datetime
from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
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

LONG_DESCRIPTION = """
The **Rate Limiter** block controls the execution frequency of a branch within a Workflow by enforcing a 
cooldown period. It ensures that the connected steps do not run more frequently than a specified interval, 
helping to manage resource usage and prevent over-execution.

### Block usage

**Rate Limiter** is useful when you have two blocks that are directly connected, as shown below:

--- input_a --> ┌───────────┐                    ┌───────────┐
--- input_b --> │   step_1  │ -->  output_a -->  │   step_2  │
--- input_c --> └───────────┘                    └───────────┘

If you want to throttle the *Step 2* execution rate - you should apply rate limiter in between:

* keep the existing blocks configuration as is (do not change connections)

* set `depends_on` reference of **Rate Limiter** into `output_a`

* set `next_steps` reference to be a list referring to `[$steps.step_2]`

* adjust `cooldown_seconds` to specify what is the number of seconds that must be awaited before next time
when `step_2` is fired 


!!! warning "Cooldown limitations"

    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect for processing HTTP requests.  
    
"""


class RateLimiterManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/rate_limiter@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Rate Limiter",
            "version": "v1",
            "short_description": "Limits the rate at which a branch of the Workflow will run.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-clock",
                "blockPriority": 2,
                "popular": True,
            },
        }
    )
    cooldown_seconds: float = Field(
        description="The minimum number of seconds between allowed executions.",
        examples=[1.0],
        default=1.0,
        ge=0.0,
    )
    depends_on: Selector() = Field(
        description="Step immediately preceding this block.",
        examples=["$steps.model"],
    )
    next_steps: List[StepSelector] = Field(
        description="Steps to execute if allowed by the rate limit.",
        examples=[["$steps.upload"]],
    )
    video_reference_image: Optional[WorkflowImageSelector] = Field(
        description="Reference to a video frame to use for timestamp generation (if running faster than realtime on recorded video).",
        examples=["$inputs.image"],
        default=None,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RateLimiterBlockV1(WorkflowBlock):
    def __init__(self):
        super().__init__()
        self._last_executed_at: Optional[datetime] = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RateLimiterManifest

    def run(
        self,
        cooldown_seconds: float,
        depends_on: any,
        next_steps: List[StepSelector],
        video_reference_image: Optional[WorkflowImageData] = None,
    ) -> BlockResult:
        current_time = datetime.now()
        try:
            metadata = video_reference_image.video_metadata
            current_time = datetime.fromtimestamp(
                1 / metadata.fps * metadata.frame_number
            )
        except Exception:
            # reference not passed, metadata not set, or not a video frame
            pass

        should_throttle = False
        if self._last_executed_at is not None:
            should_throttle = (
                current_time - self._last_executed_at
            ).total_seconds() < cooldown_seconds
        if should_throttle:
            return FlowControl(mode="terminate_branch")
        self._last_executed_at = current_time
        return FlowControl(mode="select_step", context=next_steps)
