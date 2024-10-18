from datetime import datetime
from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
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
"""


class RateLimiterManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/rate_limiter@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Rate Limiter",
            "version": "v1",
            "short_description": "Limits the rate at which a branch of the Workflow will fire.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    cooldown_seconds: float = Field(
        description="The minimum number of seconds between allowed executions.",
        examples=[1.0],
        default=1.0,
        ge=0.0,
    )
    depends_on: Union[
        WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()
    ] = Field(
        description="Reference to any output of the the step which immediately preceeds this branch.",
        examples=["$steps.model"],
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
        return ">=1.2.0,<2.0.0"


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
    ) -> BlockResult:
        current_time = datetime.now()
        should_throttle = False
        if self._last_executed_at is not None:
            should_throttle = (
                current_time - self._last_executed_at
            ).total_seconds() < cooldown_seconds
        if should_throttle:
            return FlowControl(mode="terminate_branch")
        self._last_executed_at = current_time
        return FlowControl(mode="select_step", context=next_steps)
