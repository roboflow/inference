"""Control block used by the native V2 image workflows."""

from typing import List

from pydantic import Field, StrictBool
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.kinds import BOOLEAN_KIND


class ContinueIfBlock(Block):
    """Continue to ``next_steps`` only when ``condition`` is true.

    The decision applies per invocation: bound to a per-crop boolean, each crop
    admits or denies its own continuation.
    """

    type = "v2/continue_if"

    class Params(BlockParams):
        condition: StrictBool | Ref(BOOLEAN_KIND) = Field(
            description="Decision; true selects every step in next_steps."
        )
        next_steps: List[StepRef] = Field(
            description="Steps that run only when the condition holds.",
            examples=[["$steps.invert"]],
        )

    def run(self, *, condition: bool, next_steps: List[str]) -> Select:
        """Select ``next_steps`` when ``condition`` holds.

        Args:
            condition: Decision for this invocation.
            next_steps: Step selectors as written in the step parameters.

        Returns:
            ``Select(next_steps)`` when ``condition`` is true, else ``Stop()``.
        """
        decision = Select(next_steps) if condition else Stop()

        return decision
