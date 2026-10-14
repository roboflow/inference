"""Control block used by the native V2 image workflows."""

from typing import Any, List, Mapping
from pydantic import Field, StrictBool, model_validator
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Ref,
    Select,
    StepRef,
    Stop,
)


class ContinueIfBlock(Block):
    """Gate on a ready boolean; for V1 queries use ``roboflow_core/continue_if@v1``.

    The decision applies per invocation: bound to a per-crop boolean, each crop
    admits or denies its own continuation.

    ``condition`` accepts a literal boolean or a selector of a boolean flag.
    This gate has no query evaluation or ``stop_delay``.
    """

    type = "roboflow_core/continue_if_boolean@v1"

    class Params(BlockParams):
        @model_validator(mode="before")
        @classmethod
        def _reject_query_condition(cls, values: Any) -> Any:
            if isinstance(values, Mapping) and "condition_statement" in values:
                raise ValueError(
                    "condition_statement is a field of roboflow_core/continue_if@v1 "
                    "(V1 query condition). roboflow_core/continue_if_boolean@v1 takes a boolean condition."
                )

            return values

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
