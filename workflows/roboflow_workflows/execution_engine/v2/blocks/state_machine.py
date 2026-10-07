"""Native V2 block that applies a handler-selected state machine transition.

Used inside a handler workflow. The handler named by the transition picks the
target state::

    {"type": "v2/state_machine_set", "name": "finish",
     "machine": "inspection_state", "transition": "finish_review",
     "next_state": "$steps.review.decision"}

``machine`` resolves from the handler's declaring scope; the compiler rejects
the block outside an authorized handler. ``stale`` (the record moved on since
the event that started this handler run) and ``ignored`` (the current state is
not a ``from`` state) are normal results with ``applied=False``.
"""

from typing import Any, Dict

from pydantic import Field, StrictStr
from roboflow_workflows.execution_engine.v2.context import get_execution_context
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import BOOLEAN_KIND, STRING_KIND

__all__ = ["StateMachineSetBlock"]


class StateMachineSetBlock(Block):
    """Move a state machine to a handler-selected state."""

    type = "v2/state_machine_set"
    outputs = {
        "state": Output(STRING_KIND, description="Machine state after the attempt."),
        "applied": Output(
            BOOLEAN_KIND, description="Whether the transition was applied."
        ),
        "outcome": Output(STRING_KIND, description="'applied', 'ignored' or 'stale'."),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(
            default=None,
            description=(
                "Any upstream value; the transition is attempted once per "
                "invocation of it. The value itself is ignored."
            ),
        )
        machine: StrictStr = Field(
            description="Machine name, relative to the handler's workflow.",
            examples=["inspection_state"],
        )
        transition: StrictStr = Field(
            description="Handler-selected transition of that machine.",
            examples=["finish_review"],
        )
        next_state: StrictStr | Ref(STRING_KIND) = Field(
            description="Target state; must be one of the transition's 'to' states.",
            examples=["approved", "$steps.review.decision"],
        )

    def run(
        self, *, trigger: Any, machine: str, transition: str, next_state: str
    ) -> Dict[str, Any]:
        """Attempt the transition for the handler run executing this step.

        Args:
            trigger: Ignored; only sets the cadence.
            machine: Machine name relative to the handler's workflow.
            transition: Transition name.
            next_state: Requested target state.

        Returns:
            ``state``, ``applied`` and ``outcome`` of the attempt.
        """
        result = get_execution_context().set_machine_state(
            machine, transition, next_state
        )
        outputs = {
            "state": result.state,
            "applied": result.outcome == "applied",
            "outcome": result.outcome,
        }

        return outputs
