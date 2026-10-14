"""Atomically add to a managed integer; a missing key counts as 0.

Canonical type: ``roboflow_core/state_increment@v1``.

Each state block runs one atomic single-key operation on the engine's
``managed_state`` resource, in the ``global`` scope or the ``source`` scope
of the running call. ``trigger`` only decides when the block runs; its value
is ignored. Two state steps in a row are two operations, not a transaction.
Workflow ``state`` defaults are initial values written once; after that the
block sees ordinary stored values (a deleted key stays missing).

Example::

    {"type": "roboflow_core/state_increment@v1", "name": "visits",
     "trigger": "$steps.zone.entered", "scope": "source", "key": "visits"}
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.state import ManagedState


class StateIncrementBlock(Block):
    """Atomically add to a managed integer; a missing key counts as 0."""

    type = "roboflow_core/state_increment@v1"
    outputs = {
        "value": Output(INTEGER_KIND, description="Value after the increment."),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(
            default=None,
            description=(
                "Any upstream value; the operation runs once per invocation of it. "
                "The value itself is ignored."
            ),
        )
        scope: Literal["global", "source"] = Field(
            default="global",
            description=(
                "'global' (shared by the whole execution) or 'source' (the source "
                "of this call)."
            ),
        )
        key: StrictStr | Ref(STRING_KIND) = Field(description="State key.")
        amount: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1, description="Signed 64-bit increment; negative decrements."
        )

    def __init__(self, *, managed_state: ManagedState) -> None:
        self._managed_state = managed_state

    def run(
        self,
        *,
        trigger: Any,
        scope: Literal["global", "source"],
        key: str,
        amount: int,
    ) -> Dict[str, Any]:
        """Add ``amount`` to ``key``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            amount: Signed 64-bit increment.

        Returns:
            ``value``: the new integer.

        Raises:
            StateValueError: When ``amount`` is not a signed 64-bit integer.
            StateTypeError: When the stored value is not an integer.
            StateOverflowError: When the result leaves the signed 64-bit
                range; the value is unchanged.
            StateScopeError: When ``scope`` is ``source`` and the call has no
                source or is batch-delivering.
            StateBackendError: When the backend fails; not applied.
            StateOutcomeUnknownError: When the reply was lost; maybe applied.
        """
        state = (
            self._managed_state.global_
            if scope == "global"
            else self._managed_state.source
        )
        value = state.incr(key, amount)

        return {"value": value}
