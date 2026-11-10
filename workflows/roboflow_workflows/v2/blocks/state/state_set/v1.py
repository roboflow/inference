"""Store one managed state value.

Canonical type: ``roboflow_core/state_set@v1``.

Each state block runs one atomic single-key operation on the engine's
``managed_state`` resource, in the ``global`` scope or the ``source`` scope
of the running call. ``trigger`` only decides when the block runs; its value
is ignored. Two state steps in a row are two operations, not a transaction.
Workflow ``state`` defaults are initial values written once; after that the
block sees ordinary stored values (a deleted key stays missing).
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictBool, StrictStr
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.state import ManagedState


class StateSetBlock(Block):
    """Store one managed state value."""

    type = "roboflow_core/state_set@v1"
    outputs = {
        "applied": Output(BOOLEAN_KIND, description="Whether the value was stored."),
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
        value: Any | Ref() = Field(description="Portable JSON value to store.")
        only_if_absent: StrictBool = Field(
            default=False, description="Store only when no value is stored yet."
        )

    def __init__(self, *, managed_state: ManagedState) -> None:
        self._managed_state = managed_state

    def run(
        self,
        *,
        trigger: Any,
        scope: Literal["global", "source"],
        key: str,
        value: Any,
        only_if_absent: bool,
    ) -> Dict[str, Any]:
        """Store ``value`` under ``key``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            value: Portable JSON value.
            only_if_absent: Store only when the key holds no value.

        Returns:
            ``applied``: whether the value was stored.

        Raises:
            StateValueError: When ``value`` is not portable JSON (see
                ``state.codec``).
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
        applied = state.set(key, value, only_if_absent=only_if_absent)

        return {"applied": applied}
