"""Read one managed state value; ``default`` when the key is missing.

Canonical type: ``roboflow_core/state_get@v1``.

Each state block runs one atomic single-key operation on the engine's
``managed_state`` resource, in the ``global`` scope or the ``source`` scope
of the running call. ``trigger`` only decides when the block runs; its value
is ignored. Two state steps in a row are two operations, not a transaction.
Workflow ``state`` defaults are initial values written once; after that the
block sees ordinary stored values (a deleted key stays missing).
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictStr
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.state import ManagedState


class StateGetBlock(Block):
    """Read one managed state value; ``default`` when the key is missing."""

    type = "roboflow_core/state_get@v1"
    outputs = {
        "value": Output(description="Stored value, or default when missing."),
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
        default: Any = Field(default=None, description="Value when the key is missing.")

    def __init__(self, *, managed_state: ManagedState) -> None:
        self._managed_state = managed_state

    def run(
        self,
        *,
        trigger: Any,
        scope: Literal["global", "source"],
        key: str,
        default: Any,
    ) -> Dict[str, Any]:
        """Read ``key`` from ``scope``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            default: Value when the key is missing.

        Returns:
            ``value``: a fresh copy of the value.

        Raises:
            StateScopeError: When ``scope`` is ``source`` and the call has no
                source or is batch-delivering.
            StateBackendError: When the backend fails.
        """
        state = (
            self._managed_state.global_
            if scope == "global"
            else self._managed_state.source
        )
        value = state.get(key, default)

        return {"value": value}
