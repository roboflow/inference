"""Atomically replace a managed value only if it equals ``expected``.

Canonical type: ``roboflow_core/state_compare_and_set@v1``.

Each state block runs one atomic single-key operation on the engine's
``managed_state`` resource, in the ``global`` scope or the ``source`` scope
of the running call. ``trigger`` only decides when the block runs; its value
is ignored. Two state steps in a row are two operations, not a transaction.
Workflow ``state`` defaults are initial values written once; after that the
block sees ordinary stored values (a deleted key stays missing).
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictBool, StrictStr, model_validator
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.state import MISSING, ManagedState


class StateCompareAndSetBlock(Block):
    """Atomically replace a managed value only if it equals ``expected``.

    Of several concurrent calls expecting the same value, at most one
    applies. Equality is type-sensitive (``1`` differs from ``1.0``). Each
    side needs exactly one choice, so a definition never relies on an
    implicit ``null``::

        expected: <value>   or   expected_missing: true   (expect no value)
        new: <value>        or   delete: true             (remove the key)

    A selector counts as a given value; ``expected: null`` expects JSON null.
    """

    type = "roboflow_core/state_compare_and_set@v1"
    outputs = {
        "applied": Output(
            BOOLEAN_KIND, description="Whether the value matched and was replaced."
        ),
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
        expected: Any | Ref() = Field(
            default=None,
            description="Expected current value; required unless expected_missing.",
        )
        new: Any | Ref() = Field(
            default=None, description="Value to store; required unless delete."
        )
        expected_missing: StrictBool = Field(
            default=False,
            description="Expect no stored value; excludes 'expected'.",
        )
        delete: StrictBool = Field(
            default=False, description="Delete the key; excludes 'new'."
        )

        @model_validator(mode="after")
        def _one_choice_per_side(self) -> "StateCompareAndSetBlock.Params":
            given = self.model_fields_set
            if ("expected" in given) == self.expected_missing:
                raise ValueError(
                    "give exactly one of 'expected' or 'expected_missing: true'"
                )
            if ("new" in given) == self.delete:
                raise ValueError("give exactly one of 'new' or 'delete: true'")

            return self

    def __init__(self, *, managed_state: ManagedState) -> None:
        self._managed_state = managed_state

    def run(
        self,
        *,
        trigger: Any,
        scope: Literal["global", "source"],
        key: str,
        expected: Any,
        new: Any,
        expected_missing: bool,
        delete: bool,
    ) -> Dict[str, Any]:
        """Replace ``key`` when it holds ``expected``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            expected: Expected current value.
            new: Value to store.
            expected_missing: Expect no stored value instead of ``expected``.
            delete: Delete instead of storing ``new``.

        Returns:
            ``applied``: whether the comparison matched and the change applied.

        Raises:
            StateValueError: When ``expected`` or ``new`` is not portable JSON.
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
        applied = state.compare_and_set(
            key,
            MISSING if expected_missing else expected,
            MISSING if delete else new,
        )

        return {"applied": applied}
