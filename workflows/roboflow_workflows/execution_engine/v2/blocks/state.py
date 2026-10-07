"""Native V2 blocks for managed state operations.

Each block performs one atomic single-key operation on the engine's
``managed_state`` resource, in the ``global`` scope or the ``source`` scope of
the running call::

    {"type": "v2/state_increment", "name": "visits",
     "trigger": "$steps.zone.entered", "scope": "source", "key": "visits"}

``trigger`` decides only when the block runs (once per invocation of that
upstream value); its value is ignored. Two state blocks in a
row are two separate atomic operations, not a transaction.

Values are portable JSON (see ``state.codec``); anything else fails the step
with ``StateValueError``. ``scope="source"`` fails with ``StateScopeError``
when the call has no source or is batch-delivering.

Workflow ``state`` defaults are initial values written once; after that the
blocks see ordinary stored values (a deleted key stays missing).
"""

from typing import Any, Dict, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.state import (
    MISSING,
    ManagedState,
    StateScope,
)

__all__ = [
    "StateCompareAndSetBlock",
    "StateGetBlock",
    "StateIncrementBlock",
    "StateSetBlock",
]

StateScopeName = Literal["global", "source"]

_TRIGGER_DESCRIPTION = (
    "Any upstream value; the operation runs once per invocation of it. "
    "The value itself is ignored."
)
_SCOPE_DESCRIPTION = (
    "'global' (shared by the whole execution) or 'source' (the source of this call)."
)


class _StateBlock(Block):
    """Shared constructor of the state blocks: the injected managed state."""

    def __init__(self, *, managed_state: ManagedState) -> None:
        self._managed_state = managed_state

    def _scope(self, scope: StateScopeName) -> StateScope:
        if scope == "global":
            return self._managed_state.global_

        return self._managed_state.source


class StateGetBlock(_StateBlock):
    """Read one managed state value; ``default`` when the key is missing."""

    type = "v2/state_get"
    outputs = {
        "value": Output(description="Stored value, or default when missing."),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(default=None, description=_TRIGGER_DESCRIPTION)
        scope: StateScopeName = Field(default="global", description=_SCOPE_DESCRIPTION)
        key: StrictStr | Ref(STRING_KIND) = Field(description="State key.")
        default: Any = Field(default=None, description="Value when the key is missing.")

    def run(
        self, *, trigger: Any, scope: StateScopeName, key: str, default: Any
    ) -> Dict[str, Any]:
        """Read ``key`` from ``scope``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            default: Value when the key is missing.

        Returns:
            ``value``: a fresh copy of the value.
        """
        value = self._scope(scope).get(key, default)

        return {"value": value}


class StateSetBlock(_StateBlock):
    """Store one managed state value."""

    type = "v2/state_set"
    outputs = {
        "applied": Output(BOOLEAN_KIND, description="Whether the value was stored."),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(default=None, description=_TRIGGER_DESCRIPTION)
        scope: StateScopeName = Field(default="global", description=_SCOPE_DESCRIPTION)
        key: StrictStr | Ref(STRING_KIND) = Field(description="State key.")
        value: Any | Ref() = Field(description="Portable JSON value to store.")
        only_if_absent: StrictBool = Field(
            default=False, description="Store only when no value is stored yet."
        )

    def run(
        self,
        *,
        trigger: Any,
        scope: StateScopeName,
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
        """
        applied = self._scope(scope).set(key, value, only_if_absent=only_if_absent)

        return {"applied": applied}


class StateIncrementBlock(_StateBlock):
    """Atomically add to a managed integer; a missing key counts as 0."""

    type = "v2/state_increment"
    outputs = {
        "value": Output(INTEGER_KIND, description="Value after the increment."),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(default=None, description=_TRIGGER_DESCRIPTION)
        scope: StateScopeName = Field(default="global", description=_SCOPE_DESCRIPTION)
        key: StrictStr | Ref(STRING_KIND) = Field(description="State key.")
        amount: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1, description="Signed 64-bit increment; negative decrements."
        )

    def run(
        self, *, trigger: Any, scope: StateScopeName, key: str, amount: int
    ) -> Dict[str, Any]:
        """Add ``amount`` to ``key``.

        Args:
            trigger: Ignored; only sets the cadence.
            scope: ``global`` or ``source``.
            key: State key.
            amount: Signed 64-bit increment.

        Returns:
            ``value``: the new integer.
        """
        value = self._scope(scope).incr(key, amount)

        return {"value": value}


class StateCompareAndSetBlock(_StateBlock):
    """Atomically replace a managed value only if it equals ``expected``.

    Of several concurrent calls expecting the same value, at most one
    applies. Equality is type-sensitive (``1`` differs from ``1.0``). Each
    side needs exactly one choice, so a definition never relies on an
    implicit ``null``::

        expected: <value>   or   expected_missing: true   (expect no value)
        new: <value>        or   delete: true             (remove the key)

    A selector counts as a given value; ``expected: null`` expects JSON null.
    """

    type = "v2/state_compare_and_set"
    outputs = {
        "applied": Output(
            BOOLEAN_KIND, description="Whether the value matched and was replaced."
        ),
    }

    class Params(BlockParams):
        trigger: None | Ref() = Field(default=None, description=_TRIGGER_DESCRIPTION)
        scope: StateScopeName = Field(default="global", description=_SCOPE_DESCRIPTION)
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

    def run(
        self,
        *,
        trigger: Any,
        scope: StateScopeName,
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
        """
        applied = self._scope(scope).compare_and_set(
            key,
            MISSING if expected_missing else expected,
            MISSING if delete else new,
        )

        return {"applied": applied}
