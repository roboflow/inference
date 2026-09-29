"""Author exercise: ordinary numeric V2 blocks written without engine internals.

This module shows what a block author writes. It imports only the public V2
package surface (``Batch``, ``BlockContract``, ``InputSpec``, ``OutputSpec``,
``Registry``) and never the compiler, executor or buffer carrier.

Blocks (all payloads are plain Python numbers of kind ``number``):

* ``demo/scale``: leaf. ``scaled = value * factor``. Works unchanged at any
  nesting depth because it declares an ``item`` view.
* ``demo/expand``: expander. Emits ``value + offset`` for each configured
  offset, omitting children above ``limit`` while keeping the offset's local
  index. Declares an appended dynamic axis ``children``.
* ``demo/sum_with_parent``: parent+child reducer. Receives one parent item and
  the trailing group of children; returns ``total`` and ``count`` collapsed to
  the parent level. An empty child group returns ``parent * parent_weight``
  and count 0.
* ``demo/broken``: deliberately misbehaving leaf used by the
  ``invalid-bindings`` scenario to show runtime diagnostics.

``register_author_blocks`` adds them to any registry and returns a call
counter so the demo can prove that invalid definitions fail before any
``run()`` call.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from roboflow_workflows.execution_engine.v2 import (
    Batch,
    BlockContract,
    InputSpec,
    OutputSpec,
    Registry,
)

NUMBER_KIND = "number"

SCALE_BLOCK = "demo/scale"
EXPAND_BLOCK = "demo/expand"
SUM_WITH_PARENT_BLOCK = "demo/sum_with_parent"
BROKEN_BLOCK = "demo/broken"

BROKEN_MODES = ("empty_mapping", "missing_output", "raise", "wrong_kind")


def is_number(payload: Any) -> bool:
    """Accept plain ``int``/``float`` payloads, rejecting ``bool``.

    Args:
        payload: Candidate payload.

    Returns:
        True for non-boolean numbers.
    """
    accepted = isinstance(payload, (int, float)) and not isinstance(payload, bool)

    return accepted


@dataclass
class CallCounter:
    """Counts ``run()`` invocations per block name."""

    runs: Dict[str, int] = field(default_factory=dict)

    def record(self, block_name: str) -> None:
        """Increment the counter for ``block_name``.

        Args:
            block_name: Registered block type name.
        """
        self.runs[block_name] = self.runs.get(block_name, 0) + 1

    @property
    def total(self) -> int:
        """Total number of recorded ``run()`` calls."""
        return sum(self.runs.values())


class ScaleBlock:
    """Multiply one number by a configured factor."""

    contract = BlockContract(
        reference="value",
        inputs={"value": InputSpec(kind=NUMBER_KIND, view="item")},
        outputs={"scaled": OutputSpec(kind=NUMBER_KIND, transform="preserve")},
    )

    def __init__(self, *, factor: float, counter: CallCounter):
        self._factor = factor
        self._counter = counter

    def run(self, *, value: float) -> Mapping[str, Any]:
        """Scale ``value``.

        Args:
            value: Input number.

        Returns:
            Mapping with ``scaled``.
        """
        self._counter.record(SCALE_BLOCK)
        result = {"scaled": value * self._factor}

        return result


class ExpandBlock:
    """Emit ``value + offset`` children, omitting those above ``limit``."""

    contract = BlockContract(
        reference="value",
        inputs={"value": InputSpec(kind=NUMBER_KIND, view="item")},
        outputs={
            "children": OutputSpec(
                kind=NUMBER_KIND, transform="append", axis="children"
            )
        },
    )

    def __init__(self, *, offsets: Sequence[float], limit: float, counter: CallCounter):
        self._offsets = tuple(offsets)
        self._limit = limit
        self._counter = counter

    def run(self, *, value: float) -> Mapping[str, Any]:
        """Expand ``value`` into its children.

        Args:
            value: Parent number.

        Returns:
            Mapping with ``children``: a ``Batch`` whose local indices are the
            positions of the surviving offsets (possibly sparse or empty).
        """
        self._counter.record(EXPAND_BLOCK)
        children: List[float] = []
        indices: List[Tuple[int, ...]] = []
        for position, offset in enumerate(self._offsets):
            child = value + offset
            if child > self._limit:
                continue
            children.append(child)
            indices.append((position,))

        result = {"children": Batch.of(children, indices=indices)}

        return result


class SumWithParentBlock:
    """Reduce a child group next to its parent item."""

    contract = BlockContract(
        reference="children",
        inputs={
            "parent": InputSpec(kind=NUMBER_KIND, view="item"),
            "children": InputSpec(kind=NUMBER_KIND, view="batch"),
        },
        outputs={
            "total": OutputSpec(kind=NUMBER_KIND, transform="collapse"),
            "count": OutputSpec(kind=NUMBER_KIND, transform="collapse"),
        },
    )

    def __init__(self, *, parent_weight: float, counter: CallCounter):
        self._parent_weight = parent_weight
        self._counter = counter

    def run(self, *, parent: float, children: Iterable[float]) -> Mapping[str, Any]:
        """Sum ``children`` and add the weighted ``parent``.

        Args:
            parent: The parent item the group belongs to.
            children: Trailing-axis group of child numbers; may be empty.

        Returns:
            Mapping with ``total`` and ``count``.
        """
        self._counter.record(SUM_WITH_PARENT_BLOCK)
        child_values = list(children)
        total = parent * self._parent_weight + sum(child_values)
        result = {"total": total, "count": len(child_values)}

        return result


class BrokenBlock:
    """Leaf that violates the result contract in a configured way."""

    contract = BlockContract(
        reference="value",
        inputs={"value": InputSpec(kind=NUMBER_KIND, view="item")},
        outputs={"value": OutputSpec(kind=NUMBER_KIND, transform="preserve")},
    )

    def __init__(self, *, mode: str, counter: CallCounter):
        self._mode = mode
        self._counter = counter

    def run(self, *, value: float) -> Mapping[str, Any]:
        """Return an invalid result or raise, depending on ``mode``.

        Args:
            value: Input number (ignored).

        Returns:
            An invalid mapping for ``empty_mapping``, ``missing_output`` and
            ``wrong_kind``.

        Raises:
            RuntimeError: In ``raise`` mode.
        """
        self._counter.record(BROKEN_BLOCK)
        if self._mode == "raise":
            raise RuntimeError("deliberate failure inside demo/broken")
        if self._mode == "empty_mapping":
            return {}
        if self._mode == "missing_output":
            return {"unexpected": value}

        result = {"value": "not a number"}

        return result


def _require_number(
    config: Mapping[str, Any], key: str, *, block: str, default: Any = None
) -> float:
    if key not in config:
        if default is None:
            raise ValueError(f"{block} requires config key '{key}'")
        return default

    value = config[key]
    if not is_number(value):
        raise ValueError(f"{block} config '{key}' must be a number, got {value!r}")

    return value


def register_author_blocks(registry: Registry) -> CallCounter:
    """Register the numeric author blocks on ``registry``.

    Args:
        registry: Any mutable V2 registry, for example ``native_registry()``
            or a fresh ``Registry()``.

    Returns:
        Counter incremented on every ``run()`` of these blocks.
    """
    counter = CallCounter()

    def scale_factory(config: Mapping[str, Any]) -> ScaleBlock:
        factor = _require_number(config, "factor", block=SCALE_BLOCK)
        block = ScaleBlock(factor=factor, counter=counter)

        return block

    def expand_factory(config: Mapping[str, Any]) -> ExpandBlock:
        offsets = config.get("offsets")
        if not isinstance(offsets, list) or not all(is_number(o) for o in offsets):
            raise ValueError(
                f"{EXPAND_BLOCK} config 'offsets' must be a list of numbers"
            )
        limit = _require_number(config, "limit", block=EXPAND_BLOCK)
        block = ExpandBlock(offsets=offsets, limit=limit, counter=counter)

        return block

    def sum_factory(config: Mapping[str, Any]) -> SumWithParentBlock:
        parent_weight = _require_number(
            config, "parent_weight", block=SUM_WITH_PARENT_BLOCK, default=1
        )
        block = SumWithParentBlock(parent_weight=parent_weight, counter=counter)

        return block

    def broken_factory(config: Mapping[str, Any]) -> BrokenBlock:
        mode = config.get("mode")
        if mode not in BROKEN_MODES:
            raise ValueError(
                f"{BROKEN_BLOCK} config 'mode' must be one of {list(BROKEN_MODES)}, "
                f"got {mode!r}"
            )
        block = BrokenBlock(mode=mode, counter=counter)

        return block

    registry.register_kind(NUMBER_KIND, validator=is_number)
    registry.register_block(
        SCALE_BLOCK, contract=ScaleBlock.contract, factory=scale_factory
    )
    registry.register_block(
        EXPAND_BLOCK, contract=ExpandBlock.contract, factory=expand_factory
    )
    registry.register_block(
        SUM_WITH_PARENT_BLOCK, contract=SumWithParentBlock.contract, factory=sum_factory
    )
    registry.register_block(
        BROKEN_BLOCK, contract=BrokenBlock.contract, factory=broken_factory
    )

    return counter
