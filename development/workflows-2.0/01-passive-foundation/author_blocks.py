"""Author exercise: ordinary numeric V2 blocks written as single classes.

Each block class declares its identity, ``Params``, outputs and resources.
Nothing is registered separately; ``create_author_catalogue`` only lists the
classes. All payloads are plain Python numbers of kind ``number``.

* ``demo/scale``: ``scaled = value * factor``. ``factor`` may be a literal or
  a selector. Works unchanged at any nesting depth.
* ``demo/expand``: emits ``value + offset`` per offset, omitting children above
  ``limit`` while keeping the offset's local index (a sparse ``Batch``).
* ``demo/sum_with_parent``: receives one parent and the group of its children;
  returns ``total`` and ``count`` at the parent level. An empty group returns
  ``parent * parent_weight`` and count 0.
* ``demo/broken``: violates its declaration in a configured way, for the
  ``invalid-bindings`` scenario.

Every block receives the shared ``counter`` resource in its constructor, so
the demo can prove that invalid definitions fail before any ``run()`` call.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping

from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind


def is_number(payload: Any) -> bool:
    """Accept plain ``int``/``float`` payloads, rejecting ``bool``.

    Args:
        payload: Candidate payload.

    Returns:
        True for non-boolean numbers.
    """
    accepted = isinstance(payload, (int, float)) and not isinstance(payload, bool)

    return accepted


NUMBER_KIND = Kind(name="number", description="Plain int or float.", validate=is_number)


@dataclass
class CallCounter:
    """Counts ``run()`` invocations per block type."""

    runs: Dict[str, int] = field(default_factory=dict)

    def record(self, block_type: str) -> None:
        """Increment the counter for ``block_type``.

        Args:
            block_type: Block type identity.
        """
        self.runs[block_type] = self.runs.get(block_type, 0) + 1

    @property
    def total(self) -> int:
        """Total number of recorded ``run()`` calls."""
        return sum(self.runs.values())


class ScaleBlock(Block):
    """Multiply one number by ``factor``."""

    type = "demo/scale"
    outputs = {"scaled": Output(NUMBER_KIND)}

    class Params(BlockParams):
        value: Ref(NUMBER_KIND) = Field(description="Number to scale.")
        factor: float | Ref(NUMBER_KIND) = Field(description="Multiplier.")

    def __init__(self, *, counter: CallCounter):
        self._counter = counter

    def run(self, *, value: float, factor: float) -> Mapping[str, Any]:
        self._counter.record(self.type)
        result = {"scaled": value * factor}

        return result


class ExpandBlock(Block):
    """Emit ``value + offset`` children, omitting those above ``limit``."""

    type = "demo/expand"
    outputs = {"children": Output(NUMBER_KIND, expand="children")}

    class Params(BlockParams):
        value: Ref(NUMBER_KIND) = Field(description="Parent number.")
        offsets: List[float] = Field(description="Offsets added to the parent.")
        limit: float = Field(description="Children above this value are omitted.")

    def __init__(self, *, counter: CallCounter):
        self._counter = counter

    def run(
        self, *, value: float, offsets: List[float], limit: float
    ) -> Mapping[str, Any]:
        self._counter.record(self.type)
        kept = [
            (position, value + offset)
            for position, offset in enumerate(offsets)
            if value + offset <= limit
        ]
        children = Batch.of(
            [child for _, child in kept],
            indices=[(position,) for position, _ in kept],
        )

        return {"children": children}


class SumWithParentBlock(Block):
    """Reduce a group of children next to its parent."""

    type = "demo/sum_with_parent"
    outputs = {"total": Output(NUMBER_KIND), "count": Output(NUMBER_KIND)}

    class Params(BlockParams):
        parent: Ref(NUMBER_KIND) = Field(description="Parent number.")
        children: Group(NUMBER_KIND) = Field(description="Children of the parent.")
        parent_weight: float = Field(default=1, description="Weight of the parent.")

    def __init__(self, *, counter: CallCounter):
        self._counter = counter

    def run(
        self, *, parent: float, children: Batch, parent_weight: float
    ) -> Mapping[str, Any]:
        self._counter.record(self.type)
        child_values = list(children)
        result = {
            "total": parent * parent_weight + sum(child_values),
            "count": len(child_values),
        }

        return result


class BrokenBlock(Block):
    """Violate the declared result in a configured way, or raise."""

    type = "demo/broken"
    outputs = {"value": Output(NUMBER_KIND)}

    class Params(BlockParams):
        value: Ref(NUMBER_KIND) = Field(description="Input number (ignored).")
        mode: Literal["empty_mapping", "missing_output", "raise", "wrong_kind"] = Field(
            description="How to misbehave."
        )

    def __init__(self, *, counter: CallCounter):
        self._counter = counter

    def run(self, *, value: float, mode: str) -> Mapping[str, Any]:
        self._counter.record(self.type)
        if mode == "raise":
            raise RuntimeError("deliberate failure inside demo/broken")
        if mode == "empty_mapping":
            return {}
        if mode == "missing_output":
            return {"unexpected": value}

        return {"value": "not a number"}


AUTHOR_BLOCKS = (ScaleBlock, ExpandBlock, SumWithParentBlock, BrokenBlock)


def create_author_catalogue(counter: CallCounter) -> Catalogue:
    """List the author blocks and share one call counter between them.

    Args:
        counter: Resource passed unchanged to every author block constructor.

    Returns:
        Catalogue in namespace ``demo`` with the ``number`` kind.
    """
    catalogue = Catalogue(
        AUTHOR_BLOCKS,
        kinds=[NUMBER_KIND],
        namespace="demo",
        providers={"counter": counter},
    )

    return catalogue
