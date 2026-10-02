"""V2 fixture blocks for the sequential parity examples.

Each class is an ordinary V2 block: its identity, ``Params``, outputs and
resources are declared on the class. The algorithms equal the V1 fixtures in
``reference/v1_reference_fixtures`` and the V1 core blocks the reference cases
use (ContinueIf, SwitchCase, DimensionCollapse), so both engines receive the
same work and can be compared call by call.

| V1 block | V2 block |
| --- | --- |
| ``reference/<name>@v1`` fixture | ``fixture/<name>@v1`` |
| ``roboflow_core/continue_if@v1`` with one numeric comparison | ``fixture/threshold_gate@v1`` |
| ``roboflow_core/switch_case@v1`` | ``fixture/switch@v1`` |
| ``roboflow_core/dimension_collapse@v1`` | ``fixture/collapse@v1`` |
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.execution_engine.v2.resources import Factory

FIXTURE_NAMESPACE = "fixture"


class AuditLog:
    """Side-effect record injected into ``Counter`` as a constructor resource.

    Args:
        origin: ``catalogue_provider`` or ``caller``.
    """

    created: List["AuditLog"] = []

    def __init__(self, origin: str):
        self.origin = origin
        self.events: List[Dict[str, Any]] = []
        AuditLog.created.append(self)


class Echo(Block):
    """Return the received value as ``value``."""

    type = "fixture/echo@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref() | int | float | str = Field(
            description="Literal or selected value to return unchanged."
        )

    def run(self, *, value: Any) -> dict:
        return {"value": value}


class Sink(Block):
    """Output-free side effect; only the invocation log reveals a call."""

    type = "fixture/sink@v1"

    class Params(BlockParams):
        payload: Optional[str] = Field(
            default="notice", description="Literal payload; explicit null is kept."
        )

    def run(self, *, payload: Optional[str]) -> dict:
        return {}


class RaggedExpand(Block):
    """Expand ``n`` into ``[n*10, ..., n*10 + n - 1]``; ``0`` gives no children."""

    type = "fixture/ragged_expand@v1"
    outputs = {"value": Output(expand="children")}

    class Params(BlockParams):
        value: Ref(INTEGER_KIND, FLOAT_KIND) = Field(
            description="Integer parent value and child count."
        )

    def run(self, *, value: int) -> dict:
        children = Batch.of([value * 10 + index for index in range(value)])

        return {"value": children}


class OffsetExpand(Block):
    """Expand a number into one ``value + offset`` child per offset."""

    type = "fixture/offset_expand@v1"
    outputs = {"child": Output(FLOAT_KIND, expand="children")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Parent number.")
        offsets: List[int] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=lambda: [1, 2], description="Offsets added to the parent."
        )

    def run(self, *, value: float, offsets: List[int]) -> dict:
        children = Batch.of([value + offset for offset in offsets])

        return {"child": children}


class Merge(Block):
    """Return every branch value; a missing branch arrives as ``None``."""

    type = "fixture/merge@v1"
    outputs = {"value": Output()}
    accepts_empty = True

    class Params(BlockParams):
        values: List[Ref()] = Field(description="Branch outputs to join.")

    def run(self, *, values: List[Any]) -> dict:
        return {"value": list(values)}


class Scale(Block):
    """Scalar block: called once per invocation index."""

    type = "fixture/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float | int | Ref(FLOAT_KIND) = Field(description="Number to scale.")
        factor: float | Ref(FLOAT_KIND) = Field(
            default=2.0, description="Multiplier; literal, default or selector."
        )

    def run(self, *, value: float, factor: float) -> dict:
        return {"scaled": value * factor}


class BatchScale(Block):
    """Batch ``value``: one call receives a ``Batch``; scalars are cast into one."""

    type = "fixture/batch_scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float | int | Ref(FLOAT_KIND, batch="always") = Field(
            description="Numbers to scale, always delivered as a Batch."
        )
        factor: float | Ref(FLOAT_KIND) = Field(default=2.0, description="Multiplier.")

    def run(self, *, value: Batch, factor: float) -> list:
        return [{"scaled": item * factor} for item in value]


class MixedScale(Block):
    """Scalar-or-batch ``value``: a batch source gives one call with a ``Batch``."""

    type = "fixture/mixed_scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float | int | Ref(FLOAT_KIND, batch="if_varying") = Field(
            description="A Batch when bound to varying data, else a plain number."
        )
        factor: float | Ref(FLOAT_KIND) = Field(default=2.0, description="Multiplier.")

    def run(self, *, value: Union[Batch, float], factor: float) -> Union[dict, list]:
        if isinstance(value, Batch):
            return [{"scaled": item * factor} for item in value]

        return {"scaled": value * factor}


class SumChildren(Block):
    """Add each parent to its surviving children in one vectorized call.

    V1 batch blocks always see groups whose children were all filtered; in V2
    that needs the explicit ``accepts_empty`` opt-in.
    """

    type = "fixture/sum_children@v1"
    outputs = {"total": Output(FLOAT_KIND)}
    accepts_empty = True

    class Params(BlockParams):
        parent: Ref(FLOAT_KIND, batch="always") = Field(description="Parent numbers.")
        children: Group(FLOAT_KIND, batch="always") = Field(
            description="Child numbers, one group per parent."
        )

    def run(self, *, parent: Batch, children: Batch) -> list:
        totals = [
            {"total": parent_value + sum(item for item in group if item is not None)}
            for parent_value, group in zip(parent, children)
        ]

        return totals


class Compound(Block):
    """Return the resolved compound parameters."""

    type = "fixture/compound@v1"
    outputs = {"echo": Output()}

    class Params(BlockParams):
        params: Dict[str, Ref() | float | int | str] = Field(
            default_factory=dict, description="Mapping mixing literals and selectors."
        )
        items: List[Ref() | float | int] = Field(
            default_factory=list, description="List mixing literals and selectors."
        )

    def run(self, *, params: Dict[str, Any], items: List[Any]) -> dict:
        return {"echo": {"params": dict(params), "items": list(items)}}


class Counter(Block):
    """Count invocations of this step instance and report them to ``audit``.

    Args:
        audit: Constructor resource; the catalogue provider creates one per
            step unless the caller supplies one.
    """

    type = "fixture/counter@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Selector-only number input.")

    def __init__(self, *, audit: AuditLog):
        self.audit = audit
        self.count = 0
        audit.events.append({"event": "construct"})

    def run(self, *, value: float) -> dict:
        self.count += 1
        self.audit.events.append({"event": "run", "value": value, "count": self.count})

        return {"count": self.count}


class Fail(Block):
    """Raise a ``RuntimeError`` naming the received value."""

    type = "fixture/fail@v1"
    outputs = {"never": Output()}

    class Params(BlockParams):
        value: Ref() = Field(description="Value named in the error message.")

    def run(self, *, value: Any) -> dict:
        raise RuntimeError(f"reference failure on {value!r}")


class Increment(Block):
    """Increment ``value["count"]`` in place and return the same object."""

    type = "fixture/increment@v1"
    outputs = {"value": Output(DICTIONARY_KIND)}
    mutates = ("value",)

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND) | Dict[str, Any] = Field(
            description="Mapping whose count is incremented in place."
        )

    def run(self, *, value: Dict[str, Any]) -> dict:
        value["count"] += 1

        return {"value": value}


class ReadCount(Block):
    """Return ``value["count"]`` as seen when this step runs."""

    type = "fixture/read_count@v1"
    outputs = {"seen": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND) | Dict[str, Any] = Field(
            description="Mapping to read."
        )

    def run(self, *, value: Dict[str, Any]) -> dict:
        return {"seen": value["count"]}


def _double_later(value: float) -> float:
    # The delay keeps the future pending when run() returns.
    time.sleep(0.05)

    return value * 2


class Deferred(Block):
    """Return a pending ``Future`` computed on a block-owned worker thread."""

    type = "fixture/deferred@v1"
    outputs = {"doubled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) | float = Field(description="Number to double.")

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1)

    def run(self, *, value: float) -> dict:
        doubled: Future = self._pool.submit(_double_later, value)

        return {"doubled": doubled}


class ThresholdGate(Block):
    """Continue to ``next_steps`` when ``value`` passes one numeric comparison."""

    type = "fixture/threshold_gate@v1"

    class Params(BlockParams):
        value: Ref() = Field(description="Value compared with the threshold.")
        threshold: float = Field(description="Right-hand side of the comparison.")
        comparator: Literal[">", "<"] = Field(default=">", description="Comparison.")
        next_steps: List[StepRef] = Field(description="Steps governed by the gate.")

    def run(
        self, *, value: float, threshold: float, comparator: str, next_steps: List[str]
    ) -> Select:
        passed = value > threshold if comparator == ">" else value < threshold
        decision = Select(next_steps) if passed else Stop()

        return decision


class Switch(Block):
    """Select the step registered for ``str(value)``; no match selects nothing."""

    type = "fixture/switch@v1"

    class Params(BlockParams):
        value: Ref() = Field(description="Value whose text chooses the route.")
        cases: Dict[str, StepRef] = Field(description="Route per value text.")

    def run(self, *, value: Any, cases: Dict[str, str]) -> Select:
        target = cases.get(str(value))
        decision = Select(target) if target is not None else Stop()

        return decision


class Collapse(Block):
    """Reduce one group to the list of its surviving children.

    Like V1 DimensionCollapse, a group whose children were all filtered is
    still reduced (to ``[]``); V2 needs ``accepts_empty`` for that.
    """

    type = "fixture/collapse@v1"
    outputs = {"output": Output(LIST_OF_VALUES_KIND)}
    accepts_empty = True

    class Params(BlockParams):
        data: Group() = Field(description="Children to collect.")

    def run(self, *, data: Batch) -> dict:
        return {"output": list(data)}


FIXTURE_BLOCKS = (
    Echo,
    Sink,
    RaggedExpand,
    OffsetExpand,
    Merge,
    Scale,
    BatchScale,
    MixedScale,
    SumChildren,
    Compound,
    Counter,
    Fail,
    Increment,
    ReadCount,
    Deferred,
    ThresholdGate,
    Switch,
    Collapse,
)


def create_fixture_catalogue() -> Catalogue:
    """Collect the fixture blocks with a per-step ``audit`` provider.

    Returns:
        Catalogue in namespace ``fixture``; ``Counter`` steps get a new
        ``AuditLog`` each unless the caller supplies ``fixture.audit``.
    """
    catalogue = Catalogue(
        FIXTURE_BLOCKS,
        namespace=FIXTURE_NAMESPACE,
        providers={
            "audit": Factory(
                lambda: AuditLog(origin="catalogue_provider"), scope="step"
            )
        },
    )

    return catalogue
