"""Contracts kept by the engine's per-run fast paths and release of payloads.

Readiness keeps traversing every ``Mapping``, virtual subclasses included.
A completed run frees its payloads as soon as the caller drops the result:
building entries, output trees and rows creates no reference cycle, so the
payloads do not wait for the cyclic garbage collector.
"""

import gc
import weakref
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, List

import pytest
from roboflow_workflows.execution_engine.v2.data import Batch, InputValue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.execution.inputs import _supplied
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow
from roboflow_workflows.execution_engine.v2.readiness import resolve_futures

from .execution.blocks import ContinueIf
from .execution.plans import BATCH, SCALAR, PlanBuilder


def done(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


class VirtualMapping:
    """Registered as a ``Mapping`` without inheriting from it."""

    def __init__(self, items: Dict[str, Any]):
        self._items = items

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Any:
        return self._items.items()


Mapping.register(VirtualMapping)


@pytest.mark.parametrize(
    "container",
    [
        lambda items: MappingProxyType(items),
        lambda items: VirtualMapping(items),
    ],
)
def test_readiness_resolves_futures_in_any_mapping(container: Any) -> None:
    original = container({"value": done(1.0)})

    resolved = resolve_futures(original)

    assert resolved == {"value": 1.0}
    assert isinstance(original["value"], Future)


def test_readiness_returns_future_free_mappings_unchanged() -> None:
    original = VirtualMapping({"value": 1.0})

    assert resolve_futures(original) is original


def test_supplied_input_value_resolves_futures_in_its_data() -> None:
    supplied = done(InputValue(data=[done(1.0), 2.0]))

    data, _ = _supplied("image", supplied)

    assert data == [1.0, 2.0]


def test_supplied_plain_value_resolves_nested_futures_once() -> None:
    plain = [done({"value": done(3.0)})]

    data, _ = _supplied("image", plain)

    assert data == [{"value": 3.0}]
    assert isinstance(plain[0], Future)


class Payload:
    """Weakly referenceable stand-in for an image or prediction."""


class Hold(Block):
    """Passes its payload on without remembering it."""

    type = "spc/hold@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> dict:
        return {"value": value}


class Fanout(Block):
    """Adds an axis holding the same payload twice, without remembering it."""

    type = "spc/fanout@v1"
    outputs = {"copies": Output(expand="copies")}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> dict:
        return {"copies": Batch.of([value, value])}


@contextmanager
def automatic_gc_disabled() -> Iterator[None]:
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def scalar_passthrough() -> CompiledWorkflow:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Hold, "hold", value="$inputs.value")
        .output("value", "$steps.hold.value")
        .build()
    )

    return plan


def batched_passthrough() -> CompiledWorkflow:
    plan = (
        PlanBuilder()
        .input("value", BATCH)
        .step(Hold, "hold", at=BATCH, value="$inputs.value")
        .output("value", "$steps.hold.value")
        .output("input", "$inputs.value")
        .build()
    )

    return plan


def nested_fanout() -> CompiledWorkflow:
    plan = (
        PlanBuilder()
        .input("value", BATCH)
        .step(Fanout, "fanout", at=BATCH, value="$inputs.value")
        .output("copies", "$steps.fanout.copies")
        .build()
    )

    return plan


def gated_batch() -> CompiledWorkflow:
    plan = (
        PlanBuilder()
        .input("value", BATCH)
        .input("keep", BATCH)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.keep",
            next_steps=["$steps.hold"],
        )
        .step(
            Hold,
            "hold",
            at=BATCH,
            gates=(("gate", "$steps.hold"),),
            value="$inputs.value",
        )
        .output("value", "$steps.hold.value")
        .build()
    )

    return plan


def run_and_drop_payloads(
    session: Any,
    *,
    inputs: Callable[[List[Payload]], Dict[str, Any]],
    expected_rows: Callable[[List[Payload]], List[Dict[str, Any]]],
) -> List[weakref.ref]:
    payloads = [Payload(), Payload()]
    references = [weakref.ref(payload) for payload in payloads]

    result = session.run(inputs(payloads))
    assert result.rows() == expected_rows(payloads)

    return references


@pytest.mark.parametrize(
    "build, inputs, expected_rows",
    [
        pytest.param(
            scalar_passthrough,
            lambda payloads: {"value": payloads[0]},
            lambda payloads: [{"value": payloads[0]}],
            id="scalar",
        ),
        pytest.param(
            batched_passthrough,
            lambda payloads: {"value": payloads},
            lambda payloads: [
                {"value": payload, "input": payload} for payload in payloads
            ],
            id="batched",
        ),
        pytest.param(
            nested_fanout,
            lambda payloads: {"value": payloads},
            lambda payloads: [{"copies": [payload, payload]} for payload in payloads],
            id="nested",
        ),
        pytest.param(
            gated_batch,
            lambda payloads: {"value": payloads, "keep": [1, 0]},
            lambda payloads: [{"value": payloads[0]}, {"value": None}],
            id="filtered",
        ),
    ],
)
def test_dropping_the_result_frees_payloads_without_cyclic_collection(
    build: Callable[[], CompiledWorkflow],
    inputs: Callable[[List[Payload]], Dict[str, Any]],
    expected_rows: Callable[[List[Payload]], List[Dict[str, Any]]],
) -> None:
    session = build().create_session()

    with automatic_gc_disabled():
        references = run_and_drop_payloads(
            session, inputs=inputs, expected_rows=expected_rows
        )
        alive = [reference() is not None for reference in references]

    assert alive == [False, False]
