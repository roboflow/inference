"""Call context: per-index source/time, emit rules, state resolution, test helper."""

from typing import Any, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.events import Event, EventPayloadError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.reactions.testing import (
    CapturedEvent,
    block_call,
)
from roboflow_workflows.execution_engine.v2.state import ManagedState, StateScopeError

from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
    WAIT,
    Probe,
    active_definition,
    handler,
    reacting,
    start,
)

HIT = Event({"value": FLOAT_KIND})


class Where(Block):
    """Records each call's indices, contexts and per-source state."""

    type = "test/where@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"hit": HIT}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, calls: List[Any], managed_state: ManagedState) -> None:
        self.calls = calls
        self.state = managed_state

    def run(self, value):
        context = self.execution_context
        (index,) = context.indices
        sample = context.sample_at(index)
        total = self.state.source.incr("seen") if sample is not None else None
        self.calls.append((index, sample, total, context.cause))
        self.emit("hit", value=value)
        return {"value": value}


class Vectorized(Block):
    """A batch-delivering block emitting for one chosen member."""

    type = "test/vectorized@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"hit": HIT}

    class Params(BlockParams):
        # Ref metadata strings are values; pyflakes treats them as forward refs.
        values: Ref(FLOAT_KIND, batch="always")  # noqa: F821

    def __init__(self, *, calls: List[Any], managed_state: ManagedState) -> None:
        self.calls = calls
        self.state = managed_state

    def run(self, values):
        context = self.execution_context
        indices = list(values.indices)
        self.calls.append([context.sample_at(index) for index in indices])
        for problem in (
            lambda: self.emit("hit", value=1.0),
            lambda: self.state.source.get("seen"),
        ):
            try:
                problem()
            except (EventEmissionError, StateScopeError) as error:
                self.calls.append(type(error).__name__)
        self.calls.append(self.state.at(indices[1]).incr("seen"))
        self.emit("hit", at=indices[1], value=float(values[1]))
        return [{"value": value} for value in values]


CATALOGUE = Catalogue([Where, Vectorized])


def batch_inputs(*sources: Any) -> dict:
    sample = {(position,): None if name is None else SampleContext(source_id=name)
              for position, name in enumerate(sources)}  # fmt: skip
    value = InputValue(
        [float(position) for position in range(len(sources))],
        metadata=EntryMetadata(sample=sample),
    )

    return {"values": value}


def batch_definition(block: type) -> dict:
    field = "values" if block is Vectorized else "value"
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [{"type": block.type, "name": "step", field: "$inputs.values"}],
        "outputs": [
            {"type": "JsonField", "name": "value", "selector": "$steps.step.value"}
        ],
    }

    return definition


def test_each_invocation_sees_its_own_source_and_state_scope():
    calls: List[Any] = []
    state = ManagedState()
    plan = compile_workflow(batch_definition(Where), catalogue=CATALOGUE)
    session = plan.create_session(resources={"calls": calls, "managed_state": state})

    session.run(batch_inputs("a", "b", "a", None))

    assert [
        (index, sample and sample.source_id, total) for index, sample, total, _ in calls
    ] == [
        ((0,), "a", 1),
        ((1,), "b", 1),
        ((2,), "a", 2),
        ((3,), None, None),
    ]
    assert state.for_source("a").get("seen") == 2
    assert all(cause is None for *_, cause in calls)


def test_a_batch_call_names_its_member_for_events_and_source_state():
    calls: List[Any] = []
    state = ManagedState()
    plan = compile_workflow(batch_definition(Vectorized), catalogue=CATALOGUE)
    session = plan.create_session(resources={"calls": calls, "managed_state": state})

    session.run(batch_inputs("a", "b"))

    samples, *problems, total = calls
    assert [sample.source_id for sample in samples] == ["a", "b"]
    assert problems == ["EventEmissionError", "StateScopeError"]
    assert total == 1 and state.for_source("b").get("seen") == 1


def test_a_call_without_context_carrying_arguments_uses_the_event_as_origin():
    probe = Probe()
    literal = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [{"type": "test/react@v1", "name": "react", "value": 7.0}],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.react.value"}
        ],
    }
    plan = reacting(active_definition("cam_b"), handler(depth=2, workflow=literal))
    run = start(plan, probe, {"cam_b": [1]})
    assert run.wait(WAIT)

    ((value, cause, sample, _),) = probe.seen
    assert value == 7.0
    assert sample is cause.sample and sample.source_id == "cam_b"


def test_sample_at_rejects_an_index_of_another_call():
    context = ExecutionContext(("s",), "t", "session", run_id="r", indices=((0,),))

    with pytest.raises(ContractError, match="names no index"):
        context.sample_at((1,))
    assert context.sample_at((0,)) is None
    assert context.cause is None


def test_emit_outside_a_call_or_in_a_constructor_is_an_explicit_error():
    block = Where.__new__(Where)

    with pytest.raises(EventEmissionError, match="outside an engine call"):
        block.emit("hit", value=1.0)
    with use_execution_context(ExecutionContext(("s",), Where.type, "session")):
        with pytest.raises(EventEmissionError, match="outside a block call"):
            block.emit("hit", value=1.0)


def test_block_call_captures_validated_events_with_their_source():
    calls: List[Any] = []
    block = Where(calls=calls, managed_state=ManagedState())

    with block_call(block, source_id="cam_a", indices=[(4,)]) as call:
        block.run(value=2.0)

    assert call.events == [
        CapturedEvent("hit", {"value": 2.0}, at=(4,), source_id="cam_a")
    ]
    assert calls[0][2] == 1
    with block_call(block, subscribed=()) as quiet:
        assert not block.has_subscribers("hit")
        block.run(value=2.0)
    assert quiet.events == []


def test_block_call_applies_the_engine_validation_rules():
    block = Where(calls=[], managed_state=ManagedState())

    with block_call(block) as call:
        with pytest.raises(EventEmissionError, match="undeclared event 'miss'"):
            block.emit("miss", value=1.0)
        with pytest.raises(EventPayloadError, match="missing"):
            block.emit("hit")
        with pytest.raises(EventPayloadError, match="none of the kinds"):
            block.emit("hit", value="text")
    with block_call(Vectorized, indices=[(0,), (1,)], sources={(1,): "b"}) as batch:
        with pytest.raises(EventEmissionError, match="pass at="):
            batch.context.call_scope.emit(batch.context, "hit", {"value": 1.0}, at=None)
        batch.context.call_scope.emit(batch.context, "hit", {"value": 1.0}, at=(1,))
    assert call.events == []
    assert batch.events == [
        CapturedEvent("hit", {"value": 1.0}, at=(1,), source_id="b")
    ]


def test_an_event_without_handlers_is_validated_then_dropped():
    calls: List[Any] = []
    plan = compile_workflow(batch_definition(Where), catalogue=CATALOGUE)
    session = plan.create_session(
        resources={"calls": calls, "managed_state": ManagedState()}
    )

    result = session.run(batch_inputs("a"))

    assert result.rows() == [{"value": 0.0}]


def test_an_undeclared_emission_fails_the_step():
    class Rogue(Where):
        type = "test/rogue@v1"

        def run(self, value):
            self.emit("unknown", value=value)

    plan = compile_workflow(batch_definition(Rogue), catalogue=Catalogue([Rogue]))
    session = plan.create_session(
        resources={"calls": [], "managed_state": ManagedState()}
    )

    with pytest.raises(StepExecutionError, match="undeclared event 'unknown'"):
        session.run(batch_inputs("a"))


def test_source_id_accessors_follow_the_call_delivery_mode():
    block = Where(calls=[], managed_state=ManagedState())

    with block_call(block, indices=[(2,)], sources={(2,): "cam_a"}) as single:
        assert single.context.source_id == "cam_a"
        assert single.context.source_id_at((2,)) == "cam_a"
    with block_call(Vectorized, indices=[(0,)], batched=True) as batch_of_one:
        context = batch_of_one.context
        with pytest.raises(ContractError, match="batch-delivering"):
            context.source_id
        assert context.source_id_at((0,)) == "test"
        with pytest.raises(EventEmissionError, match="pass at="):
            context.call_scope.emit(context, "hit", {"value": 1.0}, at=None)
    with block_call(block, sources={(): None}) as sourceless:
        with pytest.raises(ContractError, match="no single source"):
            sourceless.context.source_id
    constructor = ExecutionContext(("s",), Where.type, "session")
    with pytest.raises(ContractError, match="no call index"):
        constructor.source_id
