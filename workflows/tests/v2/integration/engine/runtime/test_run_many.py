"""Coalesced ``run_many`` calls of eligible scalar steps.

``Recording`` declares ``run_many``; ``Scalar`` does not. Both record every
physical call, so a test can compare what the engine passed. A step's
``Recording.respond`` replaces the default ``run_many`` result.
"""

import dataclasses
import threading
from concurrent.futures import Future
from typing import Annotated, Any, Callable, Dict, List, Optional

import pytest
from pydantic import Field
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine import (
    ImplementationContribution,
    contract_digest,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import (
    ContractError,
    InvocationBatchError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.events import Event
from roboflow_workflows.v2.engine.implementations import Implementation
from roboflow_workflows.v2.engine.phases import phase
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import (
    CompiledWorkflow,
    CompileOptions,
    ExecutionObserver,
)

from tests.v2.support.compilation.fixture_blocks import (
    Echo,
    Gate,
    batch_input,
    gate,
    parameter,
    workflow,
)

PAINT = ("paint",)


def painted(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"painted": row["value"] + row["offset"]}


class _Rows(Implementation):
    """Records each physical call of one step instance."""

    def __init__(self) -> None:
        self.scalar: List[Dict[str, Any]] = []
        self.batches: List[Dict[str, Any]] = []
        self.respond: Optional[Callable[..., Any]] = None

    def run(self, *, value, offset, style, labels) -> dict:
        row = {"value": value, "offset": offset, "style": style, "labels": labels}
        self.scalar.append(row)
        return painted(row)


class Scalar(_Rows):
    name = "scalar"


class Recording(_Rows):
    name = "recording"

    def run_many(self, *, invocations) -> Any:
        context = self.execution_context
        self.batches.append(
            {
                "invocations": invocations,
                "indices": context.indices,
                "batched": context.batched,
            }
        )
        if self.respond is not None:
            return self.respond(self, invocations)

        return [painted(row) for row in invocations]


class Phased(Recording):
    name = "phased"

    @phase
    def paint_row(self, *, value, offset):
        return {"painted": value + offset}


def paint_params() -> type:
    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        offset: Annotated[Ref(FLOAT_KIND), Field(ge=0)] | float = 0.0
        style: Dict[str, Any] = {}
        labels: List[Any | Ref()] = []

    return Params


def contract(name: str, *implementations: type, **body: Any) -> type:
    namespace = {
        "type": f"test/run_many_{name}@v1",
        "prunable": True,
        "outputs": {"painted": Output(FLOAT_KIND)},
        "Params": paint_params(),
        "implementations": implementations,
        **body,
    }
    block = type(f"RunMany_{name}", (Block,), namespace)

    return block


Paint = contract("paint", Recording)
Plain = contract("plain", Scalar)
Kept = contract("kept", Recording, prunable=False)
Retaining = contract("retaining", Recording, retains_outputs=("painted",))
PhasedPaint = contract("phased", Phased)
Wanting = contract(
    "wanting",
    Recording,
    outputs={"painted": Output(FLOAT_KIND), "extra": Output(FLOAT_KIND)},
)


class Split(Block):
    """Expands ``value`` into ``int(value)`` children ``value * 10 + k``."""

    type = "test/run_many_split@v1"
    outputs = {"parts": Output(FLOAT_KIND, expand="parts")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"parts": Batch.of([value * 10.0 + k for k in range(int(value))])}


class GatherRows(Implementation):
    name = "gather"

    def __init__(self) -> None:
        self.batches: List[Any] = []

    def run(self, *, members) -> dict:
        return {"total": float(sum(members))}

    def run_many(self, *, invocations) -> list:
        self.batches.append(invocations)
        return [{"total": float(sum(row["members"]))} for row in invocations]


class Gather(Block):
    type = "test/run_many_gather@v1"
    prunable = True
    outputs = {"total": Output(FLOAT_KIND)}
    implementations = (GatherRows,)

    class Params(BlockParams):
        members: Group(FLOAT_KIND)


class BatchRows(Implementation):
    name = "batch-rows"

    def __init__(self) -> None:
        self.calls: List[list] = []
        self.batches: List[Any] = []

    def run(self, *, value) -> list:
        self.calls.append(list(value))
        return [{"painted": item} for item in value]

    def run_many(self, *, invocations) -> list:
        self.batches.append(invocations)
        return []


class BatchPaint(Block):
    type = "test/run_many_batch@v1"
    prunable = True
    outputs = {"painted": Output(FLOAT_KIND)}
    implementations = (BatchRows,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND, batch="always")  # noqa: F821 (batch mode is a value)


BLOCKS = [
    Paint,
    Plain,
    Kept,
    Retaining,
    PhasedPaint,
    Wanting,
    Split,
    Gather,
    BatchPaint,
    Echo,
    Gate,
]
CATALOGUE = Catalogue(BLOCKS)
VALUES = batch_input("values", kind=["float"])


class Observed(ExecutionObserver):
    """Keeps the logical callbacks of a session."""

    def __init__(self, log: Optional[list] = None) -> None:
        self.invocations: List[Dict[str, Any]] = []
        self.skipped: List[tuple] = []
        self.finished: List[tuple] = []
        self.errors: List[StepExecutionError] = []
        self.log = log

    def on_invocation(self, *, step, index, arguments, result) -> None:
        self.invocations.append(
            {"step": step, "index": index, "arguments": arguments, "result": result}
        )
        if self.log is not None:
            self.log.append(("observed", step, index))

    def on_invocation_skipped(self, *, step, index, reason) -> None:
        self.skipped.append((step, index, reason))

    def on_step_finished(self, *, step, invocations, skipped) -> None:
        self.finished.append((step, invocations, skipped))

    def on_error(self, *, error) -> None:
        self.errors.append(error)

    def of(self, step: tuple) -> List[Dict[str, Any]]:
        found = [item for item in self.invocations if item["step"] == step]

        return found


def compiled(
    definition: dict, *, catalogue: Catalogue = CATALOGUE, **options: Any
) -> CompiledWorkflow:
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(**options)
    )

    return plan


def paint_step(block: type = Paint, **params: Any) -> dict:
    return {"type": block.type, "name": "paint", **params}


def simple(block: type = Paint) -> dict:
    definition = workflow(
        [paint_step(block, value="$inputs.values")],
        {"painted": "$steps.paint.painted"},
        inputs=[VALUES],
    )

    return definition


def gated_session(block: type = Paint) -> tuple:
    """A session whose paint step runs only where ``flags`` is truthy."""
    definition = workflow(
        [
            gate("keep", "$inputs.flags", ["paint"]),
            paint_step(block, value="$inputs.values"),
        ],
        {"painted": "$steps.paint.painted"},
        inputs=[VALUES, batch_input("flags")],
    )
    observer = Observed()
    session = compiled(definition).create_session(observer=observer)

    return session, observer


# --- eligibility and arguments ---------------------------------------------------


def configured(block: type) -> dict:
    definition = workflow(
        [
            paint_step(
                block,
                value="$inputs.values",
                offset="$inputs.offset",
                style={"color": "red", "sizes": [1, 2]},
                labels=["fixed", "$inputs.tags"],
            )
        ],
        {"painted": "$steps.paint.painted"},
        inputs=[VALUES, batch_input("tags"), parameter("offset")],
    )

    return definition


def test_eligible_step_makes_one_call_with_every_rows_scalar_arguments() -> None:
    tags = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]
    inputs = {"values": [1.0, 2.0, 3.0], "tags": tags, "offset": 0.5}
    plan = compiled(configured(Paint))
    many = plan.create_session()
    scalar = compiled(configured(Plain)).create_session()

    many_rows = many.run(inputs).rows()
    scalar_rows = scalar.run(inputs).rows()

    recording = many.instances[PAINT]
    (batch,) = recording.batches
    invocations = batch["invocations"]
    first, second, _ = invocations
    assert plan.step(PAINT).coalesces_invocations
    assert not plan.step(PAINT).delivers_batches
    assert recording.scalar == []
    assert list(invocations) == scalar.instances[PAINT].scalar
    assert all(row["labels"][1] is tag for row, tag in zip(invocations, tags))
    assert first["style"] == {"color": "red", "sizes": [1, 2]}
    assert first["style"] is not second["style"]
    assert first["style"]["sizes"] is not second["style"]["sizes"]
    assert first["style"] is not plan.step(PAINT).params.style
    assert batch["indices"] == ((0,), (1,), (2,))
    assert batch["batched"] is True
    assert many_rows == scalar_rows
    assert many_rows == [{"painted": 1.5}, {"painted": 2.5}, {"painted": 3.5}]


def test_contributed_hook_changes_only_the_selected_implementation() -> None:
    contributed = CATALOGUE.with_contributions(
        [
            ImplementationContribution(
                block_type=Plain.type,
                contract=contract_digest(Plain),
                implementations=(Recording,),
                provider="example-provider",
                version="1.0.0",
            )
        ]
    )
    public_plan = compiled(configured(Plain))
    contributed_plan = compiled(configured(Plain), catalogue=contributed)
    public, chosen = public_plan.step(PAINT), contributed_plan.step(PAINT)
    inputs = {"values": [1.0, 2.0], "tags": ["a", "b"], "offset": 0.0}

    rows = contributed_plan.create_session().run(inputs).rows()

    assert contract_digest(contributed.entry(Plain.type).spec) == contract_digest(Plain)
    assert (public.selected.name, chosen.selected.name) == ("scalar", "recording")
    assert (public.coalesces_invocations, chosen.coalesces_invocations) == (
        False,
        True,
    )
    assert chosen.block_type == public.block_type == Plain.type
    assert chosen.params == public.params
    assert chosen.bindings == public.bindings
    assert dict(chosen.outputs) == dict(public.outputs)
    public_described, chosen_described = public.describe(), chosen.describe()
    assert "coalesces_invocations" not in public_described
    assert chosen_described.pop("coalesces_invocations") is True
    del public_described["implementation"], chosen_described["implementation"]
    assert chosen_described == public_described
    assert rows == public_plan.create_session().run(inputs).rows()


@pytest.mark.parametrize(
    "change",
    [
        {"prunable": False},
        {"is_control": True},
        {"mutates": ("value",)},
        {"events": {"done": Event()}},
        {"retains_inputs": ("value",)},
        {"retains_outputs": ("painted",)},
    ],
)
def test_each_logical_constraint_alone_makes_a_step_ineligible(change: dict) -> None:
    step = compiled(simple()).step(PAINT)

    changed = dataclasses.replace(step, spec=dataclasses.replace(step.spec, **change))

    assert step.coalesces_invocations
    assert not changed.coalesces_invocations


@pytest.mark.parametrize(
    "block, mode, scalar_calls",
    [
        (Plain, "run", 2),
        (Kept, "run", 2),
        (Retaining, "run", 2),
        (PhasedPaint, "phases", 0),
    ],
)
def test_ineligible_steps_keep_one_call_per_row(
    block: type, mode: str, scalar_calls: int
) -> None:
    plan = compiled(simple(block), block_execution=mode)
    observer = Observed()
    session = plan.create_session(observer=observer)

    rows = session.run({"values": [1.0, 2.0]}).rows()

    instance = session.instances[PAINT]
    assert not plan.step(PAINT).coalesces_invocations
    assert rows == [{"painted": 1.0}, {"painted": 2.0}]
    assert instance.batches == []
    assert len(instance.scalar) == scalar_calls
    assert observer.finished == [(PAINT, 2, 0)]


def test_phased_implementation_coalesces_only_in_run_mode() -> None:
    assert compiled(simple(PhasedPaint)).step(PAINT).coalesces_invocations
    assert not (
        compiled(simple(PhasedPaint), block_execution="phases")
        .step(PAINT)
        .coalesces_invocations
    )


def test_existing_batch_delivery_is_unchanged() -> None:
    plan = compiled(simple(BatchPaint))
    session = plan.create_session()

    rows = session.run({"values": [1.0, 2.0]}).rows()

    instance = session.instances[PAINT]
    assert plan.step(PAINT).delivers_batches
    assert not plan.step(PAINT).coalesces_invocations
    assert instance.calls == [[1.0, 2.0]]
    assert instance.batches == []
    assert rows == [{"painted": 1.0}, {"painted": 2.0}]


# --- admitted rows -------------------------------------------------------------


@pytest.mark.parametrize(
    "flags, admitted",
    [
        ([0, 1, 0, 1], ((1,), (3,))),
        ([0, 1, 0], ((1,),)),
        ([0, 0], ()),
    ],
)
def test_only_admitted_rows_are_coalesced_at_their_original_indices(
    flags: list, admitted: tuple
) -> None:
    session, observer = gated_session()
    values = [float(position) for position in range(len(flags))]

    rows = session.run({"values": values, "flags": flags}).rows()

    recording = session.instances[PAINT]
    assert rows == [
        {"painted": value if flag else None} for value, flag in zip(values, flags)
    ]
    assert [batch["indices"] for batch in recording.batches] == (
        [admitted] if admitted else []
    )
    assert all(batch["batched"] for batch in recording.batches)
    assert recording.scalar == []
    assert [
        (item["index"], item["arguments"]["value"]) for item in observer.of(PAINT)
    ] == [(index, values[index[0]]) for index in admitted]
    assert [index for step, index, _ in observer.skipped if step == PAINT] == [
        (position,) for position, flag in enumerate(flags) if not flag
    ]
    assert observer.finished[-1] == (PAINT, len(admitted), flags.count(0))


def test_trace_records_one_physical_call_and_the_observer_every_row() -> None:
    session, observer = gated_session()

    result = session.run({"values": [0.0, 1.0, 2.0], "flags": [1, 0, 1]})

    (call,) = [
        event
        for event in result.trace
        if event["event"] == "invocation" and event["step"] == list(PAINT)
    ]
    assert (call["index"], call["indices"], call["run_many"]) == (
        None,
        [[0], [2]],
        True,
    )
    assert [item["index"] for item in observer.of(PAINT)] == [(0,), (2,)]
    assert observer.finished[-1] == (PAINT, 2, 1)
    arguments = session.instances[PAINT].batches[0]["invocations"]
    assert all(
        item["arguments"] is row for item, row in zip(observer.of(PAINT), arguments)
    )


def test_parent_values_are_broadcast_into_each_child_row() -> None:
    definition = workflow(
        [
            {"type": Split.type, "name": "split", "value": "$inputs.values"},
            paint_step(value="$steps.split.parts", offset="$inputs.values"),
        ],
        {"painted": "$steps.paint.painted"},
    )
    session = compiled(definition).create_session()

    rows = session.run({"values": [1.0, 2.0]}).rows()

    (batch,) = session.instances[PAINT].batches
    assert batch["indices"] == ((0, 0), (1, 0), (1, 1))
    assert [(row["value"], row["offset"]) for row in batch["invocations"]] == [
        (10.0, 1.0),
        (20.0, 2.0),
        (21.0, 2.0),
    ]
    assert rows == [{"painted": [11.0]}, {"painted": [22.0, 23.0]}]


def test_group_values_including_an_empty_group_arrive_per_row() -> None:
    definition = workflow(
        [
            {"type": Split.type, "name": "split", "value": "$inputs.values"},
            {"type": Gather.type, "name": "gather", "members": "$steps.split.parts"},
        ],
        {"total": "$steps.gather.total"},
    )
    session = compiled(definition).create_session()

    rows = session.run({"values": [2.0, 0.0, 1.0]}).rows()

    (invocations,) = session.instances[("gather",)].batches
    groups = [row["members"] for row in invocations]
    assert all(isinstance(group, Batch) for group in groups)
    assert [list(group) for group in groups] == [[20.0, 21.0], [], [10.0]]
    assert [group.indices for group in groups] == [((0, 0), (0, 1)), (), ((2, 0),)]
    assert [group.parent_index for group in groups] == [(0,), (1,), (2,)]
    assert rows == [{"total": 41.0}, {"total": 0.0}, {"total": 10.0}]


def test_an_invalid_later_row_fails_before_any_call() -> None:
    definition = workflow(
        [paint_step(value="$inputs.values", offset="$inputs.offsets")],
        {"painted": "$steps.paint.painted"},
        inputs=[VALUES, batch_input("offsets")],
    )
    session = compiled(definition).create_session()

    with pytest.raises(StepExecutionError, match="greater than or equal") as raised:
        session.run({"values": [1.0, 2.0, 3.0], "offsets": [0.0, 0.0, -1.0]})

    recording = session.instances[PAINT]
    assert raised.value.index == (2,)
    assert (recording.batches, recording.scalar) == ([], [])


# --- results and failures ------------------------------------------------------


def rows_of(implementation: Any, invocations: Any) -> list:
    return [painted(row) for row in invocations]


@pytest.mark.parametrize(
    "respond, got",
    [
        (lambda impl, rows: rows_of(impl, rows)[:-1], "got list of 1"),
        (lambda impl, rows: rows_of(impl, rows) * 2, "got list of 4"),
        (lambda impl, rows: {"painted": 1.0}, "got dict"),
        (lambda impl, rows: iter(rows_of(impl, rows)), "got list_iterator"),
    ],
)
def test_result_must_hold_exactly_one_result_per_row(
    respond: Callable, got: str
) -> None:
    session, observer = gated_session()
    recording = session.instances[PAINT]
    recording.respond = respond

    with pytest.raises(StepExecutionError, match="one result per invocation") as raised:
        session.run({"values": [1.0, 2.0], "flags": [1, 1]})

    assert got in str(raised.value)
    assert raised.value.index is None
    assert observer.of(PAINT) == []
    assert len(observer.errors) == 1
    assert (len(recording.batches), recording.scalar) == (1, [])


def raise_for(position: int, original: Exception) -> Callable:
    def respond(implementation: Any, invocations: Any) -> list:
        try:
            raise original
        except Exception as error:
            raise InvocationBatchError(position) from error

    return respond


def test_indexed_failure_names_the_original_index_and_cause() -> None:
    session, observer = gated_session()
    recording = session.instances[PAINT]
    original = ValueError("row is broken")
    recording.respond = raise_for(1, original)

    with pytest.raises(StepExecutionError) as raised:
        session.run({"values": [0.0, 1.0, 2.0, 3.0], "flags": [1, 0, 1, 1]})

    error = raised.value
    assert error.index == (2,)
    assert error.__cause__ is original
    assert "at index [2]" in str(error)
    assert "invocation 1 raised ValueError: row is broken" in str(error)
    assert [item.index for item in observer.errors] == [(2,)]
    assert observer.of(PAINT) == []
    assert (len(recording.batches), recording.scalar) == (1, [])


def mutated(position: Any) -> Callable:
    def respond(implementation: Any, invocations: Any) -> list:
        error = InvocationBatchError(0)
        error.position = position
        raise error from ValueError("hidden row failure")

    return respond


def rejected(position: Any) -> Callable:
    def respond(implementation: Any, invocations: Any) -> list:
        raise InvocationBatchError(position)

    return respond


@pytest.mark.parametrize(
    "respond, message",
    [
        (rejected(3), "position 3, which is not a position of its 3 invocation"),
        (mutated(True), "position True, which is not a position"),
        (mutated(-1), "position -1, which is not a position"),
        (mutated("0"), "position '0', which is not a position"),
        (rejected(True), "block raised ContractError"),
        (rejected(-1), "block raised ContractError"),
    ],
)
def test_malformed_position_fails_the_whole_call(
    respond: Callable, message: str
) -> None:
    session, observer = gated_session()
    recording = session.instances[PAINT]
    recording.respond = respond

    with pytest.raises(StepExecutionError, match=message) as raised:
        session.run({"values": [0.0, 1.0, 2.0], "flags": [1, 1, 1]})

    assert raised.value.index is None
    assert isinstance(raised.value.__cause__, (InvocationBatchError, ContractError))
    assert observer.of(PAINT) == []
    assert (len(recording.batches), recording.scalar) == (1, [])


def unindexed(implementation: Any, invocations: Any) -> list:
    raise RuntimeError("the whole batch failed")


def failing_future(implementation: Any, invocations: Any) -> list:
    late = InvocationBatchError(0)
    late.__cause__ = ValueError("late row failure")
    future: Future = Future()
    future.set_exception(late)

    return [future] + rows_of(implementation, invocations[1:])


@pytest.mark.parametrize(
    "respond, message, cause",
    [
        (unindexed, "block raised RuntimeError", RuntimeError),
        (
            failing_future,
            "a future returned by the block failed",
            InvocationBatchError,
        ),
    ],
)
def test_unindexed_and_future_failures_belong_to_the_whole_call(
    respond: Callable, message: str, cause: type
) -> None:
    session, observer = gated_session()
    recording = session.instances[PAINT]
    recording.respond = respond

    with pytest.raises(StepExecutionError, match=message) as raised:
        session.run({"values": [0.0, 1.0], "flags": [1, 1]})

    assert raised.value.index is None
    assert type(raised.value.__cause__) is cause
    assert observer.of(PAINT) == []
    assert (len(recording.batches), recording.scalar) == (1, [])


# --- readiness, context and demand ---------------------------------------------


def test_rows_are_observed_and_read_downstream_only_after_every_future() -> None:
    log: List[tuple] = []
    definition = workflow(
        [
            paint_step(value="$inputs.values"),
            {"type": Echo.type, "name": "echo", "value": "$steps.paint.painted"},
        ],
        {"echoed": "$steps.echo.value"},
        inputs=[VALUES],
    )
    observer = Observed(log=log)
    session = compiled(definition).create_session(observer=observer)

    def respond(implementation: Any, invocations: Any) -> list:
        ready: Future = Future()
        late: Future = Future()
        ready.set_result(painted(invocations[0]))
        log.append(("resolved", (0,)))

        def finish() -> None:
            log.append(("resolved", (1,)))
            late.set_result(painted(invocations[1]))

        threading.Timer(0.05, finish).start()
        return [ready, late]

    session.instances[PAINT].respond = respond

    rows = session.run({"values": [1.0, 2.0]}).rows()

    assert log == [
        ("resolved", (0,)),
        ("resolved", (1,)),
        ("observed", PAINT, (0,)),
        ("observed", PAINT, (1,)),
        ("observed", ("echo",), (0,)),
        ("observed", ("echo",), (1,)),
    ]
    assert [item["result"] for item in observer.of(PAINT)] == [
        {"painted": 1.0},
        {"painted": 2.0},
    ]
    assert rows == [{"echoed": 1.0}, {"echoed": 2.0}]


def test_call_context_names_every_original_index_and_its_source() -> None:
    session, _ = gated_session()
    seen: Dict[str, Any] = {}

    def respond(implementation: Any, invocations: Any) -> list:
        context = implementation.execution_context
        seen["sources"] = [context.source_id_at(index) for index in context.indices]
        seen["field_sources"] = [
            context.source_id_at(index, field="value") for index in context.indices
        ]
        try:
            seen["scalar_source"] = context.source_id
        except ContractError as error:
            seen["scalar_source"] = str(error)
        return rows_of(implementation, invocations)

    session.instances[PAINT].respond = respond
    values = InputValue(
        [0.0, 1.0, 2.0],
        metadata=EntryMetadata(
            sample={
                (position,): SampleContext(source_id=source)
                for position, source in enumerate("abc")
            }
        ),
    )

    session.run({"values": values, "flags": [0, 1, 1]})

    assert seen["sources"] == ["b", "c"]
    assert seen["field_sources"] == ["b", "c"]
    assert "batch-delivering call" in seen["scalar_source"]


def wanting_session() -> Any:
    definition = workflow(
        [paint_step(Wanting, value="$inputs.values")],
        {"painted": "$steps.paint.painted", "extra": "$steps.paint.extra"},
        inputs=[VALUES],
    )
    session = compiled(definition, requested_outputs=("painted",)).create_session()

    return session


def first_row_with_extra(implementation: Any, invocations: Any) -> list:
    results = rows_of(implementation, invocations)
    results[0]["extra"] = 0.0

    return results


def test_wants_answered_for_the_call_lets_any_row_omit_that_output() -> None:
    session = wanting_session()
    asked: List[bool] = []

    def respond(implementation: Any, invocations: Any) -> list:
        asked.append(implementation.wants("extra"))
        return first_row_with_extra(implementation, invocations)

    session.instances[PAINT].respond = respond

    rows = session.run({"values": [1.0, 2.0]}).rows()

    assert asked == [False]
    assert rows == [{"painted": 1.0}, {"painted": 2.0}]


def test_unqueried_omission_is_still_an_error_at_its_row() -> None:
    session = wanting_session()
    session.instances[PAINT].respond = first_row_with_extra

    with pytest.raises(StepExecutionError, match="never asked about") as raised:
        session.run({"values": [1.0, 2.0]})

    assert raised.value.index == (1,)


# --- pipelined runs ------------------------------------------------------------


def test_pipelined_runs_take_one_whole_call_turn_per_admitting_run() -> None:
    plan = compiled(simple())
    with plan.create_session() as session:
        recording = session.instances[PAINT]
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            futures = [
                pipeline.submit({"values": values})
                for values in ([1.0, 2.0], [], [3.0])
            ]
        results = [future.result(timeout=30).rows() for future in futures]
        stage = pipeline.counters.snapshot()["stages"]["$steps.paint#call"]
        indices = [batch["indices"] for batch in recording.batches]

    assert results == [[{"painted": 1.0}, {"painted": 2.0}], [], [{"painted": 3.0}]]
    assert stage["calls"] == 2
    assert indices == [((0,), (1,)), ((0,),)]


def test_failed_pipelined_call_aborts_without_retrying_rows() -> None:
    def respond(implementation: Any, invocations: Any) -> list:
        if any(row["value"] < 0 for row in invocations):
            raise RuntimeError("negative value refused")
        return rows_of(implementation, invocations)

    plan = compiled(simple())
    with plan.create_session() as session:
        recording = session.instances[PAINT]
        recording.respond = respond
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            pipeline.submit({"values": [1.0]})
            failing = pipeline.submit({"values": [-1.0, 2.0]})
        with pytest.raises(StepExecutionError, match="refused"):
            failing.result(timeout=30)
        sizes = [len(batch["invocations"]) for batch in recording.batches]

    assert isinstance(pipeline.failure, StepExecutionError)
    assert sizes == [1, 2]
    assert recording.scalar == []


def test_conditional_batch_contract_with_constant_input_keeps_scalar_route() -> None:
    class ConditionalRows(Implementation):
        name = "conditional"

        def run(self, *, value):
            return {"painted": value}

        def run_many(self, *, invocations):
            raise AssertionError("conditional batch contract must keep its route")

    class Conditional(Block):
        type = "test/run_many_conditional@v1"
        prunable = True
        outputs = {"painted": Output(FLOAT_KIND)}
        implementations = (ConditionalRows,)

        class Params(BlockParams):
            value: Ref(FLOAT_KIND, batch="if_varying")

    definition = workflow(
        [paint_step(Conditional, value="$inputs.value")],
        {"painted": "$steps.paint.painted"},
        inputs=[parameter("value", kind=["float"])],
    )
    plan = compiled(definition, catalogue=Catalogue([Conditional]))
    step = plan.step(PAINT)

    assert step.spec.accepts_batches
    assert not step.delivers_batches
    assert not step.coalesces_invocations
    assert plan.create_session().run({"value": 2.0}).rows() == [{"painted": 2.0}]


def test_root_singleton_context_keeps_empty_index_and_rejects_scalar_source() -> None:
    definition = workflow(
        [paint_step(value="$inputs.value")],
        {"painted": "$steps.paint.painted"},
        inputs=[parameter("value", kind=["float"])],
    )
    session = compiled(definition).create_session()

    def respond(implementation, invocations):
        context = implementation.execution_context
        assert context.indices == ((),)
        assert context.batched
        with pytest.raises(ContractError, match="batch-delivering call"):
            _ = context.source_id
        assert context.sample_at(()) is None
        with pytest.raises(ContractError, match="no single source"):
            context.source_id_at(())
        return rows_of(implementation, invocations)

    session.instances[PAINT].respond = respond

    assert session.run({"value": 2.0}).rows() == [{"painted": 2.0}]


def test_phased_upstream_composes_with_run_many_downstream() -> None:
    definition = workflow(
        [
            {"type": PhasedPaint.type, "name": "model", "value": "$inputs.values"},
            paint_step(value="$steps.model.painted"),
        ],
        {"painted": "$steps.paint.painted"},
    )
    plan = compiled(definition, block_execution="phases")
    session = plan.create_session()

    assert plan.step(("model",)).execution == "phases"
    assert plan.step(PAINT).execution == "run"
    assert plan.step(PAINT).coalesces_invocations
    assert session.run({"values": [1.0, 2.0]}).rows() == [
        {"painted": 1.0}, {"painted": 2.0}
    ]
    assert len(session.instances[PAINT].batches) == 1
    assert session.instances[("model",)].batches == []


@pytest.mark.parametrize("observer_fails", [False, True])
def test_pending_batch_holds_the_stage_and_observer_failure_releases_waiters(
    observer_fails: bool,
) -> None:
    from tests.v2.integration.engine.runtime.pipelining.test_phase_stages import (
        wait_for,
    )

    ready = Future()
    entered = threading.Event()
    failure = RuntimeError("observer refused second row")

    class Watching(Observed):
        def on_invocation(self, **kwargs):
            super().on_invocation(**kwargs)
            if observer_fails and kwargs["index"] == (1,):
                raise failure

    observer = Watching()
    with compiled(simple()).create_session(observer=observer) as session:
        recording = session.instances[PAINT]

        def respond(implementation, invocations):
            if invocations[0]["value"] == 1.0:
                entered.set()
                return [{"painted": ready}, {"painted": 2.0}]
            return rows_of(implementation, invocations)

        recording.respond = respond
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            first = pipeline.submit({"values": [1.0, 2.0]})
            second = pipeline.submit({"values": [3.0]})
            try:
                assert entered.wait(5)
                wait_for(
                    lambda: pipeline.counters.stage("$steps.paint#call").waited_turn
                    == 1
                )
                assert len(recording.batches) == 1
                assert not first.done() and not second.done()
                assert observer.of(PAINT) == []
            finally:
                ready.set_result(1.0)

        assert first.done() and second.done()
        assert recording.scalar == []
        if observer_fails:
            with pytest.raises(RuntimeError, match="observer refused") as raised:
                first.result(timeout=5)
            assert raised.value is failure
            assert pipeline.failure is failure
            assert (0,) in [item["index"] for item in observer.of(PAINT)]
            # The ready-call stage retires before callbacks; the successor may
            # finish or abort, but may not remain stranded behind that stage.
        else:
            assert first.result(timeout=5).rows() == [
                {"painted": 1.0}, {"painted": 2.0}
            ]
            assert second.result(timeout=5).rows() == [{"painted": 3.0}]
            assert pipeline.counters.stage("$steps.paint#call").calls == 2
