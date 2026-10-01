"""Phase execution inside the engine: equivalence, isolation, gates, batches, errors.

Each workflow is compiled twice, with ``block_execution="run"`` and
``"phases"``; the dispatch is the only difference between the plans.
"""

import gc
import inspect
import weakref
from concurrent.futures import Future
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Selected,
    Selection,
)
from roboflow_workflows.execution_engine.v2.errors import (
    StepExecutionError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from roboflow_workflows.execution_engine.v2.targets import Target

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    Gate,
    Scale,
    gate,
    nested,
    workflow,
)

MODES = ("run", "phases")


class Holder:
    """A private intermediate whose lifetime a test can watch."""

    def __init__(self, value: float) -> None:
        self.value = value


class Diamond(Block):
    """base -> (left, right) -> total. Records each phase call and intermediate."""

    type = "test/diamond@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        offset: float = 1.0

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.intermediates: List[weakref.ref] = []

    @phase
    def base(self, *, value):
        if value < 0:
            raise ValueError(f"negative value {value}")
        holder = Holder(value)
        self.intermediates.append(weakref.ref(holder))
        self.calls.append(("base", value))
        return holder

    @phase
    def left(self, *, base):
        self.calls.append(("left", base.value))
        return base.value * 2

    @phase
    def right(self, *, base, offset):
        self.calls.append(("right", base.value))
        return base.value + offset

    @phase
    def total(self, *, left, right):
        self.calls.append(("total", left))
        return {"total": left + right}

    def run(self, *, value, offset):
        base = self.base(value=value)
        left = self.left(base=base)
        right = self.right(base=base, offset=offset)
        return self.total(left=left, right=right)


class BatchedDiamond(Block):
    """Batch-delivering phases: the result phase returns one result per invocation."""

    type = "test/batched_diamond@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")  # noqa: F821 (batch mode is a value)

    def __init__(self) -> None:
        self.batches: List[list] = []

    @phase
    def doubled(self, *, values):
        self.batches.append(list(values))
        return [value * 2 for value in values]

    @phase
    def squared(self, *, values):
        return [value * value for value in values]

    @phase
    def totals(self, *, doubled, squared):
        return [{"total": a + b} for a, b in zip(doubled, squared)]

    def run(self, *, values):
        return self.totals(
            doubled=self.doubled(values=values), squared=self.squared(values=values)
        )


class Aliases(Block):
    """A phase result holding one list twice, behind futures, mutated downstream."""

    type = "test/aliases@v1"
    outputs = {"same": Output(), "seen": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    @phase
    def pair(self, *, value):
        shared: List[Any] = [_done(value)]
        return {"a": shared, "b": shared}

    @phase
    def grown(self, *, pair):
        pair["a"].append("grown")
        return pair

    @phase
    def report(self, *, grown):
        return {"same": grown["a"] is grown["b"], "seen": list(grown["b"])}

    def run(self, *, value):
        return self.report(grown=self.grown(pair=self.pair(value=value)))


class Largest(Block):
    """Chooses the largest member of a group; the value arrives as a future."""

    type = "test/largest@v1"
    outputs = {
        "largest": Output(FLOAT_KIND, source="values", context_policy="selected")
    }

    class Params(BlockParams):
        values: Group(FLOAT_KIND)

    @phase
    def position(self, *, values):
        return max(range(len(values)), key=lambda item: values[item])

    @phase
    def chosen(self, *, values, position):
        return {
            "largest": Selected(values.indices[position], value=_done(values[position]))
        }

    def run(self, *, values):
        return self.chosen(values=values, position=self.position(values=values))


class SharedChoice(Block):
    """Two selected outputs and one Selection holding one list behind a future."""

    type = "test/shared_choice@v1"
    outputs = {
        "first": Output(source="values", context_policy="selected"),
        "last": Output(source="values", context_policy="selected"),
        "both": Output(source="values", context_policy="selected", expand="picks"),
    }

    class Params(BlockParams):
        values: Group(FLOAT_KIND)

    @phase
    def shared(self, *, values):
        return [_done(values[0])]

    @phase
    def chosen(self, *, values, shared):
        first, last = values.indices[0], values.indices[-1]
        return {
            "first": Selected(first, shared),
            "last": Selected(last, shared),
            "both": Selection([first, last], values=[shared, shared]),
        }

    def run(self, *, values):
        return self.chosen(values=values, shared=self.shared(values=values))


async def _unawaited(value: float) -> float:
    return value


class CoroutineChoice(Largest):
    """Returns a coroutine as the chosen value; keeps it for inspection."""

    type = "test/coroutine_choice@v1"

    def __init__(self) -> None:
        self.coroutines: List[Any] = []

    def chosen(self, *, values, position):
        coroutine = _unawaited(values[position])
        self.coroutines.append(coroutine)
        return {"largest": Selected(values.indices[position], value=coroutine)}


class Remote(Block):
    """Unphased: returns a private payload next to a failing future."""

    type = "test/remote@v1"
    outputs = {"reply": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self) -> None:
        self.private: List[weakref.ref] = []

    def run(self, *, value):
        private = Holder(value)
        self.private.append(weakref.ref(private))
        failed: Future = Future()
        failed.set_exception(ValueError("server went away"))
        return {"reply": [private, failed]}


def _done(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


class Fast(Implementation):
    name = "fast"
    requires = ("cpu", "fast")

    def __init__(self, *, factor: float) -> None:
        self.factor = factor
        self.calls: List[str] = []

    @phase
    def scaled(self, *, value):
        self.calls.append("scaled")
        return value * self.factor

    @phase
    def result(self, *, scaled):
        self.calls.append("result")
        return {"scaled": scaled}

    def run(self, *, value):
        return self.result(scaled=self.scaled(value=value))


class Portable(Implementation):
    name = "portable"
    requires = ("cpu",)

    def run(self, *, value):
        return {"scaled": value * 10}


class Multiply(Block):
    """Logical contract; the target picks the implementation."""

    type = "test/multiply@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}
    implementations = (Fast, Portable)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


CATALOGUE = Catalogue(
    [
        Diamond,
        BatchedDiamond,
        Aliases,
        Largest,
        SharedChoice,
        CoroutineChoice,
        Remote,
        Multiply,
        Gate,
        Scale,
    ],
    namespace="test",
)


def compiled(definition: dict, *, mode: str, **options: Any):
    plan = compile_workflow(
        definition,
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode, **options),
    )

    return plan


def diamond_workflow(**params: Any) -> dict:
    definition = workflow(
        [
            {
                "type": Diamond.type,
                "name": "diamond",
                "value": "$inputs.values",
                **params,
            }
        ],
        {"total": "$steps.diamond.total"},
    )

    return definition


def phase_events(result) -> List[tuple]:
    events = [
        (event["phase"], tuple(map(tuple, event["indices"])))
        for event in result.trace
        if event["event"] == "phase"
    ]

    return events


# Equivalence and once-only execution ----------------------------------------------


def test_run_and_phase_modes_agree_and_each_phase_runs_once_per_invocation() -> None:
    results = {}
    for mode in MODES:
        plan = compiled(diamond_workflow(), mode=mode)
        session = plan.create_session()
        results[mode] = session.run({"values": [1.0, 2.0]})
        assert plan.step(("diamond",)).execution == mode
        assert [name for name, _ in session.instances[("diamond",)].calls] == [
            "base",
            "left",
            "right",
            "total",
        ] * 2

    assert (
        results["run"].rows()
        == results["phases"].rows()
        == [
            {"total": 4.0},
            {"total": 7.0},
        ]
    )
    assert phase_events(results["run"]) == []
    assert phase_events(results["phases"]) == [
        (name, (index,))
        for index in ((0,), (1,))
        for name in ("base", "left", "right", "total")
    ]


def test_intermediates_are_private_to_each_call_and_released_after_the_run() -> None:
    plan = compiled(diamond_workflow(), mode="phases")
    session = plan.create_session()
    block = session.instances[("diamond",)]

    first = session.run({"values": [1.0, 2.0]})
    second = session.run({"values": [3.0]})
    gc.collect()

    assert [row["total"] for row in first.rows() + second.rows()] == [4.0, 7.0, 10.0]
    # Each call made its own base from its own value; none survives the run.
    assert [call for call in block.calls if call[0] == "left"] == [
        ("left", 1.0),
        ("left", 2.0),
        ("left", 3.0),
    ]
    assert len(block.intermediates) == 3
    assert all(reference() is None for reference in block.intermediates)
    assert set(vars(block)) == {"calls", "intermediates"}


def test_nested_instances_of_a_phased_block_run_their_own_phases() -> None:
    child = workflow(
        [{"type": Diamond.type, "name": "diamond", "value": "$inputs.values"}],
        {"total": "$steps.diamond.total"},
    )
    definition = workflow(
        [
            nested(
                "first",
                workflow_definition=child,
                parameter_bindings={"values": "$inputs.values"},
            ),
            nested(
                "second",
                workflow_definition=child,
                parameter_bindings={"values": "$inputs.values"},
            ),
        ],
        {"first": "$steps.first.total", "second": "$steps.second.total"},
    )
    plan = compiled(definition, mode="phases")
    session = plan.create_session()

    result = session.run({"values": [1.0, 2.0]})

    assert result.rows() == [
        {"first": 4.0, "second": 4.0},
        {"first": 7.0, "second": 7.0},
    ]
    first = session.instances[("first", "diamond")]
    second = session.instances[("second", "diamond")]
    assert first is not second
    assert first.calls == second.calls
    assert [name for name, _ in first.calls] == ["base", "left", "right", "total"] * 2
    assert {event[0] for event in phase_events(result)} == {
        "base",
        "left",
        "right",
        "total",
    }
    assert len(phase_events(result)) == 16


def test_denied_invocations_run_no_phase() -> None:
    definition = workflow(
        [
            gate("admit", "$inputs.values", ["diamond"]),
            {"type": Diamond.type, "name": "diamond", "value": "$inputs.values"},
        ],
        {"total": "$steps.diamond.total"},
    )
    plan = compiled(definition, mode="phases")
    session = plan.create_session()

    result = session.run({"values": [0.0, 2.0, 0.0]})

    assert result.rows() == [{"total": None}, {"total": 7.0}, {"total": None}]
    assert session.instances[("diamond",)].calls == [
        ("base", 2.0),
        ("left", 2.0),
        ("right", 2.0),
        ("total", 4.0),
    ]
    assert {indices for _, indices in phase_events(result)} == {((1,),)}


# Batches -------------------------------------------------------------------------


def test_batch_delivering_phases_get_the_batch_once_and_return_a_list() -> None:
    definition = workflow(
        [{"type": BatchedDiamond.type, "name": "many", "values": "$inputs.values"}],
        {"total": "$steps.many.total"},
    )
    rows = {}
    for mode in MODES:
        session = compiled(definition, mode=mode).create_session()
        result = session.run({"values": [1.0, 2.0, 3.0]})
        rows[mode] = result.rows()
        assert session.instances[("many",)].batches == [[1.0, 2.0, 3.0]]
        assert session.run({"values": []}).rows() == []
        assert session.instances[("many",)].batches == [[1.0, 2.0, 3.0]]

    assert (
        rows["run"]
        == rows["phases"]
        == [{"total": 3.0}, {"total": 8.0}, {"total": 15.0}]
    )
    assert phase_events(result) == [
        (name, ((0,), (1,), (2,))) for name in ("doubled", "squared", "totals")
    ]


# Readiness and aliases ------------------------------------------------------------


def test_consumers_see_ready_values_and_shared_mutable_aliases() -> None:
    definition = workflow(
        [{"type": Aliases.type, "name": "aliases", "value": "$inputs.values"}],
        {"same": "$steps.aliases.same", "seen": "$steps.aliases.seen"},
    )
    for mode in MODES:
        session = compiled(definition, mode=mode).create_session()

        assert session.run({"values": [5.0]}).rows() == [
            {"same": True, "seen": [5.0, "grown"]}
        ]


def test_selected_results_of_the_result_phase_are_resolved_like_run_results() -> None:
    definition = workflow(
        [{"type": Largest.type, "name": "largest", "values": "$inputs.groups"}],
        {"largest": "$steps.largest.largest"},
        inputs=[
            {
                "type": "WorkflowBatchInput",
                "name": "groups",
                "kind": ["float"],
                "dimensionality": 2,
            }
        ],
    )
    for mode in MODES:
        session = compiled(definition, mode=mode).create_session()

        assert session.run({"groups": [[1.0, 3.0, 2.0], [5.0, 4.0]]}).rows() == [
            {"largest": 3.0},
            {"largest": 5.0},
        ]


GROUPS = [
    {
        "type": "WorkflowBatchInput",
        "name": "groups",
        "kind": ["float"],
        "dimensionality": 2,
    }
]


def test_wrapper_payloads_share_one_resolution_across_outputs() -> None:
    definition = workflow(
        [{"type": SharedChoice.type, "name": "choice", "values": "$inputs.groups"}],
        {
            "first": "$steps.choice.first",
            "last": "$steps.choice.last",
            "both": "$steps.choice.both",
        },
        inputs=GROUPS,
    )
    for mode in MODES:
        session = compiled(definition, mode=mode).create_session()

        (row,) = session.run({"groups": [[1.0, 2.0]]}).rows()

        assert row["first"] == [1.0]
        assert row["first"] is row["last"] is row["both"][0] is row["both"][1]


@pytest.mark.parametrize("mode", MODES)
def test_coroutine_inside_a_selected_result_fails_with_the_phase_named(mode) -> None:
    definition = workflow(
        [{"type": CoroutineChoice.type, "name": "choice", "values": "$inputs.groups"}],
        {"largest": "$steps.choice.largest"},
        inputs=GROUPS,
    )
    session = compiled(definition, mode=mode).create_session()

    with pytest.raises(StepExecutionError) as caught:
        session.run({"groups": [[1.0, 3.0]]})

    assert (caught.value.phase, caught.value.index) == ("chosen", (0,))
    assert "coroutine" in str(caught.value)
    (coroutine,) = session.instances[("choice",)].coroutines
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


# Errors ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_phase_failure_names_step_index_and_phase_and_the_session_recovers(
    mode,
) -> None:
    handled: List[StepExecutionError] = []
    session = compiled(diamond_workflow(), mode=mode).create_session(
        error_handler=handled.append
    )

    with pytest.raises(StepExecutionError) as caught:
        session.run({"values": [1.0, -1.0]})

    error = caught.value
    assert (error.step_path, error.index, error.phase) == (("diamond",), (1,), "base")
    assert str(error) == (
        "$steps.diamond at index [1] (test/diamond@v1): phase 'base' failed: "
        "ValueError: negative value -1.0"
    )
    assert isinstance(error.__cause__, ValueError)
    assert handled == [error]
    assert session.run({"values": [2.0]}).rows() == [{"total": 7.0}]


def test_failed_future_error_keeps_the_failure_but_not_the_returned_result() -> None:
    definition = workflow(
        [{"type": Remote.type, "name": "remote", "value": "$inputs.values"}],
        {"reply": "$steps.remote.reply"},
    )
    session = compiled(definition, mode="run").create_session()

    with pytest.raises(StepExecutionError) as caught:
        session.run({"values": [1.0]})
    gc.collect()

    assert isinstance(caught.value.__cause__, ValueError)
    assert [reference() for reference in session.instances[("remote",)].private] == [
        None
    ]


def test_private_phase_names_are_not_workflow_selectors() -> None:
    definition = workflow(
        [{"type": Diamond.type, "name": "diamond", "value": "$inputs.values"}],
        {"base": "$steps.diamond.base"},
    )

    with pytest.raises(WorkflowCompileError, match="base"):
        compiled(definition, mode="phases")


# Implementations --------------------------------------------------------------------


def multiply_workflow() -> dict:
    definition = workflow(
        [{"type": Multiply.type, "name": "multiply", "value": "$inputs.values"}],
        {"scaled": "$steps.multiply.scaled"},
    )

    return definition


def test_selected_implementation_is_constructed_with_its_resources_and_phases() -> None:
    target = Target(frozenset({"cpu", "fast"}))
    rows: Dict[str, list] = {}
    for mode in MODES:
        plan = compiled(multiply_workflow(), mode=mode, target=target)
        session = plan.create_session(resources={"factor": 3.0})
        instance = session.instances[("multiply",)]
        result = session.run({"values": [1.0, 2.0]})
        rows[mode] = result.rows()
        assert isinstance(instance, Fast) and instance.factor == 3.0
        assert instance.calls == ["scaled", "result"] * 2
        assert plan.step(("multiply",)).selected.name == "fast"
        assert len(phase_events(result)) == (4 if mode == "phases" else 0)

    assert rows["run"] == rows["phases"] == [{"scaled": 3.0}, {"scaled": 6.0}]


def test_unphased_selection_falls_back_to_run_in_phase_mode() -> None:
    plan = compiled(multiply_workflow(), mode="phases")
    session = plan.create_session()

    result = session.run({"values": [1.0]})

    assert plan.step(("multiply",)).selected.name == "portable"
    assert plan.step(("multiply",)).execution == "run"
    assert isinstance(session.instances[("multiply",)], Portable)
    assert result.rows() == [{"scaled": 10.0}]
    assert phase_events(result) == []
