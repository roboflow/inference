"""Passive history: Window state kept across ``session.run`` calls.

A plan with operators and no sources keeps its operators in the session.
Each run pushes the current inputs; a full window runs its consumers inline,
in the same run, and the run's result holds both the current values and the
collected history. A warming window leaves its fields ``None``.
"""

import copy
import threading
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.image.primitives.mosaic.v1 import MosaicBlock
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    InputValue,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine import Catalogue, compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.v2.engine.errors import ContractError
from roboflow_workflows.v2.engine.operators import (
    Operator,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.pipelining.passive import (
    PipelineAbortedError,
    PipelineFullError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

# Test hooks the blocks below read; cleared around every test.
HOOKS: Dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _clear_hooks():
    HOOKS.clear()
    yield
    HOOKS.clear()


class Echo(Block):
    type = "test/echo@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Any | Ref(allow_none=True)

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def run(self, *, value):
        self.calls.append(value)
        if value == 4 and "fourth_started" in HOOKS:
            HOOKS["fourth_started"].set()
        return {"out": value}


class Reduce(Block):
    """Sums (or counts) a collected window; hooks can block or fail it."""

    type = "test/reduce@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        values: Group()
        tag: str = "sum"

    def run(self, *, values, tag):
        if "barrier" in HOOKS:
            HOOKS["barrier"].wait(timeout=3)
        if "entered" in HOOKS:
            HOOKS["entered"].set()
            assert HOOKS["release"].wait(timeout=3)
        return {"out": sum(values) if tag == "sum" else len(values)}


class OwnImage(Block):
    """Clones pixels, so the history never aliases the caller's image."""

    type = "test/own_image@v1"
    outputs = {"out": Output(IMAGE_KIND)}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image):
        return {"out": image.with_pixels(image.tensor_image.clone())}


class Mutate(Block):
    type = "test/mutate@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Any | Ref()

    def run(self, *, value):
        value["n"] += 1
        return {"out": value}


class Keep(Block):
    """Continues to the targets when the value is truthy."""

    type = "test/keep@v1"

    class Params(BlockParams):
        value: Ref()
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        return Select(next_steps) if value else Stop()


class Expand(Block):
    """Expands n into the children 10n, 10n+1, ... (n children)."""

    type = "test/expand@v1"
    outputs = {"child": Output(expand="child")}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value):
        return {"child": Batch.of([10 * value + k for k in range(value)])}


class Pause(Block):
    """Echoes the value; waits for ``resume`` while ``paused`` is hooked."""

    type = "test/pause@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value):
        if "paused" in HOOKS:
            HOOKS["paused"].set()
            assert HOOKS["resume"].wait(timeout=5)
        return {"out": value}


class TwoPulses(Operator):
    """Claims one emission per push but returns two."""

    type = "test/two_pulses@v1"
    input_roles = ("input",)
    passive_max_emissions = 1

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"out": OperatorPort()}

    def push(self, arrivals):
        pulse = OperatorPulse(
            ports={"out": arrivals[0].entry}, causes=(arrivals[0].pulse,)
        )
        return [pulse, pulse]


CATALOGUE = Catalogue(
    [Echo, Reduce, OwnImage, Mutate, Keep, Expand, Pause, MosaicBlock],
    operators=[Window, TwoPulses],
)


def _history_workflow(*, batch: bool = False, nested: bool = False) -> Dict[str, Any]:
    """``x`` -> echo -> size-3 Window -> sum; outputs current, history, ref, sum."""
    steps: List[Dict[str, Any]] = [
        {"type": Echo.type, "name": "root", "value": "$inputs.x"}
    ]
    if nested:
        steps.append(
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": {
                    "version": "2.0",
                    "inputs": [{"type": "WorkflowBatchInput", "name": "history"}],
                    "steps": [
                        {
                            "type": Reduce.type,
                            "name": "sum",
                            "values": "$inputs.history",
                        }
                    ],
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "sum",
                            "selector": "$steps.sum.out",
                        }
                    ],
                },
                "parameter_bindings": {"history": "$operators.history.x"},
            }
        )
        total = "$steps.child.sum"
    else:
        steps += [
            {"type": Reduce.type, "name": "sum", "values": "$operators.history.x"},
            {
                "type": Reduce.type,
                "name": "length",
                "values": "$operators.history.x",
                "tag": "len",
            },
        ]
        total = "$steps.sum.out"
    if batch:
        camera = {"id": "camera", "kind": "sample", "stationary": True}
        workflow_input = {"name": "x", "axes": [camera]}
    else:
        workflow_input = {"type": "WorkflowParameter", "name": "x"}
    outputs = {
        "current": "$steps.root.out",
        "history": "$operators.history.x",
        "ref": "$operators.history.ref",
        "sum": total,
    }
    definition = {
        "version": "2.0",
        "inputs": [workflow_input],
        "operators": [
            {
                "type": Window.type,
                "name": "history",
                "size": 3,
                "step": 1,
                "collect": {"x": "$steps.root.out"},
                "hold": {"ref": "$steps.root.out"},
            }
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def _session(definition: Dict[str, Any]):
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    return session


WARMING = {"history": None, "ref": None, "sum": None}


def test_window_warms_up_then_slides_across_runs() -> None:
    session = _session(_history_workflow())

    results = [session.run({"x": x}) for x in (1, 2, 3, 4)]

    assert [result.rows() for result in results] == [
        [{"current": 1, **WARMING}],
        [{"current": 2, **WARMING}],
        [{"current": 3, "history": [1, 2, 3], "ref": 3, "sum": 6}],
        [{"current": 4, "history": [2, 3, 4], "ref": 4, "sum": 9}],
    ]
    # Warming: the field is filtered as a whole, not an empty group.
    assert results[0].statuses["history"] == "filtered"
    assert results[0].filtered_paths["history"] == ((),)
    # Full: a real Batch with a time axis, not a plain list.
    collected = results[2].outputs
    assert collected.layout["history"].axes[-1].kind == "time"
    assert collected.data["history"].layout == collected.layout["history"]
    session.close()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_pipeline_matches_serial_runs(workers: int, nested: bool) -> None:
    plan = compile_workflow(
        _history_workflow(batch=True, nested=nested), catalogue=CATALOGUE
    )
    serial = plan.create_session()
    expected = [serial.run({"x": [x, x * 10]}).rows() for x in range(1, 7)]
    session = plan.create_session()
    options = PipelineOptions(max_in_flight=2, max_workflow_parallelism=workers)

    with session.pipeline(options=options) as pipeline:
        futures = [pipeline.submit({"x": [x, x * 10]}) for x in range(1, 7)]
        results = [future.result(timeout=5) for future in futures]

    assert [result.rows() for result in results] == expected
    assert expected[0] == [{"current": 1, **WARMING}, {"current": 10, **WARMING}]
    assert expected[2] == [
        {"current": 3, "history": [1, 2, 3], "ref": 3, "sum": 6},
        {"current": 30, "history": [10, 20, 30], "ref": 30, "sum": 60},
    ]
    assert [axis.kind for axis in results[2].outputs.layout["history"].axes] == [
        "sample",
        "time",
    ]
    counters = pipeline.counters.snapshot()
    assert counters["peak"]["executing"] <= 2
    assert counters["peak"]["live_states"] <= 2
    assert counters["peak"]["parallel_units"] <= 2 * workers
    assert all(value == 0 for value in counters["current"].values())
    assert pipeline.outcomes == {
        "submitted": 6,
        "completed": 6,
        "failed": 0,
        "aborted": 0,
        "full": 0,
    }
    serial.close()
    session.close()


def test_emitted_steps_share_the_submission_team_without_extra_queues() -> None:
    definition = _history_workflow()
    definition["outputs"].append(
        {"type": "JsonField", "name": "length", "selector": "$steps.length.out"}
    )
    session = _session(definition)
    # Both emitted reducers must run at once on the run's two workers.
    HOOKS["barrier"] = threading.Barrier(2)
    options = PipelineOptions(max_in_flight=2, max_workflow_parallelism=2)

    with session.pipeline(options=options) as pipeline:
        for x in (1, 2):
            assert pipeline.submit({"x": x}).result(timeout=5).rows()[0]["sum"] is None
        full = [pipeline.submit({"x": x}) for x in (3, 4)]
        rows = [future.result(timeout=5).rows()[0] for future in full]

    assert [(row["sum"], row["length"]) for row in rows] == [(6, 3), (9, 3)]
    assert len(pipeline._pool.threads) == 4  # max_in_flight x parallelism
    assert not any(thread.is_alive() for thread in pipeline._pool.threads)
    counters = pipeline.counters.snapshot()
    assert counters["peak"]["parallel_units"] >= 2
    assert counters["peak"]["executing"] <= 2
    assert counters["peak"]["live_states"] <= 2
    assert counters["peak"]["queued"] == 0
    assert counters["peak"]["pending_operator_pulses"] == 0
    session.close()


def test_emitted_work_stays_inside_the_max_in_flight_budget() -> None:
    session = _session(_history_workflow())
    options = PipelineOptions(max_in_flight=2, max_workflow_parallelism=1)

    with session.pipeline(options=options) as pipeline:
        for x in (1, 2):
            pipeline.submit({"x": x}).result(timeout=5)
        HOOKS.update(
            entered=threading.Event(),
            release=threading.Event(),
            fourth_started=threading.Event(),
        )
        third = pipeline.submit({"x": 3})
        assert HOOKS["entered"].wait(timeout=3)  # third's reducer is blocked
        fourth = pipeline.submit({"x": 4})
        try:
            assert HOOKS["fourth_started"].wait(timeout=3)
            with pytest.raises(PipelineFullError):
                pipeline.submit({"x": 5}, block=False)
        finally:
            HOOKS["release"].set()
        assert third.result(timeout=5).rows()[0]["sum"] == 6
        assert fourth.result(timeout=5).rows()[0]["sum"] == 9

    counters = pipeline.counters.snapshot()
    assert counters["peak"]["executing"] == 2
    assert counters["peak"]["live_states"] == 2
    assert pipeline.outcomes["full"] == 1
    session.close()


def test_current_and_collected_values_keep_their_own_clocks() -> None:
    definition = _history_workflow()
    definition["steps"] = definition["steps"][:1]
    definition["outputs"] = definition["outputs"][:3]
    session = _session(definition)
    supplied = []

    for x, clock in enumerate(("clock-A", "clock-B", "clock-A"), start=1):
        temporal = TemporalContext(Timestamp(x, 1, clock))
        supplied.append(temporal)
        metadata = EntryMetadata(temporal={(): temporal})
        result = session.run({"x": InputValue(x, metadata=metadata)})

    assert result.rows() == [{"current": 3, "history": [1, 2, 3], "ref": 3}]
    metadata = result.outputs.metadata
    assert metadata["current"].temporal_at(()) == supplied[-1]
    assert metadata["ref"].temporal_at(()) == supplied[-1]
    assert [metadata["history"].temporal_at((i,)) for i in range(3)] == supplied
    assert metadata["history"].temporal_at(()) is None  # no common clock
    session.close()


def test_warming_operator_outputs_keep_one_row_per_input_row() -> None:
    definition = _history_workflow(batch=True)
    definition["outputs"] = [definition["outputs"][1]]
    session = _session(definition)

    assert session.run({"x": [1, 10]}).rows() == [{"history": None}] * 2
    session.close()


@pytest.mark.parametrize("submission", ["direct", "pipeline"])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_unused_input_axis_adds_no_rows(mode: str, submission: str) -> None:
    # H2-PROD-QA-003: rows follow the selected outputs' input axis. A grouped
    # input on an unrelated axis adds no rows, warming or emitted.
    definition = _history_workflow(batch=True)
    definition["outputs"] = [definition["outputs"][1]]
    unrelated = {"id": "unrelated", "kind": "sample", "stationary": True}
    definition["inputs"].append({"name": "unused", "axes": [unrelated]})
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    submissions = [{"x": [x, 10 * x], "unused": list(range(5))} for x in (1, 2, 3)]

    if submission == "direct":
        results = [session.run(inputs) for inputs in submissions]
    else:
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            futures = [pipeline.submit(inputs) for inputs in submissions]
            results = [future.result(timeout=5) for future in futures]
    *warming, emitted = results

    assert [result.rows() for result in warming] == [[{"history": None}] * 2] * 2
    assert emitted.rows() == [{"history": [1, 2, 3]}, {"history": [10, 20, 30]}]
    assert emitted.rows(serialize=True) == emitted.rows()
    session.close()


def test_image_history_feeds_a_mosaic_and_owns_its_pixels() -> None:
    camera = {"id": "camera", "kind": "sample", "stationary": True}
    definition = {
        "version": "2.0",
        "inputs": [{"name": "frame", "kind": ["image"], "axes": [camera]}],
        "steps": [
            {"type": OwnImage.type, "name": "owned", "image": "$inputs.frame"},
            {
                "type": MosaicBlock.type,
                "name": "mosaic",
                "images": "$operators.history.images",
                "tile_size": 2,
                "columns": 3,
            },
        ],
        "operators": [
            {
                "type": Window.type,
                "name": "history",
                "size": 3,
                "step": 1,
                "collect": {"images": "$steps.owned.out"},
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "history",
                "selector": "$operators.history.images",
            },
            {"type": "JsonField", "name": "mosaic", "selector": "$steps.mosaic.image"},
        ],
    }
    session = _session(definition)
    frame = ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8))

    for value in (1, 2, 3):
        frame.tensor_image.fill_(value)  # the caller reuses one buffer
        result = session.run({"frame": [frame]})

    history = result.outputs.data["history"][0]
    assert isinstance(history, Batch)
    assert history.layout.axes[-1].kind == "time"
    assert [int(image.tensor_image[0, 0, 0]) for image in history] == [1, 2, 3]
    mosaic = result.outputs.data["mosaic"][0]
    assert mosaic.tensor_image.shape == (3, 2, 6)
    assert [int(mosaic.tensor_image[0, 0, i]) for i in (0, 2, 4)] == [1, 2, 3]
    frame.tensor_image.fill_(99)
    assert [int(image.tensor_image[0, 0, 0]) for image in history] == [1, 2, 3]
    session.close()


def test_static_steps_run_once_per_run_and_history_spans_run_and_pipeline() -> None:
    definition = _history_workflow()
    definition["steps"].append({"type": Echo.type, "name": "static", "value": 100})
    definition["outputs"].append(
        {"type": "JsonField", "name": "static", "selector": "$steps.static.out"}
    )
    session = _session(definition)

    session.run({"x": 1})
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        second = pipeline.submit({"x": 2}).result(timeout=5)
        third = pipeline.submit({"x": 3}).result(timeout=5)
    fourth = session.run({"x": 4})

    assert second.rows()[0]["history"] is None
    assert third.rows()[0]["history"] == [1, 2, 3]
    assert fourth.rows()[0]["history"] == [2, 3, 4]
    assert session.instances[("static",)].calls == [100] * 4
    session.close()


def test_input_window_keeps_a_present_none_distinct_from_warming() -> None:
    definition = _history_workflow()
    definition["steps"] = []
    definition["operators"][0]["collect"] = {"x": "$inputs.x"}
    definition["operators"][0]["hold"] = {}
    definition["outputs"] = [definition["outputs"][1]]
    session = _session(definition)

    assert session.run({"x": 1}).rows() == [{"history": None}]
    assert session.run({"x": None}).statuses["history"] == "filtered"
    result = session.run({"x": 3})

    assert result.statuses["history"] == "complete"
    assert result.rows() == [{"history": [1, None, 3]}]
    assert result.outputs.layout["history"].axes[-1].kind == "time"
    session.close()


def test_chained_windows_run_inline_in_plan_order() -> None:
    definition = _history_workflow()
    definition["steps"] = definition["steps"][:2]
    definition["operators"].append(
        {
            "type": Window.type,
            "name": "sums",
            "size": 2,
            "step": 1,
            "collect": {"total": "$steps.sum.out"},
        }
    )
    definition["outputs"] = [
        {"type": "JsonField", "name": "sums", "selector": "$operators.sums.total"}
    ]
    session = _session(definition)
    options = PipelineOptions(max_in_flight=2, max_workflow_parallelism=2)

    with session.pipeline(options=options) as pipeline:
        results = [pipeline.submit({"x": x}).result(timeout=5) for x in range(1, 6)]

    assert [result.rows() for result in results] == [
        [{"sums": None}],
        [{"sums": None}],
        [{"sums": None}],
        [{"sums": [6, 9]}],
        [{"sums": [9, 12]}],
    ]
    # The second window collects scalar sums: one time axis, not two.
    assert [axis.kind for axis in results[-1].outputs.layout["sums"].axes] == ["time"]
    session.close()


def test_operator_returning_more_than_it_declared_fails_and_poisons_history() -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": [],
        "operators": [
            {"type": TwoPulses.type, "name": "op", "inputs": {"x": "$inputs.x"}}
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$operators.op.out"}
        ],
    }
    session = _session(definition)

    with pytest.raises(ContractError, match="returned 2 pulses"):
        session.run({"x": 1})
    with pytest.raises(ContractError, match="processing reset"):
        session.run({"x": 2})
    session.close()


def test_nested_literal_input_is_copied_once_per_run_despite_emission() -> None:
    # H2-CODE-01: an emitted route must not rebuild nested inputs the run
    # already prepared. The child mutates its literal input in place, so a
    # rebuilt copy would show {"n": 0} beside the mutated {"n": 1}.
    child = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": "child",
        "workflow_definition": {
            "version": "2.0",
            "inputs": [{"type": "WorkflowParameter", "name": "value"}],
            "steps": [
                {"type": Mutate.type, "name": "mutate", "value": "$inputs.value"}
            ],
            "outputs": [
                {"type": "JsonField", "name": "forward", "selector": "$inputs.value"},
                {
                    "type": "JsonField",
                    "name": "mutated",
                    "selector": "$steps.mutate.out",
                },
            ],
        },
        "parameter_bindings": {"value": {"n": 0}},
    }
    ordinary = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": [child],
        "outputs": [
            {
                "type": "JsonField",
                "name": "forward",
                "selector": "$steps.child.forward",
            },
            {
                "type": "JsonField",
                "name": "mutated",
                "selector": "$steps.child.mutated",
            },
        ],
    }
    with_history = copy.deepcopy(ordinary)
    with_history["operators"] = [
        {
            "type": Window.type,
            "name": "history",
            "size": 1,
            "collect": {"x": "$inputs.x"},
        }
    ]
    with_history["outputs"].append(
        {"type": "JsonField", "name": "history", "selector": "$operators.history.x"}
    )
    expected = {"forward": {"n": 1}, "mutated": {"n": 1}}

    for definition, extra in ((ordinary, {}), (with_history, {"history": [1]})):
        session = _session(definition)
        assert session.run({"x": 1}).rows() == [{**expected, **extra}]
        session.close()


def test_direct_runs_of_passive_history_cannot_overlap() -> None:
    session = _session(_history_workflow())
    session.run({"x": 1})
    session.run({"x": 2})
    HOOKS.update(entered=threading.Event(), release=threading.Event())
    errors: List[BaseException] = []
    runner = threading.Thread(target=lambda: session.run({"x": 3}))

    runner.start()
    try:
        assert HOOKS["entered"].wait(timeout=3)  # the first run is inside a reducer
        with pytest.raises(ContractError, match="cannot overlap"):
            session.run({"x": 4})
    except BaseException as error:  # keep the runner from hanging on failure
        errors.append(error)
    finally:
        HOOKS["release"].set()
        runner.join(timeout=5)

    assert not errors
    HOOKS.clear()
    assert session.run({"x": 4}).rows()[0]["history"] == [2, 3, 4]
    session.close()


def test_workflow_without_operators_keeps_the_ordinary_route() -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": [{"type": Echo.type, "name": "echo", "value": "$inputs.x"}],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.echo.out"}
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session()

    assert not plan.passive_history
    assert session.run({"x": 1}).rows() == [{"out": 1}]
    options = PipelineOptions(max_in_flight=2, max_workflow_parallelism=2)
    with session.pipeline(options=options) as pipeline:
        assert pipeline.submit({"x": 2}).result(timeout=5).rows() == [{"out": 2}]
    assert session._passive_history is None
    session.close()


@pytest.mark.parametrize("history", [False, True], ids=["ordinary", "passive_history"])
def test_gate_filtered_passive_group_stays_an_empty_list(history: bool) -> None:
    # A warming ``None`` belongs to operator domains only. A gate-filtered group
    # of the submission's own route keeps the ordinary ``[]`` row value.
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "x"},
            {"type": "WorkflowParameter", "name": "keep"},
        ],
        "steps": [
            {
                "type": Keep.type,
                "name": "gate",
                "value": "$inputs.keep",
                "next_steps": ["$steps.expand"],
            },
            {"type": Expand.type, "name": "expand", "value": "$inputs.x"},
        ],
        "outputs": [
            {"type": "JsonField", "name": "children", "selector": "$steps.expand.child"}
        ],
    }
    if history:
        definition["operators"] = [
            {
                "type": Window.type,
                "name": "history",
                "size": 2,
                "step": 1,
                "collect": {"x": "$inputs.x"},
            }
        ]
        definition["outputs"].append(
            {"type": "JsonField", "name": "history", "selector": "$operators.history.x"}
        )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session()

    kept = session.run({"x": 2, "keep": True})
    denied = session.run({"x": 3, "keep": False})

    assert plan.passive_history == history
    extra = ({"history": None}, {"history": [2, 3]}) if history else ({}, {})
    assert kept.rows() == [{"children": [20, 21], **extra[0]}]
    assert denied.rows() == [{"children": [], **extra[1]}]
    assert denied.rows(serialize=True) == denied.rows()
    assert denied.statuses["children"] == "filtered"
    assert denied.filtered_paths["children"] == ((),)
    session.close()


def _gated_history_workflow(*, nested: bool) -> Dict[str, Any]:
    """Size-2 Window holding ``x``/``keep``; a gate on ``keep`` filters expand."""
    route: List[Dict[str, Any]] = [
        {
            "type": Keep.type,
            "name": "gate",
            "value": "$inputs.keep" if nested else "$operators.history.keep",
            "next_steps": ["$steps.expand"],
        },
        {
            "type": Expand.type,
            "name": "expand",
            "value": "$inputs.x" if nested else "$operators.history.now",
        },
    ]
    children = "$steps.expand.child"
    if nested:
        route = [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": {
                    "version": "2.0",
                    "inputs": [
                        {"type": "WorkflowParameter", "name": "x"},
                        {"type": "WorkflowParameter", "name": "keep"},
                    ],
                    "steps": route,
                    "outputs": [
                        {"type": "JsonField", "name": "children", "selector": children}
                    ],
                },
                "parameter_bindings": {
                    "x": "$operators.history.now",
                    "keep": "$operators.history.keep",
                },
            }
        ]
        children = "$steps.child.children"
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "x"},
            {"type": "WorkflowParameter", "name": "keep"},
        ],
        "operators": [
            {
                "type": Window.type,
                "name": "history",
                "size": 2,
                "step": 1,
                "collect": {"x": "$inputs.x"},
                "hold": {"now": "$inputs.x", "keep": "$inputs.keep"},
            }
        ],
        "steps": route,
        "outputs": [
            {"type": "JsonField", "name": "children", "selector": children},
            {
                "type": "JsonField",
                "name": "history",
                "selector": "$operators.history.x",
            },
        ],
    }

    return definition


@pytest.mark.parametrize("submission", ["direct", "pipeline"])
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_gate_inside_an_emitted_route_keeps_an_empty_group(
    mode: str, nested: bool, submission: str
) -> None:
    # H2-PROD-QA-001: only an operator that emitted nothing makes its fields
    # None. A gate that filters a group of an emitted route keeps the ordinary
    # ``[]``, even though both leave the entry filtered as a whole.
    plan = compile_workflow(
        _gated_history_workflow(nested=nested),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()
    submissions = [
        {"x": 2, "keep": True},
        {"x": 3, "keep": False},
        {"x": 1, "keep": True},
    ]

    if submission == "direct":
        results = [session.run(inputs) for inputs in submissions]
    else:
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            futures = [pipeline.submit(inputs) for inputs in submissions]
            results = [future.result(timeout=5) for future in futures]
    warming, denied, kept = results

    # Warming: the gate would pass, but the operator emitted nothing.
    assert warming.rows() == [{"children": None, "history": None}]
    assert warming.unavailable == {"children", "history"}
    assert denied.rows() == [{"children": [], "history": [2, 3]}]
    assert denied.rows(serialize=True) == denied.rows()
    assert denied.statuses["children"] == "filtered"
    assert denied.filtered_paths["children"] == ((),)
    assert denied.unavailable == frozenset()
    assert kept.rows() == [{"children": [10], "history": [3, 1]}]
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_cancel_aborts_every_admitted_submission_and_poison_needs_reset(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # H2-PROD-QA-002: the cancelled first submission poisons the history
    # before the second reaches its initial guard. The second still reports
    # the cancel, not the poison; a later direct run needs a reset.
    definition = _history_workflow()
    definition["steps"][0]["type"] = Pause.type
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    session.run({"x": 1})
    pipeline = session.pipeline(options=PipelineOptions(max_in_flight=2))
    original_state = session._passive_history_state
    second_waits, release_second = threading.Event(), threading.Event()
    calls: List[int] = []

    def delayed_state(create):
        calls.append(len(calls))
        if len(calls) == 2:  # the second submission, before its initial guard
            second_waits.set()
            assert release_second.wait(timeout=5)
        return original_state(create)

    monkeypatch.setattr(session, "_passive_history_state", delayed_state)
    HOOKS.update(paused=threading.Event(), resume=threading.Event())
    try:
        first = pipeline.submit({"x": 2})
        assert HOOKS["paused"].wait(timeout=5)  # first is inside its root step
        second = pipeline.submit({"x": 3})
        assert second_waits.wait(timeout=5)
        pipeline.cancel()
        HOOKS["resume"].set()
        with pytest.raises(PipelineAbortedError):
            first.result(timeout=5)
    finally:
        HOOKS["resume"].set()
        release_second.set()
    with pytest.raises(PipelineAbortedError):
        second.result(timeout=5)
    pipeline.close()

    assert pipeline.failure is None
    assert pipeline.outcomes["aborted"] == 2
    assert pipeline.outcomes["failed"] == 0
    with pytest.raises(ContractError, match="processing reset"):
        session.run({"x": 4})
    session.update(plan, reset=True)
    assert session.run({"x": 5}).rows()[0]["history"] is None
    session.close()
