"""Live controls: nesting, refused targets, reentrancy, counters, result statuses.

Completes the cases listed as unresolved in the ``m7-controls-finish``
recovered handoff. Ordering relies on feed gates (events), never on sleeping.
"""

import threading
from typing import Any, Callable, Dict

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ControlDefinitionError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND

from tests.unit_tests.execution_engine.v2.m7_controls.blocks import (
    HeldPainter,
    Tracker,
    controls,
    enable,
    input_control,
)
from tests.unit_tests.execution_engine.v2.m7_controls.test_controls import (
    MODES,
    Merge,
    merge_definition,
)
from tests.unit_tests.execution_engine.v2.m7_demand.blocks import (
    BLOCKS,
    HANDLER,
    Double,
    Emitter,
    Gate,
    Items,
    Painter,
    Stubborn,
    definition,
    nested,
    step,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Scripted,
    active,
    emit,
    group,
    source,
)


class Reentrant(Block):
    """Prunable; calls the ``on_call`` resource, then reports the threshold it saw."""

    type = "test/m7c_reentrant@v1"
    prunable = True
    outputs = {"threshold": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        threshold: Ref(FLOAT_KIND)

    def __init__(self, *, on_call: Callable[[float], None]) -> None:
        self.on_call = on_call

    def run(self, *, value, threshold) -> dict:
        self.on_call(value)
        return {"threshold": threshold}


CATALOGUE = Catalogue(
    [*BLOCKS, Tracker, HeldPainter, Reentrant], sources=[Scripted], operators=[Merge]
)

FLOAT_VALUE = {"type": "WorkflowParameter", "name": "value", "kind": ["float"]}
KEEP = {"type": "WorkflowParameter", "name": "keep"}
THRESHOLD = {"type": "WorkflowParameter", "name": "threshold", "default_value": 0.5}


def started_resources(**resources: Any) -> Dict[str, Any]:
    started = threading.Event()
    started.set()  # scripted sources open as soon as the run starts

    return {"log": [], "started": started, **resources}


# Nested workflows -------------------------------------------------------------

CHILD = definition(
    [
        step(Double, "double", value="$inputs.value"),
        step(Painter, "painter", value="$steps.double.doubled"),
    ],
    {"doubled": "$steps.double.doubled", "overlay": "$steps.painter.overlay"},
)

GATED_CHILD = definition(
    [
        step(Gate, "gate", value="$inputs.keep", next_steps=["$steps.kept"]),
        step(Double, "kept", value="$inputs.value"),
        step(Double, "other", value="$inputs.value"),
    ],
    {"kept": "$steps.kept.doubled", "other": "$steps.other.doubled"},
    inputs=[FLOAT_VALUE, KEEP],
)


def test_a_nested_step_selector_controls_every_child_step_and_its_outputs() -> None:
    parent = definition(
        [
            nested("child", CHILD, value="$inputs.value"),
            step(Double, "outer", value="$inputs.value"),
        ],
        {
            "doubled": "$steps.child.doubled",
            "overlay": "$steps.child.overlay",
            "outer": "$steps.outer.doubled",
        },
        controls=controls(child=enable("$steps.child")),
    )
    session = compile_workflow(parent, catalogue=CATALOGUE).create_session()

    session.controls.update(child=False)
    hidden = session.run({"value": 1.0})
    session.controls.update(child=True)
    shown = session.run({"value": 1.0})

    members = session.controls.plan.controls.controls["child"].members
    assert members == (("child", "double"), ("child", "painter"))
    assert hidden.statuses == {
        "doubled": "omitted",
        "overlay": "omitted",
        "outer": "complete",
    }
    assert hidden.rows() == [{"outer": 2.0}]
    assert shown.rows() == [{"doubled": 2.0, "overlay": "paint:2", "outer": 2.0}]
    assert len(session.instances[("child", "double")].calls) == 1


def test_disabling_a_child_gate_omits_its_gated_output_unlike_a_gate_denial() -> None:
    parent = definition(
        [nested("child", GATED_CHILD, value="$inputs.value", keep="$inputs.keep")],
        {"kept": "$steps.child.kept", "other": "$steps.child.other"},
        inputs=[FLOAT_VALUE, KEEP],
        controls=controls(gate=enable("$steps.child/gate")),
    )
    session = compile_workflow(parent, catalogue=CATALOGUE).create_session()

    denied = session.run({"value": 1.0, "keep": False})
    session.controls.update(gate=False)
    disabled = session.run({"value": 1.0, "keep": True})

    closure = session.controls.plan.controls.controls["gate"].closure
    assert closure == (("child", "gate"), ("child", "kept"))
    assert denied.statuses == {"kept": "filtered", "other": "complete"}
    assert denied.rows() == [{"kept": None, "other": 2.0}]
    assert disabled.statuses == {"kept": "omitted", "other": "complete"}
    assert disabled.rows() == [{"other": 2.0}]


# Refused targets ----------------------------------------------------------------


def _with_controls(definition_: dict, **declared: dict) -> dict:
    definition_["controls"] = controls(**declared)

    return definition_


def handler_definition(**declared: dict) -> dict:
    handlers = [
        {
            "name": "notify",
            "on": "$steps.emitter.events.seen",
            "execution": {"mode": "sync"},
            "bindings": {"value": "$event.value"},
            "workflow": HANDLER,
        }
    ]
    parent = definition(
        [step(Emitter, "emitter", value="$inputs.value")],
        {"value": "$steps.emitter.value"},
        handlers=handlers,
        controls=controls(**declared),
    )

    return parent


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: _with_controls(merge_definition(), on=enable("$steps.merge")),
            "an operator",
        ),
        (lambda: _with_controls(merge_definition(), on=enable("$steps.a")), "a source"),
        (
            lambda: _with_controls(merge_definition(), on=enable("$steps.slow_a")),
            "would starve the operator",
        ),
        (lambda: handler_definition(on=enable("$steps.react")), "a step of handler"),
    ],
    ids=["operator", "source", "operator-producer", "handler-step"],
)
def test_operators_sources_and_handler_steps_are_not_controllable(
    build, message
) -> None:
    with pytest.raises(ControlDefinitionError, match=message):
        compile_workflow(build(), catalogue=CATALOGUE)


def recorded_definition(path: str, **declared: dict) -> dict:
    definition_ = active(
        [source("a", scale="$inputs.scale")],
        [
            step(Double, "double", value="$sources.a.value"),
            step(Painter, "painter", value="$steps.double.doubled"),
        ],
        [
            group("frames", "$sources.a.value", doubled="$steps.double.doubled"),
            group("archive", "$sources.a.value", overlay="$steps.painter.overlay"),
        ],
        inputs=[
            {"type": "WorkflowParameter", "name": "scale", "default_value": 1.0},
            {"type": "WorkflowParameter", "name": "directory", "default_value": path},
        ],
    )
    definition_["recording"] = {
        "type": "file",
        "directory": "$inputs.directory",
        "groups": ["archive"],
    }
    definition_["controls"] = controls(**declared)

    return definition_


@pytest.mark.parametrize(
    "declared, message",
    [
        ({"overlay": enable("$steps.painter")}, "recorded group 'archive'"),
        ({"scale": input_control("scale")}, "source 'a' reads \\$inputs.scale"),
        ({"directory": input_control("directory")}, "recording.directory"),
    ],
    ids=["recorded-field", "source-parameter", "recording-directory"],
)
def test_static_recording_and_source_settings_are_not_controllable(
    tmp_path, declared, message
) -> None:
    with pytest.raises(ControlDefinitionError, match=message):
        compile_workflow(
            recorded_definition(str(tmp_path), **declared), catalogue=CATALOGUE
        )


def test_a_control_outside_the_recorded_groups_compiles(tmp_path) -> None:
    definition_ = recorded_definition(str(tmp_path))
    definition_["recording"]["groups"] = ["frames"]
    definition_["controls"] = controls(overlay=enable("$steps.painter"))

    plan = compile_workflow(definition_, catalogue=CATALOGUE)

    assert plan.controls.controls["overlay"].closure == (("painter",),)


# Reentrant updates ----------------------------------------------------------------


def reentrant_steps(trigger: str) -> list:
    return [step(Reentrant, "probe", value=trigger, threshold="$inputs.threshold")]


def test_an_update_from_inside_a_block_call_takes_effect_at_the_next_run() -> None:
    receipts = []
    parent = definition(
        reentrant_steps("$inputs.value"),
        {"threshold": "$steps.probe.threshold"},
        inputs=[FLOAT_VALUE, THRESHOLD],
        controls=controls(threshold=input_control("threshold")),
    )
    plan = compile_workflow(parent, catalogue=CATALOGUE)
    session = plan.create_session(
        resources={
            "on_call": lambda value: receipts.append(
                session.controls.update(threshold=value)
            )
        }
    )

    first = session.run({"value": 0.7})
    second = session.run({"value": 0.9})

    assert [r.version for r in receipts] == [1, 2]
    assert (first.controls.version, first.rows()) == (0, [{"threshold": 0.5}])
    assert (second.controls.version, second.rows()) == (1, [{"threshold": 0.7}])
    assert session.controls.current.values == {"threshold": 0.9}


@MODES
def test_an_update_from_inside_an_active_block_call_reaches_later_admissions(
    pipeline,
) -> None:
    gate = threading.Event()
    receipts = []

    def on_call(value: float) -> None:
        if value == 1.0:
            receipts.append(session.controls.update(threshold=0.9))
            gate.set()  # the next emission is read only after the update

    parent = active(
        [source("a")],
        reentrant_steps("$sources.a.value"),
        [group("G", "$sources.a.value", threshold="$steps.probe.threshold")],
        inputs=[THRESHOLD],
    )
    parent["controls"] = controls(threshold=input_control("threshold"))
    plan = compile_workflow(parent, catalogue=CATALOGUE)
    feeds = {"a": [emit(value=1.0), gate, emit(value=2.0)]}
    session = plan.create_session(
        resources=started_resources(feeds=feeds, on_call=on_call)
    )
    collected = Collector()

    run = session.start(handlers=collected.handlers("G"), pipeline=pipeline)
    assert run.wait(WAIT)

    results = collected.results["G"]
    assert [r.version for r in receipts] == [1]
    assert [r.controls.version for r in results] == [0, 1]
    assert collected.rows("G") == [{"threshold": 0.5}, {"threshold": 0.9}]
    assert session.controls.in_flight.describe() == {}


# Per-source counters --------------------------------------------------------------


@MODES
def test_counters_report_omitted_groups_per_source(pipeline) -> None:
    parent = active(
        [source("a"), source("b")],
        [step(Painter, "painter", value="$sources.a.value")],
        [
            group("P", "$sources.a.value", overlay="$steps.painter.overlay"),
            group("B", "$sources.b.value", value="$sources.b.value"),
        ],
    )
    parent["controls"] = controls(overlay=enable("$steps.painter"))
    plan = compile_workflow(parent, catalogue=CATALOGUE)
    feeds = {
        "a": [emit(value=1.0), emit(value=2.0)],
        "b": [emit(value=1.0), emit(value=2.0), emit(value=3.0)],
    }
    session = plan.create_session(resources=started_resources(feeds=feeds))
    session.controls.update(overlay=False)
    collected = Collector()

    run = session.start(handlers=collected.handlers("P", "B"), pipeline=pipeline)
    assert run.wait(WAIT)

    a, b = run.counters["a"], run.counters["b"]
    assert (a.admitted, a.processed, a.delivered, a.omitted) == (2, 2, 0, 2)
    assert (b.admitted, b.processed, b.delivered, b.omitted) == (3, 3, 3, 0)
    assert "P" not in collected.results
    assert collected.rows("B") == [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}]
    assert session.instances[("painter",)].calls == []


# Result statuses ------------------------------------------------------------------


def statuses_definition() -> dict:
    parent = definition(
        [
            step(Gate, "gate", value="$inputs.keep", next_steps=["$steps.gated"]),
            step(Items, "gated", value="$inputs.value"),
            step(Items, "open", value="$inputs.value"),
            step(Painter, "painter", value="$inputs.value"),
        ],
        {
            "empty": "$steps.open.items",
            "filtered": "$steps.gated.items",
            "overlay": "$steps.painter.overlay",
        },
        inputs=[FLOAT_VALUE, KEEP],
        controls=controls(overlay=enable("$steps.painter")),
    )

    return parent


def test_filtered_omitted_and_empty_outputs_are_distinct_in_one_result() -> None:
    session = compile_workflow(
        statuses_definition(), catalogue=CATALOGUE
    ).create_session()
    session.controls.update(overlay=False)

    result = session.run({"value": -1.0, "keep": False})

    assert result.statuses == {
        "empty": "complete",
        "filtered": "filtered",
        "overlay": "omitted",
    }
    assert result.rows() == [{"empty": [], "filtered": None}]
    assert session.instances[("painter",)].calls == []


def test_a_missing_wanted_output_stays_an_error_under_a_narrowing_snapshot() -> None:
    parent = definition(
        [
            step(Stubborn, "stubborn", value="$inputs.value"),
            step(Painter, "painter", value="$inputs.value"),
        ],
        {"overlay": "$steps.stubborn.overlay", "painted": "$steps.painter.overlay"},
        controls=controls(painted=enable("$steps.painter")),
    )
    session = compile_workflow(parent, catalogue=CATALOGUE).create_session()
    session.controls.update(painted=False)

    assert session.controls.current.narrows
    with pytest.raises(StepExecutionError, match="wanted by a reader"):
        session.run({"value": 1.0})
