"""Input control values are built-in scalars (decision 011).

A published value must not change without a new version. The review of
``m7-controls-core-r2`` (C1, ``tasks/m7-controls-correctness``) showed that a
dictionary value could be mutated after publication through the caller's
object, the snapshot, a receipt or a result description. Structured values
are now refused at compile time and on update; these tests replay those
probes as explicit rejections and check that scalar versions stay isolated.
"""

import threading
from typing import Any

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
    ControlError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

from tests.unit_tests.execution_engine.v2.m7_controls.blocks import (
    controls,
    enable,
    input_control,
    step,
)
from tests.unit_tests.execution_engine.v2.m7_controls.test_controls import MODES
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Scripted,
    active,
    emit,
    group,
    source,
)


class Echo(Block):
    """Prunable; returns its value unchanged (any kind)."""

    type = "test/m7c_echo@v1"
    prunable = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        return {"value": value}


class SourceEcho(Echo):
    """``Echo`` triggered by a source pulse."""

    type = "test/m7c_source_echo@v1"

    class Params(BlockParams):
        value: Ref()
        trigger: Ref(FLOAT_KIND)

    def run(self, *, value, trigger) -> dict:
        return {"value": value}


CATALOGUE = Catalogue(
    [Echo, SourceEcho],
    sources=[Scripted],
    kinds=[DICTIONARY_KIND, LIST_OF_VALUES_KIND, STRING_KIND],
)


class Celsius(float):
    """A float subclass: carries mutable attributes, so it is not a built-in scalar."""


def echo_definition(kind: Any = None, value: Any = None, **control: Any) -> dict:
    """``$inputs.config`` (``kind``, default ``value``) is the controlled input."""
    declared = {"type": "WorkflowParameter", "name": "config", "default_value": value}
    if kind is not None:
        declared["kind"] = [kind]
    definition = {
        "version": "2.0",
        "inputs": [declared, {"type": "WorkflowParameter", "name": "payload"}],
        "steps": [
            step(Echo, "echo", value="$inputs.config"),
            step(Echo, "plain", value="$inputs.payload"),
        ],
        "outputs": [
            {"type": "JsonField", "name": "config", "selector": "$steps.echo.value"},
            {"type": "JsonField", "name": "payload", "selector": "$steps.plain.value"},
        ],
        "controls": controls(
            config=input_control("config", **control), plain=enable("$steps.plain")
        ),
    }

    return definition


# Compile-time defaults ---------------------------------------------------------


@pytest.mark.parametrize(
    "kind, default",
    [
        ("dictionary", {"threshold": 1}),
        (None, {"threshold": 1}),  # wildcard kind
        (None, [1, 2]),
        ("list_of_values", [0.5]),
    ],
    ids=["dictionary", "wildcard-dict", "wildcard-list", "list"],
)
def test_structured_input_defaults_are_refused_at_compile_time(kind, default) -> None:
    with pytest.raises(ControlDefinitionError, match="built-in scalars") as raised:
        compile_workflow(echo_definition(kind, default), catalogue=CATALOGUE)

    assert type(default).__name__ in str(raised.value)


def test_a_structured_control_default_is_refused_even_over_a_scalar_input_default() -> (
    None
):
    definition = echo_definition(None, "label", default={"threshold": 1})

    with pytest.raises(ControlDefinitionError, match="controls\\['config'\\].default"):
        compile_workflow(definition, catalogue=CATALOGUE)


@pytest.mark.parametrize("default", [None, True, 3, 0.25, "label"])
def test_scalar_defaults_compile_over_a_wildcard_input(default) -> None:
    plan = compile_workflow(echo_definition(None, default), catalogue=CATALOGUE)

    assert plan.controls.describe()["config"]["default"] == default
    assert plan.create_session().run({"payload": 1}).rows()[0]["config"] == default


# Runtime updates -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [{"threshold": 2}, [1, 2], (1, 2), {1, 2}, bytearray(b"x"), object(), Celsius(0.5)],
    ids=["dict", "list", "tuple", "set", "bytearray", "object", "float-subclass"],
)
def test_structured_or_custom_update_values_are_refused_atomically(value) -> None:
    plan = compile_workflow(echo_definition(None, "first"), catalogue=CATALOGUE)
    session = plan.create_session()
    session.controls.update(config="second")
    before = session.controls.current

    with pytest.raises(ControlError, match="built-in scalars") as raised:
        session.controls.update(config=value)
    with pytest.raises(ControlError, match="built-in scalars"):
        session.controls.update(plain=False, config=value)

    assert type(value).__name__ in str(raised.value)
    assert session.controls.current is before and session.controls.version == 1
    assert session.controls.current.enabled["plain"] is True
    assert session.run({"payload": 1}).rows()[0]["config"] == "second"


def test_ordinary_inputs_keep_accepting_structured_values() -> None:
    plan = compile_workflow(echo_definition(None, 0.5), catalogue=CATALOGUE)
    session = plan.create_session()
    payload = {"threshold": 2}

    result = session.run({"payload": payload})

    assert result.rows() == [{"config": 0.5, "payload": {"threshold": 2}}]


# Isolation of published scalar versions ---------------------------------------


def test_public_views_are_detached_and_sessions_are_independent() -> None:
    plan = compile_workflow(echo_definition("string", "initial"), catalogue=CATALOGUE)
    first = plan.create_session()
    second = plan.create_session()

    receipt = first.controls.update(config="updated")
    result = first.run({"payload": 1})
    # Every description is a fresh mapping; editing one changes nothing.
    result.controls.describe()["values"]["config"] = "edited"
    receipt.describe()["changes"]["config"]["to"] = "edited"
    first.controls.describe()["current"]["values"]["config"] = "edited"
    plan.controls.describe()["config"]["default"] = "edited"
    with pytest.raises(TypeError):
        first.controls.current.values["config"] = "edited"  # read-only mapping

    assert result.controls.values["config"] == "updated"
    assert receipt.changes["config"] == {"from": "initial", "to": "updated"}
    assert first.run({"payload": 1}).rows()[0]["config"] == "updated"
    assert first.controls.version == 1
    assert second.controls.version == 0
    assert second.run({"payload": 1}).rows()[0]["config"] == "initial"
    assert plan.create_session().controls.current.values["config"] == "initial"


@MODES
def test_an_admitted_pulse_keeps_its_value_when_a_newer_one_is_published(
    pipeline,
) -> None:
    """Review probe ``test_admitted_snapshot_value_changes_without_update``, as scalars.

    The observer holds the pulse after admission and before its first step; a
    newer value is published meanwhile and must not reach that pulse.
    """
    definition = active(
        [source("a")],
        [step(SourceEcho, "echo", value="$inputs.config", trigger="$sources.a.value")],
        [group("G", "$sources.a.value", config="$steps.echo.value")],
        inputs=[{"type": "WorkflowParameter", "name": "config", "default_value": "a"}],
    )
    definition["controls"] = controls(config=input_control("config"))
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    entered = threading.Event()
    release = threading.Event()
    started = threading.Event()
    started.set()

    class Hold(ExecutionObserver):
        def on_pulse_started(self, *, run_id, source, pulse):
            entered.set()
            assert release.wait(WAIT)

    session = plan.create_session(
        resources={"log": [], "started": started, "feeds": {"a": [emit(value=1.0)]}},
        observer=Hold(),
    )
    session.controls.update(config="admitted")
    collected = Collector()

    run = session.start(handlers=collected.handlers("G"), pipeline=pipeline)
    assert entered.wait(WAIT)  # admitted under version 1, no step called yet
    assert "1" in session.controls.in_flight.describe()
    receipt = session.controls.update(config="newer")
    release.set()
    assert run.wait(WAIT)

    (result,) = collected.results["G"]
    assert receipt.version == 2
    assert result.controls.version == 1
    assert result.rows() == [{"config": "admitted"}]
    assert session.controls.in_flight.describe() == {}
