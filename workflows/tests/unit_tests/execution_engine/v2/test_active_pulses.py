"""Per-pulse primitives and the controlled same-session two-pulse experiment.

The experiment drives ``begin_pulse`` / ``execute_step`` on two ``RunState``
objects of one session and one source from a single thread, in an order the
test chooses; its operator variant does the same with two pulses an
alignment operator actually emitted. It proves that pulse-local decisions, entries, boundary caches
and metadata never leak between pulses while block instances are shared. It
is a state-separation experiment, not a supported scheduler: the active
runtime processes pulses one at a time.
"""

import threading
from fractions import Fraction
from typing import Any, Dict

import pytest
from roboflow_workflows.execution_engine.v2.active.execution import (
    ENGINE_CLOCK_ID,
    abandon_pulse,
    begin_operator_pulse,
    begin_pulse,
    execute_pulse,
    group_result,
    operator_arrivals,
    port_entries,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.steps import execute_step
from roboflow_workflows.execution_engine.v2.operators import OperatorPulse
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.plan import PulseKey, SourcePort
from roboflow_workflows.execution_engine.v2.sources import Emission

from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Counter,
    Failing,
    Notice,
    Scale,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    CATALOGUE,
    MEDIA,
    Log,
    active,
    emit,
    group,
    parameter,
    source,
    step,
)

RUN = "run-1"


def compiled(definition: dict):
    started = threading.Event()
    started.set()
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session(
        resources={"feeds": {}, "log": Log(), "started": started}
    )

    return plan, session


def pulse(sequence: int, *, source_name: str = "a") -> PulseKey:
    return PulseKey(active_run_id=RUN, source=source_name, sequence=sequence)


def observed(ticks: int) -> Timestamp:
    return Timestamp(ticks, Fraction(1, 10**9), ENGINE_CLOCK_ID)


# Port entries ------------------------------------------------------------


def simple_plan():
    definition = active(
        [source("a")],
        [step(Scale, "double", value="$sources.a.value")],
        [group("A", "$sources.a.value", doubled="$steps.double.scaled")],
    )
    plan, session = compiled(definition)

    return plan, session


def test_present_ports_carry_the_pulse_context_and_omitted_ports_are_absent() -> None:
    plan, _ = simple_plan()
    emission = Emission(
        {"value": 2.0, "extra": None},
        media=Timestamp(40, MEDIA, "cam"),
        source_metadata={"frame": 4},
    )

    entries = port_entries(plan, "a", emission=emission, observed=observed(7))

    value = entries[SourcePort("a", "value")]
    assert value.values == {(): 2.0} and value.filtered == frozenset()
    assert value.metadata.sample_at(()) == SampleContext(
        source_id="a", source_type="test/scripted@v1", source_metadata={"frame": 4}
    )
    assert value.metadata.temporal_at(()) == TemporalContext(
        observed_coverage=observed(7), media_coverage=Timestamp(40, MEDIA, "cam")
    )
    extra = entries[SourcePort("a", "extra")]
    assert extra.values == {(): None} and extra.has_value(())
    label = entries[SourcePort("a", "label")]
    assert label.values == {} and label.is_filtered(()) and label.metadata.is_empty
    assert set(entries) == {
        SourcePort("a", name) for name in ("value", "label", "extra")
    }


def test_input_value_metadata_overlays_the_pulse_context_key_by_key() -> None:
    plan, _ = simple_plan()
    supplied = InputValue(
        3.0,
        EntryMetadata(
            sample={(): SampleContext(source_id="probe", source_type="custom")},
            temporal={(): None},
        ),
    )
    emission = Emission(
        {"value": supplied, "label": InputValue("x")}, media=Timestamp(1, MEDIA, "m")
    )

    entries = port_entries(plan, "a", emission=emission, observed=observed(1))

    value = entries[SourcePort("a", "value")]
    assert value.values == {(): 3.0}
    assert value.metadata.sample_at(()).source_id == "probe"
    assert value.metadata.temporal_at(()) is None
    label = entries[SourcePort("a", "label")]
    assert label.metadata.sample_at(()).source_id == "a"
    assert label.metadata.temporal_at(()).media_coverage == Timestamp(1, MEDIA, "m")


def test_emissions_violating_the_declaration_are_rejected() -> None:
    plan, _ = simple_plan()
    cases = [
        (Emission({"nope": 1.0}), ContractError, "undeclared ports \\['nope'\\]"),
        (Emission({"value": "text"}), WorkflowInputError, "not a valid \\['float'\\]"),
        (Emission({"value": Batch.of([1.0])}), ContractError, "Batch"),
        (
            Emission({"value": InputValue(1.0, EntryMetadata(sample={(0,): None}))}),
            ContractError,
            "deeper than the layout depth",
        ),
    ]
    for emission, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            port_entries(plan, "a", emission=emission, observed=observed(0))


def test_begin_pulse_names_the_pulse_and_shares_static_inputs() -> None:
    definition = active(
        [source("a")],
        [step(Scale, "scale", value="$sources.a.value", factor="$inputs.factor")],
        [group("A", "$sources.a.value", scaled="$steps.scale.scaled")],
        inputs=[parameter("factor", 3.0)],
    )
    plan, session = compiled(definition)
    inputs = prepare_inputs(plan, {"factor": 5.0})

    run = begin_pulse(session, pulse=pulse(2), emission=emit(value=1.0), inputs=inputs)
    execute_pulse(run)
    result = group_result(run, plan.groups_of("a")[0])

    assert run.run_id == f"{RUN}:a:2" and run.pulse == pulse(2)
    assert run.inputs["factor"] is inputs["factor"]
    assert result.rows() == [{"scaled": 5.0}]
    assert result.outputs.lineage_id == f"run:{RUN}/source:a"
    assert result.outputs.pulse_id == 2
    assert [event["event"] for event in run.trace][:2] == [
        "pulse_started",
        "step_started",
    ]
    abandon_pulse(run)
    assert (run.ports, run.outputs, run.decisions, run.constants) == ({}, {}, {}, {})
    assert result.rows() == [{"scaled": 5.0}]  # a built result stays valid


def test_filtered_emission_delivers_every_field_filtered_without_reading_entries() -> (
    None
):
    plan, session = simple_plan()
    run = begin_pulse(session, pulse=pulse(0), emission=Emission({}), inputs={})
    execute_pulse(run)

    result = group_result(run, plan.groups_of("a")[0], filtered=True)

    assert result.is_filtered and result.outputs.is_filtered
    assert result.statuses == {"doubled": "filtered"}
    assert result.filtered_paths == {"doubled": ((),)}
    assert result.rows() == [{"doubled": None}]
    assert session.instances[("double",)].calls == []


# The two-pulse experiment --------------------------------------------------


def experiment_plan():
    """Gate on the source value; a nested child forwards it; a notice fires."""
    child = {
        "version": "2.0",
        "inputs": [parameter("x", 0.0)],
        "steps": [step(Scale, "scale", value="$inputs.x")],
        "outputs": [
            {"type": "JsonField", "name": "forwarded", "selector": "$inputs.x"},
            {"type": "JsonField", "name": "scaled", "selector": "$steps.scale.scaled"},
        ],
    }
    definition = active(
        [source("a")],
        [
            step(
                ContinueIf,
                "gate",
                value="$sources.a.value",
                threshold=0.0,
                next_steps=["$steps.notice", "$steps.child"],
            ),
            step(Notice, "notice"),
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"x": "$sources.a.value"},
            },
            step(Counter, "count", value="$sources.a.value"),
        ],
        [
            group(
                "A",
                "$sources.a.value",
                forwarded="$steps.child.forwarded",
                scaled="$steps.child.scaled",
                count="$steps.count.count",
            )
        ],
    )
    plan, session = compiled(definition)

    return plan, session


def steps_by_name(plan) -> Dict[str, Any]:
    return {"/".join(item.path): item for item in plan.route("a")}


def test_two_pulses_of_one_session_keep_separate_state_while_sharing_blocks() -> None:
    plan, session = experiment_plan()
    steps = steps_by_name(plan)
    assert list(steps) == ["gate", "notice", "child/scale", "count"]
    admitting = begin_pulse(
        session, pulse=pulse(0), emission=emit(value=1.0, pts=10), inputs={}
    )
    denying = begin_pulse(
        session, pulse=pulse(1), emission=emit(value=-1.0, pts=20), inputs={}
    )

    # P0 decides first, then pauses before its branch consumes the decision.
    execute_step(admitting, steps["gate"])
    # P1 decides the opposite and runs to completion first.
    for name in steps:
        execute_step(denying, steps[name])
    denied = group_result(denying, plan.groups_of("a")[0])
    # P0 resumes with its own decision intact.
    for name in list(steps)[1:]:
        execute_step(admitting, steps[name])
    admitted = group_result(admitting, plan.groups_of("a")[0])

    assert admitted.rows() == [{"forwarded": 1.0, "scaled": 2.0, "count": 2}]
    assert denied.rows() == [{"forwarded": None, "scaled": None, "count": 1}]
    assert admitting.decisions[("gate",)].values[()] == frozenset(
        {"$steps.notice", "$steps.child"}
    )
    assert denying.decisions[("gate",)].values[()] == frozenset()
    assert admitting.trace is not denying.trace
    # The output-free action ran once, for the admitting pulse only.
    notice = session.instances[("notice",)]
    assert len(notice.calls) == 1
    # The shared counter saw P1 first, then P0: instances are shared, not state.
    assert session.instances[("count",)].calls == [{"value": -1.0}, {"value": 1.0}]
    # Each result carries its own pulse identity and media time.
    assert admitted.run_id == f"{RUN}:a:0" and denied.run_id == f"{RUN}:a:1"
    assert admitted.outputs.metadata["count"].temporal_at(
        ()
    ).media_coverage == Timestamp(10, MEDIA, "media")
    assert denied.outputs.metadata["count"].temporal_at(()).media_coverage == Timestamp(
        20, MEDIA, "media"
    )
    # The nested boundary caches are per pulse.
    assert admitting.child_outputs.keys() == denying.child_outputs.keys()
    assert admitting.child_outputs != denying.child_outputs
    abandon_pulse(admitting)
    abandon_pulse(denying)
    for run in (admitting, denying):
        assert not run.decisions and not run.child_outputs and not run.ports
    # A third pulse starts clean and makes the opposite decision again.
    third = begin_pulse(session, pulse=pulse(2), emission=emit(value=-5.0), inputs={})
    execute_pulse(third)
    assert group_result(third, plan.groups_of("a")[0]).rows() == [
        {"forwarded": None, "scaled": None, "count": 3}
    ]
    assert len(notice.calls) == 1


def test_a_failing_pulse_is_abandoned_without_touching_the_other() -> None:
    definition = active(
        [source("a")],
        [
            step(
                ContinueIf,
                "gate",
                value="$sources.a.value",
                next_steps=["$steps.notice"],
            ),
            step(Notice, "notice"),
            step(Failing, "check", value="$sources.a.value"),
            step(Counter, "count", value="$steps.check.value"),
        ],
        [group("A", "$sources.a.value", count="$steps.count.count")],
    )
    plan, session = compiled(definition)
    steps = steps_by_name(plan)
    failing = begin_pulse(session, pulse=pulse(0), emission=emit(value=-1.0), inputs={})
    healthy = begin_pulse(session, pulse=pulse(1), emission=emit(value=1.0), inputs={})

    execute_step(failing, steps["gate"])
    for name in steps:
        execute_step(healthy, steps[name])
    result = group_result(healthy, plan.groups_of("a")[0])
    with pytest.raises(StepExecutionError) as caught:
        execute_step(failing, steps["notice"])
        execute_step(failing, steps["check"])
    abandon_pulse(failing)

    assert caught.value.step_path == ("check",)
    assert result.rows() == [{"count": 1}]
    assert not failing.decisions and not failing.ports and not failing.outputs
    assert healthy.decisions[("gate",)].values[()] == frozenset({"$steps.notice"})
    assert session.instances[("notice",)].calls == [{"message": "found"}]
    # The session keeps working after the abandoned pulse.
    later = begin_pulse(session, pulse=pulse(2), emission=emit(value=2.0), inputs={})
    execute_pulse(later)
    assert group_result(later, plan.groups_of("a")[0]).rows() == [{"count": 2}]


# The operator variant ------------------------------------------------------


def operator_experiment_plan():
    """Align two sources; the same gate, child and counter run per aligned pair."""
    child = {
        "version": "2.0",
        "inputs": [parameter("x", 0.0)],
        "steps": [step(Scale, "scale", value="$inputs.x")],
        "outputs": [
            {"type": "JsonField", "name": "scaled", "selector": "$steps.scale.scaled"}
        ],
    }
    definition = active(
        [source("a"), source("b")],
        [
            step(
                ContinueIf,
                "gate",
                value="$operators.pair.a",
                next_steps=["$steps.notice", "$steps.child"],
            ),
            step(Notice, "notice"),
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"x": "$operators.pair.b"},
            },
            step(Counter, "count", value="$operators.pair.a"),
        ],
        [
            group(
                "P",
                "$operators.pair.a",
                scaled="$steps.child.scaled",
                count="$steps.count.count",
            )
        ],
    )
    definition["operators"] = [
        {
            "type": Align.type,
            "name": "pair",
            "inputs": {"a": "$sources.a.value", "b": "$sources.b.value"},
            "clock": "media",
        }
    ]
    started = threading.Event()
    started.set()
    catalogue = Catalogue.merge(CATALOGUE, Catalogue([], operators=[Align]))
    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session(
        resources={"feeds": {}, "log": Log(), "started": started}
    )

    return plan, session


def aligned_pulses(plan, session, values) -> list:
    """Feed source pulses through a real ``Align``; return what it emitted."""
    planned = plan.operator("pair")
    align = planned.spec.operator_class(
        name="pair", params=planned.params, inputs=planned.inputs
    )
    emitted = []
    for sequence, (a_value, b_value, pts) in enumerate(values):
        for name, value in (("a", a_value), ("b", b_value)):
            run = begin_pulse(
                session,
                pulse=pulse(sequence, source_name=name),
                emission=emit(value=value, pts=pts),
                inputs={},
            )
            execute_pulse(run)
            emitted += align.push(operator_arrivals(run, planned))
            abandon_pulse(run)

    return emitted


def test_two_operator_pulses_keep_separate_state_while_sharing_blocks() -> None:
    plan, session = operator_experiment_plan()
    steps = {"/".join(item.path): item for item in plan.route("pair")}
    assert list(steps) == ["gate", "notice", "child/scale", "count"]
    first, second = aligned_pulses(plan, session, [(1.0, 10.0, 0), (-1.0, 20.0, 5)])
    admitting = begin_operator_pulse(
        session, pulse=pulse(0, source_name="pair"), emission=first, inputs={}
    )
    denying = begin_operator_pulse(
        session, pulse=pulse(1, source_name="pair"), emission=second, inputs={}
    )

    execute_step(admitting, steps["gate"])
    for name in steps:
        execute_step(denying, steps[name])
    denied = group_result(denying, plan.groups_of("pair")[0])
    for name in list(steps)[1:]:
        execute_step(admitting, steps[name])
    admitted = group_result(admitting, plan.groups_of("pair")[0])

    assert admitted.rows() == [{"scaled": 20.0, "count": 2}]
    assert denied.rows() == [{"scaled": None, "count": 1}]
    assert len(session.instances[("notice",)].calls) == 1
    assert admitted.causes == (pulse(0, source_name="a"), pulse(0, source_name="b"))
    assert denied.causes == (pulse(1, source_name="a"), pulse(1, source_name="b"))
    assert admitted.run_id == f"{RUN}:pair:0" and denied.run_id == f"{RUN}:pair:1"
    assert admitting.decisions[("gate",)].values[()] == frozenset(
        {"$steps.notice", "$steps.child"}
    )
    assert denying.decisions[("gate",)].values[()] == frozenset()
    # The aligned entries are the source pulses' own, still valid after those
    # pulses were abandoned; their media time follows each pair.
    port = SourcePort("pair", "a", origin="operator")
    assert admitting.ports[port] is first.ports["a"]
    assert admitted.outputs.metadata["count"].temporal_at(
        ()
    ).media_coverage == Timestamp(0, MEDIA, "media")
    abandon_pulse(admitting)
    abandon_pulse(denying)
    assert not admitting.decisions and not admitting.ports


def test_operator_pulses_must_match_the_planned_ports() -> None:
    plan, session = operator_experiment_plan()
    (emitted,) = aligned_pulses(plan, session, [(1.0, 2.0, 0)])
    key = pulse(0, source_name="pair")

    only_a = begin_operator_pulse(
        session,
        pulse=key,
        emission=OperatorPulse(ports={"a": emitted.ports["a"]}, causes=()),
        inputs={},
    )
    assert only_a.ports[SourcePort("pair", "b", origin="operator")].is_filtered(())
    with pytest.raises(ContractError, match=r"undeclared ports \['c'\]"):
        begin_operator_pulse(
            session,
            pulse=key,
            emission=OperatorPulse(ports={"c": emitted.ports["a"]}, causes=()),
            inputs={},
        )
    wrong = Entry(
        layout=EntryLayout((Axis("extra", "static_nesting"),)),
        metadata=EntryMetadata(),
        children={(): ()},
        values={},
        filtered=frozenset(),
    )
    with pytest.raises(ContractError, match="emitted axes"):
        begin_operator_pulse(
            session,
            pulse=key,
            emission=OperatorPulse(ports={"a": wrong}, causes=()),
            inputs={},
        )
