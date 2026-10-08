"""Live controls: versions, omission, keep-ticking, causal reset (decision 009).

Barriers are events (held block phases, feed gates) or bounded waits on a
run's counters; no test uses sleeping as evidence of order.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ControlDefinitionError,
    ControlError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

from tests.unit_tests.execution_engine.v2.m7_controls.blocks import (
    HeldPainter,
    Thresholder,
    Tracker,
    controls,
    enable,
    input_control,
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

PIPELINE = PipelineOptions(max_in_flight=4)
MODES = pytest.mark.parametrize(
    "pipeline", [None, PIPELINE], ids=["serial", "pipelined"]
)


class Merge(Operator):
    """Emits every arrival of either input as one pulse; ``end_input`` re-emits the last one."""

    type = "test/m7c_merge@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        pass

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"value": OperatorPort(*inputs[0].kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.last: Dict[str, Any] = {}

    def push(self, arrivals):
        pulses = []
        for arrival in arrivals:
            self.last[arrival.input] = arrival.entry
            pulses.append(
                OperatorPulse(ports={"value": arrival.entry}, causes=(arrival.pulse,))
            )
        return pulses

    def end_input(self, name):
        if name not in self.last:
            return []
        return [OperatorPulse(ports={"value": self.last[name]}, causes=())]

    def finish(self, reason):
        return []

    def close(self):
        pass


class FinishMerge(Merge):
    """``Merge`` whose ``finish`` re-emits the last value of ``b``."""

    type = "test/m7c_finish_merge@v1"

    def finish(self, reason):
        return [OperatorPulse(ports={"value": self.last["b"]}, causes=())]


CATALOGUE = Catalogue(
    [Tracker, Thresholder, HeldPainter],
    sources=[Scripted],
    operators=[Merge, FinishMerge],
)


def parameter(name: str, default: Any, kind: Optional[str] = None) -> dict:
    declared = {"type": "WorkflowParameter", "name": name, "default_value": default}
    if kind is not None:
        declared["kind"] = [kind]

    return declared


def passive(steps: list, outputs: Dict[str, str], **sections: Any) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [
            parameter("value", 0.0),
            parameter("threshold", 0.5, FLOAT_KIND.name),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
        **sections,
    }

    return definition


def gates() -> Dict[str, threading.Event]:
    return {"entered": threading.Event(), "release": threading.Event()}


def make_session(definition: dict, feeds: Optional[dict] = None, **extra: Any):
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    started = threading.Event()
    started.set()  # scripted sources open as soon as the run starts
    resources = {"gates": gates(), "log": [], "started": started}
    if feeds is not None:
        resources["feeds"] = feeds
    session = plan.create_session(resources=resources, **extra)
    session.gates = resources["gates"]  # the HeldPainter barrier, for the test

    return session


def wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.005)


def chain_definition(state: str = "keep_ticking", **options: Any) -> dict:
    """Controlled pure preprocessing feeding a stateful member."""
    definition = passive(
        [
            step(
                Thresholder, "pre", value="$inputs.value", threshold="$inputs.threshold"
            ),
            step(Tracker, "tracker", value="$steps.pre.threshold"),
        ],
        {"above": "$steps.pre.above", "ticks": "$steps.tracker.ticks"},
        controls=controls(
            analysis=enable("$steps.pre", "$steps.tracker", state=state, **options),
            threshold=input_control("threshold", default=0.5),
        ),
    )

    return definition


# Declaration, receipts and atomic updates -------------------------------------


def test_disabled_members_stay_compiled_and_describe_lists_classes() -> None:
    definition = chain_definition(enabled=False)
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert ("pre",) in plan.demand.retained and ("tracker",) in plan.demand.retained
    described = plan.controls.describe()
    assert described["analysis"]["state_classes"] == {
        "$steps.pre": "pure",
        "$steps.tracker": "local",
    }
    session = make_session(definition)
    current = session.controls.describe()["current"]
    # The tracker keeps ticking and reads the preprocessing, so nothing is
    # skipped; only the visible outputs are omitted.
    assert current["omitted_steps"] == {}
    assert current["omitted_outputs"] == ["above", "ticks"]


def test_input_control_is_panel_owned_and_versioned() -> None:
    session = make_session(chain_definition())
    pre = session.instances[("pre",)]

    first = session.run({"value": 0.6})
    receipt = session.controls.update(threshold=0.7)
    second = session.run({"value": 0.6})

    assert (receipt.previous_version, receipt.version) == (0, 1)
    assert receipt.changes == {"threshold": {"from": 0.5, "to": 0.7}}
    assert [call["threshold"] for call in pre.calls] == [0.5, 0.7]
    assert (first.controls.version, second.controls.version) == (0, 1)
    assert first.rows()[0]["above"] == 1 and second.rows()[0]["above"] == 0
    with pytest.raises(WorkflowInputError, match="controls.update"):
        session.run({"value": 0.6, "threshold": 0.9})


def test_invalid_updates_change_nothing() -> None:
    session = make_session(chain_definition())
    before = session.controls.current

    with pytest.raises(ControlError, match="rejected 'hot'"):
        session.controls.update(analysis=False, threshold="hot")
    with pytest.raises(ControlError, match="unknown control"):
        session.controls.update(analysis=False, quality="fast")
    with pytest.raises(ControlError, match="True or False"):
        session.controls.update(analysis="no")
    with pytest.raises(ControlError, match="at least one"):
        session.controls.update()

    assert session.controls.current is before
    assert session.run({"value": 0.6}).controls.version == 0


# Keep-ticking and reset policies ----------------------------------------------


def test_keep_ticking_retains_the_preprocessing_the_stateful_member_needs() -> None:
    session = make_session(chain_definition())
    pre = session.instances[("pre",)]
    tracker = session.instances[("tracker",)]

    session.run({"value": 0.6})
    session.controls.update(analysis=False)
    hidden = session.run({"value": 0.6})
    session.controls.update(analysis=True)
    shown = session.run({"value": 0.6})

    assert len(pre.calls) == 3, "the pure member kept feeding the ticking member"
    assert [call["value"] for call in tracker.calls] == [0.5, 0.5, 0.5]
    assert hidden.statuses == {"above": "omitted", "ticks": "omitted"}
    assert hidden.rows() == [{}]
    assert tracker.asked[1] == {"value": False, "ticks": False}
    assert tracker.asked[2] == {"value": False, "ticks": True}
    assert shown.rows()[0]["ticks"] == 3 and tracker.resets == 0


def test_reset_on_enable_resets_once_per_enable_and_omits_while_disabled() -> None:
    session = make_session(chain_definition("reset_on_enable", suspends_effects=True))
    tracker = session.instances[("tracker",)]
    observed: List[tuple] = []

    class Notes(ExecutionObserver):
        def on_state_reset(self, *, step, control, epoch):
            observed.append((step, control, epoch))

    session.observer = Notes()
    session.run({"value": 0.6})
    session.run({"value": 0.6})
    session.controls.update(analysis=False)
    hidden = session.run({"value": 0.6})
    session.controls.update(analysis=True)
    first = session.run({"value": 0.6})
    second = session.run({"value": 0.6})
    session.controls.update(analysis=False)
    session.controls.update(analysis=True)
    third = session.run({"value": 0.6})

    assert len(tracker.calls) == 5 and hidden.statuses["ticks"] == "omitted"
    assert [r.rows()[0]["ticks"] for r in (first, second, third)] == [1, 2, 1]
    assert tracker.resets == 2
    assert [e["event"] for e in first.trace if e["event"] == "state_reset"] == [
        "state_reset"
    ]
    assert not any(e["event"] == "state_reset" for e in second.trace)
    assert [epoch for _, _, epoch in observed] == [2, 4]


def test_reset_on_enable_needs_consent_to_skip_a_stateful_member() -> None:
    with pytest.raises(ControlDefinitionError, match="suspends_effects: true"):
        compile_workflow(chain_definition("reset_on_enable"), catalogue=CATALOGUE)


# Active runs: version boundary, group omission, causal reset ------------------


def painter_definition(**control: Any) -> dict:
    definition = active(
        [source("a")],
        [
            step(
                HeldPainter,
                "painter",
                value="$sources.a.value",
                threshold="$inputs.threshold",
                hold=1.0,
            )
        ],
        [
            group(
                "P",
                "$sources.a.value",
                value="$steps.painter.value",
                extra="$steps.painter.extra",
            ),
            group("E", "$sources.a.value", extra="$steps.painter.extra"),
        ],
        inputs=[parameter("threshold", 0.5, FLOAT_KIND.name)],
    )
    definition["controls"] = controls(
        threshold=input_control("threshold", default=0.5),
        overlay=enable("$steps.painter", **control),
    )

    return definition


@MODES
def test_a_pulse_held_between_phases_keeps_its_version_while_a_newer_one_is_admitted(
    pipeline,
) -> None:
    gate = threading.Event()
    feeds = {"a": [emit(value=1.0), emit(value=2.0), gate, emit(value=3.0)]}
    session = make_session(painter_definition(), feeds)
    painter = session.instances[("painter",)]
    held = session.gates
    collected = Collector()

    run = session.start(
        handlers=collected.handlers("P", "E"), pipeline=pipeline, admission_bound=3
    )
    assert held["entered"].wait(WAIT)  # A (1.0) is inside its first phase
    wait_for(lambda: run.counters["a"].admitted == 2, "B admitted")
    receipt = session.controls.update(threshold=0.9)
    gate.set()  # C (3.0) is read and admitted after the update
    wait_for(lambda: run.counters["a"].admitted == 3, "C admitted")
    held["release"].set()
    assert run.wait(WAIT)

    assert receipt.version == 1
    assert [r.controls.version for r in collected.results["P"]] == [0, 0, 1]
    by_value = {p["value"]: p for p in painter.phases if p["phase"] == "second"}
    assert [p["threshold"] for p in painter.phases] == [0.5, 0.5, 0.5, 0.5, 0.9, 0.9]
    assert by_value[1.0]["extra"] is True
    assert collected.rows("E") == [
        {"extra": "extra:10"},
        {"extra": "extra:20"},
        {"extra": "extra:30"},
    ]


@MODES
def test_disabling_omits_outputs_and_whole_groups_without_waiting(pipeline) -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0)]}
    session = make_session(painter_definition(), feeds)
    session.gates["release"].set()
    collected = Collector()
    omitted: List[str] = []

    class Notes(ExecutionObserver):
        def on_group_omitted(self, *, run_id, group, source, pulse, reason):
            omitted.append(f"{group}#{pulse.sequence}")

    session.observer = Notes()
    session.controls.update(overlay=False)
    run = session.start(handlers=collected.handlers("P", "E"), pipeline=pipeline)
    assert run.wait(WAIT)

    assert "P" not in collected.results and "E" not in collected.results
    assert omitted == ["P#0", "E#0", "P#1", "E#1"]
    assert run.counters["a"].omitted == 4 and run.counters["a"].processed == 2
    assert session.instances[("painter",)].calls == []


def merge_definition(operator: type = Merge) -> dict:
    """Two sources merged by an operator feeding a resettable tracker."""
    definition = active(
        [source("a"), source("b")],
        [
            step(HeldPainter, "slow_a", value="$sources.a.value", hold=1.0),
            step(Tracker, "tracker", value="$operators.merge.value"),
        ],
        [
            group(
                "T",
                "$operators.merge.value",
                ticks="$steps.tracker.ticks",
                value="$operators.merge.value",
            )
        ],
    )
    definition["operators"] = [
        {
            "type": operator.type,
            "name": "merge",
            "inputs": {"a": "$steps.slow_a.value", "b": "$sources.b.value"},
        }
    ]
    definition["controls"] = controls(
        tracking=enable(
            "$steps.tracker", state="reset_on_enable", suspends_effects=True
        )
    )

    return definition


@MODES
def test_an_old_pulse_delayed_upstream_reaches_the_reset_member_before_the_reset(
    pipeline,
) -> None:
    gate_b = threading.Event()
    feeds = {"a": [emit(value=1.0)], "b": [gate_b, emit(value=20.0)]}
    session = make_session(merge_definition(), feeds)
    held = session.gates
    tracker = session.instances[("tracker",)]
    collected = Collector()

    run = session.start(handlers=collected.handlers("T"), pipeline=pipeline)
    assert held["entered"].wait(
        WAIT
    )  # a#0 (version 0) is held upstream of the operator
    session.controls.update(tracking=False)
    receipt = session.controls.update(tracking=True)  # epoch 2
    gate_b.set()  # b#0 is admitted under version 2 and reaches the operator first
    wait_for(lambda: run.counters["b"].admitted == 1, "b#0 admitted")
    if pipeline is not None:
        wait_for(
            lambda: session.controls.in_flight.describe() == {"0": 1, "2": 1},
            "b#0 counted in flight beside the held a#0",
        )
        assert tracker.calls == [], "the newer pulse must not run the member first"
    held["release"].set()
    assert run.wait(WAIT)

    assert receipt.version == 2 and session.controls.current.reset_epoch == 2
    # a#0 (version 0, value 1.0 painted to 10.0) ticks first. Serially, a's
    # end-of-input was sequenced at its EOS read, before the update, so its
    # re-emission (10.0) also runs under version 0; pipelined, the end is
    # scheduled when a drains, after the update. Either way every version-0
    # call precedes the one reset and every later call carries version 2.
    calls = [call["value"] for call in tracker.calls]
    assert calls[0] == 10.0 and sorted(calls[1:]) == [10.0, 20.0, 20.0]
    results = collected.results["T"]
    versions = [r.controls.version for r in results]
    ticks = [r.rows()[0]["ticks"] for r in results]
    reset_in = [any(e["event"] == "state_reset" for e in r.trace) for r in results]
    assert tracker.resets == 1 and reset_in.count(True) == 1
    first_new = reset_in.index(True)
    assert versions == sorted(versions) and versions[first_new] == 2
    assert all(version == 0 for version in versions[:first_new]) and first_new >= 1
    assert ticks[:first_new] == list(range(1, first_new + 1))
    assert ticks[first_new:] == list(range(1, len(results) - first_new + 1))
    if pipeline is not None:
        assert first_new == 1, "pipelined: a's end is scheduled after the update"


# Drain cleanup and end emissions (review probes) --------------------------------
#
# From tasks/m7-controls-correctness/test_review_probes.py (independent review of
# m7-controls-core-r2), kept as product regressions for the paths that thread
# the admitted snapshot through operator ends and finishes.


@pytest.mark.parametrize("failure", [False, True], ids=["cancel", "failure"])
def test_a_run_aborted_while_new_work_drains_completes_without_a_reset(
    failure,
) -> None:
    gate_b = threading.Event()
    feeds = {"a": [emit(value=1.0)], "b": [gate_b, emit(value=20.0)]}
    session = make_session(merge_definition(), feeds)
    registry = session.controls.in_flight
    draining = threading.Event()
    wait_drained_below = registry.wait_drained_below

    def observed(epoch, aborted=None):
        draining.set()
        return wait_drained_below(epoch, aborted)

    registry.wait_drained_below = observed
    run = session.start(handlers=Collector().handlers("T"), pipeline=PIPELINE)
    assert session.gates["entered"].wait(WAIT)  # a#0 (version 0) held upstream
    session.controls.update(tracking=False)
    session.controls.update(tracking=True)  # epoch 2
    gate_b.set()
    assert draining.wait(WAIT)  # b#0 waits before the operator for a#0
    if failure:
        run._fail(ActiveRunError("injected failure", stage="observer"))
    else:
        run.cancel()
    session.gates["release"].set()

    if failure:
        with pytest.raises(ActiveRunError, match="injected failure"):
            run.wait(WAIT)
    else:
        assert run.wait(WAIT)
    assert registry.describe() == {}
    assert session.instances[("tracker",)].resets == 0


@MODES
def test_a_finish_emission_keeps_version_order_and_resets_once(pipeline) -> None:
    gate_b = threading.Event()
    feeds = {"a": [emit(value=1.0)], "b": [gate_b, emit(value=20.0)]}
    session = make_session(merge_definition(FinishMerge), feeds)
    collected = Collector()

    run = session.start(handlers=collected.handlers("T"), pipeline=pipeline)
    assert session.gates["entered"].wait(WAIT)
    session.controls.update(tracking=False)
    session.controls.update(tracking=True)
    gate_b.set()
    wait_for(lambda: run.counters["b"].admitted == 1, "b#0 admitted")
    session.gates["release"].set()
    assert run.wait(WAIT)

    results = collected.results["T"]
    versions = [r.controls.version for r in results]
    ticks = [r.rows()[0]["ticks"] for r in results]
    reset_in = [any(e["event"] == "state_reset" for e in r.trace) for r in results]
    first_new = reset_in.index(True)
    # push a#0, push b#0, end a, end b, finish: the finish emission included.
    assert len(results) == 5 and versions == sorted(versions)
    assert reset_in.count(True) == 1 and session.instances[("tracker",)].resets == 1
    assert ticks[:first_new] == list(range(1, first_new + 1))
    assert ticks[first_new:] == list(range(1, len(results) - first_new + 1))
    assert session.controls.in_flight.describe() == {}


# Reset of a static member shared by two sources (rule 5) --------------------------
#
# From tasks/m7-handbook-reader-experienced/probe.py::static_reset_probe: a
# configuration-only tracker (domain None) runs in the routes of both sources,
# whose stage turns are ordered per domain only.


def static_tracker_definition(static: bool = True) -> dict:
    """Source a is held upstream of ``tracker``: static, or bound to source b."""
    a_fields = {"painted": "$steps.hold.value"}
    if static:
        a_fields["ticks"] = "$steps.tracker.ticks"
    definition = active(
        [source("a"), source("b")],
        [
            step(HeldPainter, "hold", value="$sources.a.value", hold=1.0),
            step(
                Tracker,
                "tracker",
                value="$inputs.value" if static else "$sources.b.value",
            ),
        ],
        [
            group("A", "$sources.a.value", **a_fields),
            group("B", "$sources.b.value", ticks="$steps.tracker.ticks"),
        ],
    )
    definition["inputs"] = [parameter("value", 1.0)]
    definition["controls"] = controls(
        tracking=enable(
            "$steps.tracker", state="reset_on_enable", suspends_effects=True
        )
    )

    return definition


def observe_drains(session) -> threading.Event:
    """Set the returned event when work starts waiting for older work."""
    registry = session.controls.in_flight
    draining = threading.Event()
    wait_drained_below = registry.wait_drained_below

    def observed(epoch, aborted=None):
        draining.set()
        return wait_drained_below(epoch, aborted)

    registry.wait_drained_below = observed

    return draining


def static_tracker_session(static: bool = True):
    """A session of ``static_tracker_definition`` whose b#0 waits for its gate."""
    gate_b = threading.Event()
    feeds = {"a": [emit(value=1.0)], "b": [gate_b, emit(value=20.0)]}
    session = make_session(static_tracker_definition(static), feeds)
    session.gate_b = gate_b

    return session


def start_with_reset_while_a_is_held(session, collected: Collector, pipeline):
    """Hold a#0 (version 0) upstream, reset-enable (epoch 2), then admit b#0."""
    run = session.start(handlers=collected.handlers("A", "B"), pipeline=pipeline)
    assert session.gates["entered"].wait(WAIT)
    session.controls.update(tracking=False)
    session.controls.update(tracking=True)
    session.gate_b.set()

    return run


@MODES
def test_an_old_pulse_of_another_source_reaches_a_static_member_before_its_reset(
    pipeline,
) -> None:
    session = static_tracker_session()
    tracker = session.instances[("tracker",)]
    assert session.plan.step(("tracker",)).domain is None
    draining = observe_drains(session)
    collected = Collector()

    run = start_with_reset_while_a_is_held(session, collected, pipeline)
    if pipeline is not None:
        assert draining.wait(WAIT)  # b#0 waits at the tracker for a#0
        assert tracker.calls == [] and tracker.resets == 0
    session.gates["release"].set()
    assert run.wait(WAIT)

    a, b = collected.results["A"], collected.results["B"]
    assert [(r.controls.version, r.rows()) for r in a] == [
        (0, [{"painted": 10.0, "ticks": 1}])
    ]
    assert [(r.controls.version, r.rows()) for r in b] == [(2, [{"ticks": 1}])]
    assert tracker.resets == 1
    assert not any(e["event"] == "state_reset" for e in a[0].trace)
    reset = [e for e in b[0].trace if e["event"] == "state_reset"]
    assert [e["epoch"] for e in reset] == [2]
    drains = [e for e in b[0].trace if e["event"] == "reset_drain"]
    if pipeline is not None:
        assert [(e["step"], e["epoch"], e["in_flight"]) for e in drains] == [
            (["tracker"], 2, {"0": 1, "2": 1})
        ]
    else:
        assert drains == []  # serial order already ran a#0 first
    assert session.controls.in_flight.describe() == {}


def test_a_domain_bound_member_resets_without_waiting_for_another_source() -> None:
    session = static_tracker_session(static=False)
    tracker = session.instances[("tracker",)]
    assert session.plan.step(("tracker",)).domain == "b"
    draining = observe_drains(session)
    collected = Collector()

    run = start_with_reset_while_a_is_held(session, collected, PIPELINE)
    try:
        wait_for(lambda: "B" in collected.results, "b#0 delivered while a#0 is held")
        assert tracker.resets == 1 and not draining.is_set()
    finally:
        session.gates["release"].set()
    assert run.wait(WAIT)

    assert [r.rows() for r in collected.results["B"]] == [[{"ticks": 1}]]
    assert [r.rows() for r in collected.results["A"]] == [[{"painted": 10.0}]]


@pytest.mark.parametrize("failure", [False, True], ids=["cancel", "failure"])
def test_a_run_aborted_while_a_static_reset_drains_completes_without_a_reset(
    failure,
) -> None:
    session = static_tracker_session()
    draining = observe_drains(session)

    run = start_with_reset_while_a_is_held(session, Collector(), PIPELINE)
    assert draining.wait(WAIT)  # b#0 waits at the tracker for a#0
    if failure:
        run._fail(ActiveRunError("injected failure", stage="observer"))
    else:
        run.cancel()
    session.gates["release"].set()

    if failure:
        with pytest.raises(ActiveRunError, match="injected failure"):
            run.wait(WAIT)
    else:
        assert run.wait(WAIT)
    assert session.controls.in_flight.describe() == {}
    assert session.instances[("tracker",)].resets == 0


# Passive pipeline ---------------------------------------------------------------


def test_passive_pipeline_takes_the_snapshot_at_submission() -> None:
    definition = passive(
        [
            step(
                HeldPainter,
                "painter",
                value="$inputs.value",
                threshold="$inputs.threshold",
                hold=1.0,
            )
        ],
        {"value": "$steps.painter.value"},
        controls=controls(threshold=input_control("threshold", default=0.5)),
    )
    session = make_session(definition)
    held = session.gates

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        first = pipeline.submit({"value": 1.0})
        assert held["entered"].wait(WAIT)
        second = pipeline.submit({"value": 2.0})
        session.controls.update(threshold=0.9)
        third = pipeline.submit({"value": 3.0})
        held["release"].set()
        results = [future.result(WAIT) for future in (first, second, third)]

    assert [r.controls.version for r in results] == [0, 0, 1]
    phases = session.instances[("painter",)].phases
    assert [p["threshold"] for p in phases if p["phase"] == "first"] == [0.5, 0.5, 0.9]
