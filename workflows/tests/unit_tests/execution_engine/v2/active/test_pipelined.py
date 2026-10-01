"""Pipelined active runs: real stage overlap, per-stage order, serial parity.

Every probe is ordered by events the test controls: a block call waits on a
``Probe`` hold until the test releases it, and the test waits for
``Probe.entered`` before it asserts. ``WAIT`` only bounds a wait so a broken
scheduler fails instead of hanging; no assertion depends on thread timing.

The harness here is shared with ``test_pipelined_lifecycle`` and
``test_pipelined_operators``.
"""

import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import (
    current_pulse_run_id,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import (
    CompileOptions,
    ExecutionObserver,
)
from roboflow_workflows.execution_engine.v2.sources import Emission

from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Counter,
    Echo,
    Failing,
    Notice,
    Scale,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Log,
    Scripted,
    active,
    emit,
    group,
    parameter,
    source,
    step,
)

PIPELINE = PipelineOptions(max_in_flight=4)


class Probe:
    """Shared by the probe blocks of one session: holds, entries and overlap.

    ``hold(unit, value)`` makes the call of ``unit`` for ``value`` wait until
    the test sets the returned event; ``entered(unit, value)`` is set when
    that call starts. ``order[unit]`` lists the values in call order and
    ``peak[unit]`` the most calls of one unit in progress at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holds: Dict[Tuple[str, float], threading.Event] = {}
        self._entered: Dict[Tuple[str, float], threading.Event] = {}
        self._active: Dict[str, int] = defaultdict(int)
        self.peak: Dict[str, int] = defaultdict(int)
        self.order: Dict[str, List[float]] = defaultdict(list)
        self.events: List[Tuple[str, str, float]] = []
        self.threads: Dict[Tuple[str, float], str] = {}

    def hold(self, unit: str, value: float) -> threading.Event:
        with self._lock:
            release = self._holds.setdefault((unit, value), threading.Event())

        return release

    def entered(self, unit: str, value: float) -> threading.Event:
        with self._lock:
            event = self._entered.setdefault((unit, value), threading.Event())

        return event

    @contextmanager
    def call(self, unit: str, value: float) -> Iterator[None]:
        with self._lock:
            self._active[unit] += 1
            self.peak[unit] = max(self.peak[unit], self._active[unit])
            self.order[unit].append(value)
            self.events.append(("enter", unit, value))
            self.threads[(unit, value)] = threading.current_thread().name
            release = self._holds.get((unit, value))
        self.entered(unit, value).set()
        try:
            if release is not None:
                assert release.wait(WAIT), f"{unit}({value}) was never released"
            yield
        finally:
            with self._lock:
                self._active[unit] -= 1
                self.events.append(("exit", unit, value))


class Held(Block):
    """Passes ``value`` through; the call is a ``Probe`` unit named after the step."""

    type = "test/pipelined_held@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    def run(self, *, value) -> dict:
        unit = "/".join(get_execution_context().step_path)
        with self.probe.call(unit, value):
            return {"value": value}


class TwoPhase(Block):
    """``first -> second``; each phase is a ``Probe`` unit of the same name."""

    type = "test/pipelined_two_phase@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    @phase
    def first(self, *, value):
        with self.probe.call("first", value):
            return value

    @phase
    def second(self, *, first):
        with self.probe.call("second", first):
            return {"value": first * 10}

    def run(self, *, value):
        return self.second(first=self.first(value=value))


class WholeCallTwoPhase(TwoPhase):
    """``TwoPhase`` whose phases must not overlap across pulses."""

    type = "test/pipelined_whole_call_two_phase@v1"
    phase_overlap = False


CATALOGUE = Catalogue(
    [
        ContinueIf,
        Counter,
        Echo,
        Failing,
        Held,
        Notice,
        Scale,
        TwoPhase,
        WholeCallTwoPhase,
    ],
    sources=[Scripted],
)


class Session:
    """A compiled active plan with its probe, feeds and session."""

    def __init__(
        self,
        definition: dict,
        feeds: Dict[str, list],
        *,
        mode: str = "run",
        observer: Optional[ExecutionObserver] = None,
        catalogue: Catalogue = CATALOGUE,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feeds = feeds
        self.log = Log()
        self.probe = Probe()
        started = threading.Event()
        started.set()
        plan = compile_workflow(
            definition,
            catalogue=catalogue,
            options=CompileOptions(block_execution=mode),
        )
        self.session = plan.create_session(
            resources={
                "feeds": feeds,
                "log": self.log,
                "started": started,
                "probe": self.probe,
                **(resources or {}),
            },
            observer=observer,
        )

    def start(self, handlers: Dict[str, Any], **options: Any):
        options.setdefault("pipeline", PIPELINE)
        run = self.session.start(handlers=handlers, **options)

        return run


def outcome(collected: Collector) -> Dict[str, list]:
    """Everything a host sees per group: sequences, rows, filtering and causes."""
    seen = {
        name: [
            (
                result.pulse.sequence,
                result.rows(),
                result.is_filtered,
                result.causes
                and [(cause.source, cause.sequence) for cause in result.causes],
            )
            for result in results
        ]
        for name, results in collected.results.items()
    }

    return seen


def run_in_both_modes(
    definition: dict,
    feeds: Dict[str, list],
    groups: List[str],
    *,
    mode: str = "run",
    catalogue: Catalogue = CATALOGUE,
    **options: Any,
) -> Tuple[Dict[str, list], Dict[str, list], Any, Any]:
    """Run serially and pipelined on fresh sessions; return both outcomes and runs."""
    outcomes = []
    runs = []
    for pipeline in (None, PIPELINE):
        session = Session(
            definition,
            {name: list(items) for name, items in feeds.items()},
            mode=mode,
            catalogue=catalogue,
        )
        collected = Collector()
        run = session.start(collected.handlers(*groups), pipeline=pipeline, **options)
        assert run.wait(WAIT)
        outcomes.append(outcome(collected))
        runs.append(run)

    return outcomes[0], outcomes[1], runs[0], runs[1]


# Serial parity --------------------------------------------------------------------


def two_sources_with_a_shared_static_step() -> Tuple[dict, dict, List[str]]:
    definition = active(
        [source("a"), source("b")],
        [
            step(Notice, "notice"),
            step(Scale, "double_a", value="$sources.a.value"),
            step(Scale, "double_b", value="$sources.b.value"),
        ],
        [
            group("A", "$sources.a.value", v="$steps.double_a.scaled"),
            group("B", "$sources.b.value", v="$steps.double_b.scaled"),
        ],
    )
    feeds = {
        "a": [emit(value=float(index)) for index in range(6)],
        "b": [emit(value=10.0 * index) for index in range(4)],
    }

    return definition, feeds, ["A", "B"]


def ports_emitted_separately() -> Tuple[dict, dict, List[str]]:
    """Groups not activated by a pulse retire its delivery turn at once."""
    definition = active(
        [source("s")],
        [
            step(Counter, "count", value="$sources.s.value"),
            step(Echo, "echo", value="$sources.s.label"),
        ],
        [
            group(
                "values",
                "$sources.s.value",
                value="$sources.s.value",
                count="$steps.count.count",
            ),
            group("labels", "$sources.s.label", label="$steps.echo.value"),
            group("extras", "$sources.s.extra", extra="$sources.s.extra"),
        ],
    )
    feeds = {
        "s": [
            emit(value=1.0),
            emit(label="x"),
            emit(extra=[]),
            emit(value=2.0, label="y"),
            Emission({}),
            emit(value=3.0),
        ]
    }

    return definition, feeds, ["values", "labels", "extras"]


def nested_gates() -> Tuple[dict, dict, List[str]]:
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
        ],
        [
            group(
                "A",
                "$sources.a.value",
                forwarded="$steps.child.forwarded",
                scaled="$steps.child.scaled",
            )
        ],
    )
    feeds = {"a": [emit(value=value) for value in (1.0, -1.0, 3.0, -2.0, 5.0)]}

    return definition, feeds, ["A"]


@pytest.mark.parametrize(
    "scenario",
    [two_sources_with_a_shared_static_step, ports_emitted_separately, nested_gates],
)
def test_pipelined_results_equal_the_serial_reference(scenario) -> None:
    definition, feeds, groups = scenario()

    serial, pipelined, serial_run, pipelined_run = run_in_both_modes(
        definition, feeds, groups
    )

    assert pipelined == serial
    assert pipelined_run.counters == serial_run.counters
    assert serial_run.pipeline_counters is None
    assert pipelined_run.pipeline_counters.peak("executing") <= PIPELINE.max_in_flight


def test_a_static_step_runs_once_per_pulse_of_every_source_in_each_source_order() -> (
    None
):
    definition, feeds, groups = two_sources_with_a_shared_static_step()
    session = Session(definition, feeds)

    run = session.start(Collector().handlers(*groups))

    assert run.wait(WAIT)
    notice = session.session.instances[("notice",)]
    assert len(notice.calls) == 10
    stage = run.pipeline_counters.snapshot()["stages"]["$steps.notice#call"]
    assert stage["calls"] == 10


# Stage overlap ------------------------------------------------------------------


def test_phases_of_different_pulses_overlap_and_each_phase_keeps_pulse_order() -> None:
    definition = active(
        [source("a")],
        [step(TwoPhase, "two", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.two.value")],
    )
    session = Session(
        definition,
        {"a": [emit(value=value) for value in (1.0, 2.0, 3.0)]},
        mode="phases",
    )
    release = session.probe.hold("second", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A"))

    assert session.probe.entered("second", 1.0).wait(WAIT)
    # Pulse 1 enters phase "first" while pulse 0 is still inside "second".
    assert session.probe.entered("first", 2.0).wait(WAIT)
    assert not session.probe.entered("second", 2.0).is_set()
    assert collected.results == {}
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("A") == [{"value": 10.0}, {"value": 20.0}, {"value": 30.0}]
    assert session.probe.order == {"first": [1.0, 2.0, 3.0], "second": [1.0, 2.0, 3.0]}
    assert session.probe.peak == {"first": 1, "second": 1}
    assert (
        session.probe.threads[("second", 1.0)] != session.probe.threads[("first", 2.0)]
    )
    counters = run.pipeline_counters
    assert counters.peak("overlapping_pulses") >= 2
    stages = counters.snapshot()["stages"]
    assert (
        stages["$steps.two#first"]["calls"] == stages["$steps.two#second"]["calls"] == 3
    )


def test_phase_overlap_false_holds_one_turn_for_the_whole_call() -> None:
    definition = active(
        [source("a")],
        [step(WholeCallTwoPhase, "two", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.two.value")],
    )
    session = Session(
        definition, {"a": [emit(value=value) for value in (1.0, 2.0)]}, mode="phases"
    )
    release = session.probe.hold("second", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A"))

    assert session.probe.entered("second", 1.0).wait(WAIT)
    session.log.wait_for(lambda: session.log.count("read", "a") == 3)  # incl. end
    release.set()
    assert run.wait(WAIT)
    assert session.probe.events == [
        ("enter", "first", 1.0),
        ("exit", "first", 1.0),
        ("enter", "second", 1.0),
        ("exit", "second", 1.0),
        ("enter", "first", 2.0),
        ("exit", "first", 2.0),
        ("enter", "second", 2.0),
        ("exit", "second", 2.0),
    ]
    assert set(run.pipeline_counters.snapshot()["stages"]) == {
        "$steps.two#call",
        "$groups.A#deliver",
    }
    assert collected.rows("A") == [{"value": 10.0}, {"value": 20.0}]


def test_different_steps_overlap_and_each_step_admits_pulses_in_order() -> None:
    definition = active(
        [source("a")],
        [
            step(Held, "early", value="$sources.a.value"),
            step(Held, "late", value="$steps.early.value"),
        ],
        [group("A", "$sources.a.value", value="$steps.late.value")],
    )
    values = [1.0, 2.0, 3.0, 4.0]
    session = Session(definition, {"a": [emit(value=value) for value in values]})
    release = session.probe.hold("late", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A"), admission_bound=4)

    assert session.probe.entered("late", 1.0).wait(WAIT)
    assert session.probe.entered("early", 2.0).wait(WAIT)
    release.set()
    assert run.wait(WAIT)
    assert session.probe.order == {"early": values, "late": values}
    assert session.probe.peak == {"early": 1, "late": 1}
    assert collected.sequences("A") == [0, 1, 2, 3]
    assert run.counters["a"].peak_admitted <= 4


def test_a_pulse_that_skips_a_held_step_still_delivers_after_the_earlier_pulse() -> (
    None
):
    definition = active(
        [source("a")],
        [
            step(
                ContinueIf,
                "gate",
                value="$sources.a.value",
                threshold=0.0,
                next_steps=["$steps.slow"],
            ),
            step(Held, "slow", value="$sources.a.value"),
        ],
        [
            group(
                "A", "$sources.a.value", a="$sources.a.value", slow="$steps.slow.value"
            )
        ],
    )
    events = StepEvents()
    session = Session(
        definition,
        {"a": [emit(value=1.0), emit(value=-1.0), emit(value=2.0)]},
        observer=events,
    )
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A"))

    assert session.probe.entered("slow", 1.0).wait(WAIT)
    # Pulse 1 is denied "slow" and finishes its route while pulse 0 is inside it.
    assert events.finished(f"{run.run_id}:a:1", ("slow",)).wait(WAIT)
    assert collected.results == {}
    release.set()
    assert run.wait(WAIT)
    assert collected.sequences("A") == [0, 1, 2]
    assert collected.rows("A") == [
        {"a": 1.0, "slow": 1.0},
        {"a": -1.0, "slow": None},
        {"a": 2.0, "slow": 2.0},
    ]


def test_independent_sources_do_not_wait_for_each_others_pulses() -> None:
    definition = active(
        [source("a"), source("b")],
        [
            step(Held, "slow_a", value="$sources.a.value"),
            step(Scale, "double_b", value="$sources.b.value"),
        ],
        [
            group("A", "$sources.a.value", v="$steps.slow_a.value"),
            group("B", "$sources.b.value", v="$steps.double_b.scaled"),
        ],
    )
    session = Session(
        definition,
        {
            "a": [emit(value=1.0), emit(value=2.0)],
            "b": [emit(value=float(index)) for index in range(5)],
        },
    )
    release = session.probe.hold("slow_a", 1.0)
    delivered_b = threading.Event()
    collected = Collector()

    def on_b(result) -> None:
        collected.handler("B")(result)
        if len(collected.results["B"]) == 5:
            delivered_b.set()

    run = session.start({"A": collected.handler("A"), "B": on_b})

    assert session.probe.entered("slow_a", 1.0).wait(WAIT)
    assert delivered_b.wait(WAIT)
    assert "A" not in collected.results
    release.set()
    assert run.wait(WAIT)
    assert collected.sequences("A") == [0, 1]
    assert collected.sequences("B") == [0, 1, 2, 3, 4]


# Attribution and callbacks --------------------------------------------------------


class StepEvents(ExecutionObserver):
    """Records step callbacks with ``current_pulse_run_id()`` and reader callbacks."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.steps: List[Tuple[str, Optional[str], Any]] = []
        self.readers: List[Optional[str]] = []
        self.pulses: List[Tuple[str, Optional[str]]] = []
        self._finished: Dict[Tuple[str, Any], threading.Event] = {}
        self.active = 0
        self.peak = 0

    def finished(self, run_id: str, step: Any) -> threading.Event:
        with self.lock:
            event = self._finished.setdefault((run_id, step), threading.Event())

        return event

    def _enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def _exit(self) -> None:
        with self.lock:
            self.active -= 1

    def on_step_started(self, *, step, block_type):
        self._enter()
        self.steps.append(("started", current_pulse_run_id(), step))
        self._exit()

    def on_step_finished(self, *, step, invocations, skipped):
        self._enter()
        run_id = current_pulse_run_id()
        self.steps.append(("finished", run_id, step))
        self.finished(run_id, step).set()
        self._exit()

    def on_source_opened(self, *, source):
        self.readers.append(current_pulse_run_id())

    def on_pulse_started(self, *, run_id, source, pulse):
        self.pulses.append((run_id, current_pulse_run_id()))


@pytest.mark.parametrize("pipeline", [None, PIPELINE], ids=["serial", "pipelined"])
def test_current_pulse_run_id_names_the_pulse_of_every_step_callback(pipeline) -> None:
    definition, feeds, groups = two_sources_with_a_shared_static_step()
    events = StepEvents()
    session = Session(definition, feeds, observer=events)

    run = session.start(Collector().handlers(*groups), pipeline=pipeline)

    assert run.wait(WAIT)
    assert events.readers == [None, None]
    assert all(run_id == current for run_id, current in events.pulses)
    pulse_ids = {f"{run.run_id}:a:{index}" for index in range(6)} | {
        f"{run.run_id}:b:{index}" for index in range(4)
    }
    assert {run_id for _, run_id, _ in events.steps} == pulse_ids
    assert current_pulse_run_id() is None


def test_handlers_and_observer_callbacks_of_a_pipelined_run_never_overlap() -> None:
    definition, feeds, groups = two_sources_with_a_shared_static_step()
    events = StepEvents()
    session = Session(definition, feeds, observer=events)

    def handler(result) -> None:
        events._enter()
        events._exit()

    run = session.start({name: handler for name in groups})

    assert run.wait(WAIT)
    assert events.peak == 1


def two_step_route() -> dict:
    """``work -> after`` on one source; group ``A`` delivers ``after``'s value."""
    definition = active(
        [source("a")],
        [
            step(Held, "work", value="$sources.a.value"),
            step(Held, "after", value="$steps.work.value"),
        ],
        [group("A", "$sources.a.value", value="$steps.after.value")],
    )

    return definition


class HeldHandler:
    """Group handler that waits in pulse 0's delivery until the test releases it."""

    def __init__(self, probe: Probe) -> None:
        self.probe = probe
        self.inside = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.sequences: List[int] = []

    def __call__(self, result) -> None:
        self.inside = True
        self.entered.set()
        if result.pulse.sequence == 0:
            assert self.release.wait(WAIT), "handler was never released"
        self.sequences.append(result.pulse.sequence)
        with self.probe._lock:
            self.probe.events.append(("handler_exit", "A", result.pulse.sequence))
        self.inside = False


def test_slow_handler_without_observer_lets_other_pulses_compute() -> None:
    """SF-1: the no-op observer is not serialized with handlers."""
    session = Session(two_step_route(), {"a": [emit(value=1.0), emit(value=2.0)]})
    handler = HeldHandler(session.probe)

    run = session.start({"A": handler})

    assert handler.entered.wait(WAIT)
    # Pulse 1 passes the work/after step boundary while pulse 0's handler runs.
    assert session.probe.entered("after", 2.0).wait(WAIT)
    assert handler.sequences == []
    handler.release.set()
    assert run.wait(WAIT)
    assert handler.sequences == [0, 1]


def test_slow_handler_with_a_user_observer_stalls_step_callbacks() -> None:
    """A user observer stays serialized with handlers (documented coupling)."""
    seen_inside: List[bool] = []

    class Watching(ExecutionObserver):
        def on_step_started(self, *, step, block_type):
            seen_inside.append(handler.inside)

        def on_step_finished(self, *, step, invocations, skipped):
            seen_inside.append(handler.inside)

    session = Session(
        two_step_route(),
        {"a": [emit(value=1.0), emit(value=2.0)]},
        observer=Watching(),
    )
    handler = HeldHandler(session.probe)
    release_after = session.probe.hold("after", 1.0)
    release_work = session.probe.hold("work", 2.0)

    run = session.start({"A": handler})

    # Pulse 1 is inside its block call before pulse 0's handler starts.
    assert session.probe.entered("work", 2.0).wait(WAIT)
    release_after.set()
    assert handler.entered.wait(WAIT)
    # Pulse 1 leaves its block call while pulse 0's handler still runs; its
    # next step callback waits for the handler.
    release_work.set()
    handler.release.set()
    assert run.wait(WAIT)

    events = session.probe.events
    assert events.index(("handler_exit", "A", 0)) < events.index(
        ("enter", "after", 2.0)
    )
    assert seen_inside and not any(seen_inside)
    assert handler.sequences == [0, 1]
