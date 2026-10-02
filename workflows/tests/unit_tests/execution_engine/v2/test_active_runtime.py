"""Lifecycle of active runs: readers, admission, delivery, stop, failure, restart.

Every wait in these tests is bounded by ``WAIT`` and ordered by events the
test controls (a reader blocked on an ``Event`` in its feed, a handler that
releases it). No assertion depends on thread timing.
"""

import gc
import threading
import warnings
import weakref
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.active.execution import ENGINE_CLOCK_ID
from roboflow_workflows.execution_engine.v2.active.runtime import (
    ActiveRun,
    GroupResult,
    start_session,
    stop_session,
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
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Counter,
    Echo,
    Failing,
    Notice,
    Scale,
    Sum,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""

MEDIA = Fraction(1, 1000)


class Log(list):
    """Append-only event log that lets a test wait for a condition on it."""

    def __init__(self) -> None:
        super().__init__()
        self.changed = threading.Condition()

    def append(self, item: Any) -> None:
        with self.changed:
            super().append(item)
            self.changed.notify_all()

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=WAIT), list(self)

    def count(self, kind: str, feed: str) -> int:
        return sum(1 for item in self if item[:2] == (kind, feed))


UNTIL_STOP = object()
"""Feed item: block ``read`` until the run's ``stop_event`` is set."""


class FeedSource(Source):
    """Reads a scripted feed: ``Emission`` values, ``Event`` gates, ``Exception``s.

    A gate blocks ``read`` until the test sets it (``UNTIL_STOP`` until the
    run is stopping); an exception is raised by ``read``. ``open`` waits for
    the ``started`` resource, which the harness sets once ``start`` returned,
    so a handler never runs before the test holds the run. Every lifecycle
    call is appended to the ``log`` resource.
    """

    class Params(SourceParams):
        feed: str
        scale: float | Ref(FLOAT_KIND) = 1.0
        fail_open: bool = False
        fail_close: bool = False

    def __init__(
        self, *, feeds: Dict[str, list], log: Log, started: threading.Event
    ) -> None:
        self.feeds = feeds
        self.log = log
        self.started = started
        self.feed = ""
        self.fail_close = False
        self.items = iter(())

    def open(self, *, feed, scale, fail_open, fail_close) -> None:
        self.started.wait(WAIT)
        open_gate = self.feeds.get(f"{feed}:open")
        if open_gate is not None:
            open_gate.wait(WAIT)
        self.feed = feed
        self.fail_close = fail_close
        self.log.append(("open", feed, scale, id(self)))
        if fail_open:
            raise RuntimeError(f"cannot open {feed}")
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event) or item is UNTIL_STOP:
            gate = self.stop_event if item is UNTIL_STOP else item
            self.log.append(("wait", self.feed))
            gate.wait(WAIT)
            item = next(self.items, None)
        if isinstance(item, Exception):
            raise item
        self.log.append(
            ("read", self.feed, None if item is None else sorted(item.data))
        )
        return item

    def close(self) -> None:
        self.log.append(("close", self.feed))
        if self.fail_close:
            raise RuntimeError(f"cannot close {self.feed}")


class Scripted(FeedSource):
    """Three ungrouped ports emitted together or separately."""

    type = "test/scripted@v1"
    outputs = {
        "value": SourceOutput(FLOAT_KIND),
        "label": SourceOutput(STRING_KIND),
        "extra": SourceOutput(),
    }


class Pairs(FeedSource):
    """One grouped port: a batch of floats per pulse."""

    type = "test/pairs@v1"
    outputs = {
        "pair": SourceOutput(
            FLOAT_KIND, layout=EntryLayout((Axis(id="items", kind="dynamic_nesting"),))
        )
    }


ITEMS = EntryLayout((Axis(id="items", kind="dynamic_nesting"),))


class Tagged(FeedSource):
    """Two grouped ports declared over one source-local axis."""

    type = "test/tagged@v1"
    outputs = {
        "pair": SourceOutput(FLOAT_KIND, layout=ITEMS),
        "tag": SourceOutput(STRING_KIND, layout=ITEMS),
    }


class Hold(Block):
    """Blocks inside ``run`` on the ``release`` gate for values at or above a threshold."""

    type = "test/hold@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        threshold: float = 0.0

    def __init__(self, *, gates: Dict[str, threading.Event]) -> None:
        self.gates = gates
        self.calls: List[float] = []

    def run(self, *, value, threshold) -> dict:
        self.calls.append(value)
        if value >= threshold:
            self.gates["entered"].set()
            self.gates["release"].wait(WAIT)
        return {"value": value}


CATALOGUE = Catalogue(
    [ContinueIf, Counter, Echo, Failing, Hold, Notice, Scale, Sum],
    sources=[Scripted, Pairs, Tagged],
)


def emit(
    *, pts: Optional[int] = None, meta: Optional[dict] = None, **ports
) -> Emission:
    media = Timestamp(pts, MEDIA, "media") if pts is not None else None
    return Emission(ports, media=media, source_metadata=meta or {})


def source(name: str, *, feed: Optional[str] = None, **params: Any) -> dict:
    return {"type": Scripted.type, "name": name, "feed": feed or name, **params}


def step(block: type, name: str, **params: Any) -> dict:
    return {"type": block.type, "name": name, **params}


def group(name: str, anchor: str, **fields: str) -> dict:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def active(sources: list, steps: list, groups: list, *, inputs: list = ()) -> dict:
    definition = {
        "version": "2.0",
        "inputs": list(inputs),
        "sources": sources,
        "steps": steps,
        "outputs": groups,
    }

    return definition


def parameter(name: str, default: Any) -> dict:
    return {"type": "WorkflowParameter", "name": name, "default_value": default}


class Collector:
    """Handlers that keep every result per group, in delivery order."""

    def __init__(self) -> None:
        self.results: Dict[str, List[GroupResult]] = {}

    def handler(self, name: str) -> Callable[[GroupResult], None]:
        def handle(result: GroupResult) -> None:
            self.results.setdefault(name, []).append(result)

        return handle

    def handlers(self, *names: str) -> Dict[str, Callable[[GroupResult], None]]:
        return {name: self.handler(name) for name in names}

    def rows(self, name: str) -> List[dict]:
        return [result.rows()[0] for result in self.results.get(name, [])]

    def sequences(self, name: str) -> List[int]:
        return [result.pulse.sequence for result in self.results.get(name, [])]


class Events(ExecutionObserver):
    """Records active-run observer events; sets an event per closed source."""

    def __init__(self) -> None:
        self.events: List[tuple] = []
        self.closed: Dict[str, threading.Event] = {}

    def closed_event(self, source: str) -> threading.Event:
        return self.closed.setdefault(source, threading.Event())

    def on_run_started(self, *, session_id, run_id):
        self.events.append(("run_started",))

    def on_source_opened(self, *, source):
        self.events.append(("source_opened", source))

    def on_pulse_started(self, *, run_id, source, pulse):
        self.events.append(("pulse_started", source, pulse.sequence))

    def on_step_started(self, *, step, block_type):
        self.events.append(("step_started", step))

    def on_group_delivered(self, *, run_id, group, source, pulse):
        self.events.append(("group_delivered", group, pulse.sequence))

    def on_pulse_finished(self, *, run_id, source, pulse, error):
        self.events.append(("pulse_finished", source, pulse.sequence, error is None))

    def on_source_closed(self, *, source, error):
        self.events.append(("source_closed", source, error is None))
        self.closed_event(source).set()

    def on_run_finished(self, *, run_id, result, error):
        self.events.append(("run_finished", error is None))


class Harness:
    """A compiled active plan, its session and one started run."""

    def __init__(
        self,
        definition: dict,
        feeds: Dict[str, list],
        *,
        observer: Optional[ExecutionObserver] = None,
        error_handler: Optional[Callable[[StepExecutionError], None]] = None,
    ) -> None:
        self.log = Log()
        self.started = threading.Event()
        self.plan = compile_workflow(definition, catalogue=CATALOGUE)
        self.session = self.plan.create_session(
            resources={
                "feeds": feeds,
                "log": self.log,
                "started": self.started,
                "gates": {"entered": threading.Event(), "release": threading.Event()},
            },
            observer=observer,
            error_handler=error_handler,
        )
        self.run: Optional[ActiveRun] = None

    def start(self, handlers: Dict[str, Any], **options: Any) -> ActiveRun:
        self.started.clear()
        try:
            self.run = self.session.start(handlers=handlers, **options)
        finally:
            self.started.set()
        return self.run

    def instance(self, name: str):
        return self.session.instances[(name,)]


def doubling(sources: list, groups: list) -> dict:
    """Each source's ``value`` doubled by its own ``Scale`` step."""
    steps = [
        step(Scale, f"double_{item['name']}", value=f"$sources.{item['name']}.value")
        for item in sources
    ]
    definition = active(sources, steps, groups)

    return definition


# Independent sources -----------------------------------------------------


def test_sources_deliver_independently_and_continue_after_the_other_ends() -> None:
    hold_a, hold_b = threading.Event(), threading.Event()
    feeds = {
        "a": [emit(value=1.0), hold_a, emit(value=2.0), emit(value=3.0)],
        "b": [hold_b, emit(value=10.0)],
    }
    definition = doubling(
        [source("a"), source("b")],
        [
            group("A", "$sources.a.value", doubled="$steps.double_a.scaled"),
            group("B", "$sources.b.value", doubled="$steps.double_b.scaled"),
        ],
    )
    events = Events()
    harness = Harness(definition, feeds, observer=events)
    collected = Collector()
    b_closed_at_delivery: List[bool] = []

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        b_closed_at_delivery.append(harness.run.counters["b"].closed)
        hold_b.set()  # b may read only after a's first result arrived

    def on_b(result: GroupResult) -> None:
        collected.handler("B")(result)
        events.closed_event("b").wait(WAIT)  # b ends; only then a resumes
        hold_a.set()

    run = harness.start({"A": on_a, "B": on_b})

    assert run.wait(WAIT)
    assert collected.rows("A") == [{"doubled": 2.0}, {"doubled": 4.0}, {"doubled": 6.0}]
    assert collected.rows("B") == [{"doubled": 20.0}]
    assert collected.sequences("A") == [0, 1, 2]
    assert collected.sequences("B") == [0]
    assert b_closed_at_delivery == [False, True, True]
    assert {name: item.ended for name, item in run.counters.items()} == {
        "a": True,
        "b": True,
    }
    assert run.state == "finished"
    assert harness.log.count("close", "a") == harness.log.count("close", "b") == 1


def test_ports_emitted_separately_route_terminally_and_deliver_once() -> None:
    feeds = {
        "s": [
            emit(value=1.0),
            emit(label="x"),
            emit(extra=[]),
            emit(extra=None),
            emit(value=2.0, label="y"),
            Emission({}),
        ]
    }
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
                label="$sources.s.label",
            ),
            group(
                "labels",
                "$sources.s.label",
                label="$steps.echo.value",
                count="$steps.count.count",
            ),
            group("extras", "$sources.s.extra", extra="$sources.s.extra"),
        ],
    )
    harness = Harness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("values", "labels", "extras"))

    assert run.wait(WAIT)
    assert collected.sequences("values") == [0, 4, 5]
    assert collected.sequences("labels") == [1, 4, 5]
    assert collected.sequences("extras") == [2, 3, 5]
    assert collected.rows("values") == [
        {"value": 1.0, "count": 1, "label": None},
        {"value": 2.0, "count": 2, "label": "y"},
        {"value": None, "count": None, "label": None},
    ]
    assert collected.rows("labels") == [
        {"label": "x", "count": None},
        {"label": "y", "count": 2},
        {"label": None, "count": None},
    ]
    assert collected.rows("extras") == [{"extra": []}, {"extra": None}, {"extra": None}]
    first, _, last = collected.results["values"]
    assert first.statuses == {
        "value": "complete",
        "count": "complete",
        "label": "filtered",
    }
    assert first.filtered_paths["label"] == ((),)
    assert not first.is_filtered
    assert last.is_filtered and last.outputs.is_filtered
    assert last.statuses == {
        "value": "filtered",
        "count": "filtered",
        "label": "filtered",
    }
    present_empty, present_none, _ = collected.results["extras"]
    assert present_empty.statuses == {"extra": "complete"}
    assert present_none.statuses == {"extra": "complete"}
    # The shared stateful step ran once per pulse that supplied its input.
    assert len(harness.instance("count").calls) == 2
    assert run.counters["s"].delivered == 9


def test_output_free_action_runs_once_per_admitted_pulse_of_every_source() -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0)], "b": [emit(value=3.0)]}
    definition = active(
        [source("a"), source("b")],
        [step(Notice, "notice"), step(Scale, "double", value="$sources.a.value")],
        [
            group("A", "$sources.a.value", doubled="$steps.double.scaled"),
            group("A2", "$sources.a.value", doubled="$steps.double.scaled"),
        ],
    )
    harness = Harness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("A", "A2"))

    assert run.wait(WAIT)
    assert len(harness.instance("notice").calls) == 3
    assert len(harness.instance("double").calls) == 2
    assert (
        collected.rows("A")
        == collected.rows("A2")
        == [{"doubled": 2.0}, {"doubled": 4.0}]
    )


def test_static_inputs_reach_source_parameters_and_steps() -> None:
    feeds = {"a": [emit(value=1.0)]}
    definition = active(
        [source("a", scale="$inputs.factor")],
        [step(Scale, "scale", value="$sources.a.value", factor="$inputs.factor")],
        [group("A", "$sources.a.value", scaled="$steps.scale.scaled")],
        inputs=[parameter("factor", 3.0)],
    )
    harness = Harness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("A"), inputs={"factor": 4.0})
    assert run.wait(WAIT)
    again = harness.start(collected.handlers("A"))
    assert again.wait(WAIT)

    opens = [item[:3] for item in harness.log if item[0] == "open"]
    assert opens == [("open", "a", 4.0), ("open", "a", 3.0)]
    assert collected.rows("A") == [{"scaled": 4.0}, {"scaled": 3.0}]


def test_zero_length_source_opens_ends_and_closes_without_a_pulse() -> None:
    harness = Harness(
        doubling([source("z")], [group("Z", "$sources.z.value", v="$sources.z.value")]),
        {"z": []},
    )
    collected = Collector()

    run = harness.start(collected.handlers("Z"))

    assert run.wait(WAIT)
    assert collected.results == {}
    counters = run.counters["z"]
    assert (counters.opened, counters.ended, counters.closed) == (True, True, True)
    assert (counters.read, counters.admitted) == (0, 0)
    assert list(harness.log)[1:] == [("read", "z", None), ("close", "z")]


# Stop, backpressure and callbacks -------------------------------------------


def test_stop_closes_admission_drains_admitted_work_and_is_idempotent() -> None:
    hold = threading.Event()
    feeds = {"a": [emit(value=1.0), hold, emit(value=2.0), emit(value=3.0)]}
    harness = Harness(
        doubling(
            [source("a")], [group("A", "$sources.a.value", v="$steps.double_a.scaled")]
        ),
        feeds,
    )
    collected = Collector()
    stopped = threading.Event()
    wait_errors: List[Exception] = []

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        harness.run.stop()
        harness.run.stop()
        try:
            harness.run.wait()
        except ContractError as error:
            wait_errors.append(error)
        stopped.set()

    run = harness.start({"A": on_a})

    assert stopped.wait(WAIT)
    assert run.state == "stopping"
    hold.set()  # the reader now returns the second emission, after stop
    assert run.wait(WAIT)
    assert run.state == "finished"
    assert collected.rows("A") == [{"v": 2.0}]
    counters = run.counters["a"]
    assert (counters.read, counters.admitted, counters.unadmitted) == (2, 1, 1)
    assert (counters.processed, counters.delivered, counters.closed) == (1, 1, True)
    assert len(wait_errors) == 1 and "would wait for itself" in str(wait_errors[0])
    run.stop()
    assert run.wait(WAIT) and run.state == "finished"


def test_stop_before_any_pulse_processes_nothing_and_closes_sources() -> None:
    hold = threading.Event()
    feeds = {"a": [hold, emit(value=1.0)]}
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        feeds,
    )
    collected = Collector()

    run = harness.start(collected.handlers("A"))
    harness.log.wait_for(lambda: harness.log.count("wait", "a") == 1)
    run.stop()
    hold.set()

    assert run.wait(WAIT)
    assert collected.results == {}
    assert run.counters["a"].unadmitted == 1
    assert run.counters["a"].closed


def test_slow_synchronous_handler_backpressures_readers_within_the_bound() -> None:
    bound = 2
    feeds = {"a": [emit(value=float(index)) for index in range(10)]}
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        feeds,
    )
    release = threading.Event()
    invariants: List[bool] = []

    def on_a(result: GroupResult) -> None:
        counters = harness.run.counters["a"]
        invariants.append(counters.read <= counters.processed + bound + 1)
        if result.pulse.sequence == 0:
            release.wait(WAIT)

    run = harness.start({"A": on_a}, admission_bound=bound)

    # One pulse in the handler, ``bound - 1`` queued, one read and waiting.
    harness.log.wait_for(lambda: harness.log.count("read", "a") == bound + 1)
    assert run.wait(timeout=0.2) is False
    assert harness.log.count("read", "a") == bound + 1
    assert run.counters["a"].admitted == bound
    release.set()

    assert run.wait(WAIT)
    assert run.counters["a"].delivered == 10
    assert invariants == [True] * 10


def test_context_manager_stops_and_waits() -> None:
    hold = threading.Event()
    feeds = {"a": [emit(value=1.0), hold, emit(value=2.0)]}
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        feeds,
    )
    collected = Collector()
    delivered = threading.Event()

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        delivered.set()

    with harness.start({"A": on_a}) as run:
        assert delivered.wait(WAIT)
        hold.set()

    assert run.done and run.state == "finished"
    assert len(collected.results["A"]) >= 1


# Failures ----------------------------------------------------------------


def test_read_failure_cancels_undelivered_work_and_closes_every_source() -> None:
    hold_a, hold_b = threading.Event(), threading.Event()
    feeds = {
        "a": [emit(value=1.0), emit(value=2.0), hold_a, RuntimeError("boom")],
        "b": [hold_b, emit(value=5.0)],
    }
    events = Events()
    harness = Harness(
        doubling(
            [source("a"), source("b")],
            [
                group("A", "$sources.a.value", v="$sources.a.value"),
                group("B", "$sources.b.value", v="$sources.b.value"),
            ],
        ),
        feeds,
        observer=events,
    )
    collected = Collector()

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        hold_a.set()  # a's read now fails, while this pulse is being delivered
        events.closed_event("a").wait(WAIT)

    run = harness.start({"A": on_a, "B": collected.handler("B")}, admission_bound=2)
    assert events.closed_event("a").wait(WAIT)
    hold_b.set()

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    failure = caught.value
    assert (failure.stage, failure.source, failure.pulse) == ("read", "a", None)
    assert isinstance(failure.__cause__, RuntimeError) and "boom" in str(failure)
    assert run.state == "failed" and run.failure is failure
    a, b = run.counters["a"], run.counters["b"]
    # Pulse 0's group was delivered before its route ran; the failure then
    # cut the rest of that pulse, so it counts as cancelled, not processed.
    assert (a.admitted, a.processed, a.cancelled, a.closed) == (2, 0, 2, True)
    assert a.delivered == 1
    assert (b.read, b.unadmitted, b.delivered, b.closed) == (1, 1, 0, True)
    assert collected.sequences("A") == [0] and "B" not in collected.results
    with pytest.raises(ActiveRunError) as again:
        run.wait(WAIT)
    assert again.value is failure


def test_open_failure_closes_the_partially_opened_source_and_the_others() -> None:
    hold_a, open_b = threading.Event(), threading.Event()
    feeds = {"a": [hold_a, emit(value=1.0)], "b": [], "b:open": open_b}
    events = Events()
    harness = Harness(
        doubling(
            [source("a"), source("b", fail_open=True)],
            [group("A", "$sources.a.value", v="$sources.a.value")],
        ),
        feeds,
        observer=events,
    )

    run = harness.start({"A": Collector().handler("A")})
    harness.log.wait_for(lambda: harness.log.count("wait", "a") == 1)
    open_b.set()  # b fails to open while a is blocked in read
    assert events.closed_event("b").wait(WAIT)
    hold_a.set()

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert (caught.value.stage, caught.value.source) == ("open", "b")
    assert harness.log.count("close", "b") == harness.log.count("close", "a") == 1
    assert not run.counters["b"].opened and run.counters["b"].closed
    assert run.counters["a"].opened and run.counters["a"].closed
    assert run.counters["a"].unadmitted == 1 and run.counters["a"].delivered == 0


def test_step_failure_is_attributed_to_its_pulse_and_step() -> None:
    feeds = {"a": [emit(value=1.0), emit(value=-1.0), emit(value=2.0)]}
    handled: List[StepExecutionError] = []
    harness = Harness(
        active(
            [source("a")],
            [step(Failing, "check", value="$sources.a.value")],
            [group("A", "$sources.a.value", v="$steps.check.value")],
        ),
        feeds,
        error_handler=handled.append,
    )
    collected = Collector()

    run = harness.start(collected.handlers("A"))

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    failure = caught.value
    assert (failure.stage, failure.source, failure.pulse) == ("step", "a", 1)
    assert failure.step_path == ("check",)
    assert isinstance(failure.__cause__, StepExecutionError)
    assert handled == [failure.__cause__]
    assert collected.rows("A") == [{"v": 1.0}]
    counters = run.counters["a"]
    # The failing pulse itself counts as cancelled, like the ones dropped after it.
    assert counters.admitted == counters.processed + counters.cancelled
    assert counters.cancelled >= 1
    assert counters.read == counters.admitted + counters.unadmitted
    assert counters.closed


def test_handler_failure_is_terminal_and_a_close_failure_is_kept() -> None:
    # The reader reaches its failing close only once the handler failure has
    # set the stop event, so the close error is the secondary one.
    feeds = {"a": [emit(value=1.0), emit(value=2.0), UNTIL_STOP]}
    harness = Harness(
        doubling(
            [source("a", fail_close=True)],
            [group("A", "$sources.a.value", v="$sources.a.value")],
        ),
        feeds,
    )

    def on_a(result: GroupResult) -> None:
        if result.pulse.sequence == 1:
            raise ValueError("cannot handle")

    run = harness.start({"A": on_a})

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    failure = caught.value
    assert (failure.stage, failure.source, failure.pulse, failure.group) == (
        "handler",
        "a",
        1,
        "A",
    )
    assert isinstance(failure.__cause__, ValueError)
    assert [error.stage for error in failure.suppressed] == ["close"]
    assert run.counters["a"].delivered == 1  # the failing delivery is not counted


def test_close_failure_alone_fails_the_run_after_delivery() -> None:
    hold = threading.Event()
    harness = Harness(
        doubling(
            [source("a", fail_close=True)],
            [group("A", "$sources.a.value", v="$sources.a.value")],
        ),
        {"a": [emit(value=1.0), hold]},
    )
    collected = Collector()

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        hold.set()  # the source ends, and fails to close, after delivery

    run = harness.start({"A": on_a})

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert (caught.value.stage, caught.value.source, caught.value.suppressed) == (
        "close",
        "a",
        (),
    )
    assert collected.rows("A") == [{"v": 1.0}]


def test_async_handlers_are_rejected_and_awaitables_never_leak() -> None:
    definition = doubling(
        [source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]
    )
    harness = Harness(definition, {"a": [emit(value=1.0)]})

    async def coroutine_handler(result: GroupResult) -> None:
        pass

    with pytest.raises(ContractError, match="coroutine function"):
        harness.start({"A": coroutine_handler})
    assert list(harness.log) == []  # rejected before any source opened

    async def later() -> None:
        pass

    def returns_awaitable(result: GroupResult):
        return later()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        run = harness.start({"A": returns_awaitable})
        with pytest.raises(ActiveRunError) as caught:
            run.wait(WAIT)
    assert caught.value.stage == "handler" and "awaitable" in str(caught.value)
    assert not [w for w in caught_warnings if issubclass(w.category, RuntimeWarning)]


def test_start_validates_handlers_bound_and_plan_kind_before_opening() -> None:
    definition = doubling(
        [source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]
    )
    harness = Harness(definition, {"a": [emit(value=1.0)]})

    with pytest.raises(ContractError, match="unknown output groups \\['B'\\]"):
        harness.start({"B": lambda result: None})
    with pytest.raises(ContractError, match="must be callable"):
        harness.start({"A": "not callable"})
    with pytest.raises(ContractError, match="at least 1"):
        harness.start({}, admission_bound=0)
    with pytest.raises(WorkflowInputError, match="Missing"):
        harness.start({}, inputs={"unknown": 1})
    assert list(harness.log) == []
    with pytest.raises(WorkflowInputError, match="start"):
        harness.session.run({})

    passive = compile_workflow(
        {
            "version": "2.0",
            "inputs": [parameter("x", 1.0)],
            "steps": [step(Scale, "scale", value="$inputs.x")],
            "outputs": [
                {"type": "JsonField", "name": "y", "selector": "$steps.scale.scaled"}
            ],
        },
        catalogue=CATALOGUE,
    )
    with pytest.raises(WorkflowInputError, match="no sources"):
        start_session(passive.create_session())


def test_second_start_waits_for_the_first_and_block_state_continues() -> None:
    hold = threading.Event()
    feeds = {"a": [emit(value=1.0), hold, emit(value=2.0)]}
    harness = Harness(
        active(
            [source("a")],
            [step(Counter, "count", value="$sources.a.value")],
            [group("A", "$sources.a.value", count="$steps.count.count")],
        ),
        feeds,
    )
    collected = Collector()
    delivered = threading.Event()

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        delivered.set()

    first = harness.start({"A": on_a})
    assert delivered.wait(WAIT)
    with pytest.raises(ContractError, match="already has active run"):
        harness.start({"A": on_a})
    hold.set()
    assert first.wait(WAIT)
    second = harness.start(collected.handlers("A"))
    assert second.wait(WAIT)

    assert collected.rows("A") == [
        {"count": 1},
        {"count": 2},
        {"count": 3},
        {"count": 4},
    ]
    assert first.run_id != second.run_id
    instances = {item[3] for item in harness.log if item[0] == "open"}
    assert len(instances) == 2  # a fresh source instance per start
    assert (
        collected.results["A"][0].pulse.sequence
        == collected.results["A"][2].pulse.sequence
        == 0
    )


# Identity, metadata and grouped ports ---------------------------------------


def test_results_carry_pulse_identity_source_and_media_context() -> None:
    supplied = InputValue(
        "x",
        EntryMetadata(
            sample={(): SampleContext(source_id="probe-7", source_type="custom")},
            temporal={(): None},
        ),
    )
    feeds = {
        "a": [
            Emission(
                {"value": 1.0, "label": supplied},
                media=Timestamp(40, MEDIA, "cam"),
                source_metadata={"frame": 4},
            )
        ]
    }
    harness = Harness(
        active(
            [source("a")],
            [step(Scale, "double", value="$sources.a.value")],
            [
                group(
                    "A",
                    "$sources.a.value",
                    value="$sources.a.value",
                    doubled="$steps.double.scaled",
                    label="$sources.a.label",
                )
            ],
        ),
        feeds,
    )
    collected = Collector()

    run = harness.start(collected.handlers("A"))

    assert run.wait(WAIT)
    (result,) = collected.results["A"]
    assert result.pulse.active_run_id == run.run_id
    assert (result.source, result.pulse.sequence) == ("a", 0)
    assert result.run_id == f"{run.run_id}:a:0"
    assert result.outputs.lineage_id == f"run:{run.run_id}/source:a"
    assert result.outputs.pulse_id == 0
    assert result.session_id == harness.session.session_id
    for key in ("value", "doubled"):
        metadata = result.outputs.metadata[key]
        sample = metadata.sample_at(())
        assert (sample.source_id, sample.source_type) == ("a", Scripted.type)
        assert dict(sample.source_metadata) == {"frame": 4}
        temporal = metadata.temporal_at(())
        assert temporal.media_coverage == Timestamp(40, MEDIA, "cam")
        assert temporal.observed_coverage.clock_id == ENGINE_CLOCK_ID
        assert temporal.capture_coverage is None
        assert result.outputs.layout[key].axes == ()
    label = result.outputs.metadata["label"]
    assert label.sample_at(()).source_id == "probe-7"
    assert label.temporal_at(()) is None
    assert result.rows() == [{"value": 1.0, "doubled": 2.0, "label": "x"}]


def test_grouped_port_keeps_its_batch_and_feeds_group_consumers() -> None:
    feeds = {
        "p": [
            Emission({"pair": Batch.of([1.0, 2.0])}),
            Emission({"pair": Batch.of([])}),
        ]
    }
    definition = active(
        [{"type": Pairs.type, "name": "p", "feed": "p"}],
        [step(Sum, "total", values="$sources.p.pair")],
        [
            group(
                "P",
                "$sources.p.pair",
                pair="$sources.p.pair",
                total="$steps.total.total",
            )
        ],
    )
    harness = Harness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("P"))

    assert run.wait(WAIT)
    first, second = collected.results["P"]
    assert first.rows() == [{"pair": 1.0, "total": 3.0}, {"pair": 2.0, "total": 3.0}]
    assert first.outputs.layout["pair"].depth == 1
    assert first.outputs.data["pair"].indices == ((0,), (1,))
    # A genuinely empty group is a present value: no rows, like an empty
    # batch input, but the reducer ran and both fields are complete.
    assert second.rows() == []
    assert second.statuses == {"pair": "complete", "total": "complete"}
    assert len(second.outputs.data["pair"]) == 0
    assert second.outputs.data["total"] == 0.0


def test_emission_violating_the_declaration_fails_at_stage_emission() -> None:
    cases = [
        (Emission({"nope": 1.0}), "undeclared ports"),
        (Emission({"value": "text"}), "not a valid"),
        (Emission({"value": Batch.of([1.0])}), "Batch"),
    ]
    for emission, message in cases:
        harness = Harness(
            doubling(
                [source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]
            ),
            {"a": [emission]},
        )
        run = harness.start({"A": Collector().handler("A")})
        with pytest.raises(ActiveRunError, match=message) as caught:
            run.wait(WAIT)
        assert (caught.value.stage, caught.value.pulse) == ("emission", 0)
        assert run.counters["a"].closed


# Gates, nesting and observer ------------------------------------------------


def test_gates_and_nested_forwarding_follow_each_pulse() -> None:
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
    harness = Harness(
        definition, {"a": [emit(value=1.0), emit(value=-1.0), emit(value=3.0)]}
    )
    collected = Collector()

    run = harness.start(collected.handlers("A"))

    assert run.wait(WAIT)
    assert collected.rows("A") == [
        {"forwarded": 1.0, "scaled": 2.0},
        {"forwarded": None, "scaled": None},
        {"forwarded": 3.0, "scaled": 6.0},
    ]
    assert [result.is_filtered for result in collected.results["A"]] == [
        False,
        True,
        False,
    ]
    assert len(harness.instance("notice").calls) == 2
    assert len(harness.session.instances[("child", "scale")].calls) == 2


def test_observer_sees_the_run_sources_pulses_and_groups_in_order() -> None:
    events = Events()
    harness = Harness(
        doubling(
            [source("a")], [group("A", "$sources.a.value", v="$steps.double_a.scaled")]
        ),
        {"a": [emit(value=1.0)]},
        observer=events,
    )

    run = harness.start({"A": Collector().handler("A")})

    assert run.wait(WAIT)
    # The reader closes the source while the processor may still be busy
    # with the pulse, so only each thread's own order is fixed.
    reader = [event for event in events.events if event[0].startswith("source_")]
    processor = [
        event
        for event in events.events
        if event[0].startswith(("pulse_", "step_", "group_"))
    ]
    assert events.events[0] == ("run_started",)
    assert events.events[-1] == ("run_finished", True)
    assert reader == [("source_opened", "a"), ("source_closed", "a", True)]
    assert processor == [
        ("pulse_started", "a", 0),
        ("step_started", ("double_a",)),
        ("group_delivered", "A", 0),
        ("pulse_finished", "a", 0, True),
    ]
    assert events.events.index(("source_opened", "a")) < events.events.index(
        ("pulse_started", "a", 0)
    )


# Cancellation, readiness, observers, retention, shared axes ------------------


def test_failure_during_a_pulse_cancels_its_remaining_steps_and_deliveries() -> None:
    hold_read = threading.Event()
    feeds = {"a": [emit(value=1.0), emit(value=2.0), hold_read, RuntimeError("boom")]}
    events = Events()
    harness = Harness(
        active(
            [source("a")],
            [
                step(Hold, "hold", value="$sources.a.value", threshold=2.0),
                step(Scale, "after", value="$steps.hold.value"),
            ],
            [group("A", "$sources.a.value", after="$steps.after.scaled")],
        ),
        feeds,
        observer=events,
    )
    collected = Collector()
    hold = harness.session.instances[("hold",)]

    run = harness.start(collected.handlers("A"))
    assert hold.gates["entered"].wait(WAIT)  # pulse 1 is inside the block
    hold_read.set()  # the reader now fails
    assert events.closed_event("a").wait(WAIT)
    hold.gates["release"].set()  # the block returns after the failure

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert caught.value.stage == "read"
    assert collected.rows("A") == [{"after": 2.0}]  # pulse 0 stays delivered
    assert hold.calls == [1.0, 2.0]
    assert len(harness.instance("after").calls) == 1  # pulse 1 cut at the boundary
    counters = run.counters["a"]
    assert (counters.processed, counters.cancelled, counters.delivered) == (1, 1, 1)


def test_groups_are_delivered_at_their_own_ready_boundary() -> None:
    harness = Harness(
        active(
            [source("a")],
            [
                step(Hold, "hold", value="$sources.a.value"),
                step(Notice, "notice"),
            ],
            [
                group("direct", "$sources.a.value", value="$sources.a.value"),
                group("held", "$sources.a.value", value="$steps.hold.value"),
            ],
        ),
        {"a": [emit(value=1.0)]},
    )
    hold = harness.session.instances[("hold",)]
    order: List[str] = []

    def on_direct(result: GroupResult) -> None:
        order.append(("direct", hold.calls == []))
        hold.gates["release"].set()  # the later action may only run now

    def on_held(result: GroupResult) -> None:
        order.append(("held", hold.calls == [1.0]))

    run = harness.start({"direct": on_direct, "held": on_held})

    assert run.wait(WAIT)
    assert order == [("direct", True), ("held", True)]
    assert len(harness.instance("notice").calls) == 1
    assert run.counters["a"].delivered == 2


class Raising(Events):
    """Observer raising ``RuntimeError`` in one chosen callback."""

    def __init__(self, failing: str) -> None:
        super().__init__()
        self.failing = failing

    def _maybe_raise(self, name: str) -> None:
        if name == self.failing:
            raise RuntimeError(f"observer {name} failed")

    def on_run_started(self, *, session_id, run_id):
        self._maybe_raise("on_run_started")

    def on_source_opened(self, *, source):
        self._maybe_raise("on_source_opened")

    def on_pulse_started(self, *, run_id, source, pulse):
        self._maybe_raise("on_pulse_started")

    def on_pulse_finished(self, *, run_id, source, pulse, error):
        self._maybe_raise("on_pulse_finished")

    def on_group_delivered(self, *, run_id, group, source, pulse):
        self._maybe_raise("on_group_delivered")

    def on_source_closed(self, *, source, error):
        super().on_source_closed(source=source, error=error)
        self._maybe_raise("on_source_closed")

    def on_run_finished(self, *, run_id, result, error):
        self._maybe_raise("on_run_finished")


@pytest.mark.parametrize(
    "failing",
    [
        "on_source_opened",
        "on_pulse_started",
        "on_pulse_finished",
        "on_group_delivered",
        "on_source_closed",
        "on_run_finished",
    ],
)
def test_observer_exceptions_fail_the_run_and_never_strand_it(failing: str) -> None:
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        {"a": [emit(value=1.0)]},
        observer=Raising(failing),
    )

    run = harness.start({"A": Collector().handler("A")})

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert caught.value.stage == "observer"
    if failing == "on_group_delivered":
        assert (caught.value.source, caught.value.pulse, caught.value.group) == (
            "a",
            0,
            "A",
        )
    assert f"observer {failing} failed" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert run.done and run.state == "failed"
    assert harness.log.count("close", "a") == 1


def test_observer_failure_at_run_start_raises_before_any_source_opens() -> None:
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        {"a": [emit(value=1.0)]},
        observer=Raising("on_run_started"),
    )

    with pytest.raises(ActiveRunError) as caught:
        harness.start({"A": Collector().handler("A")})

    assert caught.value.stage == "start"
    assert list(harness.log) == []
    # The session is free for another start.
    harness.session.observer = Events()
    run = harness.start({"A": Collector().handler("A")})
    assert run.wait(WAIT)


def test_completed_runs_do_not_keep_their_session_alive() -> None:
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        {"a": [emit(value=1.0)]},
    )
    run = harness.start({"A": Collector().handler("A")})
    assert run.wait(WAIT)
    session_ref = weakref.ref(harness.session)

    del harness, run
    gc.collect()

    assert session_ref() is None


def test_co_emitted_ports_sharing_an_axis_must_agree_on_their_groups() -> None:
    sparse = [(0,), (2,)]
    feeds = {
        "t": [
            Emission(
                {
                    "pair": Batch.of([1.0, 3.0], indices=sparse),
                    "tag": Batch.of(["x", "z"], indices=sparse),
                }
            ),
            Emission({"pair": Batch.of([5.0])}),  # tag omitted: no correspondence check
            Emission(
                {
                    "pair": Batch.of([1.0, 2.0]),
                    "tag": Batch.of(["x", "z"], indices=sparse),
                }
            ),
        ]
    }
    harness = Harness(
        active(
            [{"type": Tagged.type, "name": "t", "feed": "t"}],
            [],
            [
                group(
                    "T", "$sources.t.pair", pair="$sources.t.pair", tag="$sources.t.tag"
                )
            ],
        ),
        feeds,
    )
    collected = Collector()

    run = harness.start(collected.handlers("T"))

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert caught.value.stage == "emission" and caught.value.pulse == 2
    assert "share axes" in str(caught.value) and "groups at depth 0 differ" in str(
        caught.value
    )
    first, second = collected.results["T"]
    assert first.outputs.data["pair"].indices == ((0,), (2,))
    assert first.outputs.data["tag"].indices == ((0,), (2,))
    assert first.rows() == [
        {"pair": 1.0, "tag": "x"},
        {"pair": None, "tag": None},
        {"pair": 3.0, "tag": "z"},
    ]
    assert second.statuses == {"pair": "complete", "tag": "filtered"}


# Session stop, synchronous lifecycle, launch failures ------------------------


def test_session_stop_from_a_handler_that_runs_before_start_returns(
    monkeypatch,
) -> None:
    feeds = {"a": [emit(value=float(index)) for index in range(6)]}
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        feeds,
    )
    harness.started.set()  # no startup barrier: the source opens right away
    session = harness.session
    collected = Collector()
    handled = threading.Event()
    seen_before_return: List[bool] = []

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        session.stop()
        session.stop()
        handled.set()

    real_launch = ActiveRun._launch

    def launch_then_wait_for_the_handler(run: ActiveRun) -> None:
        real_launch(run)
        seen_before_return.append(handled.wait(WAIT))

    monkeypatch.setattr(ActiveRun, "_launch", launch_then_wait_for_the_handler)
    run = session.start(handlers={"A": on_a})

    assert seen_before_return == [True]  # the handler completed before start returned
    assert run.wait(WAIT)
    counters = run.counters["a"]
    assert counters.admitted == counters.processed >= 1
    assert counters.delivered == counters.processed
    assert counters.read == counters.admitted + counters.unadmitted
    assert harness.log.count("close", "a") == 1
    assert collected.sequences("A") == list(range(counters.processed))
    session.stop()  # no unfinished run: a no-op
    assert run.state == "finished"


def test_session_stop_without_a_run_is_a_no_op_and_targets_the_current_run() -> None:
    hold = threading.Event()
    feeds = {"a": [emit(value=1.0), hold, emit(value=2.0)]}
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        feeds,
    )
    session = harness.session
    collected = Collector()

    session.stop()  # before any start: nothing to stop, and no effect on the next start
    hold.set()  # the first run passes the gate without stopping
    first = harness.start(collected.handlers("A"))
    assert first.wait(WAIT)
    assert collected.rows("A") == [{"v": 1.0}, {"v": 2.0}]

    hold.clear()
    delivered = threading.Event()

    def on_a(result: GroupResult) -> None:
        collected.handler("A")(result)
        delivered.set()

    second = harness.start({"A": on_a})
    assert delivered.wait(WAIT)
    first.stop()  # an old handle cannot stop the current run
    assert second.state == "running"
    session.stop()
    assert second.state == "stopping"
    hold.set()
    assert second.wait(WAIT)
    assert second.counters["a"].unadmitted == 1


def test_session_stop_rejects_passive_sessions() -> None:
    passive = compile_workflow(
        {
            "version": "2.0",
            "inputs": [parameter("x", 1.0)],
            "steps": [step(Scale, "scale", value="$inputs.x")],
            "outputs": [
                {"type": "JsonField", "name": "y", "selector": "$steps.scale.scaled"}
            ],
        },
        catalogue=CATALOGUE,
    ).create_session()

    with pytest.raises(WorkflowInputError, match="no sources"):
        passive.stop()
    with pytest.raises(WorkflowInputError, match="no sources"):
        stop_session(passive)


class WrappedAsyncOpen(Scripted):
    """Synchronous signature returning a coroutine: only visible at call time."""

    type = "test/wrapped_async_open@v1"

    async def _open(self) -> None:
        pass

    def open(self, *, feed, scale, fail_open, fail_close):
        self.feed = feed
        return self._open()


class WrappedAsyncRead(Scripted):
    type = "test/wrapped_async_read@v1"

    async def _read(self) -> None:
        pass

    def read(self):
        return self._read()


class WrappedAsyncClose(Scripted):
    type = "test/wrapped_async_close@v1"

    async def _close(self) -> None:
        self.log.append(("close-body", self.feed))

    def close(self):
        return self._close()


ASYNC_CATALOGUE = Catalogue(
    [Scale],
    sources=[
        Scripted,
        WrappedAsyncOpen,
        WrappedAsyncRead,
        WrappedAsyncClose,
    ],
)


def async_harness(source_class: type, feed: list) -> Harness:
    definition = active(
        [{"type": source_class.type, "name": "a", "feed": "a"}],
        [step(Scale, "double", value="$sources.a.value")],
        [group("A", "$sources.a.value", v="$steps.double.scaled")],
    )
    monkey = Harness.__new__(Harness)
    monkey.log = Log()
    monkey.started = threading.Event()
    monkey.started.set()
    monkey.plan = compile_workflow(definition, catalogue=ASYNC_CATALOGUE)
    monkey.session = monkey.plan.create_session(
        resources={
            "feeds": {"a": feed},
            "log": monkey.log,
            "started": monkey.started,
            "gates": {},
        }
    )
    monkey.run = None

    return monkey


@pytest.mark.parametrize("method", ["open", "read", "close"])
def test_coroutine_lifecycle_methods_are_rejected_before_any_source_opens(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = async_harness(Scripted, [emit(value=1.0)])

    async def substituted_method(*args, **kwargs):
        return None

    # Declaration checks already passed; exercise the runtime's defensive guard.
    with monkeypatch.context() as patch:
        patch.setattr(Scripted, method, substituted_method)
        with pytest.raises(ActiveRunError) as caught:
            harness.session.start(handlers={"A": Collector().handler("A")})

    assert (caught.value.stage, caught.value.source) == ("start", "a")
    assert f"{method}()" in str(caught.value) and "synchronous" in str(caught.value)
    assert list(harness.log) == []
    # The session is not left registered: a later start of a healthy plan works.
    healthy = Harness(
        doubling([source("b")], [group("B", "$sources.b.value", v="$sources.b.value")]),
        {"b": [emit(value=1.0)]},
    )
    assert healthy.start(Collector().handlers("B")).wait(WAIT)


@pytest.mark.parametrize(
    "source_class, stage, opened, closed",
    [
        (WrappedAsyncOpen, "open", False, True),
        (WrappedAsyncRead, "read", True, True),
        (WrappedAsyncClose, "close", True, False),
    ],
)
def test_lifecycle_calls_returning_awaitables_fail_truthfully(
    source_class: type, stage: str, opened: bool, closed: bool
) -> None:
    # After a failed open or read the ordinary synchronous close still runs
    # and is reported; a close returning an awaitable is never reported done.
    harness = async_harness(source_class, [emit(value=1.0)])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        run = harness.session.start(handlers={"A": Collector().handler("A")})
        with pytest.raises(ActiveRunError) as caught:
            run.wait(WAIT)
        gc.collect()

    assert (caught.value.stage, caught.value.source) == (stage, "a")
    assert "awaitable" in str(caught.value)
    counters = run.counters["a"]
    assert counters.opened is opened
    assert counters.closed is closed
    assert harness.log.count("close", "a") == (1 if closed else 0)
    assert harness.log.count("close-body", "a") == 0
    assert not [w for w in caught_warnings if issubclass(w.category, RuntimeWarning)]


def test_callable_instance_with_async_call_is_rejected_as_a_handler() -> None:
    class AsyncCallable:
        async def __call__(self, result: GroupResult) -> None:
            pass

    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        {"a": [emit(value=1.0)]},
    )

    with pytest.raises(ContractError, match="coroutine function"):
        harness.start({"A": AsyncCallable()})
    assert list(harness.log) == []


def _no_runtime_threads_remain() -> bool:
    for thread in threading.enumerate():
        if thread.name.startswith("workflows-v2-"):
            thread.join(WAIT)
            if thread.is_alive():
                return False

    return True


def test_processor_start_failure_releases_the_session_without_opening_sources(
    monkeypatch,
) -> None:
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value", v="$sources.a.value")]),
        {"a": [emit(value=1.0)]},
    )
    real_start = threading.Thread.start

    def refuse_processor(thread: threading.Thread) -> None:
        if thread.name.startswith("workflows-v2-run-"):
            raise RuntimeError("simulated thread creation refusal")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_processor)
    with pytest.raises(ActiveRunError) as caught:
        harness.start(Collector().handlers("A"))
    monkeypatch.undo()

    assert caught.value.stage == "start"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert list(harness.log) == []  # no source was opened or closed
    assert _no_runtime_threads_remain()
    collected = Collector()
    assert harness.start(collected.handlers("A")).wait(WAIT)  # registry was released
    assert collected.rows("A") == [{"v": 1.0}]


def test_reader_start_failure_stops_started_readers_and_releases_the_session(
    monkeypatch,
) -> None:
    feeds = {"a": [UNTIL_STOP, emit(value=1.0)], "b": [emit(value=2.0)]}
    harness = Harness(
        doubling(
            [source("a"), source("b")],
            [
                group("A", "$sources.a.value", v="$sources.a.value"),
                group("B", "$sources.b.value", v="$sources.b.value"),
            ],
        ),
        feeds,
    )
    harness.started.set()
    real_start = threading.Thread.start

    def refuse_second_reader(thread: threading.Thread) -> None:
        if thread.name == "workflows-v2-source-b":
            # Source a is already open and blocked in read when b cannot start.
            harness.log.wait_for(lambda: harness.log.count("wait", "a") == 1)
            raise RuntimeError("simulated thread creation refusal")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_second_reader)
    collected = Collector()
    with pytest.raises(ActiveRunError) as caught:
        harness.session.start(handlers=collected.handlers("A", "B"))
    monkeypatch.undo()

    assert (caught.value.stage, caught.value.source) == ("start", "b")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.suppressed == ()
    assert harness.log.count("open", "a") == harness.log.count("close", "a") == 1
    assert harness.log.count("open", "b") == harness.log.count("close", "b") == 0
    assert collected.results == {}
    assert _no_runtime_threads_remain()
    feeds["a"] = [emit(value=1.0)]  # the retry gets a feed that ends on its own
    again = harness.start(collected.handlers("A", "B"))  # registry was released
    assert again.wait(WAIT)
    assert collected.rows("A") == [{"v": 1.0}] and collected.rows("B") == [{"v": 2.0}]


def test_source_locations_do_not_collide_with_nested_step_identities() -> None:
    from roboflow_workflows.execution_engine.v2.context import ExecutionContext
    from roboflow_workflows.execution_engine.v2.errors import format_step_path
    from roboflow_workflows.execution_engine.v2.introspection import describe_workflow

    child = {
        "version": "2.0",
        "inputs": [parameter("x", 1.0)],
        "steps": [step(Scale, "s", value="$inputs.x")],
        "outputs": [],
    }
    definition = active(
        [source("s")],
        [
            {
                "type": "inner_workflow",
                "name": "sources",
                "workflow_definition": child,
                "parameter_bindings": {"x": 1.0},
            }
        ],
        [],
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    description = describe_workflow(plan)

    assert description["sources"]["s"]["node_id"] == "$sources.s"
    assert description["steps"][0]["node_id"] == "$steps.sources/s"
    assert format_step_path(plan.source("s").step_path) == "$sources.s"
    context = ExecutionContext(("sources", "s"), Scale.type, "session")
    assert context.step_selector == "$steps.sources/s"
