"""Pipelined active runs: overload, stop, cancel, failure, self-wait and counters.

Every source counter ends with ``read == admitted + dropped + unadmitted`` and
``admitted == processed + cancelled``; each test asserts it on its run.
Probes are ordered by events, never by sleeping (see ``test_pipelined``).
"""

import threading
from contextlib import contextmanager
from typing import Any, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError, ContractError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    PipelinedCoordination,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    CATALOGUE,
    PIPELINE,
    Held,
    Probe,
    Session,
)
from tests.unit_tests.execution_engine.v2.execution.blocks import Failing, Scale
from tests.unit_tests.execution_engine.v2.pipelining.test_workers import (
    live_threads,
    refuse_second_worker,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    UNTIL_STOP,
    WAIT,
    Collector,
    _no_runtime_threads_remain,
    active,
    emit,
    group,
    source,
    step,
)

WINDOWED = Catalogue.merge(CATALOGUE, Catalogue([], operators=[Window]))


def assert_balanced(run: Any) -> None:
    """The documented source counter invariants of a finished run."""
    for name, counters in run.counters.items():
        assert counters.read == (
            counters.admitted + counters.dropped + counters.unadmitted
        ), name
        assert counters.admitted == counters.processed + counters.cancelled, name


class Notes(ExecutionObserver):
    """Writes lifecycle callbacks into the probe's event list, in order."""

    def __init__(self) -> None:
        self.probe: Optional[Probe] = None
        self.failed = threading.Event()

    def note(self, *event: Any) -> None:
        with self.probe._lock:
            self.probe.events.append(event)

    def on_pulse_finished(self, *, run_id, source, pulse, error):
        if error is not None:
            self.failed.set()

    def on_operator_finished(self, *, operator, error):
        self.note("operator_finished", operator)

    def on_run_finished(self, *, run_id, result, error):
        self.note("run_finished")


def noted_session(definition: dict, feeds: dict, **options: Any) -> Session:
    notes = Notes()
    session = Session(definition, feeds, observer=notes, **options)
    notes.probe = session.probe

    return session


def slow_definition() -> dict:
    definition = active(
        [source("a")],
        [step(Held, "slow", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.slow.value")],
    )

    return definition


# Overload ---------------------------------------------------------------------------


def test_latest_admits_only_the_newest_pending_value_while_every_worker_is_busy() -> (
    None
):
    session = Session(slow_definition(), {"a": []})
    entered = session.probe.entered("slow", 1.0)
    # The reader reads values 2..10 only once pulse 0 holds the only worker.
    session.feeds["a"].extend(
        [emit(value=1.0), entered] + [emit(value=float(v)) for v in range(2, 11)]
    )
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(
        collected.handlers("A"),
        admission_bound=1,
        pipeline=PipelineOptions(max_in_flight=1, overload="latest"),
    )

    assert entered.wait(WAIT)
    session.log.wait_for(lambda: ("read", "a", None) in session.log)
    counters = run.counters["a"]
    assert (counters.read, counters.admitted, counters.dropped) == (10, 1, 8)
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("A") == [{"value": 1.0}, {"value": 10.0}]
    assert collected.sequences("A") == [0, 1]
    assert (counters.admitted, counters.dropped, counters.unadmitted) == (2, 8, 0)
    assert counters.peak_admitted == 1
    assert_balanced(run)


def test_stop_discards_the_pending_latest_value_and_drains_the_admitted_one() -> None:
    session = Session(slow_definition(), {"a": []})
    entered = session.probe.entered("slow", 1.0)
    session.feeds["a"].extend(
        [emit(value=1.0), entered, emit(value=2.0), emit(value=3.0), UNTIL_STOP]
        + [emit(value=4.0)]
    )
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(
        collected.handlers("A"),
        admission_bound=1,
        pipeline=PipelineOptions(max_in_flight=2, source_overload={"a": "latest"}),
    )

    assert entered.wait(WAIT)
    session.log.wait_for(lambda: session.log.count("wait", "a") == 2)
    run.stop()
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("A") == [{"value": 1.0}]
    counters = run.counters["a"]
    assert (counters.read, counters.admitted, counters.dropped) == (4, 1, 1)
    assert counters.unadmitted == 2  # the pending 3.0 and the 4.0 read after stop
    assert run.state == "finished"
    assert_balanced(run)


def test_block_overload_is_lossless_and_bounded_by_admission_and_workers() -> None:
    values = [float(value) for value in range(8)]
    session = Session(slow_definition(), {"a": [emit(value=v) for v in values]})
    release = session.probe.hold("slow", 0.0)
    reads_while_held: List[int] = []
    collected = Collector()

    run = session.start(
        collected.handlers("A"),
        admission_bound=2,
        pipeline=PipelineOptions(max_in_flight=4),
    )

    assert session.probe.entered("slow", 0.0).wait(WAIT)
    reads_while_held.append(session.log.count("read", "a"))
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("A") == [{"value": value} for value in values]
    # Two admitted plus one read waiting for a slot, whatever the timing.
    assert reads_while_held[0] <= 3
    counters = run.counters["a"]
    assert (counters.read, counters.admitted, counters.processed) == (8, 8, 8)
    assert counters.dropped == counters.unadmitted == 0
    assert counters.peak_admitted <= 2
    gauges = run.pipeline_counters
    assert gauges.peak("executing") <= 4
    assert gauges.peak("queued") <= 2
    assert gauges.peak("live_states") <= 4
    assert gauges.snapshot()["totals"]["end_tasks"] == 1
    assert set(gauges.snapshot()["result_age_ns"]) == {"A"}
    assert_balanced(run)


# Stop, cancel and failure ----------------------------------------------------------


def windowed_definition() -> dict:
    definition = slow_definition()
    definition["outputs"].append(group("W", "$operators.clip.x", x="$operators.clip.x"))
    definition["operators"] = [
        {
            "type": Window.type,
            "name": "clip",
            "size": 3,
            "partial": "emit",
            "collect": {"x": "$steps.slow.value"},
        }
    ]

    return definition


@pytest.mark.parametrize("pipeline", [None, PIPELINE], ids=["serial", "pipelined"])
def test_stop_drains_admitted_pulses_then_finishes_operators_with_stop(
    pipeline,
) -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0), UNTIL_STOP, emit(value=3.0)]}
    session = Session(windowed_definition(), feeds, catalogue=WINDOWED)
    collected = Collector()

    def on_a(result) -> None:
        collected.handler("A")(result)
        if len(collected.results["A"]) == 2:
            session.session.stop()

    run = session.start({"A": on_a, "W": collected.handler("W")}, pipeline=pipeline)

    assert run.wait(WAIT)
    assert collected.rows("A") == [{"value": 1.0}, {"value": 2.0}]
    assert collected.rows("W") == [{"x": [1.0, 2.0]}]
    assert run.operator_counters["clip"].finished
    assert run.counters["a"].unadmitted == 1
    assert run.state == "finished"
    assert_balanced(run)


def test_cancel_lets_the_running_call_return_before_closing_operators() -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0), emit(value=3.0), UNTIL_STOP]}
    session = noted_session(windowed_definition(), feeds, catalogue=WINDOWED)
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A", "W"), admission_bound=3)

    assert session.probe.entered("slow", 1.0).wait(WAIT)
    session.log.wait_for(lambda: session.log.count("wait", "a") == 1)
    run.cancel()
    assert run.state == "stopping" and not run.done
    assert ("operator_finished", "clip") not in session.probe.events
    release.set()
    assert run.wait(WAIT) is True
    assert run.state == "cancelled" and run.failure is None
    events = session.probe.events
    assert events.index(("exit", "slow", 1.0)) < events.index(
        ("operator_finished", "clip")
    )
    assert events[-1] == ("run_finished",)
    assert collected.results == {}
    counters = run.counters["a"]
    assert (counters.admitted, counters.processed, counters.cancelled) == (3, 0, 3)
    clip = run.operator_counters["clip"]
    assert (clip.finished, clip.closed) == (False, True)
    assert session.probe.order["slow"] == [1.0]
    assert_balanced(run)
    assert _no_runtime_threads_remain()


def test_cancel_of_a_serial_run_stops_at_the_next_pulse() -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0), UNTIL_STOP]}
    session = Session(slow_definition(), feeds)
    release = session.probe.hold("slow", 1.0)

    run = session.start(Collector().handlers("A"), pipeline=None)

    assert session.probe.entered("slow", 1.0).wait(WAIT)
    run.cancel()
    release.set()
    assert run.wait(WAIT) is True
    assert run.state == "cancelled"
    assert run.counters["a"].processed == 0
    assert_balanced(run)


def test_a_failure_waits_for_running_calls_and_cancels_the_rest() -> None:
    definition = active(
        [source("a"), source("b")],
        [
            step(Held, "slow", value="$sources.a.value"),
            step(Failing, "check", value="$sources.b.value"),
        ],
        [
            group("A", "$sources.a.value", value="$steps.slow.value"),
            group("B", "$sources.b.value", value="$steps.check.value"),
        ],
    )
    session = noted_session(
        definition,
        {
            "a": [emit(value=1.0), emit(value=2.0), UNTIL_STOP],
            "b": [emit(value=-1.0), emit(value=5.0)],
        },
    )
    notes = session.session.observer
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(collected.handlers("A", "B"), admission_bound=2)

    assert session.probe.entered("slow", 1.0).wait(WAIT)
    assert notes.failed.wait(WAIT)
    assert not run.done  # the held call cannot be interrupted
    release.set()
    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert (caught.value.stage, caught.value.source, caught.value.pulse) == (
        "step",
        "b",
        0,
    )
    assert caught.value.step_path == ("check",)
    assert run.state == "failed"
    assert collected.results.get("B") is None
    assert session.probe.events[-1] == ("run_finished",)
    assert_balanced(run)
    assert _no_runtime_threads_remain()


def test_a_handler_failure_aborts_pulses_waiting_for_their_delivery_turn() -> None:
    session = Session(
        slow_definition(), {"a": [emit(value=float(value)) for value in range(4)]}
    )

    def failing(result) -> None:
        raise RuntimeError("cannot deliver")

    run = session.start({"A": failing}, admission_bound=4)

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert (caught.value.stage, caught.value.group, caught.value.pulse) == (
        "handler",
        "A",
        0,
    )
    assert run.counters["a"].delivered == 0
    assert_balanced(run)


# Self-wait, options and restart ----------------------------------------------------


class Waiting(Block):
    """Calls ``wait()`` of the run in ``waits`` from inside a block call."""

    type = "test/pipelined_waiting@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, waits: "SelfWaits") -> None:
        self.waits = waits

    def run(self, *, value) -> dict:
        self.waits.attempt("block")
        return {"value": value}


class SelfWaits(ExecutionObserver):
    """Tries ``run.wait()`` from every kind of thread a run owns."""

    def __init__(self, operation: str = "wait") -> None:
        self.operation = operation
        self.run: Any = None
        self.ready = threading.Event()
        self.outcomes: dict = {}

    def attempt(self, where: str) -> None:
        assert self.ready.wait(WAIT)
        try:
            if self.operation == "wait":
                self.run.wait(WAIT)
            else:
                with self.run:
                    if self.operation == "exceptional_exit":
                        raise ValueError("body failed")
            self.outcomes.setdefault(where, "returned")
        except ContractError as error:
            assert "would wait for itself" in str(error)
            self.outcomes.setdefault(where, "rejected")

    def on_source_opened(self, *, source):
        self.attempt("reader")

    def on_step_started(self, *, step, block_type):
        self.attempt("observer")

    def on_run_finished(self, *, run_id, result, error):
        self.attempt("finisher")


@pytest.mark.parametrize("pipeline", [None, PIPELINE], ids=["serial", "pipelined"])
@pytest.mark.parametrize("operation", ["wait", "context_exit", "exceptional_exit"])
def test_waiting_from_any_thread_of_the_run_is_rejected(pipeline, operation) -> None:
    waits = SelfWaits(operation)
    definition = active(
        [source("a")],
        [step(Waiting, "wait", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.wait.value")],
    )
    session = Session(
        definition,
        {"a": [emit(value=1.0)]},
        observer=waits,
        catalogue=Catalogue.merge(CATALOGUE, Catalogue([Waiting])),
        resources={"waits": waits},
    )

    run = session.start(
        {"A": lambda result: waits.attempt("handler")}, pipeline=pipeline
    )
    waits.run = run
    waits.ready.set()

    assert run.wait(WAIT)
    assert waits.outcomes == dict.fromkeys(
        ["reader", "observer", "block", "handler", "finisher"], "rejected"
    )


def test_pipeline_options_are_checked_before_any_source_opens() -> None:
    session = Session(slow_definition(), {"a": [emit(value=1.0)]})

    with pytest.raises(ContractError, match="unknown source"):
        session.start(
            Collector().handlers("A"),
            pipeline=PipelineOptions(source_overload={"camera": "latest"}),
        )
    with pytest.raises(ContractError, match="PipelineOptions"):
        session.start(Collector().handlers("A"), pipeline={"max_in_flight": 2})

    assert list(session.log) == []
    collected = Collector()
    assert session.start(collected.handlers("A")).wait(WAIT)
    assert collected.rows("A") == [{"value": 1.0}]


def test_a_restarted_pipelined_run_reuses_block_state_and_leaves_no_threads() -> None:
    feeds = {"a": [emit(value=1.0), emit(value=2.0)]}
    definition = active(
        [source("a")],
        [step(Scale, "double", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.double.scaled")],
    )
    session = Session(definition, feeds)

    for _ in range(2):
        feeds["a"] = [emit(value=1.0), emit(value=2.0)]
        collected = Collector()
        assert session.start(collected.handlers("A")).wait(WAIT)
        assert collected.rows("A") == [{"value": 2.0}, {"value": 4.0}]

    assert len(session.session.instances[("double",)].calls) == 4
    assert _no_runtime_threads_remain()


@pytest.mark.parametrize("pipeline", [None, PIPELINE], ids=["serial", "pipelined"])
def test_external_exceptional_context_exit_drains_and_preserves_body_error(
    pipeline,
) -> None:
    session = Session(slow_definition(), {"a": [emit(value=1.0), UNTIL_STOP]})
    collected = Collector()
    run = session.start(collected.handlers("A"), pipeline=pipeline)
    assert session.probe.entered("slow", 1.0).wait(WAIT)

    with pytest.raises(ValueError, match="body failed"):
        with run:
            raise ValueError("body failed")

    assert run.done
    assert collected.rows("A") == [{"value": 1.0}]
    assert_balanced(run)
    assert _no_runtime_threads_remain()


def test_a_failed_worker_launch_fails_start_and_releases_the_session(
    monkeypatch,
) -> None:
    definition = active(
        [source("a")],
        [step(Scale, "double", value="$sources.a.value")],
        [group("A", "$sources.a.value", v="$steps.double.scaled")],
    )
    session = Session(definition, {"a": [emit(value=1.0)]})
    refuse_second_worker(monkeypatch, "workflows-v2-worker-")

    with pytest.raises(ActiveRunError) as caught:
        session.start(Collector().handlers("A"))
    monkeypatch.undo()

    assert caught.value.stage == "start"
    assert str(caught.value.__cause__) == "injected second-worker launch failure"
    assert live_threads("workflows-v2-worker-") == []
    assert list(session.log) == []  # no source was opened
    collected = Collector()
    assert session.start(collected.handlers("A")).wait(WAIT)
    assert collected.rows("A") == [{"v": 2.0}]


@pytest.mark.parametrize("abort", ["cancel", "source_failure"])
def test_abort_prevents_delivery_waiting_for_callback_lock(monkeypatch, abort) -> None:
    reached_lock = threading.Event()
    fail_source = threading.Event()
    held_locks = []
    delivered = []
    initialize = PipelinedCoordination.__init__
    callbacks = PipelinedCoordination.callbacks

    def hold_callbacks(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._callbacks.acquire()
        held_locks.append(self._callbacks)

    @contextmanager
    def observe_lock_wait(self):
        reached_lock.set()
        with callbacks(self):
            yield

    monkeypatch.setattr(PipelinedCoordination, "__init__", hold_callbacks)
    monkeypatch.setattr(PipelinedCoordination, "callbacks", observe_lock_wait)
    feed = [emit(value=1.0)]
    if abort == "source_failure":
        feed.extend([fail_source, RuntimeError("source failed")])
    session = Session(
        active(
            [source("a")],
            [],
            [group("A", "$sources.a.value", value="$sources.a.value")],
        ),
        {"a": feed},
    )
    run = None
    try:
        run = session.start(
            {"A": lambda result: delivered.append(result.pulse.sequence)}
        )
        assert reached_lock.wait(WAIT)
        if abort == "cancel":
            run.cancel()
        else:
            fail_source.set()
            assert run._coordination._aborted.wait(WAIT)
        held_locks.pop().release()
        if abort == "cancel":
            assert run.wait(WAIT)
            assert run.state == "cancelled"
        else:
            with pytest.raises(ActiveRunError, match="source failed"):
                run.wait(WAIT)
        assert delivered == []
        assert run.counters["a"].cancelled == 1
        assert_balanced(run)
    finally:
        fail_source.set()
        for lock in held_locks:
            lock.release()
        if run is not None:
            run.cancel()
            try:
                assert run.wait(WAIT)
            except ActiveRunError:
                pass
    assert _no_runtime_threads_remain()
