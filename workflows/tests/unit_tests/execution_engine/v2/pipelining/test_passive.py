"""Passive pipeline: bounded submissions, overlap, order, failure, cancel, exclusion.

Every wait is bounded by ``WAIT`` and ordered by events the test controls: a
block call records that it was reached and, when the test holds its key,
waits until the test releases it. No assertion depends on thread timing.
"""

import threading
from collections import defaultdict
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import current_pulse_run_id
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions, passive
from roboflow_workflows.execution_engine.v2.pipelining.passive import (
    PassivePipeline,
    PipelineAbortedError,
    PipelineFullError,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompileOptions,
    ExecutionObserver,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
)

from tests.unit_tests.execution_engine.v2.pipelining.test_workers import (
    live_threads,
    refuse_second_worker,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""


class Probe:
    """Shared by the blocks of one session: records visits, holds keys.

    ``visit(key)`` appends the key, signals that it was reached and, when the
    test called ``hold(key)``, waits for ``release(key)``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.visits: List[str] = []
        self.run_ids: Dict[str, Optional[str]] = {}
        self._reached: Dict[str, threading.Event] = defaultdict(threading.Event)
        self._held: Dict[str, threading.Event] = {}

    def hold(self, *keys: str) -> None:
        for key in keys:
            self._held[key] = threading.Event()

    def release(self, *keys: str) -> None:
        for key in keys:
            self._held[key].set()

    def reached(self, key: str) -> bool:
        with self._lock:
            event = self._reached[key]

        return event.wait(WAIT)

    def visit(self, key: str) -> None:
        with self._lock:
            self.visits.append(key)
            self.run_ids[key] = current_pulse_run_id()
            event = self._reached[key]
            held = self._held.get(key)
        event.set()
        if held is not None:
            assert held.wait(WAIT), f"{key} was never released"


class Stage(Block):
    """Run-mode step: visits ``<name>:<value>``; fails when value == fail_on."""

    type = "test/pipe_stage@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        label: str
        fail_on: float = -1.0

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe
        self.order: List[float] = []

    def run(self, *, value: float, label: str, fail_on: float) -> dict:
        self.probe.visit(f"{label}:{value:g}")
        self.order.append(value)
        if value == fail_on:
            raise RuntimeError(f"{label} rejects {value:g}")

        return {"value": value}


class TwoPhases(Block):
    """Phased step ``first`` -> ``second``; each phase visits ``<phase>:<value>``."""

    type = "test/pipe_two_phases@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    @phase
    def first(self, *, value):
        self.probe.visit(f"first:{value:g}")
        return value

    @phase
    def second(self, *, first):
        self.probe.visit(f"second:{first:g}")
        return {"value": first * 10}

    def run(self, *, value):
        return self.second(first=self.first(value=value))


class Ticks(Source):
    type = "test/pipe_ticks@v1"
    outputs = {"tick": SourceOutput(FLOAT_KIND)}

    def open(self) -> None:
        pass

    def read(self) -> Optional[Emission]:
        return None


CATALOGUE = Catalogue([Stage, TwoPhases], sources=[Ticks], namespace="test")


def _definition(*, fail_on: float = -1.0) -> dict:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": [
            {
                "type": Stage.type,
                "name": "load",
                "value": "$inputs.value",
                "label": "load",
                "fail_on": fail_on,
            },
            {"type": TwoPhases.type, "name": "model", "value": "$steps.load.value"},
            {
                "type": Stage.type,
                "name": "save",
                "value": "$steps.model.value",
                "label": "save",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.save.value"}
        ],
    }


def _session(probe: Probe, *, observer=None, fail_on: float = -1.0):
    plan = compile_workflow(
        _definition(fail_on=fail_on),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution="phases"),
    )
    session = plan.create_session({"probe": probe}, observer=observer)

    return session


def _out(future: Future) -> float:
    result = future.result(timeout=WAIT)
    rows = result.rows()

    return rows[0]["out"]


def test_pipeline_results_and_stateful_order_equal_serial_runs() -> None:
    serial_probe, piped_probe = Probe(), Probe()
    serial = _session(serial_probe)
    expected = [
        serial.run({"value": float(value)}).rows()[0]["out"] for value in range(6)
    ]

    session = _session(piped_probe)
    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [pipeline.submit({"value": float(value)}) for value in range(6)]

    assert [_out(future) for future in futures] == expected
    for step in ("load", "save"):
        assert session.instances[(step,)].order == [
            float(value) * (10 if step == "save" else 1) for value in range(6)
        ]
    assert sorted(piped_probe.visits) == sorted(serial_probe.visits)
    assert pipeline.outcomes == {
        "submitted": 6,
        "completed": 6,
        "failed": 0,
        "aborted": 0,
        "full": 0,
    }


def test_next_submission_runs_phase_first_while_the_previous_is_in_second() -> None:
    probe = Probe()
    probe.hold("second:0")
    session = _session(probe)

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        first = pipeline.submit({"value": 0.0})
        assert probe.reached("second:0")
        second = pipeline.submit({"value": 1.0})
        # Run 1 enters phase "first" while run 0 is still inside "second".
        assert probe.reached("first:1")
        assert "second:1" not in probe.visits
        probe.release("second:0")

        assert _out(first) == 0.0 and _out(second) == 10.0

    assert pipeline.counters.peak("overlapping_pulses") >= 2
    stages = pipeline.counters.snapshot()["stages"]
    assert stages["$steps.model#first"]["calls"] == 2
    assert stages["$steps.model#second"]["calls"] == 2


def test_a_later_run_waits_for_its_turn_at_a_stage_an_earlier_run_holds() -> None:
    probe = Probe()
    probe.hold("load:0")
    session = _session(probe)

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        first = pipeline.submit({"value": 0.0})
        assert probe.reached("load:0")
        second = pipeline.submit({"value": 1.0})
        probe.release("load:0")
        results = [_out(first), _out(second)]

    assert results == [0.0, 10.0]
    assert probe.visits.index("load:0") < probe.visits.index("load:1")
    assert probe.visits.index("save:0") < probe.visits.index("save:10")


def test_submit_is_bounded_by_idle_workers_without_a_queue() -> None:
    probe = Probe()
    probe.hold("load:0")
    session = _session(probe)

    with session.pipeline(options=PipelineOptions(max_in_flight=1)) as pipeline:
        first = pipeline.submit({"value": 0.0})
        assert probe.reached("load:0")
        with pytest.raises(PipelineFullError, match="1 pipeline workers are busy"):
            pipeline.submit({"value": 1.0}, block=False)
        with pytest.raises(PipelineFullError):
            pipeline.submit({"value": 1.0}, timeout=0.01)
        assert pipeline.counters.current("executing") == 1

        waiting: List[Future] = []
        submitter = threading.Thread(
            target=lambda: waiting.append(pipeline.submit({"value": 2.0}))
        )
        submitter.start()
        probe.release("load:0")
        submitter.join(WAIT)
        assert not submitter.is_alive()

        assert _out(first) == 0.0 and _out(waiting[0]) == 20.0

    assert pipeline.outcomes["full"] == 2
    assert pipeline.outcomes["submitted"] == 2
    assert pipeline.counters.peak("executing") == 1
    assert pipeline.counters.peak("live_states") == 1


def test_invalid_inputs_raise_at_submit_and_consume_no_turn() -> None:
    session = _session(Probe())

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        with pytest.raises(WorkflowInputError):
            pipeline.submit({"unknown": 1.0})
        future = pipeline.submit({"value": 3.0})

        assert _out(future) == 30.0

    assert pipeline.outcomes["submitted"] == 1


def test_a_failure_aborts_waiting_runs_and_closes_submission() -> None:
    probe = Probe()
    probe.hold("load:0", "first:0")
    session = _session(probe, fail_on=1.0)

    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        first = pipeline.submit({"value": 0.0})
        assert probe.reached("load:0")
        failing = pipeline.submit({"value": 1.0})
        third = pipeline.submit({"value": 2.0})
        probe.release("load:0")
        # Run 0 is held inside phase "first"; run 1 fails at "load" while
        # run 2 waits for its turn there.
        assert probe.reached("first:0")

        with pytest.raises(StepExecutionError, match="load rejects 1") as failed:
            failing.result(timeout=WAIT)
        with pytest.raises(
            PipelineAbortedError, match="a submission failed: StepExecutionError"
        ):
            third.result(timeout=WAIT)
        with pytest.raises(
            PipelineAbortedError, match="accepts no submissions: a submission failed"
        ):
            pipeline.submit({"value": 4.0})
        probe.release("first:0")

    # Run 0 was inside a call when the abort came: the call finished, the
    # run then stopped at its next stage.
    with pytest.raises(PipelineAbortedError) as aborted:
        first.result(timeout=WAIT)
    assert aborted.value.failure is failed.value
    assert pipeline.failure is failed.value
    outcomes = pipeline.outcomes
    assert outcomes["submitted"] == 3
    assert outcomes["failed"] == 1 and outcomes["aborted"] == 2
    assert "load:2" not in probe.visits


def test_cancel_stops_waiting_runs_lets_running_calls_finish() -> None:
    probe = Probe()
    probe.hold("load:0")
    session = _session(probe)

    pipeline = session.pipeline(options=PipelineOptions(max_in_flight=2))
    first = pipeline.submit({"value": 0.0})
    assert probe.reached("load:0")
    second = pipeline.submit({"value": 1.0})
    pipeline.cancel()
    with pytest.raises(PipelineAbortedError, match="cancelled"):
        pipeline.submit({"value": 2.0})
    probe.release("load:0")
    pipeline.close()

    for future in (first, second):
        with pytest.raises(PipelineAbortedError, match="was cancelled") as aborted:
            future.result(timeout=WAIT)
        assert aborted.value.failure is None
    assert pipeline.failure is None
    assert probe.visits == ["load:0"]
    assert pipeline.outcomes["aborted"] == 2


def test_body_error_cancels_and_the_session_runs_again_afterwards() -> None:
    probe = Probe()
    session = _session(probe)

    with pytest.raises(KeyError):
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            pipeline.submit({"value": 1.0}).result(timeout=WAIT)
            raise KeyError("caller bug")

    assert pipeline.outcomes["completed"] == 1
    assert session.run({"value": 2.0}).rows()[0]["out"] == 20.0


def test_session_excludes_direct_runs_and_a_second_pipeline_while_open() -> None:
    probe = Probe()
    session = _session(probe)

    with session.pipeline(options=PipelineOptions(max_in_flight=1)):
        with pytest.raises(ContractError):
            session.run({"value": 1.0})
        with pytest.raises(ContractError):
            session.pipeline(options=PipelineOptions(max_in_flight=1))

    probe.hold("load:5")
    direct: List[Any] = []
    runner = threading.Thread(target=lambda: direct.append(session.run({"value": 5.0})))
    runner.start()
    assert probe.reached("load:5")
    with pytest.raises(ContractError):
        session.pipeline(options=PipelineOptions(max_in_flight=1))
    probe.release("load:5")
    runner.join(WAIT)

    assert direct[0].rows()[0]["out"] == 50.0
    with session.pipeline() as pipeline:
        assert _out(pipeline.submit({"value": 6.0})) == 60.0


def test_close_and_submit_from_a_worker_raise_instead_of_waiting() -> None:
    attempts: List[BaseException] = []
    pipelines: List[PassivePipeline] = []

    class SelfWaiting(ExecutionObserver):
        def on_run_started(self, *, session_id: str, run_id: str) -> None:
            for call in (
                pipelines[0].close,
                lambda: pipelines[0].submit({"value": 1.0}),
            ):
                try:
                    call()
                except ContractError as error:
                    attempts.append(error)

    session = _session(Probe(), observer=SelfWaiting())
    with session.pipeline(options=PipelineOptions(max_in_flight=1)) as pipeline:
        pipelines.append(pipeline)
        assert _out(pipeline.submit({"value": 1.0})) == 10.0

    assert [str(error).split(" was called")[0] for error in attempts] == [
        "PassivePipeline.close()",
        "PassivePipeline.submit()",
    ]


def test_block_code_sees_its_run_id_on_the_worker_thread() -> None:
    probe = Probe()
    session = _session(probe)

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        futures = [pipeline.submit({"value": float(value)}) for value in range(2)]
        results = [future.result(timeout=WAIT) for future in futures]

    for value, result in enumerate(results):
        assert probe.run_ids[f"load:{value}"] == result.run_id
        assert probe.run_ids[f"second:{value}"] == result.run_id
    assert current_pulse_run_id() is None


def test_active_plans_and_wrong_options_are_rejected() -> None:
    active = compile_workflow(
        {
            "version": "2.0",
            "sources": [{"type": Ticks.type, "name": "ticks"}],
            "steps": [],
            "outputs": [],
        },
        catalogue=CATALOGUE,
    )
    with pytest.raises(WorkflowInputError, match="pipeline=PipelineOptions"):
        active.create_session().pipeline()

    session = _session(Probe())
    with pytest.raises(ContractError, match="PipelineOptions"):
        session.pipeline(options={"max_in_flight": 2})
    with session.pipeline() as pipeline:
        assert pipeline.options == PipelineOptions()


class NonZero(Block):
    """Control: continue to the targets for a non-zero value."""

    type = "test/pipe_non_zero@v1"

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        return Select(next_steps) if value else Stop()


class Record(Block):
    """Stateful per-invocation step: remembers every value it was called with."""

    type = "test/pipe_record@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self) -> None:
        self.calls: List[float] = []

    def run(self, *, value: float) -> dict:
        self.calls.append(value)
        return {"value": value * 2}


class Triple(Block):
    """Batch-delivering step: one call for every admitted value of a run."""

    type = "test/pipe_triple@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND, batch="always")  # noqa: F821 (batch mode is a value)

    def __init__(self) -> None:
        self.batches: List[List[float]] = []

    def run(self, *, value) -> list:
        self.batches.append(list(value))
        return [{"value": item * 3} for item in value]


NESTED_CATALOGUE = Catalogue([NonZero, Record, Triple], namespace="test")


def _nested_gated_batched_plan():
    child = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [
            {
                "type": NonZero.type,
                "name": "gate",
                "value": "$inputs.values",
                "next_steps": ["$steps.record"],
            },
            {"type": Record.type, "name": "record", "value": "$inputs.values"},
        ],
        "outputs": [
            {"type": "JsonField", "name": "doubled", "selector": "$steps.record.value"}
        ],
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "parameter_bindings": {"values": "$inputs.values"},
                "workflow_definition": child,
            },
            {"type": Triple.type, "name": "triple", "value": "$steps.child.doubled"},
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "doubled",
                "selector": "$steps.child.doubled",
            },
            {"type": "JsonField", "name": "tripled", "selector": "$steps.triple.value"},
        ],
    }
    plan = compile_workflow(definition, catalogue=NESTED_CATALOGUE)

    return plan


def test_nested_gates_and_batches_equal_serial_including_zero_call_runs() -> None:
    plan = _nested_gated_batched_plan()
    submissions = [[1.0, 0.0, 2.0], [0.0, 0.0], [3.0], [0.0, 4.0]]
    serial = plan.create_session()
    expected = [serial.run({"values": values}).rows() for values in submissions]

    session = plan.create_session()
    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [pipeline.submit({"values": values}) for values in submissions]
        rows = [future.result(timeout=WAIT).rows() for future in futures]

    assert rows == expected
    assert rows[1] == [
        {"doubled": None, "tripled": None},
        {"doubled": None, "tripled": None},
    ]
    record = ("child", "record")
    assert session.instances[record].calls == serial.instances[record].calls
    assert (
        session.instances[("triple",)].batches == serial.instances[("triple",)].batches
    )
    assert pipeline.outcomes["completed"] == len(submissions)


class CountingLock:
    """Wraps a pipeline lock; ``waiters(n)`` returns once n acquisitions began."""

    def __init__(self, lock: Any) -> None:
        self._lock = lock
        self._condition = threading.Condition()
        self._attempts = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        with self._condition:
            self._attempts += 1
            self._condition.notify_all()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "CountingLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def waiters(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(lambda: self._attempts >= count, WAIT)


def test_blocked_submitters_prepare_nothing_beyond_one_waiting_submission(
    monkeypatch,
) -> None:
    probe = Probe()
    probe.hold("load:0")
    session = _session(probe)
    prepared: List[float] = []
    second_prepared = threading.Event()
    real_prepare = passive.prepare_inputs

    def recording_prepare(plan, inputs):
        prepared.append(inputs["value"])
        if len(prepared) == 2:
            second_prepared.set()
        return real_prepare(plan, inputs)

    monkeypatch.setattr(passive, "prepare_inputs", recording_prepare)

    with session.pipeline(options=PipelineOptions(max_in_flight=1)) as pipeline:
        lock = CountingLock(pipeline._submit_lock)
        pipeline._submit_lock = lock
        first = pipeline.submit({"value": 0.0})
        assert probe.reached("load:0")

        futures: Dict[float, Future] = {}
        submitters = [
            threading.Thread(
                target=lambda v=value: futures.__setitem__(
                    v, pipeline.submit({"value": v})
                )
            )
            for value in (1.0, 2.0)
        ]
        submitters[0].start()
        assert second_prepared.wait(WAIT)
        submitters[1].start()
        lock.waiters(3)
        # Submission 1 is prepared and waits for the busy worker; submission 2
        # waits for the submission lock and has prepared nothing.
        assert prepared == [0.0, 1.0]

        probe.release("load:0")
        for submitter in submitters:
            submitter.join(WAIT)
            assert not submitter.is_alive()

        assert [_out(first), _out(futures[1.0]), _out(futures[2.0])] == [
            0.0,
            10.0,
            20.0,
        ]
    assert prepared == [0.0, 1.0, 2.0]


def test_concurrent_closes_wait_and_a_stale_close_keeps_a_newer_pipeline() -> None:
    probe = Probe()
    probe.hold("load:0")
    session = _session(probe)
    old = session.pipeline(options=PipelineOptions(max_in_flight=1))
    running = old.submit({"value": 0.0})
    assert probe.reached("load:0")
    lock = CountingLock(old._close_lock)
    old._close_lock = lock

    closers = [threading.Thread(target=old.close) for _ in range(2)]
    for closer in closers:
        closer.start()
    lock.waiters(2)
    probe.release("load:0")
    for closer in closers:
        closer.join(WAIT)
        assert not closer.is_alive()
    assert _out(running) == 0.0

    newer = session.pipeline(options=PipelineOptions(max_in_flight=1))
    old.close()
    with pytest.raises(ContractError, match="open pipeline"):
        session.run({"value": 1.0})

    newer.close()
    assert session.run({"value": 1.0}).rows()[0]["out"] == 10.0


def test_an_interrupted_close_keeps_the_session_claimed_until_a_retry() -> None:
    session = _session(Probe())
    pipeline = session.pipeline(options=PipelineOptions(max_in_flight=1))
    assert _out(pipeline.submit({"value": 1.0})) == 10.0
    join_workers = pipeline._pool.shutdown
    attempts: List[int] = []

    def interrupted_once() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("join interrupted")
        join_workers()

    pipeline._pool.shutdown = interrupted_once

    with pytest.raises(RuntimeError, match="join interrupted"):
        pipeline.close()
    with pytest.raises(ContractError, match="open pipeline"):
        session.run({"value": 2.0})

    pipeline.close()
    assert len(attempts) == 2
    assert session.run({"value": 2.0}).rows()[0]["out"] == 20.0


def test_a_failed_worker_launch_leaves_no_worker_and_releases_the_session(
    monkeypatch,
) -> None:
    session = _session(Probe())
    prefix = f"v2-pipeline-{session.session_id[:8]}"
    refuse_second_worker(monkeypatch, prefix)

    with pytest.raises(RuntimeError, match="injected second-worker"):
        session.pipeline(options=PipelineOptions(max_in_flight=2))
    monkeypatch.undo()

    assert live_threads(prefix) == []
    # The claim was released only after the cleanup: the session runs again.
    assert session.run({"value": 2.0}).rows()[0]["out"] == 20.0
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        assert _out(pipeline.submit({"value": 3.0})) == 30.0
