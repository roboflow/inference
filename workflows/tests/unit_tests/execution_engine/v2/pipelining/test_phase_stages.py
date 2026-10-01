"""Stages of real steps: phase overlap, whole-call exclusion and nth-call retirement.

Each pulse is a passive run executed by ``execute_step`` on its own thread,
with one ``PipelinedCoordination`` shared by the pulses. The phased block
reports each phase entry to a ``Probe``; a test hooks a phase entry to set
or wait for events. Waits are bounded by ``TIMEOUT`` and nothing depends
on sleeping: "blocked" is observed through the gates' counters.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    PASSIVE_DOMAIN,
    SERIAL,
    Coordination,
    PipelinedCoordination,
    RunAborted,
    Ticket,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompileOptions,
    ExecutionObserver,
)

TIMEOUT = 5.0

Hook = Callable[[], None]


class Probe:
    """Records phase entries and runs the hook registered for one entry."""

    def __init__(self) -> None:
        self.log: List[Tuple[str, float]] = []
        self.hooks: Dict[Tuple[str, float], Hook] = {}
        self._lock = threading.Lock()

    def enter(self, name: str, value: float) -> None:
        with self._lock:
            self.log.append((name, value))
        hook = self.hooks.get((name, value))
        if hook is not None:
            hook()


class TwoPhase(Block):
    """first -> second. ``value < 0`` fails in ``second``."""

    type = "test/pipelining/two_phase@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    @phase
    def first(self, *, value):
        self.probe.enter("first", value)
        return value

    @phase
    def second(self, *, first):
        self.probe.enter("second", first)
        if first < 0:
            raise ValueError(f"negative {first}")
        return {"total": first * 10}

    def run(self, *, value):
        return self.second(first=self.first(value=value))


class WholeCall(TwoPhase):
    """The same phases, sharing ``self`` across a call: no phase overlap."""

    type = "test/pipelining/whole_call@v1"
    phase_overlap = False


CATALOGUE = Catalogue([TwoPhase, WholeCall])


def session_of(block: type, *, mode: str = "phases", batch: bool = False, **extra):
    value_input = (
        {"type": "WorkflowBatchInput", "name": "value", "kind": ["float"]}
        if batch
        else {"type": "WorkflowParameter", "name": "value", "kind": ["float"]}
    )
    definition = {
        "version": "2.0",
        "inputs": [value_input],
        "steps": [{"type": block.type, "name": "two", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "total", "selector": "$steps.two.total"}
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    probe = Probe()
    session = plan.create_session(resources={"probe": probe}, **extra)

    return session, probe


def execute(session, coordination: Coordination, ordinal: int, value: Any) -> RunState:
    """One passive run as one pulse of the ``$passive`` domain."""
    run = RunState(
        session=session,
        run_id=f"run-{ordinal}",
        inputs=prepare_inputs(session.plan, {"value": value}),
        coordination=coordination,
        ticket=Ticket(PASSIVE_DOMAIN, ordinal),
    )
    for step in session.plan.steps:
        execute_step(run, step)

    return run


class Pulse(threading.Thread):
    """Runs ``execute`` on a thread and keeps its run or error."""

    def __init__(self, session, coordination, ordinal: int, value: Any) -> None:
        super().__init__(daemon=True)
        self.arguments = (session, coordination, ordinal, value)
        self.run_state: Optional[RunState] = None
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self.run_state = execute(*self.arguments)
        except BaseException as error:  # RunAborted is a BaseException
            self.error = error


def started(session, coordination, ordinal: int, value: Any) -> Pulse:
    pulse = Pulse(session, coordination, ordinal, value)
    pulse.start()

    return pulse


def wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


def totals(run: RunState) -> Dict[tuple, Any]:
    values = dict(run.outputs[(("two",), "total")].values)

    return values


def pipelined(session) -> PipelinedCoordination:
    coordination = PipelinedCoordination(session, options=PipelineOptions())

    return coordination


def test_phase_a_of_pulse_1_overlaps_phase_b_of_pulse_0() -> None:
    """The critical probe: retirement after the nth call, not at step end."""
    session, probe = session_of(TwoPhase)
    coordination = pipelined(session)
    pulse0_in_second = threading.Event()
    pulse1_in_first = threading.Event()
    overlapped: List[bool] = []

    def pulse0_second() -> None:
        pulse0_in_second.set()
        # Pulse 0 stays in phase "second" until pulse 1 is inside "first".
        overlapped.append(pulse1_in_first.wait(TIMEOUT))

    probe.hooks[("second", 0.0)] = pulse0_second
    probe.hooks[("first", 1.0)] = pulse1_in_first.set

    pulses = [started(session, coordination, 0, 0.0)]
    assert pulse0_in_second.wait(TIMEOUT)
    pulses.append(started(session, coordination, 1, 1.0))
    for pulse in pulses:
        pulse.join(TIMEOUT)

    assert overlapped == [True]
    assert [pulse.error for pulse in pulses] == [None, None]
    assert totals(pulses[0].run_state) == {(): 0.0}
    assert totals(pulses[1].run_state) == {(): 10.0}
    counters = coordination.counters.snapshot()
    assert counters["peak"]["overlapping_pulses"] == 2
    assert set(counters["stages"]) == {"$steps.two#first", "$steps.two#second"}
    assert counters["stages"]["$steps.two#first"]["calls"] == 2


@pytest.mark.parametrize(
    "block, mode",
    [(WholeCall, "phases"), (TwoPhase, "run")],
    ids=["phase_overlap_false", "run_mode"],
)
def test_whole_call_holds_every_phase_until_the_call_is_ready(block, mode) -> None:
    session, probe = session_of(block, mode=mode)
    coordination = pipelined(session)
    gate = coordination.gate("$steps.two#call")
    seen_blocked: List[bool] = []

    def pulse0_second() -> None:
        # Pulse 1 is observed waiting for its turn at the whole-call stage.
        wait_for(lambda: coordination.counters.stage(gate.name).waited_turn == 1)
        seen_blocked.append(("first", 1.0) not in probe.log)

    probe.hooks[("second", 0.0)] = pulse0_second
    pulse0 = started(session, coordination, 0, 0.0)
    wait_for(lambda: ("first", 0.0) in probe.log)
    pulse1 = started(session, coordination, 1, 1.0)
    pulse0.join(TIMEOUT)
    pulse1.join(TIMEOUT)

    assert seen_blocked == [True]
    assert probe.log == [
        ("first", 0.0),
        ("second", 0.0),
        ("first", 1.0),
        ("second", 1.0),
    ]
    assert set(coordination.counters.snapshot()["stages"]) == {"$steps.two#call"}


def test_multi_call_pulse_keeps_each_phase_until_its_last_call() -> None:
    session, probe = session_of(TwoPhase, batch=True)
    coordination = pipelined(session)
    pulse1_in_first = threading.Event()
    overlapped: List[bool] = []

    # Pulse 0 makes two calls; it waits in its LAST "second" for pulse 1.
    probe.hooks[("second", 1.0)] = lambda: overlapped.append(
        pulse1_in_first.wait(TIMEOUT)
    )
    probe.hooks[("first", 10.0)] = pulse1_in_first.set

    pulse0 = started(session, coordination, 0, [0.0, 1.0])
    wait_for(lambda: ("first", 0.0) in probe.log)
    pulse1 = started(session, coordination, 1, [10.0, 11.0])
    pulse0.join(TIMEOUT)
    pulse1.join(TIMEOUT)

    assert overlapped == [True]
    log = probe.log
    # Pulse 1 entered "first" only after pulse 0's last "first".
    assert log.index(("first", 10.0)) > log.index(("first", 1.0))
    assert totals(pulse0.run_state) == {(0,): 0.0, (1,): 10.0}
    assert totals(pulse1.run_state) == {(0,): 100.0, (1,): 110.0}


def test_zero_call_pulse_retires_without_waiting_for_earlier_pulses() -> None:
    session, probe = session_of(TwoPhase, batch=True)
    coordination = pipelined(session)

    # Pulse 1 has no invocation; it completes although pulse 0 never started.
    execute(session, coordination, 1, [])
    first = coordination.gate("$steps.two#first")
    assert first._finished == {PASSIVE_DOMAIN: {1}}

    execute(session, coordination, 0, [2.0])
    run2 = execute(session, coordination, 2, [3.0])

    assert totals(run2) == {(0,): 30.0}
    assert first._next == {PASSIVE_DOMAIN: 3}
    assert first._finished == {}


class Recorder(ExecutionObserver):
    def __init__(self) -> None:
        self.errors: List[BaseException] = []

    def on_error(self, *, error) -> None:
        self.errors.append(error)


def test_failure_aborts_waiting_pulses_without_reporting_them_as_errors() -> None:
    observer = Recorder()
    handled: List[BaseException] = []
    session, probe = session_of(
        TwoPhase, observer=observer, error_handler=handled.append
    )
    coordination = pipelined(session)
    release = threading.Event()
    probe.hooks[("first", -1.0)] = lambda: release.wait(TIMEOUT)

    pulse0 = started(session, coordination, 0, -1.0)
    wait_for(lambda: ("first", -1.0) in probe.log)
    pulse1 = started(session, coordination, 1, 1.0)
    wait_for(lambda: coordination.counters.stage("$steps.two#first").waited_turn == 1)
    release.set()
    pulse0.join(TIMEOUT)
    pulse1.join(TIMEOUT)

    assert isinstance(pulse0.error, StepExecutionError)
    assert pulse0.error.phase == "second"
    assert isinstance(pulse1.error, RunAborted)
    assert coordination.aborted
    # The attributed step error is the cause, recorded before waiters woke.
    assert coordination.abort_cause is pulse0.error
    coordination.abort(ValueError("later failure"))
    assert coordination.abort_cause is pulse0.error
    assert len(observer.errors) == 1 and len(handled) == 1
    with pytest.raises(RunAborted):
        execute(session, coordination, 2, 2.0)


def test_serial_and_pipelined_runs_produce_the_same_outputs() -> None:
    session, _ = session_of(TwoPhase, batch=True)
    serial = [execute(session, SERIAL, number, [number, 1.5]) for number in range(3)]
    coordination = pipelined(session)
    pulses = [
        started(session, coordination, number, [number, 1.5]) for number in range(3)
    ]
    for pulse in pulses:
        pulse.join(TIMEOUT)

    assert [pulse.error for pulse in pulses] == [None, None, None]
    assert [totals(pulse.run_state) for pulse in pulses] == [
        totals(run) for run in serial
    ]
    assert serial[0].coordination is SERIAL


def test_pipelined_observer_callbacks_never_run_concurrently() -> None:
    inside = []
    overlaps = []

    class Exclusive(ExecutionObserver):
        def on_step_started(self, *, step, block_type) -> None:
            inside.append(step)
            overlaps.append(len(inside) > 1)
            time.sleep(0.001)
            inside.pop()

    session, _ = session_of(TwoPhase, observer=Exclusive())
    coordination = pipelined(session)
    pulses = [started(session, coordination, number, 1.0) for number in range(4)]
    for pulse in pulses:
        pulse.join(TIMEOUT)

    assert [pulse.error for pulse in pulses] == [None] * 4
    assert overlaps == [False] * 4


def test_cancel_aborts_without_a_cause() -> None:
    session, _ = session_of(TwoPhase)
    coordination = pipelined(session)

    coordination.abort()

    assert coordination.aborted and coordination.abort_cause is None
    with pytest.raises(RunAborted):
        execute(session, coordination, 0, 1.0)
