"""Reusable pulse teams: P teams of W workers, branches of one pulse at once.

``PipelineOptions(max_in_flight=P, max_workflow_parallelism=W)``: each of the
P pipeline workers leads a team of W threads. Independent steps (and, in an
active pulse, group deliveries) of one pulse run on the team at once; W = 1
keeps the plan-order loop. Serial ``session.run`` is the reference oracle.

Every overlap claim is shown with events the test controls: a held call
waits until the test releases it, and the test waits for another call to
start meanwhile. ``WAIT`` only bounds each wait, so a broken scheduler fails
instead of hanging; no assertion depends on thread timing.
"""

import contextvars
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.active.pulses import _RouteGraph
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import current_pulse_run_id
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
)
from roboflow_workflows.execution_engine.v2.execution.steps import RunState
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    GAUGES,
    SERIAL,
    Coordination,
    PipelineCounters,
    RunAborted,
)
from roboflow_workflows.execution_engine.v2.pipelining.teams import (
    Team,
    bound_team,
    team_for,
)
from roboflow_workflows.execution_engine.v2.pipelining.workers import (
    OwnedThreads,
    WorkerPool,
)
from roboflow_workflows.execution_engine.v2.plan import Constant

from tests.unit_tests.execution_engine.v2.active import test_pipelined_operators
from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    CATALOGUE as PIPELINED_CATALOGUE,
)
from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    Held,
    Probe,
    Session,
    nested_gates,
    outcome,
    ports_emitted_separately,
    two_sources_with_a_shared_static_step,
)
from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Deferred,
    Failing,
    Scale,
)
from tests.unit_tests.execution_engine.v2.pipelining.test_workers import live_threads
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    active,
    emit,
    group,
    parameter,
    source,
    step,
)

SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2)]
"""(max_in_flight, max_workflow_parallelism) combinations every matrix covers."""


class Join(Block):
    """Adds two branches; the call is the ``Probe`` unit ``join``."""

    type = "test/threading_join@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        left: Ref(FLOAT_KIND)
        right: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe
        self.run_ids: List[Optional[str]] = []

    def run(self, *, left, right) -> dict:
        with self.probe.call("join", left + right):
            self.run_ids.append(current_pulse_run_id())
            return {"value": left + right}


CATALOGUE = Catalogue.merge(PIPELINED_CATALOGUE, Catalogue([Join, Deferred]))


def options(in_flight: int, parallelism: int) -> PipelineOptions:
    shaped = PipelineOptions(
        max_in_flight=in_flight, max_workflow_parallelism=parallelism
    )

    return shaped


# Teams ----------------------------------------------------------------------------


class Graph:
    """A hand-written ``UnitGraph``; priorities default to unit order."""

    def __init__(self, predecessors: List[Tuple[int, ...]], priorities=None):
        self.predecessors = tuple(predecessors)
        successors: List[List[int]] = [[] for _ in predecessors]
        for unit, before in enumerate(predecessors):
            for predecessor in before:
                successors[predecessor].append(unit)
        self.successors = tuple(tuple(after) for after in successors)
        self.priorities = tuple(priorities or range(len(predecessors)))


DIAMOND = Graph([(), (0,), (0,), (1, 2)])
"""0 forks into 1 and 2; 3 joins them."""


def team_of(size: int, **kwargs: Any) -> Team:
    team = Team(size, name="team-test", mark_owned=OwnedThreads().mark, **kwargs)

    return team


def test_a_team_runs_both_branches_at_once_and_joins_after_them() -> None:
    meet = threading.Barrier(2, timeout=WAIT)
    ran: List[Tuple[int, str]] = []
    counters = PipelineCounters()
    team = team_of(2, counters=counters)

    def execute(unit: int) -> None:
        if unit in (1, 2):
            meet.wait()  # both branches must be in progress at once
        ran.append((unit, threading.current_thread().name))

    try:
        assert team.run(DIAMOND, execute=execute, stop=lambda: False) is True
    finally:
        team.close()

    assert [unit for unit, _ in ran][0] == 0 and ran[-1][0] == 3
    assert {thread for unit, thread in ran if unit in (1, 2)} == {
        threading.current_thread().name,
        "team-test.1",
    }
    assert counters.peak("parallel_units") == 2
    assert counters.current("parallel_units") == 0


def test_the_worker_finishing_a_unit_continues_its_path_depth_first() -> None:
    # 0 -> 1 -> 3 and 0 -> 2; priorities rank the path 1, 3 before 2.
    graph = Graph([(), (0,), (0,), (1,)], priorities=[0, 1, 3, 2])
    threads: Dict[int, str] = {}
    took_two = threading.Event()
    team = team_of(2)

    def execute(unit: int) -> None:
        threads[unit] = threading.current_thread().name
        if unit == 2:
            took_two.set()
        if unit == 1:
            assert took_two.wait(WAIT)  # the idle helper took the other branch

    try:
        team.run(graph, execute=execute, stop=lambda: False)
    finally:
        team.close()

    # Either worker may claim the root; the one that did continues its path.
    assert set(threads.values()) == {threading.current_thread().name, "team-test.1"}
    assert threads[0] == threads[1] == threads[3] != threads[2]


def test_the_first_real_failure_halts_the_job_after_running_units_settle() -> None:
    started_one = threading.Event()
    release_one = threading.Event()
    ran: List[int] = []
    team = team_of(2)

    def execute(unit: int) -> None:
        ran.append(unit)
        if unit == 1:
            started_one.set()
            assert release_one.wait(WAIT)
            raise RunAborted("aborted after the failure")
        if unit == 2:
            assert started_one.wait(WAIT)
            release_one.set()
            raise ValueError("branch two failed")

    try:
        with pytest.raises(ValueError, match="branch two failed"):
            team.run(DIAMOND, execute=execute, stop=lambda: False)
        assert team.idle
    finally:
        team.close()

    assert sorted(ran) == [0, 1, 2]  # the join never started


def test_stop_halts_before_the_next_unit_without_an_error() -> None:
    ran: List[int] = []
    team = team_of(2)

    try:
        completed = team.run(
            Graph([(), (0,), (1,)]), execute=ran.append, stop=lambda: len(ran) >= 2
        )
    finally:
        team.close()

    assert completed is False
    assert ran == [0, 1]


def test_a_cycle_fails_instead_of_waiting_forever() -> None:
    team = team_of(2)

    try:
        with pytest.raises(RuntimeError, match="can never become ready"):
            team.run(Graph([(), (2,), (1,)]), execute=lambda unit: None, stop=bool)
    finally:
        team.close()


def test_helpers_run_units_in_the_callers_context() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    marker.set("pulse-7")
    meet = threading.Barrier(2, timeout=WAIT)
    seen: Dict[int, str] = {}
    team = team_of(2)

    def execute(unit: int) -> None:
        if unit in (1, 2):
            meet.wait()
        seen[unit] = marker.get("missing")

    try:
        team.run(DIAMOND, execute=execute, stop=lambda: False)
    finally:
        team.close()

    assert seen == dict.fromkeys(range(4), "pulse-7")


class Pipelined(Coordination):
    """Only what ``team_for`` reads of a pipelined run's coordination."""

    pipelined = True


PIPELINED = Pipelined()


def test_a_busy_or_single_worker_team_is_not_offered_again() -> None:
    team = team_of(2)
    seen: List[Optional[Team]] = []

    try:
        assert team_for(PIPELINED) is None  # this thread leads no team
        with bound_team(team):
            assert team_for(PIPELINED) is team
            team.run(
                Graph([()]),
                execute=lambda unit: seen.append(team_for(PIPELINED)),
                stop=lambda: False,
            )
        with bound_team(team_of(1)):
            assert team_for(PIPELINED) is None
    finally:
        team.close()

    assert seen == [None]  # re-entrance runs in plan order on the caller


def test_a_serial_run_never_takes_the_idle_team_of_its_thread() -> None:
    # E.g. a block of a pipelined run starts a serial session on its worker.
    team = team_of(2)

    try:
        with bound_team(team):
            assert team.idle
            assert team_for(SERIAL) is None
    finally:
        team.close()


def test_deliveries_follow_their_steps_and_start_before_ready_steps() -> None:
    graph = _RouteGraph.of(DIAMOND, deliveries=[(1,), ()])

    assert graph.predecessors[4:] == ((1,), ())
    assert 4 in graph.successors[1]
    assert graph.priorities[4] == graph.priorities[5] < min(graph.priorities[:4])


def test_a_pool_owns_exactly_p_times_w_threads_and_joins_them_all() -> None:
    pool = WorkerPool(3, name="budget-test", owner=OwnedThreads(), team_size=2)

    assert len(live_threads("budget-test")) == 6
    assert len(pool.threads) == 6
    pool.shutdown()
    assert live_threads("budget-test") == []


def test_a_failed_team_launch_joins_every_started_thread(monkeypatch) -> None:
    real_start = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name == "team-launch-1.1":
            raise RuntimeError("injected helper launch failure")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)

    with pytest.raises(RuntimeError, match="injected helper launch failure"):
        WorkerPool(2, name="team-launch", owner=OwnedThreads(), team_size=2)

    assert live_threads("team-launch") == []


def test_parallel_units_is_a_gauge_of_every_pipeline() -> None:
    assert "parallel_units" in GAUGES


@pytest.mark.parametrize("parallelism", [0, True, 1.5])
def test_max_workflow_parallelism_is_a_positive_int(parallelism) -> None:
    with pytest.raises(ContractError, match="max_workflow_parallelism"):
        PipelineOptions(max_workflow_parallelism=parallelism)
    assert PipelineOptions().max_workflow_parallelism == 1


# Run state ------------------------------------------------------------------------


class SlowCopy:
    """Its first deep copy waits until the test releases it; copies are counted."""

    def __init__(self) -> None:
        self.copies = 0
        self.copying = threading.Event()
        self.release = threading.Event()

    def __deepcopy__(self, memo: dict) -> "SlowCopy":
        self.copies += 1
        if self.copies == 1:
            self.copying.set()
            assert self.release.wait(WAIT)
        return SlowCopy()


def test_concurrent_first_readers_share_one_constant_copy() -> None:
    session, _ = passive_session(uneven_fork_join())
    run = RunState(session=session, run_id="run", inputs={})
    constant = Constant(SlowCopy())
    entries: List[Any] = []

    def read() -> None:
        entries.append(run.entry_for(constant))

    first = threading.Thread(target=read)
    first.start()
    assert constant.value.copying.wait(WAIT)
    others = [threading.Thread(target=read) for _ in range(4)]
    for thread in others:
        thread.start()
    constant.value.release.set()
    for thread in [first, *others]:
        thread.join(WAIT)

    assert len(entries) == 5
    assert all(entry is entries[0] for entry in entries)
    assert constant.value.copies == 1


# Passive pipelines ----------------------------------------------------------------

CHILD = {
    "version": "2.0",
    "inputs": [parameter("x", 0.0)],
    "steps": [
        step(
            ContinueIf,
            "inner_gate",
            value="$inputs.x",
            threshold=5.0,
            next_steps=["$steps.big"],
        ),
        step(Scale, "big", value="$inputs.x", factor=10.0),
    ],
    "outputs": [
        {"type": "JsonField", "name": "forwarded", "selector": "$inputs.x"},
        {"type": "JsonField", "name": "scaled", "selector": "$steps.big.scaled"},
    ],
}


def uneven_fork_join() -> dict:
    """A long branch (left, tripled) and a short gated one with a nested gate.

    load -> left -> tripled ----------------------------> join
         -> right -> gate(right > 0) -> child(x > 5: big) -> mark
                 \\---------------------------------------> join
    """
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": [
            step(Held, "load", value="$inputs.value"),
            step(Held, "left", value="$steps.load.value"),
            step(Scale, "tripled", value="$steps.left.value", factor=3.0),
            step(Held, "right", value="$steps.load.value"),
            step(
                ContinueIf,
                "gate",
                value="$steps.right.value",
                threshold=0.0,
                next_steps=["$steps.child"],
            ),
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": CHILD,
                "parameter_bindings": {"x": "$steps.right.value"},
            },
            step(Held, "mark", value="$steps.child.scaled"),
            step(
                Join, "join", left="$steps.tripled.scaled", right="$steps.right.value"
            ),
        ],
        "outputs": [
            {"type": "JsonField", "name": "join", "selector": "$steps.join.value"},
            {"type": "JsonField", "name": "mark", "selector": "$steps.mark.value"},
            {
                "type": "JsonField",
                "name": "forwarded",
                "selector": "$steps.child.forwarded",
            },
        ],
    }

    return definition


VALUES = [-2.0, 0.0, 3.0, 7.0, 9.0, 12.0]
"""Gate denies <= 0; the nested gate admits > 5."""


def passive_session(definition: dict, **kwargs: Any):
    probe = Probe()
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session({"probe": probe}, **kwargs)

    return session, probe


def thread_prefix(session) -> str:
    return f"v2-pipeline-{session.session_id[:8]}"


def test_w1_keeps_plan_order_on_one_thread_per_submission() -> None:
    serial, serial_probe = passive_session(uneven_fork_join())
    for value in VALUES:
        serial.run({"value": value})
    session, probe = passive_session(uneven_fork_join())

    with session.pipeline(options=options(1, 1)) as pipeline:
        assert len(live_threads(thread_prefix(session))) == 1
        futures = [pipeline.submit({"value": value}) for value in VALUES]
    for future in futures:
        future.result(WAIT)

    assert probe.events == serial_probe.events
    assert len(set(probe.threads.values())) == 1
    assert pipeline.counters.peak("parallel_units") == 0


@pytest.mark.parametrize("in_flight, parallelism", SHAPES)
def test_every_shape_equals_the_serial_reference_and_keeps_stage_order(
    in_flight, parallelism
) -> None:
    serial, serial_probe = passive_session(uneven_fork_join())
    expected = [serial.run({"value": value}).rows() for value in VALUES]
    session, probe = passive_session(uneven_fork_join())

    with session.pipeline(options=options(in_flight, parallelism)) as pipeline:
        threads = live_threads(thread_prefix(session))
        futures = [pipeline.submit({"value": value}) for value in VALUES]
    results = [future.result(WAIT).rows() for future in futures]

    assert results == expected
    assert dict(probe.order) == dict(serial_probe.order)  # per stage, submission order
    assert len(threads) == in_flight * parallelism
    assert live_threads(thread_prefix(session)) == []
    counters = pipeline.counters
    assert counters.peak("executing") <= in_flight
    assert counters.peak("parallel_units") <= in_flight * parallelism
    assert counters.current("parallel_units") == 0


def test_a_short_branch_completes_while_the_long_branch_of_its_pulse_waits() -> None:
    session, probe = passive_session(uneven_fork_join())
    release = probe.hold("left", 7.0)

    with session.pipeline(options=options(1, 2)) as pipeline:
        future = pipeline.submit({"value": 7.0})
        assert probe.entered("left", 7.0).wait(WAIT)
        # The short branch runs to its end, nested gate included, meanwhile.
        assert probe.entered("mark", 70.0).wait(WAIT)
        assert not probe.entered("join", 28.0).is_set()
        release.set()
        rows = future.result(WAIT).rows()

    assert rows == [{"join": 28.0, "mark": 70.0, "forwarded": 7.0}]
    assert probe.threads[("left", 7.0)] != probe.threads[("mark", 70.0)]
    assert pipeline.counters.peak("parallel_units") == 2


def test_a_later_submission_advances_but_never_overtakes_an_earlier_stage() -> None:
    session, probe = passive_session(uneven_fork_join())
    release = probe.hold("left", 7.0)

    with session.pipeline(options=options(2, 2)) as pipeline:
        first = pipeline.submit({"value": 7.0})
        assert probe.entered("left", 7.0).wait(WAIT)
        second = pipeline.submit({"value": 9.0})
        # Submission 1 passes every stage submission 0 already retired ...
        assert probe.entered("mark", 90.0).wait(WAIT)
        # ... but waits for its turn where submission 0 is still in progress.
        assert not probe.entered("left", 9.0).is_set()
        release.set()
        results = [first.result(WAIT).rows(), second.result(WAIT).rows()]

    assert results == [
        [{"join": 28.0, "mark": 70.0, "forwarded": 7.0}],
        [{"join": 36.0, "mark": 90.0, "forwarded": 9.0}],
    ]
    assert probe.order["left"] == [7.0, 9.0]
    assert probe.order["mark"] == [70.0, 90.0]


def failing_branch() -> dict:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": [
            step(Held, "left", value="$inputs.value"),
            step(Failing, "check", value="$inputs.value"),
            step(Join, "join", left="$steps.left.value", right="$steps.check.value"),
        ],
        "outputs": [
            {"type": "JsonField", "name": "join", "selector": "$steps.join.value"}
        ],
    }

    return definition


def test_a_branch_failure_waits_for_its_running_sibling_and_skips_the_join() -> None:
    failed = threading.Event()
    session, probe = passive_session(
        failing_branch(), error_handler=lambda error: failed.set()
    )
    release = probe.hold("left", -1.0)

    with session.pipeline(options=options(1, 2)) as pipeline:
        future = pipeline.submit({"value": -1.0})
        assert probe.entered("left", -1.0).wait(WAIT)
        assert failed.wait(WAIT)
        assert not future.done()  # its sibling still runs on the team
        release.set()
        with pytest.raises(StepExecutionError, match="negative -1") as caught:
            future.result(WAIT)

    assert caught.value.step_path == ("check",)
    assert "join" not in probe.order
    assert pipeline.failure is caught.value
    assert live_threads(thread_prefix(session)) == []


def deferred_branches(*, fail: bool) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": [
            step(Held, "left", value="$inputs.value"),
            step(Deferred, "deferred", value="$inputs.value", fail=fail),
            step(
                Join, "join", left="$steps.left.value", right="$steps.deferred.doubled"
            ),
        ],
        "outputs": [
            {"type": "JsonField", "name": "join", "selector": "$steps.join.value"}
        ],
    }

    return definition


def test_futures_resolve_in_their_branch_and_their_failure_is_the_steps() -> None:
    session, _ = passive_session(deferred_branches(fail=False))
    with session.pipeline(options=options(2, 2)) as pipeline:
        future = pipeline.submit({"value": 2.0})
        assert future.result(WAIT).rows() == [{"join": 6.0}]

    failing, _ = passive_session(deferred_branches(fail=True))
    with failing.pipeline(options=options(2, 2)) as pipeline:
        future = pipeline.submit({"value": 2.0})
        with pytest.raises(StepExecutionError, match="a future returned by the block"):
            future.result(WAIT)


def test_every_unit_of_a_submission_sees_its_run_id() -> None:
    session, probe = passive_session(uneven_fork_join())

    with session.pipeline(options=options(2, 2)) as pipeline:
        futures = [pipeline.submit({"value": value}) for value in (7.0, 9.0)]
    for future in futures:
        future.result(WAIT)

    join = session.instances[("join",)]
    assert len(set(join.run_ids)) == 2 and None not in join.run_ids


# Active runs ----------------------------------------------------------------------


def active_fork_join() -> Tuple[dict, dict, List[str]]:
    definition = active(
        [source("a")],
        [
            step(Held, "left", value="$sources.a.value"),
            step(Held, "right", value="$sources.a.value"),
            step(Join, "join", left="$steps.left.value", right="$steps.right.value"),
        ],
        [
            group("R", "$sources.a.value", r="$steps.right.value"),
            group("J", "$sources.a.value", j="$steps.join.value"),
        ],
    )
    feeds = {"a": [emit(value=float(index)) for index in range(1, 5)]}

    return definition, feeds, ["R", "J"]


def run_outcome(scenario, pipeline: Optional[PipelineOptions]) -> Dict[str, list]:
    definition, feeds, groups = scenario()
    session = Session(definition, feeds, catalogue=CATALOGUE)
    collected = Collector()
    run = session.start(collected.handlers(*groups), pipeline=pipeline)
    assert run.wait(WAIT)

    return outcome(collected)


@pytest.mark.parametrize("in_flight, parallelism", SHAPES)
@pytest.mark.parametrize(
    "scenario",
    [
        active_fork_join,
        two_sources_with_a_shared_static_step,
        ports_emitted_separately,
        nested_gates,
    ],
)
def test_active_shapes_equal_the_serial_reference(
    scenario, in_flight, parallelism
) -> None:
    serial = run_outcome(scenario, None)

    pipelined = run_outcome(scenario, options(in_flight, parallelism))

    assert pipelined == serial


def test_a_group_is_delivered_while_an_unrelated_branch_of_its_pulse_runs() -> None:
    definition, _, groups = active_fork_join()
    session = Session(definition, {"a": [emit(value=1.0)]}, catalogue=CATALOGUE)
    release = session.probe.hold("left", 1.0)
    delivered = threading.Event()
    collected = Collector()
    handlers = collected.handlers(*groups)
    deliver_r = handlers["R"]
    handlers["R"] = lambda result: (deliver_r(result), delivered.set())

    run = session.start(handlers, pipeline=options(2, 2))

    assert session.probe.entered("left", 1.0).wait(WAIT)
    assert delivered.wait(WAIT)  # R needs only "right"
    assert len(live_threads(f"workflows-v2-worker-{run.run_id[:8]}")) == 4
    assert "J" not in collected.results
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("R") == [{"r": 1.0}]
    assert collected.rows("J") == [{"j": 2.0}]
    assert live_threads(f"workflows-v2-worker-{run.run_id[:8]}") == []
    assert run.pipeline_counters.peak("parallel_units") >= 2


def test_an_active_branch_failure_is_attributed_to_its_pulse_and_step() -> None:
    definition = active(
        [source("a")],
        [
            step(Held, "left", value="$sources.a.value"),
            step(Failing, "check", value="$sources.a.value"),
        ],
        [group("A", "$sources.a.value", left="$steps.left.value")],
    )
    session = Session(definition, {"a": [emit(value=-1.0)]}, catalogue=CATALOGUE)
    release = session.probe.hold("left", -1.0)
    run = session.start(Collector().handlers("A"), pipeline=options(1, 2))
    assert session.probe.entered("left", -1.0).wait(WAIT)
    release.set()

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)

    assert (caught.value.stage, caught.value.source, caught.value.pulse) == (
        "step",
        "a",
        0,
    )
    assert caught.value.step_path == ("check",)
    assert live_threads(f"workflows-v2-worker-{run.run_id[:8]}") == []


def test_inline_operator_emissions_equal_the_serial_reference_with_teams() -> None:
    operators = test_pipelined_operators

    def run_chain(pipeline: Optional[PipelineOptions]) -> Tuple[dict, list]:
        operators.LIFECYCLE.clear()
        session = Session(
            operators.reverse_completion_chain(),
            operators.chain_feeds(),
            catalogue=operators.OPERATORS,
        )
        collected = Collector()
        run = session.start(collected.handlers("P", "C", "S"), pipeline=pipeline)
        assert run.wait(WAIT)

        return outcome(collected), list(operators.LIFECYCLE)

    try:
        serial = run_chain(None)
        pipelined = run_chain(options(2, 2))
    finally:
        operators.LIFECYCLE.clear()

    assert pipelined == serial
