"""Graph updates against concurrent use of the session.

This version updates idle sessions only. A direct run, an open passive
pipeline or an unfinished active run makes ``apply_update`` raise
``SessionBusyError`` at once, also when called from inside the run, so a
synchronous update never waits for itself. Every wait is bounded by ``WAIT``
and ordered by events the test controls.
"""

import threading
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.active import runtime
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    SessionBusyError,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    CATALOGUE,
    WAIT,
    Count,
    Hold,
    Scale,
    active,
    group,
    passive,
    step,
)

HOLD = passive(
    [step(Hold, "hold", value="$inputs.value")], {"value": "$steps.hold.value"}
)
HOLD_AND_SCALE = passive(
    [
        step(Hold, "hold", value="$inputs.value"),
        step(Scale, "scale", value="$inputs.value"),
    ],
    {"value": "$steps.hold.value", "scaled": "$steps.scale.scaled"},
)
TICKS = [step(Count, "count", value="$sources.ticks.value")]
COUNTS = active(TICKS, [group("counts", count="$steps.count.count")], after=1)
ATTACHED = active(
    [*TICKS, step(Scale, "scale", value="$sources.ticks.value")],
    [
        group("counts", count="$steps.count.count"),
        group("scaled", scaled="$steps.scale.scaled"),
    ],
    after=1,
)


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


@pytest.fixture
def gates() -> Dict[str, Any]:
    names = ("entered", "release", "paused", "resume")
    events = {name: threading.Event() for name in names}

    return events


@pytest.fixture
def resources(gates) -> Dict[str, Any]:
    log: List[Any] = []
    values = {"gates": gates, "log": log, "model": object()}

    return values


def test_direct_run_in_progress_rejects_the_commit_and_keeps_the_candidate(
    gates, resources
) -> None:
    session = compiled(HOLD).create_session(resources)
    prepared = session.prepare_update(compiled(HOLD_AND_SCALE))
    worker = threading.Thread(target=session.run, args=({"value": 1.0},))
    worker.start()
    assert gates["entered"].wait(WAIT)

    with pytest.raises(SessionBusyError, match="direct run"):
        session.apply_update(prepared)

    gates["release"].set()
    worker.join(WAIT)
    assert prepared.state == "prepared"
    assert session.apply_update(prepared).graph_version == 1


def test_update_from_inside_a_running_block_is_rejected_without_waiting(
    gates, resources
) -> None:
    session = compiled(HOLD).create_session(resources)
    prepared = session.prepare_update(compiled(HOLD_AND_SCALE))
    gates["release"].set()

    def apply_inside() -> Any:
        try:
            session.apply_update(prepared)
        except SessionBusyError as error:
            return error

    gates["inside"] = apply_inside

    session.run({"value": 1.0})

    assert isinstance(session.instances[("hold",)].inside, SessionBusyError)
    assert session.graph_version == 0


def test_open_passive_pipeline_rejects_the_commit(gates, resources) -> None:
    session = compiled(HOLD).create_session(resources)
    gates["release"].set()

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        pipeline.submit({"value": 1.0}).result(WAIT)
        with pytest.raises(SessionBusyError, match="open pipeline"):
            session.update(compiled(HOLD_AND_SCALE))

    assert session.update(compiled(HOLD_AND_SCALE)).graph_version == 1


@pytest.mark.parametrize("pipeline", [None, PipelineOptions(max_in_flight=2)])
def test_unfinished_active_run_rejects_the_commit_also_from_its_handler(
    gates, resources, pipeline
) -> None:
    session = compiled(COUNTS).create_session(resources)
    prepared = session.prepare_update(compiled(ATTACHED))
    from_handler: List[BaseException] = []

    def on_counts(result: Any) -> None:
        try:
            session.apply_update(prepared)
        except SessionBusyError as error:
            from_handler.append(error)

    run = session.start(handlers={"counts": on_counts}, pipeline=pipeline)
    assert gates["paused"].wait(WAIT)

    with pytest.raises(SessionBusyError, match="unfinished active run"):
        session.apply_update(prepared)

    gates["resume"].set()
    run.wait(timeout=WAIT)
    assert len(from_handler) == 3
    assert session.apply_update(prepared).graph_version == 1


def test_two_candidates_prepared_together_commit_exactly_once(resources) -> None:
    session = compiled(HOLD).create_session(resources)
    barrier = threading.Barrier(2, timeout=WAIT)
    outcomes: List[Any] = []
    lock = threading.Lock()

    def prepare_and_apply() -> None:
        barrier.wait()
        prepared = session.prepare_update(compiled(HOLD_AND_SCALE))
        barrier.wait()
        try:
            outcome = session.apply_update(prepared)
        except UpdateConflictError as error:
            outcome = error
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=prepare_and_apply) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)

    receipts = [o for o in outcomes if not isinstance(o, BaseException)]
    conflicts = [o for o in outcomes if isinstance(o, UpdateConflictError)]
    assert len(receipts) == 1 and len(conflicts) == 1
    assert session.graph_version == 1


def test_start_that_raced_an_update_fails_and_never_runs_the_old_plan(
    gates, resources, monkeypatch
) -> None:
    session = compiled(COUNTS).create_session(resources)
    registering, proceed = threading.Event(), threading.Event()
    register = runtime._register_handlers

    def held_register(plan: Any, handlers: Any) -> Any:
        registering.set()
        assert proceed.wait(WAIT)
        return register(plan, handlers)

    monkeypatch.setattr(runtime, "_register_handlers", held_register)
    failures: List[BaseException] = []

    def start() -> None:
        try:
            session.start(handlers={"counts": lambda result: None})
        except ContractError as error:
            failures.append(error)

    starter = threading.Thread(target=start)
    starter.start()
    assert registering.wait(WAIT)

    session.update(compiled(ATTACHED))
    proceed.set()
    starter.join(WAIT)

    (failure,) = failures
    assert "switched to graph version 1" in str(failure)
    assert [entry for entry in resources["log"] if entry[0] == "open"] == []
    gates["resume"].set()
    monkeypatch.setattr(runtime, "_register_handlers", register)
    delivered: List[Any] = []
    session.start(handlers={"scaled": delivered.append}).wait(timeout=WAIT)
    assert [result.graph_version for result in delivered] == [1, 1, 1]
