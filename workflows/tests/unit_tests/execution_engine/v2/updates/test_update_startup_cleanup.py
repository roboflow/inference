"""A failed start keeps the session reserved until its cleanup settled.

When the observer refuses a run, ``start`` closes the run's operators and
releases the caller's resources before it raises. Until then the run stays
registered: a graph update and another start are refused. ``close`` becomes
allowed once the run no longer uses the session, before ``finalize``. The
startup failure stays the raised error, also when cleanup fails.
Every wait is bounded by ``WAIT`` and ordered by events the test controls.
"""

import threading
from typing import Any, Callable, Dict

import pytest
from roboflow_workflows.execution_engine.v2.active.runtime import start_session
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    SessionBusyError,
    SessionClosedError,
)
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver
from roboflow_workflows.execution_engine.v2.updates import PREPARED

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    WAIT,
    Scale,
    Stateful,
    Ticks,
    active,
    group,
    step,
)


class BlockingClose(Operator):
    """Signals ``entered`` in ``close`` and waits for ``release``."""

    type = "test/update_blocking_close@v1"
    input_roles = ("input",)
    entered = threading.Event()
    release = threading.Event()

    class Params(OperatorParams):
        pass

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"out": OperatorPort(*inputs[0].kinds)}

    def push(self, arrivals):
        return []

    def finish(self, reason):
        return []

    def close(self) -> None:
        type(self).entered.set()
        assert type(self).release.wait(WAIT)


class RefusingObserver(ExecutionObserver):
    def on_run_started(self, *, session_id, run_id) -> None:
        raise ValueError("observer refuses the run")


CATALOGUE = Catalogue([Stateful, Scale], sources=[Ticks], operators=[BlockingClose])
RESOURCES = {"log": [], "gates": {}, "model": object()}


@pytest.fixture(autouse=True)
def reset_operator_events() -> None:
    BlockingClose.entered = threading.Event()
    BlockingClose.release = threading.Event()


def plans(*, operator: bool) -> Any:
    old = active(
        [step(Stateful, "state", value="$sources.ticks.value")],
        [group("state", value="$steps.state.value")],
        count=0,
    )
    if operator:
        old["operators"] = [
            {
                "name": "cleanup",
                "type": BlockingClose.type,
                "inputs": {"value": "$sources.ticks.value"},
            }
        ]
    new = {
        **old,
        "steps": [*old["steps"], step(Scale, "added", value="$sources.ticks.value")],
        "outputs": [*old["outputs"], group("added", value="$steps.added.scaled")],
    }
    compiled = (
        compile_workflow(old, catalogue=CATALOGUE),
        compile_workflow(new, catalogue=CATALOGUE),
    )

    return compiled


def in_thread(target: Callable[[], Any], outcome: Dict[str, Any]) -> threading.Thread:
    def call() -> None:
        try:
            outcome["result"] = target()
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=call)
    thread.start()

    return thread


def assert_startup_failure(error: Any) -> None:
    assert isinstance(error, ActiveRunError)
    assert error.stage == "start"
    assert isinstance(error.__cause__, ValueError)


def test_failed_start_reserves_the_session_until_operator_cleanup() -> None:
    old, new = plans(operator=True)
    session = old.create_session(RESOURCES, observer=RefusingObserver())
    prepared = session.prepare_update(new)
    started: Dict[str, Any] = {}
    starting = in_thread(session.start, started)
    assert BlockingClose.entered.wait(WAIT)

    with pytest.raises(SessionBusyError, match="unfinished active run"):
        session.apply_update(prepared)
    with pytest.raises(ContractError, match="already has active run"):
        session.start()
    with pytest.raises(ContractError, match="unfinished active run"):
        session.close()
    BlockingClose.release.set()
    starting.join(WAIT)

    assert_startup_failure(started["error"])
    assert prepared.state == PREPARED and not session.closed
    assert session.apply_update(prepared).graph_version == 1
    session.close()


def test_failed_start_finalizer_can_close_the_session() -> None:
    old, new = plans(operator=False)
    session = old.create_session(RESOURCES, observer=RefusingObserver())
    prepared = session.prepare_update(new)
    entered, release = threading.Event(), threading.Event()

    def finalize() -> None:
        entered.set()
        assert release.wait(WAIT)
        session.close()

    started: Dict[str, Any] = {}
    starting = in_thread(lambda: start_session(session, finalize=finalize), started)
    assert entered.wait(WAIT)

    with pytest.raises(SessionBusyError, match="unfinished active run"):
        session.apply_update(prepared)
    with pytest.raises(ContractError, match="already has active run"):
        session.start()
    release.set()
    starting.join(WAIT)

    assert_startup_failure(started["error"])
    assert session.closed
    with pytest.raises(SessionClosedError):
        session.start()


def test_failing_cleanup_still_settles_the_registration() -> None:
    old, new = plans(operator=False)
    session = old.create_session(RESOURCES, observer=RefusingObserver())
    prepared = session.prepare_update(new)

    def finalize() -> None:
        raise RuntimeError("finalize failed")

    with pytest.raises(ActiveRunError) as raised:
        start_session(session, finalize=finalize)

    assert_startup_failure(raised.value)
    assert [type(item.__cause__) for item in raised.value.suppressed] == [RuntimeError]
    assert session.apply_update(prepared).graph_version == 1
    session.close()


@pytest.mark.parametrize("earlier_failure", [False, True])
def test_cleanup_failure_is_recorded_before_run_completion(
    earlier_failure: bool,
) -> None:
    old, new = plans(operator=False)

    class FinishObserver(ExecutionObserver):
        def on_run_finished(self, *, run_id, result, error) -> None:
            if earlier_failure:
                raise ValueError("finish observer failed")

    session = old.create_session(RESOURCES, observer=FinishObserver())
    prepared = session.prepare_update(new)
    entered, release = threading.Event(), threading.Event()

    def finalize() -> None:
        entered.set()
        assert release.wait(WAIT)
        raise RuntimeError("finalize failed")

    run = start_session(session, finalize=finalize)
    try:
        assert entered.wait(WAIT)
        assert run.releasing and not run.done
        with pytest.raises(SessionBusyError):
            session.apply_update(prepared)
    finally:
        release.set()

    with pytest.raises(ActiveRunError) as raised:
        run.wait(WAIT)

    failure = raised.value
    assert run.done and run.state == "failed" and run.failure is failure
    if earlier_failure:
        assert failure.stage == "observer"
        assert isinstance(failure.__cause__, ValueError)
        assert len(failure.suppressed) == 1
        cleanup = failure.suppressed[0]
    else:
        cleanup = failure
    assert cleanup.stage == "finalize"
    assert isinstance(cleanup.__cause__, RuntimeError)
    assert session.apply_update(prepared).graph_version == 1
    session.close()
