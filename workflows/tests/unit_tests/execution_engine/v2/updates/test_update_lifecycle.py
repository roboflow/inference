"""Graph updates against session close, discard and control updates.

``close`` and ``apply_update`` serialize: an update either commits before the
close or is rejected after it, and a busy session refuses to close. A
concurrent ``discard`` or control update never sees half of a commit. A
discarded or stale candidate releases everything only it holds. Every wait is
bounded by ``WAIT`` and ordered by events the test controls.
"""

import gc
import threading
import weakref
from typing import Any, Callable, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import Block, Output
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    SessionClosedError,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.updates import (
    APPLIED,
    DISCARDED,
    PREPARED,
)

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    WAIT,
    Count,
    Hold,
    Scale,
    Stateful,
    Ticks,
    active,
    group,
    passive,
    step,
)


class Payload:
    """A large resource, e.g. a model, observed through a weak reference."""


class UsesPayload(Block):
    """Takes a new ``payload`` resource that only an update provides."""

    type = "test/update_uses_payload@v1"
    outputs = {"value": Output()}

    def __init__(self, *, payload: Payload) -> None:
        self.payload = payload

    def run(self) -> dict:
        return {"value": 1}


LIFECYCLE = Catalogue([Count, Scale, Stateful, Hold, UsesPayload], sources=[Ticks])
STATEFUL = passive(
    [step(Stateful, "state", value="$inputs.value")], {"state": "$steps.state.value"}
)
STATEFUL_AND_SCALE = passive(
    [
        step(Stateful, "state", value="$inputs.value"),
        step(Scale, "scale", value="$inputs.value"),
    ],
    {"state": "$steps.state.value", "scaled": "$steps.scale.scaled"},
)
COUNT = passive(
    [step(Count, "count", value="$inputs.value")], {"count": "$steps.count.count"}
)
COUNT_AND_PAYLOAD = passive(
    [step(Count, "count", value="$inputs.value"), step(UsesPayload, "payload")],
    {"count": "$steps.count.count", "payload": "$steps.payload.value"},
)


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=LIFECYCLE)

    return plan


@pytest.fixture
def stateful() -> Any:
    session = compiled(STATEFUL).create_session({"log": [], "model": object()})

    return session


def blocked(
    target: Callable[..., Any], entered: threading.Event, release: threading.Event
) -> Callable[..., Any]:
    """Wrap ``target`` so it signals ``entered`` and waits for ``release``."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(WAIT)
        return target(*args, **kwargs)

    return wrapper


def in_thread(target: Callable[[], Any], outcome: Dict[str, Any]) -> threading.Thread:
    def call() -> None:
        try:
            outcome["result"] = target()
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=call)
    thread.start()

    return thread


def test_closed_session_rejects_update_run_and_reuse(stateful) -> None:
    prepared = stateful.prepare_update(compiled(STATEFUL_AND_SCALE))

    stateful.close()
    stateful.close()

    with pytest.raises(SessionClosedError):
        stateful.apply_update(prepared)
    with pytest.raises(SessionClosedError):
        stateful.prepare_update(compiled(STATEFUL_AND_SCALE))
    with pytest.raises(SessionClosedError):
        stateful.run({"value": 1.0})
    with pytest.raises(SessionClosedError):
        stateful.pipeline()
    assert prepared.state == PREPARED
    assert stateful.graph_version == 0
    assert stateful.closed


def test_apply_that_holds_the_session_finishes_before_close(stateful) -> None:
    prepared = stateful.prepare_update(compiled(STATEFUL_AND_SCALE))
    entered, release = threading.Event(), threading.Event()
    stateful.controls._rebasing = blocked(stateful.controls._rebasing, entered, release)
    applied: Dict[str, Any] = {}
    applying = in_thread(lambda: stateful.apply_update(prepared), applied)
    assert entered.wait(WAIT)

    closed: Dict[str, Any] = {}
    closing = in_thread(stateful.close, closed)
    closing.join(0.2)
    waited_for_commit = closing.is_alive()
    release.set()
    applying.join(WAIT)
    closing.join(WAIT)

    assert waited_for_commit
    assert applied["result"].graph_version == 1
    assert "error" not in closed
    assert stateful.closed and stateful.graph_version == 1
    assert prepared.state == APPLIED


def test_close_that_holds_the_session_rejects_a_later_apply(stateful) -> None:
    prepared = stateful.prepare_update(compiled(STATEFUL_AND_SCALE))
    entered, release = threading.Event(), threading.Event()
    stateful._busy_reason = blocked(stateful._busy_reason, entered, release)
    closed: Dict[str, Any] = {}
    closing = in_thread(stateful.close, closed)
    assert entered.wait(WAIT)
    del stateful._busy_reason

    applied: Dict[str, Any] = {}
    applying = in_thread(lambda: stateful.apply_update(prepared), applied)
    applying.join(0.2)
    waited_for_close = applying.is_alive()
    release.set()
    closing.join(WAIT)
    applying.join(WAIT)

    assert waited_for_close
    assert isinstance(applied["error"], SessionClosedError)
    assert stateful.closed and stateful.graph_version == 0
    assert prepared.state == PREPARED


def test_busy_session_refuses_to_close_and_keeps_its_state() -> None:
    gates = {"entered": threading.Event(), "release": threading.Event()}
    session = compiled(
        passive(
            [
                step(Stateful, "state", value="$inputs.value"),
                step(Hold, "hold", value="$inputs.value"),
            ],
            {"state": "$steps.state.value", "hold": "$steps.hold.value"},
        )
    ).create_session({"gates": gates})
    ran: Dict[str, Any] = {}
    running = in_thread(lambda: session.run({"value": 1.0}), ran)
    assert gates["entered"].wait(WAIT)

    with pytest.raises(ContractError, match="direct run"):
        session.close()
    gates["release"].set()
    running.join(WAIT)
    session.managed_state.global_.set("after", 1)
    session.close()

    assert ran["result"].rows() == [{"state": 1.0, "hold": 1.0}]
    assert session.closed


def test_active_session_closes_only_after_its_run_and_never_starts_again() -> None:
    log: List[Any] = []
    plan = compiled(
        active(
            [step(Count, "count", value="$sources.ticks.value")],
            [group("counts", count="$steps.count.count")],
            after=1,
        )
    )
    resumed = threading.Event()
    resumed.set()
    gates = {"paused": threading.Event(), "resume": resumed}
    session = plan.create_session({"log": log, "model": object(), "gates": gates})
    stopped = threading.Event()
    inside: Dict[str, Any] = {}

    def on_counts(result: Any) -> None:
        if stopped.is_set():
            return
        try:
            session.close()
        except ContractError as error:
            inside["error"] = error
        session.stop()
        stopped.set()

    run = session.start(handlers={"counts": on_counts})
    assert stopped.wait(WAIT)
    run.wait(WAIT)
    session.close()
    opened = [entry for entry in log if entry[0] == "open"]

    with pytest.raises(SessionClosedError):
        session.start()
    assert "unfinished active run" in str(inside["error"])
    assert [entry for entry in log if entry[0] == "open"] == opened


def test_discard_during_a_commit_waits_and_sees_it_applied(stateful) -> None:
    prepared = stateful.prepare_update(compiled(STATEFUL_AND_SCALE))
    entered, release = threading.Event(), threading.Event()
    stateful.controls._rebasing = blocked(stateful.controls._rebasing, entered, release)
    applied: Dict[str, Any] = {}
    applying = in_thread(lambda: stateful.apply_update(prepared), applied)
    assert entered.wait(WAIT)

    discarded: Dict[str, Any] = {}
    discarding = in_thread(prepared.discard, discarded)
    discarding.join(0.2)
    waited_for_commit = discarding.is_alive()
    release.set()
    applying.join(WAIT)
    discarding.join(WAIT)

    assert waited_for_commit
    assert isinstance(discarded["error"], UpdateConflictError)
    assert prepared.state == APPLIED
    assert stateful.graph_version == 1
    assert ("scale",) in stateful.instances


def test_control_update_during_a_commit_is_kept() -> None:
    definition = passive(
        [step(Scale, "scale", value="$inputs.value", factor="$inputs.factor")],
        {"scaled": "$steps.scale.scaled"},
    )
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "factor", "default_value": 2.0}
    )
    definition["controls"] = {"factor": {"type": "input", "input": "$inputs.factor"}}
    attached = {
        **definition,
        "steps": [*definition["steps"], step(Count, "count", value="$inputs.value")],
    }
    attached["outputs"] = [
        *definition["outputs"],
        {"type": "JsonField", "name": "count", "selector": "$steps.count.count"},
    ]
    session = compiled(definition).create_session({"log": [], "model": object()})
    prepared = session.prepare_update(compiled(attached))
    entered, release = threading.Event(), threading.Event()
    session.controls._rebasing = blocked(session.controls._rebasing, entered, release)
    applying = in_thread(lambda: session.apply_update(prepared), {})
    assert entered.wait(WAIT)

    session.controls.update(factor=3.0)
    release.set()
    applying.join(WAIT)
    result = session.run({"value": 1.0})

    assert result.rows() == [{"scaled": 3.0, "count": 1}]
    assert (result.controls.version, result.graph_version) == (1, 1)


def watched_payloads(created: List[weakref.ref]) -> Factory:
    def create() -> Payload:
        payload = Payload()
        created.append(weakref.ref(payload))
        return payload

    return Factory(create)


def test_discarded_candidate_releases_its_factory_and_caller_values() -> None:
    models: List[weakref.ref] = []
    session = compiled(COUNT).create_session(
        {"log": [], "model": watched_payloads(models)}
    )
    created: List[weakref.ref] = []
    caller_value = Payload()
    caller_ref = weakref.ref(caller_value)
    prepared = session.prepare_update(
        compiled(COUNT_AND_PAYLOAD),
        resources={"payload": watched_payloads(created), "unused": caller_value},
    )
    del caller_value

    prepared.discard()
    gc.collect()

    assert prepared.state == DISCARDED
    assert created[0]() is None and caller_ref() is None
    assert models[0]() is session.instances[("count",)].model


def test_stale_candidate_releases_its_values_and_the_applied_one_keeps_them() -> None:
    session = compiled(COUNT).create_session({"log": [], "model": object()})
    first, second = [], []
    applied = session.prepare_update(
        compiled(COUNT_AND_PAYLOAD), resources={"payload": watched_payloads(first)}
    )
    stale = session.prepare_update(
        compiled(COUNT_AND_PAYLOAD), resources={"payload": watched_payloads(second)}
    )
    session.apply_update(applied)

    with pytest.raises(UpdateConflictError, match="prepare it again"):
        session.apply_update(stale)
    gc.collect()

    assert stale.state == DISCARDED and second[0]() is None
    assert first[0]() is session.instances[("payload",)].payload
