"""Machine controller semantics on memory and a real, owned Redis server.

Machines, transitions, origins and causes are the real compiled types.
``_Plan`` is the only stand-in: a ``ReactionPlan`` with handler-selected
transitions needs compiled handler plans, so ``_Plan`` exposes just
``machines``, ``transitions_for`` and the real ``authorize_setter``. One test
at the end runs the controller on a compiled workflow's ``ReactionPlan``.
"""

import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.blocks import state_machine
from roboflow_workflows.execution_engine.v2.blocks.state_machine import (
    StateMachineSetBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import SampleContext
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
    StepPath,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.reactions.dispatch import EventCause
from roboflow_workflows.execution_engine.v2.reactions.machines import (
    MachineRuntime,
    MachineStamp,
    MachineStateError,
    TransitionCounters,
    TransitionResult,
)
from roboflow_workflows.execution_engine.v2.reactions.plan import (
    EventOrigin,
    PlannedMachine,
    PlannedTransition,
    authorize_setter,
)
from roboflow_workflows.execution_engine.v2.state import (
    InMemoryStateBackend,
    ManagedState,
    StateOutcomeUnknownError,
    StateScopeError,
    StateValueError,
)
from roboflow_workflows.execution_engine.v2.state.codec import (
    INT64_MAX,
    machine_storage_key,
)

from ..state._redis_server import OwnedRedisServer

MACHINE: StepPath = ("inspection",)
HANDLER: StepPath = ("decide",)
REQUESTED = EventOrigin("step", "review_requested", ("camera_check",))
TICK = EventOrigin("step", "tick", ("clock",))


class _Plan:
    """``ReactionPlan`` surface the controller reads, without handler plans."""

    def __init__(self, machines: Tuple[PlannedMachine, ...]) -> None:
        self.machines = machines

    def transitions_for(self, origin: EventOrigin) -> Tuple[PlannedTransition, ...]:
        return tuple(
            t for m in self.machines for t in m.transitions if t.trigger == origin
        )

    def setter(
        self, handler: StepPath, machine_ref: str, transition: str
    ) -> PlannedTransition:
        machines = {machine.path: machine for machine in self.machines}
        return authorize_setter(
            machines, handler=handler, machine_ref=machine_ref, transition=transition
        )


def _origin(machine: StepPath, event: str) -> EventOrigin:
    return EventOrigin("machine", event, machine)


def _review_machine(
    path: StepPath = MACHINE, scope: str = "source", handler: StepPath = HANDLER
) -> PlannedMachine:
    return PlannedMachine(
        path=path,
        scope=scope,
        initial="idle",
        states=("idle", "reviewing", "approved", "rejected"),
        transitions=(
            PlannedTransition(
                "start_review",
                path,
                frozenset({"idle", "approved", "rejected"}),
                ("reviewing",),
                trigger=REQUESTED,
                emits=_origin(path, "decision_requested"),
                fields={
                    "image": ("event", "frame"),
                    "from": ("transition", "from"),
                    "label": ("literal", "review"),
                },
            ),
            PlannedTransition(
                "finish_review",
                path,
                frozenset({"reviewing"}),
                ("approved", "rejected"),
                handler=handler,
                emits=_origin(path, "decided"),
                fields={"state": ("transition", "to"), "via": ("transition", "name")},
            ),
        ),
        events={
            "decision_requested": Event({"image": (), "from": (), "label": ()}),
            "decided": Event({"state": (), "via": ()}),
        },
    )


def _chain_machine() -> PlannedMachine:
    # Two transitions on one trigger with disjoint "from": a -> b, b -> c.
    return PlannedMachine(
        path=("chain",),
        scope="global",
        initial="a",
        states=("a", "b", "c"),
        transitions=(
            PlannedTransition("ab", ("chain",), {"a"}, ("b",), trigger=TICK),
            PlannedTransition("bc", ("chain",), {"b"}, ("c",), trigger=TICK),
        ),
    )


class _Recorder:
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, Dict[str, Any], Any, MachineStamp]] = []

    def __call__(self, origin, fields, *, parent, stamp) -> None:
        self.calls.append((origin, dict(fields), parent, stamp))


@pytest.fixture(scope="module")
def redis_server() -> Iterator[OwnedRedisServer]:
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server is not installed")
    pytest.importorskip("redis")

    directory = Path(tempfile.mkdtemp(prefix="wf2m-", dir="/tmp"))
    server = OwnedRedisServer(directory)
    try:
        server.wait_ready()
        yield server
    finally:
        server.stop()
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(params=["memory", "redis"])
def state(request) -> Iterator[ManagedState]:
    namespace = f"test-{uuid.uuid4().hex}"
    if request.param == "memory":
        managed = ManagedState(InMemoryStateBackend(), namespace=namespace)
    else:
        from roboflow_workflows.execution_engine.v2.state.redis import RedisStateBackend

        server = request.getfixturevalue("redis_server")
        managed = ManagedState(RedisStateBackend(server.url), namespace=namespace)
    try:
        yield managed
    finally:
        managed.backend.close()


def _cause(
    source: Optional[str] = "cam-1",
    *,
    event: str = "review_requested",
    parent: Optional[EventCause] = None,
    stamp: Optional[MachineStamp] = None,
) -> EventCause:
    return EventCause(
        event=event,
        emitter=stamp.machine if stamp is not None else ("camera_check",),
        session_id="session",
        run_id="run",
        pulse=None,
        index=(),
        sample=SampleContext(source) if source is not None else None,
        temporal=None,
        sequence=0,
        parent=parent,
        kind="machine" if stamp is not None else "step",
        stamp=stamp,
    )


def _runtime(
    state: ManagedState, *machines: PlannedMachine
) -> Tuple[MachineRuntime, _Recorder]:
    recorder = _Recorder()
    runtime = MachineRuntime(
        _Plan(machines or (_review_machine(),)), state, emit=recorder
    )
    return runtime, recorder


def test_fixed_transition_applies_once_and_emits_exact_event(state) -> None:
    runtime, recorder = _runtime(state)
    cause = _cause()

    runtime.on_event(REQUESTED, cause, {"frame": "F1", "unused": 1})

    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 1)
    stamp = MachineStamp(MACHINE, "cam-1", "start_review", "idle", "reviewing", 1)
    assert recorder.calls == [
        (
            _origin(MACHINE, "decision_requested"),
            {"image": "F1", "from": "idle", "label": "review"},
            cause,
            stamp,
        )
    ]
    runtime.on_event(REQUESTED, cause, {"frame": "F2"})
    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 1)
    assert len(recorder.calls) == 1
    assert runtime.counters()["inspection.start_review"] == TransitionCounters(1, 1, 0)


def test_one_event_selects_at_most_one_transition_per_machine(state) -> None:
    runtime, _ = _runtime(state, _chain_machine())
    cause = _cause(source=None)

    runtime.on_event(TICK, cause, {})
    assert runtime.current(("chain",)) == ("b", 1)
    runtime.on_event(TICK, cause, {})
    assert runtime.current(("chain",)) == ("c", 2)
    runtime.on_event(TICK, cause, {})

    assert runtime.current(("chain",)) == ("c", 2)
    assert dict(runtime.counters()) == {
        "chain.ab": TransitionCounters(applied=1, ignored=2, stale=0),
        "chain.bc": TransitionCounters(applied=1, ignored=2, stale=0),
    }


def test_machines_are_visited_and_emitted_in_declaration_order(state) -> None:
    first = _review_machine(("child_a", "m"))
    second = _review_machine(("child_b", "m"))
    order: List[str] = []

    def emit(origin, fields, *, parent, stamp) -> None:
        order.append("/".join(stamp.machine))
        # Emission runs without controller locks: re-entering must not block.
        assert runtime.counters() is not None
        runtime.current(stamp.machine, source_id="cam-1")

    runtime = MachineRuntime(_Plan((first, second)), state, emit=emit)
    worker = threading.Thread(
        target=runtime.on_event, args=(REQUESTED, _cause(), {"frame": 0})
    )
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert order == ["child_a/m", "child_b/m"]


def test_scopes_sources_and_nested_identities_are_isolated(state) -> None:
    nested = _review_machine(("child", "inspection"))
    global_machine = _review_machine(("plant",), scope="global")
    runtime, _ = _runtime(state, _review_machine(), nested, global_machine)

    runtime.on_event(REQUESTED, _cause("cam-1"), {"frame": 0})
    runtime.on_event(REQUESTED, _cause("cam-2"), {"frame": 0})
    runtime.on_event(REQUESTED, _cause("cam-2"), {"frame": 0})

    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 1)
    assert runtime.current(MACHINE, source_id="cam-3") == ("idle", 0)
    assert runtime.current(("child", "inspection"), source_id="cam-2") == (
        "reviewing",
        1,
    )
    assert runtime.current(("plant",)) == ("reviewing", 1)
    with pytest.raises(StateScopeError):
        runtime.current(MACHINE)
    with pytest.raises(StateScopeError):
        runtime.current(("plant",), source_id="cam-1")
    with pytest.raises(EventEmissionError):
        runtime.on_event(REQUESTED, _cause(source=None), {"frame": 0})


def test_reserved_record_never_aliases_public_state_keys(state) -> None:
    runtime, _ = _runtime(state, _review_machine(("plant",), scope="global"))
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    for name in (
        "plant",
        "inspection",
        machine_storage_key(state.namespace, None, "plant"),
    ):
        state.global_.set(name, {"state": "approved", "version": 99})
        state.for_source("cam-1").set(name, "x")
        state.global_.delete(name)

    assert runtime.current(("plant",)) == ("reviewing", 1)
    assert state.global_.get("plant") is None


def test_handler_decision_with_matching_stamp_applies_and_emits(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})
    machine_cause = _cause(event="decision_requested", stamp=recorder.calls[0][3])
    handler_cause = _cause(event="inner", parent=machine_cause)

    result = runtime.set_state(
        handler=HANDLER,
        cause=handler_cause,
        machine_ref="inspection",
        transition="finish_review",
        next_state="approved",
    )

    stamp = MachineStamp(MACHINE, "cam-1", "finish_review", "reviewing", "approved", 2)
    assert result == TransitionResult("applied", state="approved", stamp=stamp)
    assert recorder.calls[-1] == (
        _origin(MACHINE, "decided"),
        {"state": "approved", "via": "finish_review"},
        handler_cause,
        stamp,
    )


def test_stale_stamp_after_a_b_a_is_rejected_without_emission(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})
    old_stamp = recorder.calls[0][3]
    first_review = _cause(event="decision_requested", stamp=old_stamp)
    runtime.set_state(
        handler=HANDLER,
        cause=first_review,
        machine_ref="inspection",
        transition="finish_review",
        next_state="rejected",
    )
    runtime.on_event(REQUESTED, _cause(), {"frame": 1})  # reviewing again, v3
    emitted = len(recorder.calls)

    result = runtime.set_state(
        handler=HANDLER,
        cause=first_review,
        machine_ref="inspection",
        transition="finish_review",
        next_state="approved",
    )

    assert result == TransitionResult("stale", state="reviewing")
    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 3)
    assert len(recorder.calls) == emitted
    assert runtime.counters()["inspection.finish_review"] == TransitionCounters(1, 0, 1)


def _finish(runtime: MachineRuntime, cause: EventCause, next_state: str, **kwargs):
    return runtime.set_state(
        handler=kwargs.get("handler", HANDLER),
        cause=cause,
        machine_ref=kwargs.get("machine_ref", "inspection"),
        transition="finish_review",
        next_state=next_state,
    )


def test_foreign_source_stamp_never_selects_its_record(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause("cam-a"), {"frame": 0})
    runtime.on_event(REQUESTED, _cause("cam-b"), {"frame": 0})
    stamp_a = recorder.calls[0][3]
    # Constructed inconsistent chain: a cam-b handler run below a cam-a stamp.
    foreign = _cause("cam-a", event="decision_requested", stamp=stamp_a)
    handler_cause = _cause("cam-b", event="inner", parent=foreign)

    result = _finish(runtime, handler_cause, "approved")

    # No cam-b stamp in the chain: current-state mode on cam-b's record.
    assert result.outcome == "applied"
    assert result.stamp.source_id == "cam-b"
    assert runtime.current(MACHINE, source_id="cam-b") == ("approved", 2)
    assert runtime.current(MACHINE, source_id="cam-a") == ("reviewing", 1)


def test_matching_stamp_behind_a_foreign_stamp_decides(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause("cam-a"), {"frame": 0})
    runtime.on_event(REQUESTED, _cause("cam-b"), {"frame": 0})
    stamp_a, stamp_b = recorder.calls[0][3], recorder.calls[1][3]
    _finish(runtime, _cause("cam-b"), "rejected")  # cam-b: rejected, v2
    runtime.on_event(REQUESTED, _cause("cam-b"), {"frame": 1})  # reviewing, v3
    older = _cause("cam-b", event="decision_requested", stamp=stamp_b)
    nearer = _cause("cam-a", event="decision_requested", stamp=stamp_a, parent=older)
    emitted = len(recorder.calls)

    result = _finish(runtime, _cause("cam-b", event="inner", parent=nearer), "approved")

    # The old cam-b stamp is found past cam-a's and is stale (A -> B -> A).
    assert result == TransitionResult("stale", state="reviewing")
    assert runtime.current(MACHINE, source_id="cam-b") == ("reviewing", 3)
    assert runtime.current(MACHINE, source_id="cam-a") == ("reviewing", 1)
    assert len(recorder.calls) == emitted


def test_global_stamp_is_matched_through_source_causes(state) -> None:
    runtime, recorder = _runtime(state, _review_machine(("plant",), scope="global"))
    runtime.on_event(REQUESTED, _cause("cam-1"), {"frame": 0})
    stamp = recorder.calls[0][3]
    review = _cause("cam-1", event="decision_requested", stamp=stamp)

    applied = _finish(
        runtime,
        _cause("cam-2", event="inner", parent=review),
        "approved",
        machine_ref="plant",
    )
    runtime.on_event(REQUESTED, _cause("cam-3"), {"frame": 1})  # reviewing, v3
    stale = _finish(runtime, review, "rejected", machine_ref="plant")

    assert stamp.source_id is None
    assert applied.outcome == "applied"
    assert applied.stamp == MachineStamp(
        ("plant",), None, "finish_review", "reviewing", "approved", 2
    )
    assert stale == TransitionResult("stale", state="reviewing")
    assert runtime.current(("plant",)) == ("reviewing", 3)


def test_matching_stamp_still_needs_a_legal_from_state(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})
    review = _cause(event="decision_requested", stamp=recorder.calls[0][3])
    approved = _finish(runtime, review, "approved")
    emitted = len(recorder.calls)
    # The decided event's stamp equals the record (approved, v2), but
    # finish_review only leaves "reviewing".
    decided = _cause(event="decided", stamp=approved.stamp)

    result = _finish(runtime, _cause(event="inner", parent=decided), "rejected")

    assert result == TransitionResult("ignored", state="approved")
    assert runtime.current(MACHINE, source_id="cam-1") == ("approved", 2)
    assert len(recorder.calls) == emitted
    assert runtime.counters()["inspection.finish_review"] == TransitionCounters(1, 1, 0)


@pytest.mark.parametrize("source_id", ["", 17, b"cam-1"])
def test_invalid_source_id_is_rejected_before_any_record(state, source_id) -> None:
    proxied, proxy = _interfered(state)
    runtime, _ = _runtime(proxied)
    proxy.used.clear()

    with pytest.raises(StateValueError, match="source id"):
        runtime.current(MACHINE, source_id=source_id)

    assert proxy.used == []
    assert runtime.current(MACHINE, source_id="cam-1") == ("idle", 0)


def test_unstamped_setter_uses_current_state(state) -> None:
    runtime, recorder = _runtime(state)
    plain = _cause()

    ignored = runtime.set_state(
        handler=HANDLER,
        cause=plain,
        machine_ref="inspection",
        transition="finish_review",
        next_state="approved",
    )
    runtime.on_event(REQUESTED, plain, {"frame": 0})
    applied = runtime.set_state(
        handler=HANDLER,
        cause=plain,
        machine_ref="inspection",
        transition="finish_review",
        next_state="approved",
    )

    assert ignored == TransitionResult("ignored", state="idle")
    assert applied.outcome == "applied"
    assert runtime.current(MACHINE, source_id="cam-1") == ("approved", 2)
    with pytest.raises(StateScopeError):
        runtime.set_state(
            handler=HANDLER,
            cause=_cause(source=None),
            machine_ref="inspection",
            transition="finish_review",
            next_state="approved",
        )


def test_unauthorized_handler_and_illegal_target_raise(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    with pytest.raises(ContractError):
        runtime.set_state(
            handler=("intruder",),
            cause=_cause(),
            machine_ref="inspection",
            transition="finish_review",
            next_state="approved",
        )
    with pytest.raises(ContractError, match="not a target"):
        runtime.set_state(
            handler=HANDLER,
            cause=_cause(),
            machine_ref="inspection",
            transition="finish_review",
            next_state="idle",
        )

    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 1)
    assert len(recorder.calls) == 1


class _Interfering:
    """Backend proxy that runs ``before_cas`` once, right before the next CAS."""

    def __init__(self, backend, before_cas=None, cas_result=None, cas_error=None):
        self._backend = backend
        self.before_cas = before_cas
        self.cas_result = cas_result
        self.cas_error = cas_error
        self.cas_calls = 0
        self.used: List[str] = []

    def __getattr__(self, name):
        self.used.append(name)
        return getattr(self._backend, name)

    def compare_and_set(self, key, expected, new):
        self.cas_calls += 1
        if self.before_cas is not None:
            action, self.before_cas = self.before_cas, None
            action()
        if self.cas_error is not None:
            raise self.cas_error
        if self.cas_result is not None:
            return self.cas_result
        return self._backend.compare_and_set(key, expected, new)


def _interfered(state: ManagedState, **kwargs) -> Tuple[ManagedState, _Interfering]:
    proxy = _Interfering(state.backend, **kwargs)
    return ManagedState(proxy, namespace=state.namespace), proxy


def test_lost_cas_re_reads_and_ignores_when_state_moved(state) -> None:
    rival, _ = _runtime(state)
    proxied, proxy = _interfered(
        state,
        before_cas=lambda: rival.on_event(REQUESTED, _cause(), {"frame": "rival"}),
    )
    runtime, recorder = _runtime(proxied)

    runtime.on_event(REQUESTED, _cause(), {"frame": "mine"})

    assert proxy.cas_calls == 1
    assert recorder.calls == []
    assert runtime.current(MACHINE, source_id="cam-1") == ("reviewing", 1)


def test_lost_cas_on_stamped_decision_is_stale_without_retry(state) -> None:
    runtime, recorder = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})
    review = _cause(event="decision_requested", stamp=recorder.calls[0][3])
    rival, _ = _runtime(state)
    proxied, proxy = _interfered(
        state,
        before_cas=lambda: rival.set_state(
            handler=HANDLER,
            cause=review,
            machine_ref="inspection",
            transition="finish_review",
            next_state="rejected",
        ),
    )
    late, late_recorder = _runtime(proxied)

    result = late.set_state(
        handler=HANDLER,
        cause=review,
        machine_ref="inspection",
        transition="finish_review",
        next_state="approved",
    )

    assert result == TransitionResult("stale", state="rejected")
    assert proxy.cas_calls == 1
    assert late_recorder.calls == []


def test_unknown_cas_outcome_propagates_without_retry_or_emission(state) -> None:
    proxied, proxy = _interfered(
        state, cas_error=StateOutcomeUnknownError("lost reply")
    )
    runtime, recorder = _runtime(proxied)

    with pytest.raises(StateOutcomeUnknownError):
        runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    assert proxy.cas_calls == 1
    assert recorder.calls == []
    assert runtime.counters()["inspection.start_review"] == TransitionCounters()


def test_endless_cas_misses_stop_with_explicit_error(state) -> None:
    proxied, proxy = _interfered(state, cas_result=False)
    runtime, recorder = _runtime(proxied)

    with pytest.raises(MachineStateError, match="compare-and-set"):
        runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    assert proxy.cas_calls == 64
    assert recorder.calls == []


def test_concurrent_events_apply_exactly_one_transition(state) -> None:
    runtimes = [_runtime(state) for _ in range(8)]
    barrier = threading.Barrier(len(runtimes))
    errors: List[BaseException] = []

    def fire(runtime: MachineRuntime) -> None:
        try:
            barrier.wait(timeout=10)
            runtime.on_event(REQUESTED, _cause(), {"frame": 0})
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=fire, args=(r,)) for r, _ in runtimes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert errors == []
    assert sum(len(recorder.calls) for _, recorder in runtimes) == 1
    assert runtimes[0][0].current(MACHINE, source_id="cam-1") == ("reviewing", 1)


def test_restart_keeps_record_and_deleted_record_is_explicit(state) -> None:
    runtime, _ = _runtime(state)
    runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    restarted, _ = _runtime(state)
    assert restarted.current(MACHINE, source_id="cam-1") == ("reviewing", 1)

    state.backend.delete(machine_storage_key(state.namespace, "cam-1", "inspection"))
    with pytest.raises(MachineStateError, match="missing"):
        restarted.current(MACHINE, source_id="cam-1")

    new_service = ManagedState(state.backend, namespace=state.namespace)
    resumed, _ = _runtime(new_service)
    assert resumed.current(MACHINE, source_id="cam-1") == ("idle", 0)


@pytest.mark.parametrize(
    "stored",
    [
        "not json",
        '"idle"',
        '{"state":"idle"}',
        '{"state":"unknown","version":1}',
        '{"state":"idle","version":true}',
        '{"state":"idle","version":-1}',
        '{"extra":1,"state":"idle","version":1}',
    ],
)
def test_corrupt_record_raises_explicitly(state, stored: str) -> None:
    runtime, recorder = _runtime(state)
    key = machine_storage_key(state.namespace, "cam-1", "inspection")
    runtime.current(MACHINE, source_id="cam-1")
    state.backend.set(key, stored, only_if_absent=False)

    with pytest.raises(MachineStateError, match="corrupt"):
        runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    assert recorder.calls == []


def test_version_overflow_raises_without_change(state) -> None:
    runtime, recorder = _runtime(state)
    key = machine_storage_key(state.namespace, "cam-1", "inspection")
    runtime.current(MACHINE, source_id="cam-1")
    stored = f'{{"state":"idle","version":{INT64_MAX}}}'
    state.backend.set(key, stored, only_if_absent=False)

    with pytest.raises(MachineStateError, match="64-bit"):
        runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    assert state.backend.get(key) == stored
    assert recorder.calls == []


def test_counters_are_independent_snapshots(state) -> None:
    runtime, _ = _runtime(state)
    before = runtime.counters()

    runtime.on_event(REQUESTED, _cause(), {"frame": 0})

    assert before["inspection.start_review"] == TransitionCounters()
    assert runtime.counters()["inspection.start_review"] == TransitionCounters(1, 0, 0)
    with pytest.raises(TypeError):
        before["x"] = TransitionCounters()  # type: ignore[index]


class _Inspection(Block):
    type = "test/machine_inspection@v1"
    outputs = {"seen": Output(FLOAT_KIND)}
    events = {"review_requested": Event({"frame": FLOAT_KIND})}

    class Params(BlockParams):
        frame: Ref(FLOAT_KIND)

    def run(self, frame):
        return {"seen": frame}


class _Review(Block):
    type = "test/machine_review@v1"
    outputs = {"next_state": Output(STRING_KIND)}

    class Params(BlockParams):
        image: Ref(FLOAT_KIND)

    def run(self, image):
        return {"next_state": "approved"}


def _compiled_plan():
    """Plan of a compiled workflow: draft example 05's review loop, passive."""
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "frame"}],
        "steps": [
            {"type": _Inspection.type, "name": "inspection", "frame": "$inputs.frame"}
        ],
        "outputs": [
            {"type": "JsonField", "name": "seen", "selector": "$steps.inspection.seen"}
        ],
        "state_machines": [
            {
                "name": "inspection_state",
                "scope": "source",
                "initial_state": "idle",
                "states": ["idle", "reviewing", "approved", "rejected"],
                "transitions": [
                    {
                        "name": "start_review",
                        "from": ["idle", "approved", "rejected"],
                        "on": "$steps.inspection.events.review_requested",
                        "to": "reviewing",
                        "emit": {
                            "name": "decision_requested",
                            "fields": {"image": "$event.frame"},
                        },
                    },
                    {
                        "name": "finish_review",
                        "from": ["reviewing"],
                        "handler": "decide",
                        "to": ["approved", "rejected"],
                    },
                ],
            }
        ],
        "handlers": [
            {
                "name": "decide",
                "on": "$state_machines.inspection_state.events.decision_requested",
                "bindings": {"image": "$event.image"},
                "workflow": {
                    "inputs": [{"name": "image", "kind": ["float"]}],
                    "steps": [
                        {
                            "type": _Review.type,
                            "name": "review",
                            "image": "$inputs.image",
                        },
                        {
                            "type": StateMachineSetBlock.type,
                            "name": "save",
                            "machine": "inspection_state",
                            "transition": "finish_review",
                            "next_state": "$steps.review.next_state",
                        },
                    ],
                    "outputs": [],
                },
            }
        ],
    }
    catalogue = Catalogue([_Inspection, _Review, StateMachineSetBlock])

    return compile_workflow(definition, catalogue=catalogue).reactions


def test_compiled_plan_matches_stamps_by_machine_and_source(state) -> None:
    plan = _compiled_plan()
    recorder = _Recorder()
    runtime = MachineRuntime(plan, state, emit=recorder)
    machine = ("inspection_state",)
    trigger = EventOrigin("step", "review_requested", ("inspection",))
    for source in ("cam-a", "cam-b"):
        runtime.on_event(trigger, _cause(source), {"frame": 0.5})
    stamp_a, stamp_b = recorder.calls[0][3], recorder.calls[1][3]
    review_b = _cause("cam-b", event="decision_requested", stamp=stamp_b)

    def finish(cause: EventCause, handler: StepPath = ("decide",)):
        return runtime.set_state(
            handler=handler,
            cause=cause,
            machine_ref="inspection_state",
            transition="finish_review",
            next_state="approved",
        )

    nearer_a = _cause(
        "cam-a", event="decision_requested", stamp=stamp_a, parent=review_b
    )
    applied = finish(_cause("cam-b", event="inner", parent=nearer_a))
    repeated = finish(review_b)

    assert recorder.calls[0][0] == plan.machine(machine).origin("decision_requested")
    assert applied.stamp == MachineStamp(
        machine, "cam-b", "finish_review", "reviewing", "approved", 2
    )
    assert repeated == TransitionResult("stale", state="approved")
    assert runtime.current(machine, source_id="cam-a") == ("reviewing", 1)
    with pytest.raises(ContractError):
        finish(review_b, handler=("intruder",))
    with pytest.raises(StateValueError):
        runtime.current(machine, source_id="")


def test_setter_block_returns_result_fields(monkeypatch) -> None:
    calls = []

    class _Context:
        def set_machine_state(self, machine, transition, next_state):
            calls.append((machine, transition, next_state))
            return TransitionResult("stale", state="rejected")

    monkeypatch.setattr(state_machine, "get_execution_context", _Context)

    outputs = StateMachineSetBlock().run(
        trigger=None,
        machine="inspection",
        transition="finish_review",
        next_state="approved",
    )

    assert calls == [("inspection", "finish_review", "approved")]
    assert outputs == {"state": "rejected", "applied": False, "outcome": "stale"}
