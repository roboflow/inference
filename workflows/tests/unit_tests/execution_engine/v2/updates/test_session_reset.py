"""Explicit processing reset of an idle ``ExecutionSession``.

A reset constructs every step and handler session again, in a fresh
resolver: block-local state restarts, ``Factory`` values are created again,
caller values are passed again. Engine-owned managed state starts fresh;
a caller's is kept or explicitly replaced, never cleared. The control panel
object stays. A failed, discarded or stale candidate leaves the session as
it was and closes only state it created.
"""

import gc
import threading
import weakref
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    IncompatibleUpdateError,
    ResourceError,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.execution_engine.v2.reactions import runtime as reactions
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.state import InMemoryStateBackend
from roboflow_workflows.execution_engine.v2.state import session as state_session
from roboflow_workflows.execution_engine.v2.state.api import ManagedState

from tests.unit_tests.execution_engine.v2.m7_controls.blocks import (
    Thresholder,
    Tracker,
    controls,
    enable,
    input_control,
)
from tests.unit_tests.execution_engine.v2.updates.blocks import (
    WAIT,
    Broken,
    Count,
    Scale,
    Ticks,
    active,
    child,
    group,
    nested,
    passive,
    step,
)


class Tally(Block):
    """Counts its calls in managed state under the global key ``n``."""

    type = "test/reset_tally@v1"
    outputs = {"n": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, managed_state: Any) -> None:
        self.state = managed_state

    def run(self, *, value) -> dict:
        return {"n": self.state.global_.incr("n")}


class Ping(Block):
    """Emits ``entered`` with its value."""

    type = "test/reset_ping@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"entered": Event({"value": FLOAT_KIND})}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.emit("entered", value=value)
        return {"value": value}


class Slow(Block):
    """Its constructor waits on ``gates["release"]`` after setting ``entered``."""

    type = "test/reset_slow@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, gates: Dict[str, threading.Event]) -> None:
        gates["entered"].set()
        assert gates["release"].wait(WAIT)

    def run(self, *, value) -> dict:
        return {"value": value}


CATALOGUE = Catalogue(
    [Count, Scale, Broken, Tally, Ping, Slow, Tracker, Thresholder], sources=[Ticks]
)
OUTPUTS = {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"}


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def scaled(factor: float = 10.0) -> dict:
    steps = [
        step(Count, "count", value="$inputs.value"),
        step(Scale, "scale", value="$inputs.value", factor=factor),
    ]

    return passive(steps, OUTPUTS)


class Models:
    """Session ``Factory`` that counts how often it creates the model."""

    def __init__(self) -> None:
        self.created: List[object] = []

    def __call__(self) -> object:
        model = object()
        self.created.append(model)
        return model


@pytest.fixture
def log() -> List[Any]:
    return []


@pytest.fixture
def models() -> Models:
    return Models()


@pytest.fixture
def session(log, models) -> Any:
    session = compiled(scaled()).create_session({"log": log, "model": Factory(models)})

    return session


def row(result) -> Dict[str, Any]:
    (only,) = result.rows()

    return only


def test_reset_constructs_every_step_again_also_an_unchanged_one(
    session, log, models
) -> None:
    old_count = session.instances[("count",)]
    session.run({"value": 1.0})
    session.run({"value": 1.0})
    log.clear()

    receipt = session.update(compiled(scaled(factor=3.0)), reset=True)
    after = session.run({"value": 1.0})

    assert row(after) == {"count": 1, "scaled": 3.0}
    assert session.instances[("count",)] is not old_count
    assert [entry[1] for entry in log] == ["count", "scale"]
    assert len(models.created) == 2
    assert session.instances[("count",)].model is session.instances[("scale",)].model
    assert (receipt.graph_version, receipt.processing_version, receipt.reset) == (
        1,
        1,
        True,
    )
    assert receipt.cleanup.state == "done" and receipt.cleanup_failures == ()
    assert (after.graph_version, after.processing_version) == (1, 1)


def test_reset_of_a_compatible_plan_then_a_preserving_update_keeps_processing(
    session, log
) -> None:
    session.run({"value": 1.0})
    session.update(compiled(scaled()), reset=True)
    reset_count = session.instances[("count",)]
    session.run({"value": 1.0})

    with_extra = scaled()
    with_extra["steps"].append(step(Count, "extra", value="$inputs.value"))
    with_extra["outputs"].append(
        {"type": "JsonField", "name": "extra", "selector": "$steps.extra.count"}
    )
    receipt = session.update(compiled(with_extra))
    after = session.run({"value": 1.0})

    assert (receipt.reset, receipt.processing_version, receipt.graph_version) == (
        False,
        1,
        2,
    )
    assert session.instances[("count",)] is reset_count
    assert row(after) == {"count": 2, "scaled": 10.0, "extra": 1}


def test_reset_removes_steps_and_rebuilds_nested_ones(log, models) -> None:
    inputs = [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}]
    inner = child(
        [
            step(Count, "inner_count", value="$inputs.v"),
            step(Scale, "inner_scale", value="$inputs.v", factor=4.0),
        ],
        {"n": "$steps.inner_count.count", "s": "$steps.inner_scale.scaled"},
        inputs,
    )
    smaller = child(
        [step(Count, "inner_count", value="$inputs.v")],
        {"n": "$steps.inner_count.count"},
        inputs,
    )
    outer = passive(
        [
            step(Count, "count", value="$inputs.value"),
            nested("inner", inner, v="$inputs.value"),
        ],
        {"count": "$steps.count.count", "n": "$steps.inner.n", "s": "$steps.inner.s"},
    )
    reduced = passive(
        [nested("inner", smaller, v="$inputs.value")], {"n": "$steps.inner.n"}
    )
    session = compiled(outer).create_session({"log": log, "model": Factory(models)})
    old_inner = session.instances[("inner", "inner_count")]
    session.run({"value": 1.0})

    session.update(compiled(reduced), reset=True)
    after = session.run({"value": 1.0})

    assert set(session.instances) == {("inner", "inner_count")}
    assert session.instances[("inner", "inner_count")] is not old_inner
    assert row(after) == {"n": 1}


def test_preparation_failure_keeps_the_old_processing_and_closes_new_state(
    monkeypatch,
) -> None:
    closed: List[Any] = []

    class Tracked(ManagedState):
        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(state_session, "ManagedState", Tracked)
    definition = passive(
        [step(Tally, "tally", value="$inputs.value")], {"n": "$steps.tally.n"}
    )
    session = compiled(definition).create_session()
    session.run({"value": 1.0})
    broken = passive(
        [
            step(Tally, "tally", value="$inputs.value"),
            step(Broken, "broken", value="$inputs.value"),
        ],
        {"n": "$steps.tally.n", "b": "$steps.broken.value"},
    )

    with pytest.raises(ResourceError, match="cannot load"):
        session.prepare_update(compiled(broken), reset=True, resources={"log": []})

    assert len(closed) == 1 and closed[0] is not session.owned_state
    assert (session.graph_version, session.processing_version) == (0, 0)
    assert row(session.run({"value": 1.0})) == {"n": 2}


# Managed state -----------------------------------------------------------------


def tally(**sections: Any) -> dict:
    definition = passive(
        [step(Tally, "tally", value="$inputs.value")],
        {"n": "$steps.tally.n"},
        **sections,
    )

    return definition


class SpyBackend(InMemoryStateBackend):
    """Counts deletes and closes; never expected from the engine here."""

    def __init__(self) -> None:
        super().__init__()
        self.deletes = 0
        self.closes = 0

    def delete(self, key: str) -> bool:
        self.deletes += 1
        return super().delete(key)

    def close(self) -> None:
        self.closes += 1
        super().close()


def test_engine_owned_state_starts_fresh_and_the_old_one_closes() -> None:
    session = compiled(tally(state={"global": {"n": 10}})).create_session()
    old_owned = session.owned_state
    session.run({"value": 1.0})

    session.update(compiled(tally(state={"global": {"n": 100}})), reset=True)

    assert row(session.run({"value": 1.0})) == {"n": 101}
    assert session.owned_state is not old_owned
    assert old_owned.backend._closed


def test_caller_state_is_retained_with_its_values_and_never_closed() -> None:
    backend = SpyBackend()
    service = ManagedState(backend, namespace="line-1")
    session = compiled(tally()).create_session({"managed_state": service})
    session.run({"value": 1.0})
    session.run({"value": 1.0})

    receipt = session.update(compiled(tally()), reset=True)

    assert row(session.run({"value": 1.0})) == {"n": 3}
    assert session.managed_state is service
    assert (backend.deletes, backend.closes, receipt.cleanup_failures) == (0, 0, ())
    assert receipt.processing_version == 1


def test_an_isolated_replacement_starts_at_defaults_and_leaves_old_keys() -> None:
    backend = SpyBackend()
    old = ManagedState(backend, namespace="line-1")
    session = compiled(tally(state={"global": {"n": 10}})).create_session(
        {"managed_state": old}
    )
    session.run({"value": 1.0})
    replacement = ManagedState(backend, namespace="line-2")

    session.update(
        compiled(tally(state={"global": {"n": 50}})),
        reset=True,
        resources={"managed_state": replacement},
    )

    assert row(session.run({"value": 1.0})) == {"n": 51}
    assert old.global_.get("n") == 11
    assert (backend.deletes, backend.closes) == (0, 0)
    assert session.owned_state is None


def test_handler_sessions_and_the_passive_reaction_runtime_are_replaced() -> None:
    seen: List[Any] = []

    def handler(factor: float) -> dict:
        return {
            "name": "react",
            "on": "$steps.ping.events.entered",
            "bindings": {"value": "$event.value"},
            "workflow": {
                "inputs": [{"name": "value", "kind": ["float"]}],
                "steps": [step(Scale, "scale", value="$inputs.value", factor=factor)],
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": "s",
                        "selector": "$steps.scale.scaled",
                    }
                ],
            },
        }

    def definition(factor: float) -> dict:
        return passive(
            [step(Ping, "ping", value="$inputs.value")],
            {"value": "$steps.ping.value"},
            handlers=[handler(factor)],
        )

    session = compiled(definition(2.0)).create_session({"log": seen, "model": None})
    session.run({"value": 1.0})
    old_handler_session = session.handler_sessions[("react",)]
    old_runtime = reactions.session_reactions(session)

    session.update(compiled(definition(3.0)), reset=True)
    session.run({"value": 1.0})

    assert session.handler_sessions[("react",)] is not old_handler_session
    assert old_runtime._state == "closed"
    assert reactions.session_reactions(session) is not old_runtime
    assert [entry[1] for entry in seen if entry[0] == "init"] == ["scale", "scale"]


# Controls ----------------------------------------------------------------------


def controlled(threshold_default: float = 0.5, extra: bool = False) -> dict:
    declared = {
        "analysis": enable(
            "$steps.pre",
            "$steps.tracker",
            state="reset_on_enable",
            suspends_effects=True,
        ),
        "threshold": input_control("threshold", default=threshold_default),
    }
    if extra:
        declared["gain"] = input_control("gain", default=1.0)
    definition = passive(
        [
            step(
                Thresholder, "pre", value="$inputs.value", threshold="$inputs.threshold"
            ),
            step(Tracker, "tracker", value="$steps.pre.threshold"),
        ],
        {"above": "$steps.pre.above", "ticks": "$steps.tracker.ticks"},
        controls=controls(**declared),
    )
    # Both plans declare both inputs: a reset keeps the input schema.
    for name in ("threshold", "gain"):
        definition["inputs"].append(
            {
                "type": "WorkflowParameter",
                "name": name,
                "kind": ["float"],
                "default_value": 0.5,
            }
        )

    return definition


def test_saved_panel_keeps_values_current_at_commit_and_resets_epochs() -> None:
    session = compiled(controlled()).create_session()
    panel = session.controls
    panel.update(threshold=0.7)
    panel.update(analysis=False)
    panel.update(analysis=True)

    update = session.prepare_update(compiled(controlled(extra=True)), reset=True)
    panel.update(threshold=0.9)
    version = panel.version
    session.apply_update(update)
    result = session.run({"value": 1.0})

    assert session.controls is panel
    assert panel.version == version + 1
    assert dict(panel.current.values) == {"threshold": 0.9, "gain": 1.0}
    assert dict(panel.current.enabled) == {"analysis": True}
    assert dict(panel.current.epochs) == {"analysis": 0}
    assert session.reset_epoch(("tracker",)) == 0
    assert session.instances[("tracker",)].resets == 0
    assert row(result)["ticks"] == 1
    assert update.assessment.reset.controls_carried == ("analysis", "threshold")
    assert update.assessment.reset.controls_initialized == ("gain",)


def test_a_changed_control_declaration_starts_at_its_new_default() -> None:
    session = compiled(controlled()).create_session()
    session.controls.update(threshold=0.7)

    session.update(compiled(controlled(threshold_default=0.2)), reset=True)

    assert dict(session.controls.current.values) == {"threshold": 0.2}


# Candidate lifecycle -----------------------------------------------------------


def test_a_stale_reset_candidate_is_discarded_and_closes_its_state() -> None:
    session = compiled(tally()).create_session()
    first = session.prepare_update(compiled(tally()), reset=True)
    second = session.prepare_update(compiled(tally()), reset=True)
    second_state = second._reset_parts.owned_state

    session.apply_update(first)

    with pytest.raises(UpdateConflictError, match="prepare it again"):
        session.apply_update(second)
    assert second.state == "discarded"
    assert second_state.backend._closed
    assert not session.owned_state.backend._closed


def test_discarding_a_reset_candidate_closes_its_state_once(monkeypatch) -> None:
    closes: List[Any] = []
    session = compiled(tally()).create_session()
    update = session.prepare_update(compiled(tally()), reset=True)
    owned = update._reset_parts.owned_state
    monkeypatch.setattr(owned, "close", lambda: closes.append(owned))

    update.discard()
    update.discard()

    assert closes == [owned]
    assert session.processing_version == 0
    assert row(session.run({"value": 1.0})) == {"n": 1}


def test_overlapping_reset_preparation_is_refused() -> None:
    gates = {"entered": threading.Event(), "release": threading.Event()}
    session = compiled(scaled()).create_session(
        {"log": [], "model": None, "gates": gates}
    )
    slow = passive(
        [step(Slow, "slow", value="$inputs.value")], {"v": "$steps.slow.value"}
    )
    prepared: List[Any] = []
    worker = threading.Thread(
        target=lambda: prepared.append(
            session.prepare_update(compiled(slow), reset=True)
        )
    )
    worker.start()
    assert gates["entered"].wait(WAIT)

    try:
        with pytest.raises(UpdateConflictError, match="being prepared"):
            session.prepare_update(compiled(scaled()), reset=True)
    finally:
        gates["release"].set()
        worker.join(WAIT)

    assert prepared[0].reset
    prepared[0].discard()
    assert session.prepare_update(compiled(scaled()), reset=True).reset


def test_applied_candidates_and_receipts_keep_no_old_generation_alive(
    session,
) -> None:
    kept: List[Any] = []
    retired: List[weakref.ref] = []

    for factor in (2.0, 3.0, 4.0):
        retired.append(weakref.ref(session.instances[("count",)]))
        update = session.prepare_update(compiled(scaled(factor)), reset=True)
        kept.append((update, session.apply_update(update)))
    gc.collect()

    assert [ref() for ref in retired] == [None, None, None]
    assert all(update.instances == {} for update, _ in kept)
    assert [receipt.processing_version for _, receipt in kept] == [1, 2, 3]


def test_a_cleanup_failure_is_reported_and_the_reset_still_applies(
    monkeypatch,
) -> None:
    session = compiled(tally()).create_session()

    def fail() -> None:
        raise RuntimeError("backend gone")

    monkeypatch.setattr(session.owned_state, "close", fail)

    receipt = session.update(compiled(tally()), reset=True)

    assert receipt.cleanup.state == "failed"
    assert receipt.cleanup_failures == (
        "closing the old managed state: RuntimeError: backend gone",
    )
    assert session.processing_version == 1
    assert row(session.run({"value": 1.0})) == {"n": 1}


# Sources -----------------------------------------------------------------------


def test_sources_keep_their_resources_and_a_running_run_resets_live(
    log, models
) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    counted = active(
        [step(Count, "count", value="$sources.ticks.value")],
        [group("counts", count="$steps.count.count")],
        after=1,
    )
    scaled_counts = active(
        [step(Scale, "count", value="$sources.ticks.value", factor=2.0)],
        [group("counts", count="$steps.count.scaled")],
        after=1,
    )
    session = compiled(counted).create_session(
        {"log": log, "model": Factory(models), "gates": gates}
    )
    source_resources = dict(session.source_resources)
    delivered: List[Any] = []
    run = session.start(handlers={"counts": delivered.append})
    assert gates["paused"].wait(WAIT)
    update = session.prepare_update(compiled(scaled_counts), reset=True)

    receipt = run.apply_update(update)
    gates["resume"].set()
    assert run.wait(timeout=WAIT)

    assert receipt.reset and receipt.processing_version == 1
    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    assert receipt.called_at <= receipt.cut_at <= receipt.drained_at
    assert receipt.drained_at <= receipt.resumed_at
    assert dict(session.source_resources) == source_resources
    assert [entry[0] for entry in log if entry[1] == "ticks"] == ["open", "close"]
    assert [(r.processing_version, r.rows()[0]["count"]) for r in delivered] == [
        (0, 1),
        (1, 4.0),
        (1, 6.0),
    ]


def test_a_session_factory_shared_with_a_source_rules_a_reset_out(log) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    counted = active(
        [step(Count, "count", value="$sources.ticks.value")],
        [group("counts", count="$steps.count.count")],
    )
    session = compiled(counted).create_session(
        {"log": Factory(list), "gates": gates, "model": Factory(object)}
    )

    assessment = session.assess_update(compiled(counted))
    shared = session.source_resources["ticks"]["log"].value
    allowed = session.assess_update(compiled(counted), resources={"log": shared})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_shared", "provided:log")
    ]
    assert "session.source_resources['ticks']['log'].value" in (
        assessment.blockers[0].detail
    )
    assert allowed.blockers == ()
    session.update(compiled(counted), reset=True, resources={"log": shared})
    assert [entry[:2] for entry in shared] == [("init", "count")] * 2


def test_overriding_a_value_a_source_keeps_rules_a_reset_out(log, models) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    counted = active(
        [step(Count, "count", value="$sources.ticks.value")],
        [group("counts", count="$steps.count.count")],
    )
    session = compiled(counted).create_session(
        {"log": log, "gates": gates, "model": Factory(models)}
    )

    with pytest.raises(IncompatibleUpdateError, match="source_resource_overridden"):
        session.prepare_update(compiled(counted), reset=True, resources={"log": []})
    kept = session.assess_update(compiled(counted), resources={"log": log})

    assert kept.blockers == ()


class ExternalStore:
    """A backend the engine cannot see into, like a Redis client."""

    def __init__(self) -> None:
        self._inner = InMemoryStateBackend()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.parametrize(
    "replacement, kind",
    [
        (lambda old: ManagedState(SpyBackend(), namespace="line-1"), "reset"),
        (lambda old: ManagedState(old.backend, namespace="line-2"), "reset"),
        (lambda old: ManagedState(ExternalStore(), namespace="line-2"), "reset"),
        (lambda old: ManagedState(ExternalStore(), namespace="line-1"), "unsupported"),
    ],
    ids=["private-memory", "new-namespace", "external-new-ns", "external-same-ns"],
)
def test_a_replacement_state_counts_as_isolated_only_when_known(
    replacement, kind
) -> None:
    old = ManagedState(SpyBackend(), namespace="line-1")
    session = compiled(tally()).create_session({"managed_state": old})

    assessment = session.assess_update(
        compiled(tally(state={"global": {"n": 5}})),
        resources={"managed_state": replacement(old)},
    )

    assert assessment.kind == kind
    reasons = [blocker.reason for blocker in assessment.blockers]
    assert reasons == ([] if kind == "reset" else ["state_isolation_unknown"])


def test_a_running_run_resets_a_nested_workflow(log, models) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    inputs = [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}]
    counting = child(
        [step(Count, "inner", value="$inputs.v")], {"n": "$steps.inner.count"}, inputs
    )
    scaling = child(
        [step(Scale, "inner", value="$inputs.v", factor=3.0)],
        {"n": "$steps.inner.scaled"},
        inputs,
    )

    def boxed(inner: dict) -> dict:
        return active(
            [nested("box", inner, v="$sources.ticks.value")],
            [group("counts", count="$steps.box.n")],
            after=1,
        )

    session = compiled(boxed(counting)).create_session(
        {"log": log, "model": Factory(models), "gates": gates}
    )
    delivered: List[Any] = []
    run = session.start(handlers={"counts": delivered.append})
    assert gates["paused"].wait(WAIT)
    old_inner = session.instances[("box", "inner")]

    receipt = run.apply_update(
        session.prepare_update(compiled(boxed(scaling)), reset=True)
    )
    gates["resume"].set()
    assert run.wait(timeout=WAIT)

    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    assert session.instances[("box", "inner")] is not old_inner
    assert [(r.processing_version, r.rows()[0]["count"]) for r in delivered] == [
        (0, 1),
        (1, 6.0),
        (1, 9.0),
    ]


# Storage and resource sharing that a reset must not split ---------------------


def test_a_replacement_on_the_engine_owned_backend_is_refused_before_building() -> None:
    session = compiled(tally(state={"global": {"n": 10}})).create_session()
    session.run({"value": 1.0})
    owned_backend = session.owned_state.backend
    borrowing = ManagedState(owned_backend, namespace="replacement")
    new = compiled(tally(state={"global": {"n": 50}}))

    assessment = session.assess_update(new, resources={"managed_state": borrowing})
    with pytest.raises(IncompatibleUpdateError, match="state_backend_closes"):
        session.prepare_update(new, reset=True, resources={"managed_state": borrowing})

    assert assessment.kind == "unsupported" and assessment.reset is None
    assert [blocker.reason for blocker in assessment.blockers] == [
        "state_backend_closes"
    ]
    assert row(session.run({"value": 1.0})) == {"n": 12}
    assert not owned_backend._closed
    assert borrowing.global_.get("n") is None  # nothing was seeded


def test_a_factory_service_on_the_engine_owned_backend_fails_preparation() -> None:
    session = compiled(tally()).create_session()
    session.run({"value": 1.0})
    owned_backend = session.owned_state.backend
    providing = Catalogue(
        [Tally],
        kinds=[FLOAT_KIND],
        providers={
            "managed_state": Factory(
                lambda: ManagedState(owned_backend, namespace="from-factory")
            )
        },
    )
    new = compile_workflow(tally(), catalogue=providing)

    with pytest.raises(ResourceError, match="backend of the engine-owned state"):
        session.prepare_update(new, reset=True)

    assert row(session.run({"value": 1.0})) == {"n": 2}
    assert not owned_backend._closed


def test_a_caller_backend_serves_a_new_namespace_on_every_reset() -> None:
    backend = SpyBackend()
    session = compiled(tally(state={"global": {"n": 0}})).create_session(
        {"managed_state": ManagedState(backend, namespace="g0")}
    )
    session.run({"value": 1.0})

    for generation in (1, 2, 3):
        receipt = session.update(
            compiled(tally(state={"global": {"n": 10 * generation}})),
            reset=True,
            resources={
                "managed_state": ManagedState(backend, namespace=f"g{generation}")
            },
        )
        assert receipt.cleanup.state == "done"
        assert row(session.run({"value": 1.0})) == {"n": 10 * generation + 1}

    assert (backend.closes, backend.deletes) == (0, 0)
    assert ManagedState(backend, namespace="g0").global_.get("n") == 1


def test_each_engine_owned_state_closes_once_and_never_the_current_one(
    monkeypatch,
) -> None:
    session = compiled(tally()).create_session()
    closes: Dict[int, int] = {}
    owned: List[Any] = []

    def counted(state: Any) -> None:
        close = state.close

        def counting() -> None:
            closes[id(state)] = closes.get(id(state), 0) + 1
            close()

        monkeypatch.setattr(state, "close", counting)
        owned.append(state)

    counted(session.owned_state)
    for _ in range(3):
        session.update(compiled(tally()), reset=True)
        counted(session.owned_state)
        assert row(session.run({"value": 1.0})) == {"n": 1}

    assert [closes.get(id(state), 0) for state in owned] == [1, 1, 1, 0]
    assert not session.owned_state.backend._closed


DEMO = Catalogue([Count], sources=[Ticks], namespace="demo")


def counted_in(catalogue: Catalogue) -> Any:
    definition = active(
        [step(Count, "count", value="$sources.ticks.value")],
        [group("counts", count="$steps.count.count")],
    )
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def test_a_scoped_reset_value_cannot_shadow_what_a_source_shares(log) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    plan = counted_in(DEMO)
    session = plan.create_session({"log": log, "gates": gates, "model": object()})

    assessment = session.assess_update(plan, resources={"demo.log": []})
    with pytest.raises(IncompatibleUpdateError, match="source_resource_overridden"):
        session.prepare_update(plan, reset=True, resources={"demo.log": []})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_overridden", "provided:log")
    ]
    assert "provided:demo.log" in assessment.blockers[0].detail
    assert session.resources[("count",)]["log"].value is log
    assert session.source_resources["ticks"]["log"].value is log


def test_a_caller_value_cannot_shadow_a_catalogue_value_a_source_shares() -> None:
    shared = []
    catalogue = Catalogue(
        [Count], sources=[Ticks], namespace="demo", providers={"log": shared}
    )
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    plan = counted_in(catalogue)
    session = plan.create_session({"gates": gates, "model": object()})
    assert session.resources[("count",)]["log"].value is shared

    assessment = session.assess_update(plan, resources={"log": []})
    kept = session.assess_update(plan, resources={"log": shared})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_overridden", "catalogue:demo.log")
    ]
    assert kept.blockers == () and kept.reset is not None


def test_a_processing_only_resource_can_still_be_replaced(log) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    plan = counted_in(DEMO)
    session = plan.create_session({"log": log, "gates": gates, "model": object()})
    replacement = object()

    assessment = session.assess_update(plan, resources={"model": replacement})
    receipt = session.update(plan, reset=True, resources={"model": replacement})

    assert assessment.blockers == () and assessment.kind == "preserve"
    assert receipt.processing_version == 1
    assert session.resources[("count",)]["model"].value is replacement
    assert session.resources[("count",)]["log"].value is log


# The source resolves ``log``; the step in ``demo`` resolves ``demo.log``.
SPLIT_DEMO = Catalogue.merge(
    Catalogue([Count], namespace="demo"), Catalogue(sources=[Ticks])
)


def test_one_value_under_two_keys_is_shared_with_the_source(log) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    plan = counted_in(SPLIT_DEMO)
    session = plan.create_session(
        {"log": log, "demo.log": log, "gates": gates, "model": object()}
    )
    assert session.resources[("count",)]["log"].source == "provided:demo.log"

    assessment = session.assess_update(plan, resources={"demo.log": []})
    with pytest.raises(IncompatibleUpdateError, match="source_resource_overridden"):
        session.prepare_update(plan, reset=True, resources={"demo.log": []})
    session.update(plan, reset=True, resources={"demo.log": log})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_overridden", "provided:demo.log")
    ]
    assert session.resources[("count",)]["log"].value is log
    assert session.source_resources["ticks"]["log"].value is log


def test_a_different_value_under_another_key_can_still_be_replaced(log) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    plan = counted_in(SPLIT_DEMO)
    session = plan.create_session(
        {"log": log, "demo.log": [], "gates": gates, "model": object()}
    )
    replacement = []

    assessment = session.assess_update(plan, resources={"demo.log": replacement})
    session.update(plan, reset=True, resources={"demo.log": replacement})

    assert assessment.blockers == () and assessment.reset is not None
    assert session.resources[("count",)]["log"].value is replacement
    assert session.source_resources["ticks"]["log"].value is log


def test_a_running_run_refuses_the_owned_backend_and_resets_fresh_instead(
    log,
) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    tallied = active(
        [step(Tally, "tally", value="$sources.ticks.value")],
        [group("counts", n="$steps.tally.n")],
        after=1,
    )
    session = compiled(tallied).create_session({"log": log, "gates": gates})
    old_backend = session.owned_state.backend
    delivered: List[Any] = []
    run = session.start(handlers={"counts": delivered.append})
    assert gates["paused"].wait(WAIT)
    borrowing = {"managed_state": ManagedState(old_backend, namespace="other")}

    with pytest.raises(IncompatibleUpdateError, match="state_backend_closes"):
        session.prepare_update(compiled(tallied), reset=True, resources=borrowing)
    receipt = run.apply_update(session.prepare_update(compiled(tallied), reset=True))
    gates["resume"].set()
    assert run.wait(timeout=WAIT)
    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    again: List[Any] = []
    assert session.start(handlers={"counts": again.append}).wait(timeout=WAIT)

    assert [(r.processing_version, r.rows()[0]["n"]) for r in delivered] == [
        (0, 1),
        (1, 1),
        (1, 2),
    ]
    assert [r.rows()[0]["n"] for r in again] == [3, 4, 5]
    assert old_backend._closed and not session.owned_state.backend._closed
