"""``ExecutionSession`` graph updates of an idle session.

An applied update keeps the session, its retained instances with their
block-local state, its managed state and its session resources. A rejected
update leaves every one of them, and the current plan, in use.
"""

import threading
from typing import Any, List

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    IncompatibleUpdateError,
    ResourceError,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.resources import Factory

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    CATALOGUE,
    WAIT,
    Broken,
    Count,
    Scale,
    active,
    group,
    passive,
    step,
)

COUNTER = passive(
    [step(Count, "count", value="$inputs.value")], {"count": "$steps.count.count"}
)
WITH_SCALE = passive(
    [
        step(Count, "count", value="$inputs.value"),
        step(Scale, "scale", value="$inputs.value", factor=10.0),
    ],
    {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"},
)


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


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
    session = compiled(COUNTER).create_session({"log": log, "model": Factory(models)})

    return session


def counts(result) -> int:
    (row,) = result.rows()

    return row["count"]


def test_added_consumer_reuses_counter_state_instance_and_session_factory(
    session, log, models
) -> None:
    counter = session.instances[("count",)]
    first = session.run({"value": 1.0})
    second = session.run({"value": 1.0})

    receipt = session.update(compiled(WITH_SCALE))
    third = session.run({"value": 2.0})

    assert [counts(first), counts(second)] == [1, 2]
    assert third.rows() == [{"count": 3, "scaled": 20.0}]
    assert session.instances[("count",)] is counter
    assert [entry[1] for entry in log] == ["count", "scale"]
    assert len(models.created) == 1
    assert session.instances[("scale",)].model is counter.model
    assert (receipt.previous_version, receipt.graph_version) == (0, 1)
    assert receipt.diff.added == (("scale",),)


def test_results_keep_the_graph_version_and_plan_that_produced_them(session) -> None:
    before = session.run({"value": 1.0})
    old_plan = session.plan

    session.update(compiled(WITH_SCALE))
    after = session.run({"value": 1.0})

    assert (before.graph_version, before.plan) == (0, old_plan)
    assert before.rows() == [{"count": 1}]
    assert (after.graph_version, session.graph_version) == (1, 1)
    assert after.plan is session.plan is not old_plan


def test_breaking_parameter_edit_is_rejected_and_old_graph_keeps_running(
    log, models
) -> None:
    session = compiled(WITH_SCALE).create_session(
        {"log": log, "model": Factory(models)}
    )
    session.run({"value": 1.0})
    edited = WITH_SCALE["steps"][1] | {"factor": 99.0}
    breaking = passive(
        [WITH_SCALE["steps"][0], edited], {"scaled": "$steps.scale.scaled"}
    )

    with pytest.raises(IncompatibleUpdateError, match="params_changed") as raised:
        session.update(compiled(breaking))

    assert raised.value.diff.breaking[0].name == "$steps.scale"
    assert session.graph_version == 0
    assert session.run({"value": 1.0}).rows() == [{"count": 2, "scaled": 10.0}]
    assert len(log) == 2, "nothing new was constructed"


def test_constructor_failure_rejects_before_commit_and_can_repeat(session, log) -> None:
    broken = passive(
        [
            step(Count, "count", value="$inputs.value"),
            step(Broken, "broken", value="$inputs.value"),
        ],
        {"count": "$steps.count.count", "broken": "$steps.broken.value"},
    )

    for _ in range(2):
        with pytest.raises(ResourceError, match="cannot load"):
            session.update(compiled(broken))

    assert session.graph_version == 0
    assert [entry[1] for entry in log] == ["count", "broken", "broken"]
    assert counts(session.run({"value": 1.0})) == 1


def test_a_discarded_candidate_never_reaches_the_session(session, log, models) -> None:
    prepared = session.prepare_update(compiled(WITH_SCALE))
    added = prepared.instances[("scale",)]

    prepared.discard()

    assert prepared.state == "discarded"
    assert prepared.instances == {}
    assert ("scale",) not in session.instances
    with pytest.raises(UpdateConflictError, match="already discarded"):
        session.apply_update(prepared)
    receipt = session.update(compiled(WITH_SCALE))
    assert receipt.graph_version == 1
    assert session.instances[("scale",)] is not added
    assert len(models.created) == 1, "the counter's model was created before both"


def test_stale_and_reused_candidates_conflict(session) -> None:
    first = session.prepare_update(compiled(WITH_SCALE))
    second = session.prepare_update(compiled(WITH_SCALE))

    session.apply_update(first)

    with pytest.raises(UpdateConflictError, match="already applied"):
        session.apply_update(first)
    with pytest.raises(UpdateConflictError, match="already applied"):
        first.discard()
    with pytest.raises(UpdateConflictError, match="graph version 0"):
        session.apply_update(second)
    assert second.state == "discarded"
    assert session.graph_version == 1


def test_candidate_of_another_session_conflicts(session, log, models) -> None:
    other = compiled(COUNTER).create_session({"log": log, "model": Factory(models)})
    prepared = other.prepare_update(compiled(WITH_SCALE))

    with pytest.raises(UpdateConflictError, match="prepared for session"):
        session.apply_update(prepared)

    assert prepared.state == "prepared"
    assert other.apply_update(prepared).graph_version == 1


def test_update_resources_may_add_keys_but_not_replace_them(session) -> None:
    with pytest.raises(ContractError, match="new resource keys only"):
        session.prepare_update(compiled(WITH_SCALE), resources={"log": []})

    prepared = session.prepare_update(compiled(WITH_SCALE), resources={"extra": 1})

    assert prepared.state == "prepared"


def test_new_step_factory_is_created_once_and_joins_the_session(log) -> None:
    created: List[object] = []
    plan = compiled(
        passive([step(Scale, "a", value="$inputs.value")], {"a": "$steps.a.scaled"})
    )
    session = plan.create_session(
        {"log": log, "model": Factory(lambda: created.append(1) or object())}
    )
    first = passive(
        [
            step(Scale, "a", value="$inputs.value"),
            step(Scale, "b", value="$inputs.value"),
        ],
        {"a": "$steps.a.scaled", "b": "$steps.b.scaled"},
    )
    second = passive(
        [
            step(Scale, "a", value="$inputs.value"),
            step(Scale, "b", value="$inputs.value"),
            step(Scale, "c", value="$inputs.value"),
        ],
        {"a": "$steps.a.scaled", "b": "$steps.b.scaled", "c": "$steps.c.scaled"},
    )

    session.update(compiled(first))
    session.update(compiled(second))

    models = {session.instances[(name,)].model for name in "abc"}
    assert len(created) == 1 and len(models) == 1


def test_controls_keep_values_and_version_while_graph_version_moves(
    log, models
) -> None:
    controlled = {
        "inputs": [
            {"type": "WorkflowParameter", "name": "value", "kind": ["float"]},
            {"type": "WorkflowParameter", "name": "factor", "default_value": 2.0},
        ],
        "controls": {"factor": {"type": "input", "input": "$inputs.factor"}},
    }

    def definition(*extra):
        scale = step(Scale, "scale", value="$inputs.value", factor="$inputs.factor")
        outputs = {"scaled": "$steps.scale.scaled"} | {
            name: f"$steps.{name}.count" for name in extra
        }
        steps = [scale, *(step(Count, name, value="$inputs.value") for name in extra)]
        return passive(steps, outputs) | controlled

    session = compiled(definition()).create_session(
        {"log": log, "model": Factory(models)}
    )
    session.controls.update(factor=4.0)
    session.controls.update(factor=5.0)

    session.update(compiled(definition("count")))
    result = session.run({"value": 1.0})

    assert result.rows() == [{"scaled": 5.0, "count": 1}]
    assert (result.controls.version, result.graph_version) == (2, 1)


def test_active_session_attaches_a_group_and_keeps_the_counter(log, models) -> None:
    gates = {"paused": threading.Event(), "resume": threading.Event()}
    counter = [step(Count, "count", value="$sources.ticks.value")]
    counts_only = active(counter, [group("counts", count="$steps.count.count")])
    attached = active(
        [*counter, step(Scale, "scale", value="$sources.ticks.value")],
        [
            group("counts", count="$steps.count.count"),
            group("scaled", scaled="$steps.scale.scaled"),
        ],
    )
    session = compiled(counts_only).create_session(
        {"log": log, "model": Factory(models), "gates": gates}
    )
    delivered: List[Any] = []

    session.start(handlers={"counts": delivered.append}).wait(timeout=WAIT)
    session.update(compiled(attached))
    session.start(
        handlers={"counts": delivered.append, "scaled": delivered.append}
    ).wait(timeout=WAIT)

    counted = [r.rows()[0]["count"] for r in delivered if r.group == "counts"]
    assert counted == [1, 2, 3, 4, 5, 6]
    assert [r.graph_version for r in delivered if r.group == "counts"] == [0] * 3 + [
        1
    ] * 3
    assert {r.graph_version for r in delivered if r.group == "scaled"} == {1}
    assert [entry[1] for entry in log if entry[0] == "init"] == ["count", "scale"]
    assert len(models.created) == 1
