"""Classification of updates and the pure ``ExecutionSession.assess_update``.

Every change says the least disruptive update that applies it: a
preserving update, a reset, or no update at all. The assessment adds what only the session knows, its
managed state and resources, and describes what a reset would replace. It
builds nothing, so a UI may call it for every edit.
"""

import json
from typing import Any, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    IncompatibleUpdateError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.sources import (
    Source,
    SourceOutput,
    SourceParams,
)
from roboflow_workflows.execution_engine.v2.state import InMemoryStateBackend
from roboflow_workflows.execution_engine.v2.state.api import ManagedState
from roboflow_workflows.execution_engine.v2.updates import (
    PRESERVE,
    RESET,
    UNSUPPORTED,
    PlanChange,
    compare_plans,
)

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    Broken,
    Count,
    Label,
    Scale,
    Stateful,
    Ticks,
    active,
    child,
    group,
    nested,
    passive,
    step,
)


class StatefulTicks(Source):
    """A source that asks for managed state."""

    type = "test/reset_stateful_ticks@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        pass

    def __init__(self, *, managed_state: Any) -> None:
        self.state = managed_state

    def open(self) -> None:
        pass

    def read(self):
        return None

    def close(self) -> None:
        pass


CATALOGUE = Catalogue(
    [Count, Scale, Label, Broken, Stateful], sources=[Ticks, StatefulTicks]
)
COUNTER = [step(Count, "count", value="$inputs.value")]
SCALED = [*COUNTER, step(Scale, "scale", value="$inputs.value", factor=10.0)]
OUTPUTS = {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"}


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def scaled(factor: float = 10.0, **sections: Any) -> dict:
    steps = [*COUNTER, step(Scale, "scale", value="$inputs.value", factor=factor)]

    return passive(steps, OUTPUTS, **sections)


@pytest.fixture
def log() -> List[Any]:
    return []


@pytest.fixture
def created() -> List[object]:
    return []


@pytest.fixture
def session(log, created) -> Any:
    model = Factory(lambda: created.append(object()) or created[-1])
    session = compiled(scaled()).create_session({"log": log, "model": model})
    log.clear()

    return session


def changes(diff) -> List[tuple]:
    described = [(c.component, c.name, c.reason, c.requires) for c in diff.changes]

    return described


def test_each_change_says_whether_nothing_a_reset_or_no_update_applies_it() -> None:
    base = compiled(scaled())

    added = compare_plans(
        compiled(passive(COUNTER, {"count": "$steps.count.count"})), base
    )
    tuned = compare_plans(base, compiled(scaled(factor=3.0)))
    removed = compare_plans(
        base, compiled(passive(COUNTER, {"count": "$steps.count.count"}))
    )
    pruned = compare_plans(
        base, compiled(passive(SCALED, {"count": "$steps.count.count"}))
    )
    other_input = scaled()
    other_input["inputs"] = [
        {
            "type": "WorkflowParameter",
            "name": "value",
            "kind": ["float"],
            "default_value": 1.0,
        }
    ]
    retyped = compare_plans(base, compiled(other_input))

    assert added.kind == PRESERVE and added.compatible
    assert ("step", "$steps.scale", "added", "preserve") in changes(added)
    assert tuned.kind == RESET and not tuned.compatible
    assert changes(tuned) == [("step", "$steps.scale", "params_changed", "reset")]
    assert removed.kind == pruned.kind == RESET
    assert ("step", "$steps.scale", "removed", "reset") in changes(removed)
    assert ("step", "$steps.scale", "newly_pruned", "reset") in changes(pruned)
    assert retyped.kind == UNSUPPORTED
    assert ("input", "$inputs.value", "changed", "unsupported") in changes(retyped)
    assert [c.name for c in retyped.unsupported] == ["$inputs.value"]
    assert all(c.breaking == (c.requires != PRESERVE) for c in retyped.changes)


def test_requires_is_the_one_classification_and_breaking_is_read_from_it() -> None:
    written_before_requires = PlanChange(
        "step", "$steps.a", "params_changed", True, "d"
    )
    compatible = PlanChange("step", "$steps.a", "added", breaking=False)
    reset = PlanChange("step", "$steps.a", "params_changed", requires=RESET)

    assert (written_before_requires.requires, written_before_requires.detail) == (
        UNSUPPORTED,
        "d",
    )
    assert (compatible.requires, compatible.breaking) == (PRESERVE, False)
    assert (reset.breaking, reset.describe()["breaking"]) == (True, True)
    with pytest.raises(ContractError):
        PlanChange("step", "$steps.a", "added", False, requires=PRESERVE)
    with pytest.raises(ContractError):
        PlanChange("step", "$steps.a", "added")


def test_a_changed_or_added_source_is_unsupported_and_operators_reset() -> None:
    counted = [step(Count, "count", value="$sources.ticks.value")]
    groups = [group("counts", count="$steps.count.count")]
    base = compiled(active(counted, groups))

    longer = compare_plans(base, compiled(active(counted, groups, count=9)))
    relabelled = compare_plans(
        base,
        compiled(active([step(Label, "count", value="$sources.ticks.value")], [])),
    )

    assert longer.kind == UNSUPPORTED
    assert changes(longer) == [("source", "$sources.ticks", "changed", "unsupported")]
    assert relabelled.kind == RESET
    assert {c.reason for c in relabelled.changes if c.name == "$steps.count"} >= {
        "block_changed"
    }


def test_nested_step_changes_and_removals_require_a_reset() -> None:
    inner = child(
        [step(Count, "inner_count", value="$inputs.v")],
        {"n": "$steps.inner_count.count"},
        [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}],
    )
    tuned_inner = child(
        [
            step(Count, "inner_count", value="$inputs.v"),
            step(Scale, "inner_scale", value="$inputs.v", factor=4.0),
        ],
        {"n": "$steps.inner_count.count", "s": "$steps.inner_scale.scaled"},
        [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}],
    )
    retuned_inner = child(
        [step(Scale, "inner_scale", value="$inputs.v", factor=5.0)],
        {"s": "$steps.inner_scale.scaled"},
        [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}],
    )

    def outer(definition: dict, output: str) -> dict:
        return passive(
            [nested("inner", definition, v="$inputs.value")],
            {"out": f"$steps.inner.{output}"},
        )

    grown = compiled(outer(tuned_inner, "s"))
    diff = compare_plans(grown, compiled(outer(retuned_inner, "s")))

    assert diff.kind == RESET
    assert ("step", "$steps.inner/inner_scale", "params_changed", "reset") in changes(
        diff
    )
    assert ("step", "$steps.inner/inner_count", "removed", "reset") in changes(diff)
    assert compare_plans(compiled(outer(inner, "n")), grown).kind == PRESERVE


def test_assessment_builds_nothing_and_describes_a_reset_as_json(
    session, log, created
) -> None:
    before = (session.graph_version, dict(session.instances), len(created))

    assessment = session.assess_update(compiled(scaled(factor=3.0)))
    described = json.loads(json.dumps(assessment.describe()))

    assert (session.graph_version, dict(session.instances), len(created)) == before
    assert log == []
    assert assessment.kind == RESET and assessment.base_version == 0
    assert [c["reason"] for c in described["preserve_blocked_by"]] == ["params_changed"]
    assert described["reset_blocked_by"] == []
    reset = described["reset"]
    assert reset["steps_constructed"] == ["$steps.count", "$steps.scale"]
    assert reset["steps_removed"] == [] and reset["sources_kept"] == []
    assert reset["managed_state"]["action"] == "none"
    assert reset["resources"]["$steps.count"] == [
        {"name": "log", "source": "provided:log", "factory": None},
        {"name": "model", "source": "provided:model", "factory": "session"},
    ]


def test_a_compatible_plan_is_preserve_with_a_reset_still_described(session) -> None:
    assessment = session.assess_update(compiled(scaled()))

    assert assessment.kind == PRESERVE
    assert assessment.preserve_blocked_by == assessment.reset_blocked_by == ()
    assert assessment.reset is not None


def test_a_preserving_request_for_a_reset_plan_is_refused_and_points_to_reset(
    session, log
) -> None:
    with pytest.raises(IncompatibleUpdateError, match=r"reset=True") as raised:
        session.prepare_update(compiled(scaled(factor=3.0)))

    assert raised.value.diff.kind == RESET
    assert log == [] and session.graph_version == 0


def test_an_unsupported_reset_is_refused_before_any_constructor_runs(
    session, log
) -> None:
    other_input = scaled(factor=3.0)
    other_input["inputs"] = [{"type": "WorkflowParameter", "name": "value"}]

    with pytest.raises(IncompatibleUpdateError, match=r"\$inputs.value") as raised:
        session.prepare_update(compiled(other_input), reset=True)

    assert raised.value.assessment.kind == UNSUPPORTED
    assert raised.value.assessment.reset is None
    assert log == [] and session.graph_version == 0


def test_replacing_a_caller_resource_is_reported_per_key(session) -> None:
    assessment = session.assess_update(
        compiled(scaled(factor=3.0)), resources={"model": "replacement"}
    )

    assert assessment.reset.resources_replaced == ("model",)
    _, model = assessment.reset.resources["$steps.scale"]
    assert (model.source, model.factory) == ("provided:model", None)


# Managed state -----------------------------------------------------------------

DEFAULTS = {"global": {"n": 0}}


def stateful(**sections: Any) -> dict:
    definition = passive(
        [step(Stateful, "keep", value="$inputs.value")],
        {"kept": "$steps.keep.value"},
        **sections,
    )

    return definition


def state_action(assessment) -> tuple:
    blockers = tuple(blocker.reason for blocker in assessment.blockers)
    action = None if assessment.reset is None else assessment.reset.managed_state.action

    return action, blockers


def test_engine_owned_state_is_replaced_fresh() -> None:
    session = compiled(stateful(state=DEFAULTS)).create_session()

    assessment = session.assess_update(compiled(stateful(state={"global": {"n": 5}})))

    assert state_action(assessment) == ("fresh", ())
    assert assessment.kind == RESET


def test_retained_caller_state_with_changed_defaults_needs_an_isolated_replacement() -> (
    None
):
    backend = InMemoryStateBackend()
    service = ManagedState(backend, namespace="line-1")
    session = compiled(stateful(state=DEFAULTS)).create_session(
        {"managed_state": service}
    )
    changed = compiled(stateful(state={"global": {"n": 5}}))

    kept = session.assess_update(changed)
    same_place = session.assess_update(
        changed, resources={"managed_state": ManagedState(backend, namespace="line-1")}
    )
    isolated = session.assess_update(
        changed, resources={"managed_state": ManagedState(backend, namespace="line-2")}
    )
    unchanged = session.assess_update(compiled(stateful(state=DEFAULTS)))

    assert state_action(kept) == (None, ("retained_state_schema_changed",))
    assert kept.kind == UNSUPPORTED
    assert state_action(same_place) == (None, ("retained_state_schema_changed",))
    assert state_action(isolated) == ("replaced", ())
    assert state_action(unchanged) == ("retained", ())


def test_passing_the_engine_owned_state_back_is_not_isolation() -> None:
    session = compiled(stateful(state=DEFAULTS)).create_session()

    assessment = session.assess_update(
        compiled(stateful(state=DEFAULTS)),
        resources={"managed_state": session.owned_state},
    )

    assert state_action(assessment) == (None, ("state_not_isolated",))
    # The plan is unchanged: a preserving update still applies it.
    assert assessment.kind == PRESERVE and assessment.preserve_blocked_by == ()
    assert [c.reason for c in assessment.reset_blocked_by] == ["state_not_isolated"]


def test_a_source_sharing_engine_state_rules_a_reset_out() -> None:
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": StatefulTicks.type, "name": "ticks"}],
        "steps": [step(Count, "count", value="$sources.ticks.value")],
        "outputs": [group("counts", count="$steps.count.count")],
    }
    session = compiled(definition).create_session({"log": [], "model": None})
    tuned = {
        **definition,
        "steps": [step(Scale, "count", value="$sources.ticks.value")],
        "outputs": [group("counts", count="$steps.count.scaled")],
    }

    assessment = session.assess_update(compiled(tuned))

    assert state_action(assessment) == (None, ("source_shares_state",))
    assert assessment.kind == UNSUPPORTED
