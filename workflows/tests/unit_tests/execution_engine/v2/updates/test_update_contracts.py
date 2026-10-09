"""Contracts a graph update keeps: one managed state, unchanged block contracts.

Added steps share the session's managed-state service exactly as steps do at
session creation. A retained instance is reused only when the block contract
and the selected implementation contract are unchanged, also for validated
plans built by hand rather than by the compiler.
"""

from dataclasses import replace
from typing import Any

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
)
from roboflow_workflows.execution_engine.v2.resources import ResourceSpec
from roboflow_workflows.execution_engine.v2.state import ManagedState
from roboflow_workflows.execution_engine.v2.updates import compare_plans

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    CATALOGUE,
    Label,
    Stateful,
    passive,
    step,
)


def untyped(steps: list, outputs: dict) -> dict:
    """``passive`` without input kinds, for a catalogue that declares none."""
    definition = passive(steps, outputs)
    definition["inputs"] = [{"type": "WorkflowParameter", "name": "value"}]

    return definition


class ScopedStateful(Stateful):
    """``Stateful`` in the ``new`` namespace; records every construction."""

    type = "test/update_scoped_stateful@v1"
    constructed = 0

    def __init__(self, *, managed_state: Any) -> None:
        super().__init__(managed_state=managed_state)
        type(self).constructed += 1


class EmptySensitive(Block):
    """Counts the calls that reach it."""

    type = "test/update_empty_sensitive@v1"
    outputs = {"calls": Output()}

    class Params(BlockParams):
        value: object | Ref()

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, value) -> dict:
        self.calls += 1
        return {"calls": self.calls}


SCOPED = Catalogue.merge(
    Catalogue([Stateful, EmptySensitive]),
    Catalogue([ScopedStateful], namespace="new"),
)
OLD = untyped(
    [step(Stateful, "old", value="$inputs.value")], {"old": "$steps.old.value"}
)
WITH_NEW = untyped(
    [
        step(Stateful, "old", value="$inputs.value"),
        step(ScopedStateful, "new", value="$inputs.value"),
    ],
    {"old": "$steps.old.value", "new": "$steps.new.value"},
)


@pytest.fixture(autouse=True)
def reset_constructed() -> None:
    ScopedStateful.constructed = 0


def compiled(definition: dict, catalogue: Catalogue = SCOPED) -> Any:
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


@pytest.mark.parametrize("value", ["session_service", None])
def test_added_step_shares_the_session_managed_state(value: Any) -> None:
    session = compiled(OLD).create_session()
    service = session.managed_state
    scoped = service if value == "session_service" else None

    session.update(compiled(WITH_NEW), resources={"new.managed_state": scoped})

    assert session.instances[("new",)].state is service
    assert session.instances[("old",)].state is service
    assert session.run({"value": 1.0}).rows() == [{"old": 1.0, "new": 1.0}]
    session.close()


def test_scoped_none_given_at_creation_still_means_the_session_service() -> None:
    session = compiled(OLD).create_session({"new.managed_state": None})

    session.update(compiled(WITH_NEW))

    assert session.instances[("new",)].state is session.managed_state
    session.close()


def test_added_step_with_another_managed_state_is_rejected_before_construction() -> (
    None
):
    session = compiled(OLD).create_session()
    service, retained = session.managed_state, session.instances[("old",)]
    other = ManagedState(namespace="other")

    with pytest.raises(ResourceError, match="not the session's managed state"):
        session.update(compiled(WITH_NEW), resources={"new.managed_state": other})

    assert ScopedStateful.constructed == 0
    assert session.graph_version == 0
    assert session.managed_state is service
    assert session.instances == {("old",): retained}
    assert retained.state is service
    assert session.run({"value": 2.0}).rows() == [{"old": 2.0}]
    session.close()
    other.close()


def hand_built_empty_sensitive() -> Any:
    plan = compiled(
        {
            "version": "2.0",
            "inputs": [{"name": "values", "type": "WorkflowBatchInput"}],
            "steps": [step(EmptySensitive, "probe", value="$inputs.values")],
            "outputs": [{"name": "calls", "selector": "$steps.probe.calls"}],
        }
    )

    return plan


def test_hand_built_plan_with_another_block_contract_is_rejected() -> None:
    old = hand_built_empty_sensitive()
    probe = old.steps[0]
    changed = replace(
        probe, spec=replace(probe.spec, accepts_empty=not probe.spec.accepts_empty)
    )
    new = replace(old, steps=(changed,))
    session = old.create_session()
    session.run({"values": [None]})
    calls = session.instances[("probe",)].calls

    diff = compare_plans(old, new)
    with pytest.raises(IncompatibleUpdateError):
        session.update(new)
    session.run({"values": [None]})

    assert [(c.name, c.reason) for c in diff.breaking] == [
        ("$steps.probe", "contract_changed")
    ]
    assert session.plan is old
    assert session.instances[("probe",)].calls == calls


def test_hand_built_plan_with_another_implementation_contract_is_rejected() -> None:
    definition = passive(
        [step(Label, "label", value="$inputs.value")], {"label": "$steps.label.label"}
    )
    old = compiled(definition, catalogue=CATALOGUE)
    label = old.steps[0]
    selected = label.implementation.spec
    changes = (
        {"phase_overlap": not selected.phase_overlap},
        {
            "resources": (
                *selected.resources,
                ResourceSpec("extra", required=False, default=None),
            )
        },
    )

    for change in changes:
        # A valid plan selects one of its block's implementations, so a hand
        # edit of the selection also edits the block contract that lists it.
        changed = replace(selected, **change)
        spec = replace(
            label.spec,
            implementations=tuple(
                changed if item.name == selected.name else item
                for item in label.spec.implementations
            ),
        )
        implementation = replace(label.implementation, spec=changed)
        new = replace(
            old, steps=(replace(label, spec=spec, implementation=implementation),)
        )
        diff = compare_plans(old, new)

        assert {(c.name, c.reason) for c in diff.breaking} == {
            ("$steps.label", "contract_changed"),
            ("$steps.label", "implementation_changed"),
        }


def test_recompiled_plan_with_the_same_contracts_is_compatible() -> None:
    old = hand_built_empty_sensitive()

    diff = compare_plans(old, hand_built_empty_sensitive())

    assert diff.compatible
    assert diff.retained == (("probe",),)
