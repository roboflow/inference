"""A graph update compares retention like the rest of the block contract.

A retained instance is reused only when its block contract is unchanged, so a
plan whose step keeps other inputs or outputs is a breaking update.
"""

from dataclasses import replace
from typing import Any

import pytest
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import IncompatibleUpdateError
from roboflow_workflows.v2.engine.updates import compare_plans


class Holder(Block):
    """Keeps the last value it saw and returns it."""

    type = "test/update_holder@v1"
    retains_inputs = ("value",)
    outputs = {"held": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self) -> None:
        self.held = None

    def run(self, *, value: Any) -> dict:
        self.held = value
        return {"held": value}


CATALOGUE = Catalogue([Holder], namespace="test")
DEFINITION = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [{"type": Holder.type, "name": "hold", "value": "$inputs.value"}],
    "outputs": [{"type": "JsonField", "name": "held", "selector": "$steps.hold.held"}],
}


def compiled() -> Any:
    plan = compile_workflow(DEFINITION, catalogue=CATALOGUE)

    return plan


def with_retention(plan: Any, **retention: tuple) -> Any:
    """``plan`` with its one step's block keeping other payloads."""
    [hold] = plan.steps
    changed = replace(
        plan, steps=(replace(hold, spec=replace(hold.spec, **retention)),)
    )

    return changed


@pytest.mark.parametrize(
    "retention",
    [{"retains_inputs": ()}, {"retains_outputs": ("held",)}],
    ids=["input-no-longer-kept", "output-now-kept"],
)
def test_changed_retention_is_a_contract_change(retention: dict) -> None:
    old = compiled()
    new = with_retention(old, **retention)

    diff = compare_plans(old, new)

    assert [(change.name, change.reason) for change in diff.breaking] == [
        ("$steps.hold", "contract_changed")
    ]


def test_a_session_rejects_an_update_with_changed_retention() -> None:
    old = compiled()
    session = old.create_session()
    session.run({"value": 1})
    holder = session.instances[("hold",)]

    with pytest.raises(IncompatibleUpdateError):
        session.update(with_retention(old, retains_outputs=("held",)))

    assert session.plan is old
    assert session.instances[("hold",)] is holder
    session.close()


def test_the_same_retention_is_a_compatible_update() -> None:
    old = compiled()
    session = old.create_session()
    holder = session.instances[("hold",)]

    diff = compare_plans(old, compiled())
    session.update(compiled())

    assert diff.compatible
    assert session.instances[("hold",)] is holder
    session.close()
