"""Retained payloads in compiled sessions: warnings are honest, effects are effects.

The cache repro (retention design C5): a getter hands out an object its
resource keeps, and a later step mutates it in place. Compilation now reports
it; running shows the engine copies nothing, so the kept object does change.
"""

from typing import Any, Dict

import pytest
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, DICTIONARY_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ControlDefinitionError,
    MutationConflictError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions


class Store:
    """A caller-owned object store shared by the getter and the setter."""

    def __init__(self) -> None:
        self.values: Dict[str, Any] = {}


class Put(Block):
    """Stores its input by reference."""

    type = "test/retained_put@v1"
    retains_inputs = ("value",)
    outputs = {"stored": Output(BOOLEAN_KIND)}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)
        key: str

    def __init__(self, *, store: Store) -> None:
        self.store = store

    def reset_state(self) -> None:
        pass

    def run(self, *, value: Any, key: str) -> dict:
        self.store.values[key] = value
        return {"stored": True}


class Get(Block):
    """Returns the stored object itself."""

    type = "test/retained_get@v1"
    retains_outputs = ("value",)
    prunable = True
    outputs = {"value": Output(DICTIONARY_KIND)}

    class Params(BlockParams):
        key: str
        trigger: Ref()

    def __init__(self, *, store: Store) -> None:
        self.store = store

    def run(self, *, key: str, trigger: Any) -> dict:
        return {"value": self.store.values[key]}


class Bump(Block):
    """Mutates a dictionary in place."""

    type = "test/retained_bump@v1"
    mutates = ("value",)
    outputs = {"count": Output()}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value: Any) -> dict:
        value["count"] = value.get("count", 0) + 1
        return {"count": value["count"]}


CATALOGUE = Catalogue([Put, Get, Bump], namespace="test")


def definition(steps: list, outputs: Dict[str, str], **sections: Any) -> dict:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
        **sections,
    }


GET_THEN_BUMP = definition(
    [
        {"type": Get.type, "name": "get", "key": "k", "trigger": "$inputs.x"},
        {"type": Bump.type, "name": "bump", "value": "$steps.get.value"},
    ],
    {"count": "$steps.bump.count"},
)


def test_the_cache_repro_is_reported_and_the_kept_object_still_changes() -> None:
    store = Store()
    kept = {"count": 0}
    store.values["k"] = kept
    plan = compile_workflow(GET_THEN_BUMP, catalogue=CATALOGUE)
    session = plan.create_session({"store": store})

    first = session.run({"x": 1}).rows()
    second = session.run({"x": 1}).rows()

    [warning] = plan.warnings
    assert "after $steps.get keeps ['output value']" in warning
    assert "The engine adds no copy" in warning
    # The warning changes nothing at run time: each run mutates the kept object.
    assert (first, second) == ([{"count": 1}], [{"count": 2}])
    assert store.values["k"] is kept and kept == {"count": 2}
    session.close()


def test_strict_policy_rejects_the_cache_repro() -> None:
    with pytest.raises(MutationConflictError, match=r"\$steps.get keeps"):
        compile_workflow(
            GET_THEN_BUMP,
            catalogue=CATALOGUE,
            options=CompileOptions(mutation_conflicts="error"),
        )


def controlled(step: dict, **options: Any) -> dict:
    control = {"type": "enable", "steps": [f"$steps.{step['name']}"], **options}
    controlled_definition = definition(
        [step], {}, controls={"switch": {"state": "reset_on_enable", **control}}
    )

    return controlled_definition


PUT = {"type": Put.type, "name": "put", "key": "k", "value": "$inputs.x"}
GET = {"type": Get.type, "name": "get", "key": "k", "trigger": "$inputs.x"}


def test_disabling_a_step_that_keeps_inputs_needs_effect_consent() -> None:
    with pytest.raises(
        ControlDefinitionError, match="kept inputs of \\['\\$steps.put'\\]"
    ):
        compile_workflow(controlled(PUT), catalogue=CATALOGUE)

    compile_workflow(controlled(PUT, suspends_effects=True), catalogue=CATALOGUE)


def test_disabling_a_step_that_only_keeps_outputs_needs_no_consent() -> None:
    plan = compile_workflow(controlled(GET), catalogue=CATALOGUE)

    assert plan.controls.describe()["switch"]["state_classes"] == {"$steps.get": "pure"}
