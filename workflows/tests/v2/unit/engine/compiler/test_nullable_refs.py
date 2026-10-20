"""Compiling ``Ref(..., allow_none=True)``: bindings, introspection, plan checks.

The compiler copies the leaf's flag into every binding it creates (selected,
literal cast, list/dict leaf, source parameter). Plans and the editor's
workflow introspection publish it only when set, so default plans describe
exactly as before. A plan whose binding disagrees with its declaration is
rejected, and a plan update never retains a step whose flag changed.
"""

import dataclasses
import json
from typing import Any, Optional

import pytest
from pydantic import StrictStr
from roboflow_workflows.v2.data.batch import EntryLayout
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import ContractError
from roboflow_workflows.v2.engine.introspection import describe_workflow
from roboflow_workflows.v2.engine.plan import Binding, InputPort
from roboflow_workflows.v2.engine.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)
from roboflow_workflows.v2.engine.updates import compare_plans

from tests.v2.support.compilation.fixture_blocks import (
    batch_input,
    parameter,
    step,
    workflow,
)


def nullable_block(allow_none: bool) -> type:
    """One block type; ``allow_none`` decides its selector leaves' flag."""

    class Take(Block):
        type = "test/take@v1"
        outputs = {"out": Output(STRING_KIND, source="value")}

        class Params(BlockParams):
            value: Optional[StrictStr] | Ref(
                STRING_KIND, batch="if_varying", allow_none=allow_none  # noqa: F821
            )
            listed: list[StrictStr | Ref(STRING_KIND, allow_none=allow_none)] = []
            mapped: dict[str, StrictStr | Ref(STRING_KIND, allow_none=allow_none)] = {}
            cast: Optional[StrictStr] | Ref(
                STRING_KIND, batch="always", allow_none=allow_none  # noqa: F821
            ) = None
            plain: Optional[StrictStr] | Ref(STRING_KIND) = None

        def run(self, **arguments: Any) -> Any:
            return [{"out": value} for value in arguments["value"]]

    return Take


class Labelled(Source):
    """A source with an opted-in parameter selector."""

    type = "test/labelled@v1"
    outputs = {"label": SourceOutput(STRING_KIND)}

    class Params(SourceParams):
        label: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = None

    def open(self, *, label) -> None:
        self.label = label

    def read(self) -> Optional[Emission]:
        return None

    def close(self) -> None:
        pass


DEFINITION = workflow(
    [
        step(
            "take",
            "take",
            value="$inputs.value",
            listed=["a", "$inputs.word"],
            mapped={"k": "$inputs.word"},
            cast="literal",
            plain="$inputs.word",
        )
    ],
    {"out": "$steps.take.out"},
    inputs=[batch_input("value", kind=["string"]), parameter("word", kind=["string"])],
)


def _compile(allow_none: bool) -> Any:
    block = nullable_block(allow_none)
    plan = compile_workflow(
        DEFINITION, catalogue=Catalogue([block], kinds=[STRING_KIND])
    )

    return plan


def _flags(plan: Any) -> dict:
    (planned,) = plan.steps
    flags = {
        (binding.field, binding.position, binding.mode): binding.allow_none
        for binding in planned.bindings
    }

    return flags


def test_every_binding_copies_its_leaf_flag() -> None:
    flags = _flags(_compile(True))

    assert flags == {
        ("value", (), "element"): True,
        ("listed", (1,), "constant"): True,
        ("mapped", ("k",), "constant"): True,
        ("cast", (), "constant"): True,  # the literal the compiler casts
        ("plain", (), "constant"): False,
    }
    assert set(_flags(_compile(False)).values()) == {False}


def test_both_plan_serializers_publish_true_only_where_it_is_set() -> None:
    plan = _compile(True)
    (planned,) = plan.steps
    bindings = {binding.field: binding.describe() for binding in planned.bindings}
    (editor_step,) = describe_workflow(plan)["steps"]
    editor = {
        name: [use.get("allow_none") for use in parameter["selectors"]]
        for name, parameter in editor_step["parameters"].items()
        if parameter["selectors"]
    }

    assert {name: item.get("allow_none") for name, item in bindings.items()} == {
        "value": True,
        "listed": True,
        "mapped": True,
        "cast": True,
        "plain": None,
    }
    assert editor == {
        "value": [True],
        "listed": [True],
        "mapped": [True],
        "cast": [True],
        "plain": [None],
    }


def test_default_plans_describe_without_the_flag() -> None:
    plan = _compile(False)
    (planned,) = plan.steps

    assert "allow_none" not in json.dumps(describe_workflow(plan), default=repr)
    for binding in planned.bindings:
        assert "allow_none" not in binding.describe()


def test_a_source_parameter_binding_copies_the_flag() -> None:
    definition = {
        "version": "2.0",
        "inputs": [parameter("label", kind=["string"])],
        "sources": [{"type": Labelled.type, "name": "cam", "label": "$inputs.label"}],
        "steps": [],
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "G",
                "anchor": "$sources.cam.label",
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": "label",
                        "selector": "$sources.cam.label",
                    }
                ],
            }
        ],
    }
    plan = compile_workflow(
        definition, catalogue=Catalogue([], kinds=[STRING_KIND], sources=[Labelled])
    )

    (binding,) = plan.sources["cam"].bindings
    assert binding.allow_none is True
    assert binding.describe()["allow_none"] is True


def test_a_binding_disagreeing_with_its_declaration_is_rejected() -> None:
    (planned,) = _compile(False).steps
    tampered = tuple(
        dataclasses.replace(binding, allow_none=binding.field == "value")
        for binding in planned.bindings
    )

    with pytest.raises(ContractError, match="copies allow_none=True"):
        dataclasses.replace(planned, bindings=tampered)


def test_a_binding_needs_a_bool_flag() -> None:
    with pytest.raises(ContractError, match="allow_none must be a bool"):
        Binding(
            field="value",
            position=(),
            selector="$inputs.value",
            source=InputPort(name="value"),
            source_layout=EntryLayout(),
            mode="constant",
            allow_none=1,
        )


def test_an_update_retains_an_unchanged_flag_and_resets_a_changed_one() -> None:
    block = nullable_block(True)
    catalogue = Catalogue([block], kinds=[STRING_KIND])
    old = compile_workflow(DEFINITION, catalogue=catalogue)
    same = compile_workflow(DEFINITION, catalogue=catalogue)
    flipped = _compile(False)

    unchanged = compare_plans(old, same)
    changed = compare_plans(old, flipped)

    assert unchanged.kind != "reset" and unchanged.changes == ()
    assert changed.kind == "reset"
    reasons = {
        change.reason for change in changed.changes if change.name == "$steps.take"
    }
    assert {"contract_changed", "bindings_changed"} <= reasons
