"""Literals at batch-only and Group positions are cast like selected scalars."""

from typing import Dict, Optional

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import INTEGER_KIND
from roboflow_workflows.execution_engine.v2.plan import Constant, InputPort

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    parameter,
    workflow,
)


class LiteralBatch(Block):
    """Batch-only, scalar-or-batch and per-call fields that all accept literals."""

    type = "test/literal_batch@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: int | Ref(INTEGER_KIND, batch="always") = 1
        maybe: int | Ref(INTEGER_KIND, batch="if_varying") = 2
        plain: Optional[int] | Ref(INTEGER_KIND) = None

    def run(self, *, value, maybe, plain) -> list:
        return []


class LiteralGroups(Block):
    """Parent reference plus whole and compound groups accepting literals."""

    type = "test/literal_groups@v1"
    outputs = {"kept": Output(preserve="groups"), "whole": Output(preserve="single")}

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, int | Group(INTEGER_KIND)]
        single: int | Group(INTEGER_KIND) = 5

    def run(self, *, parent, groups, single) -> dict:
        return {}


LITERALS = Catalogue.merge(CATALOGUE, Catalogue([LiteralBatch, LiteralGroups]))


def _plan(steps, inputs=None):
    plan = compile_workflow(workflow(steps, inputs=inputs), catalogue=LITERALS)

    return plan


def test_batch_only_literals_and_defaults_become_constant_batch_bindings() -> None:
    plan = _plan(
        [
            {"type": "test/literal_batch@v1", "name": "literal", "value": 7},
            {"type": "test/literal_batch@v1", "name": "default"},
        ],
        inputs=[],
    )

    for name, value in (("literal", 7), ("default", 1)):
        step = plan.step((name,))
        assert [
            (b.field_path, b.source, b.mode, b.batch, b.selector) for b in step.bindings
        ] == [(("value",), Constant(value), "constant", "always", "")]
        assert axes_of(step.invocation_layout) == []
        assert step.delivers_batches


def test_scalar_or_batch_and_per_call_literals_stay_plain() -> None:
    plan = _plan(
        [
            {
                "type": "test/literal_batch@v1",
                "name": "s",
                "value": "$inputs.values",
                "maybe": 3,
                "plain": 4,
            },
        ],
        inputs=[batch_input("values", kind=["integer"])],
    )

    step = plan.step(("s",))
    assert [b.field_path for b in step.bindings] == [("value",)]
    assert step.params.maybe == 3 and step.params.plain == 4


def test_group_literals_are_cast_per_parent_beside_selected_leaves() -> None:
    plan = _plan(
        [
            {
                "type": "test/literal_groups@v1",
                "name": "g",
                "parent": "$inputs.values",
                "groups": {"literal": 8, "selected": "$inputs.offset"},
            },
        ],
        inputs=[
            batch_input("values"),
            parameter("offset", default=9, kind=["integer"]),
        ],
    )

    step = plan.step(("g",))
    groups = step.bindings_for("groups")
    assert [(b.position, b.source, b.mode) for b in groups] == [
        (("literal",), Constant(8), "constant_group"),
        (("selected",), InputPort("offset"), "constant_group"),
    ]
    assert {axes_of(b.cast_layout)[-1] for b in groups} == {"g/groups/cast"}
    assert axes_of(step.outputs["kept"].layout) == ["inputs", "g/groups/cast"]
    single = step.binding_for("single")
    assert (single.source, single.mode) == (Constant(5), "constant_group")
    assert axes_of(step.outputs["whole"].layout) == ["inputs", "g/single/cast"]
    assert [b.field for b in step.bindings] == ["parent", "groups", "groups", "single"]
