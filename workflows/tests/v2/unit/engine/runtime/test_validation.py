"""Decision 018: each logical invocation is validated before any call of its step.

``BlockSpec.validate_resolved_arguments`` checks selected payloads by their
kinds, shared field constraints and the author's field/model validators. It
never converts or copies: blocks receive the selected objects themselves.
"""

from typing import Annotated

import pytest
from pydantic import Field
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ResolvedParameterError,
    StepExecutionError,
)

from tests.v2.support.execution.blocks import (
    InvertMany,
    StringGroups,
    TextOrMapping,
    Thresholds,
)
from tests.v2.support.execution.plans import BATCH, SCALAR, PlanBuilder


def calls(session, name):
    return session.instances[(name,)].calls


def thresholds_plan(**params):
    plan = (
        PlanBuilder()
        .input("low", BATCH)
        .input("high", BATCH)
        .step(Thresholds, "span", at=BATCH, **params)
        .output("span", "$steps.span.span")
        .build()
    )

    return plan


def test_field_validator_rejects_a_selected_value_before_any_call() -> None:
    session = thresholds_plan(high="$inputs.high").create_session()

    with pytest.raises(StepExecutionError, match="high must be at most 100") as caught:
        session.run({"low": [0, 0], "high": [5, 500]})

    assert caught.value.index == (1,)
    assert isinstance(caught.value.__cause__, ResolvedParameterError)
    assert caught.value.__cause__.field_path == ("high",)
    assert calls(session, "span") == []


def test_model_validator_sees_the_whole_resolved_invocation() -> None:
    session = thresholds_plan(low="$inputs.low", high="$inputs.high").create_session()

    rows = session.run({"low": [1, 2], "high": [5, 6]}).rows()
    with pytest.raises(StepExecutionError, match="low must not exceed high") as caught:
        session.run({"low": [1, 9], "high": [5, 6]})

    assert rows == [{"span": 4}, {"span": 4}]
    assert caught.value.index == (1,)
    assert len(calls(session, "span")) == 2


def test_batch_delivering_step_is_validated_per_logical_invocation() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(InvertMany, "invert", at=BATCH, values="$inputs.values", offset=1)
        .build()
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match="parameter values") as caught:
        session.run({"values": [1, "two", 3]})

    assert caught.value.index == (1,)
    assert calls(session, "invert") == []


def test_selected_dict_and_list_keep_identity_beside_unrelated_literals() -> None:
    plan = (
        PlanBuilder()
        .input("payload", SCALAR)
        .input("items", SCALAR)
        .step(TextOrMapping, "pick", payload="$inputs.payload", items="$inputs.items")
        .output("payload", "$steps.pick.payload")
        .output("items", "$steps.pick.items")
        .build()
    )
    payload, items = {"a": 1}, [1, 2, 3]

    result = plan.create_session().run({"payload": payload, "items": items})

    # A literal "str" or "Tuple[int, int]" alternative neither rejects nor
    # narrows the selected dict and three-element list.
    assert result.outputs.data["payload"] is payload
    assert result.outputs.data["items"] is items


def test_literal_cast_into_a_group_is_not_checked_against_the_group_kind() -> None:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "parents"},
            {"type": "WorkflowParameter", "name": "label", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": "test/string_groups@v1",
                "name": "groups",
                "parent": "$inputs.parents",
                "groups": {"literal": 8, "selected": "$inputs.label"},
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "members",
                "selector": "$steps.groups.members",
            }
        ],
    }
    plan = compile_workflow(
        definition, catalogue=Catalogue([StringGroups], kinds=(STRING_KIND,))
    )
    session = plan.create_session()

    rows = session.run({"parents": [1, 2], "label": "x"}).rows()

    groups = [call["groups"] for call in calls(session, "groups")]
    assert [group["literal"].indices for group in groups] == [((0, 0),), ((1, 0),)]
    assert rows == [{"members": {"literal": [8], "selected": ["x"]}}] * 2


class AtLeastTwo(Block):
    """Bound written on the selector arm; the literal arm accepts any int."""

    type = "test/at_least_two@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        count: Annotated[Ref(INTEGER_KIND), Field(ge=2)] | int

    def __init__(self):
        self.calls = []

    def run(self, *, count) -> dict:
        self.calls.append(count)

        return {"count": count}


def test_selector_arm_bound_rejects_one_invocation_before_any_call() -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "counts"}],
        "steps": [{"type": AtLeastTwo.type, "name": "at", "count": "$inputs.counts"}],
        "outputs": [],
    }
    plan = compile_workflow(definition, catalogue=Catalogue([AtLeastTwo]))
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match="greater than or equal") as caught:
        session.run({"counts": [5, 1]})

    assert caught.value.index == (1,)
    assert caught.value.__cause__.field_path == ("count",)
    assert calls(session, "at") == []


def test_selected_group_child_of_wrong_kind_names_field_and_child() -> None:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "parents"},
            {"type": "WorkflowParameter", "name": "label"},
        ],
        "steps": [
            {
                "type": "test/string_groups@v1",
                "name": "groups",
                "parent": "$inputs.parents",
                "groups": {"selected": "$inputs.label"},
            }
        ],
        "outputs": [],
    }
    plan = compile_workflow(
        definition, catalogue=Catalogue([StringGroups], kinds=(STRING_KIND,))
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match=r"groups\.selected") as caught:
        session.run({"parents": [1], "label": 5})

    assert caught.value.index == (0,)
    assert "child [0, 0]" in str(caught.value)
    assert calls(session, "groups") == []
