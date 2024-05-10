import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.operators import Operator
from inference.core.workflows.core_steps.flow_control.condition import (
    BlockManifest,
    ConditionBlock,
)
from inference.core.workflows.entities.types import FlowControl


def test_manifest_parsing_when_valid_input_provided() -> None:
    # given
    data = {
        "type": "Condition",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "==",
        "step_if_true": "$steps.a",
        "step_if_false": "$steps.b",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="Condition",
        name="some",
        left="$inputs.left",
        right="$inputs.right",
        operator="==",
        step_if_true="$steps.a",
        step_if_false="$steps.b",
    )


def test_manifest_parsing_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "==",
        "step_if_true": "$steps.a",
        "step_if_false": "$steps.b",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_invalid_operator_provided() -> None:
    # given
    data = {
        "type": "Condition",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "invalid",
        "step_if_true": "$steps.a",
        "step_if_false": "$steps.b",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_invalid_step_selector_provided() -> None:
    # given
    data = {
        "type": "Condition",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "==",
        "step_if_true": "invalid",
        "step_if_false": "$steps.b",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.asyncio
async def test_run_condition_step() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="cat",
        operator=Operator.EQUAL,
        right="cat",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_2",
    )


@pytest.mark.asyncio
async def test_run_condition_step_for_outputs_with_batch_size_1_and_compared_parameter_is_list() -> (
    None
):
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left=1,
        operator=Operator.IN,
        right=[1, 2, 3],
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_2",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_prefix_to_be_matched_correctly() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_STARTS_WITH,
        right="ab",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_2",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_prefix_not_to_be_matched() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_STARTS_WITH,
        right="abd",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_3",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_postfix_to_be_matched_correctly() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_ENDS_WITH,
        right="cd",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_2",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_postfix_not_to_be_matched() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_ENDS_WITH,
        right="cde",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_3",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_infix_to_be_matched_correctly() -> None:
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_CONTAINS,
        right="bc",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_2",
    )


@pytest.mark.asyncio
async def test_run_condition_step_when_string_infix_not_to_be_matched_correctly() -> (
    None
):
    # given
    step = ConditionBlock()

    # when
    _, flow_control = await step.run_locally(
        left="abcd",
        operator=Operator.STR_CONTAINS,
        right="bce",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )
    # then
    assert flow_control == FlowControl(
        mode="select_step",
        context="$steps.step_3",
    )
