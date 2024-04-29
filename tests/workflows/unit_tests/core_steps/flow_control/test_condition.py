import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.flow_control.condition import (
    BlockManifest,
)


def test_manifest_parsing_when_valid_input_provided() -> None:
    # given
    data = {
        "type": "Condition",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "equal",
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
        operator="equal",
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
        "operator": "equal",
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
        "operator": "equal",
        "step_if_true": "invalid",
        "step_if_false": "$steps.b",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)
